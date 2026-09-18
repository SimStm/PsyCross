#include "PsyX_GPU.h"

#include "PsyX/PsyX_public.h"
#include "PsyX/PsyX_globals.h"
#include "PsyX/PsyX_render.h"

#include "../PsyX_main.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <SDL_timer.h>

#define GET_TPAGE_FORMAT(tpage) ((TexFormat)((tpage >> 7) & 0x3))
#define GET_TPAGE_BLEND(tpage)  ((BlendMode)(((tpage >> 5) & 3) + 1))

#define GET_TPAGE_DITHER(tpage) ((tpage >> 9) & 0x1)

#define GET_CLUT_X(clut)        ((clut & 0x3F) << 4)
#define GET_CLUT_Y(clut)        (clut >> 6)

OT_TAG prim_terminator = { -1, 0 }; // P_TAG with zero primLength

DISPENV currentDispEnv;
DISPENV activeDispEnv;
DRAWENV activeDrawEnv;

static const char* currentSplitDebugText = nullptr;
TextureID overrideTexture = 0;
int overrideTextureWidth = 0;
int overrideTextureHeight = 0;

static const int MAX_TEXTURE_OVERRIDES = 128;
static const int OVERRIDE_TEXTURE_UV_SIZE = 255;

struct RegisteredTextureOverride
{
	int id;
	PsyXTextureOverride descriptor;
};

RegisteredTextureOverride g_textureOverrides[MAX_TEXTURE_OVERRIDES];
int g_textureOverrideCount = 0;
int g_nextTextureOverrideId = 1;
int g_textureOverridesEnabled = 1;

TextureID g_automaticOverrideTexture = 0;
int g_automaticOverrideTextureWidth = 0;
int g_automaticOverrideTextureHeight = 0;

int g_GPUDisabledState = 0;
int g_DrawPrimMode = 0;

// Compact cutout coverage for override images. The renderer discards override
// fragments whose alpha is below 0.5 (item 01); picking has no GPU feedback, so
// the same coverage is kept on the CPU as a one-bit-per-texel mask and sampled
// with the shader's own override UV mapping. Only textures with transparency are
// retained, and only up to a bounded size.
namespace
{
const int MAX_CUTOUT_MASKS = 32;
const int MAX_CUTOUT_MASK_TEXELS = 256 * 256;

struct CutoutMask
{
	TextureID textureId;
	int width;
	int height;
	unsigned char* bits;		// 1 = alpha >= 128 (binary cutout coverage)
	unsigned char* coverage;	// 1 = alpha > 0 (proportional-mode coverage)
};

CutoutMask g_cutoutMasks[MAX_CUTOUT_MASKS];
int g_cutoutMaskCount = 0;

void RetainCutoutMask(TextureID textureId, int width, int height, const unsigned char* rgbaPixels)
{
	if (textureId == 0 || !rgbaPixels || width <= 0 || height <= 0 ||
		width * height > MAX_CUTOUT_MASK_TEXELS || g_cutoutMaskCount >= MAX_CUTOUT_MASKS)
	{
		return;
	}

	const int texels = width * height;
	const int byteCount = (texels + 7) / 8;
	unsigned char* bits = (unsigned char*)malloc(byteCount);
	unsigned char* coverage = (unsigned char*)malloc(byteCount);
	if (!bits || !coverage)
	{
		free(bits);
		free(coverage);
		return;
	}

	memset(bits, 0, byteCount);
	memset(coverage, 0, byteCount);
	for (int i = 0; i < texels; ++i)
	{
		const unsigned char alpha = rgbaPixels[i * 4 + 3];
		if (alpha >= 128)
			bits[i >> 3] |= (unsigned char)(1 << (i & 7));
		if (alpha > 0)
			coverage[i >> 3] |= (unsigned char)(1 << (i & 7));
	}

	CutoutMask& mask = g_cutoutMasks[g_cutoutMaskCount++];
	mask.textureId = textureId;
	mask.width = width;
	mask.height = height;
	mask.bits = bits;
	mask.coverage = coverage;
}

void ReleaseCutoutMask(TextureID textureId)
{
	for (int i = 0; i < g_cutoutMaskCount; ++i)
	{
		if (g_cutoutMasks[i].textureId != textureId)
			continue;

		free(g_cutoutMasks[i].bits);
		free(g_cutoutMasks[i].coverage);
		memmove(&g_cutoutMasks[i], &g_cutoutMasks[i + 1], sizeof(CutoutMask) * (g_cutoutMaskCount - i - 1));
		--g_cutoutMaskCount;
		return;
	}
}

int HasCutoutMask(TextureID textureId)
{
	for (int i = 0; i < g_cutoutMaskCount; ++i)
	{
		if (g_cutoutMasks[i].textureId == textureId)
			return 1;
	}
	return 0;
}

// Normalised override UV in [0,1], matching the 32-bit RGBA shader's
// tc = v_texcoord.xy / 255 over the source region. Returns 1 (drawable),
// 0 (cut out) or -1 when no mask is retained for the texture. In binary cutout
// mode a texel below half alpha is a hole; in proportional mode only a fully
// transparent texel is a hole, mirroring overrideAlphaMode in the shader.
int SampleCutoutMask(TextureID textureId, float u, float v, int proportional)
{
	for (int i = 0; i < g_cutoutMaskCount; ++i)
	{
		if (g_cutoutMasks[i].textureId != textureId)
			continue;

		const CutoutMask& mask = g_cutoutMasks[i];
		int x = (int)(u * mask.width);
		int y = (int)(v * mask.height);
		if (x < 0) x = 0;
		if (x >= mask.width) x = mask.width - 1;
		if (y < 0) y = 0;
		if (y >= mask.height) y = mask.height - 1;

		const int index = y * mask.width + x;
		const unsigned char* source = proportional ? mask.coverage : mask.bits;
		return (source[index >> 3] >> (index & 7)) & 1;
	}

	return -1;
}
}

struct GPUDrawSplit
{
	DRAWENV			drawenv;
	DISPENV			dispenv;

	BlendMode		blendMode;

	TexFormat		texFormat;
	TextureID		textureId;

	int				drawPrimMode;

	u_short			startVertex;
	u_short			numVerts;

	int				textureOverridden;
	unsigned short		inspectorSourceU;
	unsigned short		inspectorSourceV;
	unsigned short		inspectorSourceWidth;
	unsigned short		inspectorSourceHeight;

	const char*		debugText;
};

#define MAX_DRAW_SPLITS	 4096

GrVertex g_vertexBuffer[MAX_VERTEX_BUFFER_SIZE];
GPUDrawSplit g_splits[MAX_DRAW_SPLITS];

int g_vertexIndex = 0;
int g_splitIndex = 0;
PsyXRenderStats g_psyxRenderStats = { 0, 0 };

static const int MAX_INSPECTOR_RANGES = 2048;

struct InspectorRange
{
	uintptr_t begin;
	uintptr_t end;
	char label[PSYX_INSPECTOR_LABEL_LENGTH];
	PsyXInspectorObject object;
};

InspectorRange g_inspectorRanges[MAX_INSPECTOR_RANGES];
int g_inspectorRangeCount = 0;
// Vertices actually attributed to each registered range in the current frame.
// A zero count means the range was registered but no parsed primitive mapped to
// it, so the source is unreachable by picking and must be reported as such.
int g_inspectorRangeVertexCount[MAX_INSPECTOR_RANGES];
int g_vertexInspectorRange[MAX_VERTEX_BUFFER_SIZE];
int g_inspectorPickPending = 0;
// Set for the next resolved pick only: the selection keeps the parent of the
// picked component instead of the part itself.
int g_inspectorWholeObjectPick = 0;
int g_inspectorPickX = 0;
int g_inspectorPickY = 0;
PsyXInspectorSelection g_inspectorSelection = {};
static PsyXInspectorTriangle g_inspectorTriangles[4096];
static int g_inspectorTriangleCount = 0;

int g_automaticOverrideSourceU = 0;
int g_automaticOverrideSourceV = 0;
int g_automaticOverrideSourceWidth = 0;
int g_automaticOverrideSourceHeight = 0;

void PsyX_ResetRenderStats()
{
	g_inspectorTriangleCount = 0;
	g_psyxRenderStats.vertexCount = 0;
	g_psyxRenderStats.drawSplitCount = 0;
}

void PsyX_RecordRenderStats()
{
	g_psyxRenderStats.vertexCount += g_vertexIndex;
	g_psyxRenderStats.drawSplitCount += g_splitIndex;
}

void PsyX_GetRenderStats(PsyXRenderStats* stats)
{
	if (stats)
		*stats = g_psyxRenderStats;
}

void PsyX_Inspector_RequestPick(int windowX, int windowY)
{
	g_inspectorPickX = windowX;
	g_inspectorPickY = windowY;
	g_inspectorPickPending = 1;
}

void PsyX_Inspector_SetWholeObjectPick(int enabled)
{
	g_inspectorWholeObjectPick = enabled != 0;
}

void PsyX_Inspector_ClearSelection()
{
	g_inspectorSelection = {};
	g_inspectorTriangleCount = 0;
	g_inspectorPickPending = 0;
}

int PsyX_Inspector_GetTriangle(int index, PsyXInspectorTriangle* triangle)
{
	if (!triangle || index < 0 || index >= g_inspectorTriangleCount) return 0;
	*triangle = g_inspectorTriangles[index];
	return 1;
}

int PsyX_Inspector_GetSelection(PsyXInspectorSelection* selection)
{
	if (!selection)
		return 0;
	*selection = g_inspectorSelection;
	return selection->valid;
}

int PsyX_Inspector_GetRangeInfo(int index, char* label, int labelCapacity, int* vertexCount)
{
	if (index < 0 || index >= g_inspectorRangeCount)
		return 0;

	if (label && labelCapacity > 0)
	{
		strncpy(label, g_inspectorRanges[index].label, labelCapacity - 1);
		label[labelCapacity - 1] = '\0';
	}
	if (vertexCount)
		*vertexCount = g_inspectorRangeVertexCount[index];

	return 1;
}

void PsyX_Inspector_RegisterPrimitiveRange(const void* begin, const void* end, const char* label)
{
	if (!begin || !end || !label || g_inspectorRangeCount >= MAX_INSPECTOR_RANGES)
		return;

	const uintptr_t beginAddress = reinterpret_cast<uintptr_t>(begin);
	const uintptr_t endAddress = reinterpret_cast<uintptr_t>(end);
	if (endAddress <= beginAddress)
		return;

	const int rangeIndex = g_inspectorRangeCount++;
	g_inspectorRangeVertexCount[rangeIndex] = 0;

	InspectorRange& range = g_inspectorRanges[rangeIndex];
	memset(&range.object, 0, sizeof(range.object));
	range.begin = beginAddress;
	range.end = endAddress;
	strncpy(range.label, label, sizeof(range.label) - 1);
	range.label[sizeof(range.label) - 1] = '\0';
}

void PsyX_Inspector_RegisterObjectRange(const void* begin, const void* end, const char* label, const PsyXInspectorObject* object)
{
	const int before = g_inspectorRangeCount;
	PsyX_Inspector_RegisterPrimitiveRange(begin, end, label);
	if (object && g_inspectorRangeCount > before)
	{
		g_inspectorRanges[before].object = *object;
		g_inspectorRanges[before].object.key[127] = '\0';
		g_inspectorRanges[before].object.modelName[63] = '\0';
	}
}

static int FindInspectorRange(const void* packet)
{
	const uintptr_t address = reinterpret_cast<uintptr_t>(packet);
	for (int i = g_inspectorRangeCount - 1; i >= 0; --i)
	{
		if (address >= g_inspectorRanges[i].begin && address < g_inspectorRanges[i].end)
			return i;
	}
	return -1;
}

unsigned int PsyX_CreateRGBATexture(int width, int height, const unsigned char* rgbaPixels)
{
	if (width <= 0 || height <= 0 || !rgbaPixels)
		return 0;

	// Fully opaque overrides get mipmaps to stop minification shimmer.
	// Textures with transparency keep plain filtering so mip averaging cannot
	// bleed the transparent colour into cutout edges.
	bool fullyOpaque = true;
	const unsigned char* pixel = rgbaPixels;
	for (int i = 0; i < width * height; ++i, pixel += 4)
	{
		if (pixel[3] != 255)
		{
			fullyOpaque = false;
			break;
		}
	}

	const unsigned int textureId = fullyOpaque
		? GR_CreateRGBATextureMipmapped(width, height, (u_char*)rgbaPixels)
		: GR_CreateRGBATexture(width, height, (u_char*)rgbaPixels);

	// Textures with transparency are the ones a cutout can discard; keep their
	// coverage so picking can reject the same texels the shader discards.
	if (!fullyOpaque && textureId != 0)
		RetainCutoutMask(textureId, width, height, rgbaPixels);

	return textureId;
}

void PsyX_DestroyRGBATexture(unsigned int textureId)
{
	ReleaseCutoutMask(textureId);
	if (textureId != 0)
		GR_DestroyTexture(textureId);
}

int PsyX_RegisterTextureOverride(const PsyXTextureOverride* textureOverride)
{
	if (!textureOverride || textureOverride->textureId == 0 ||
		textureOverride->width == 0 || textureOverride->height == 0 ||
		g_textureOverrideCount >= MAX_TEXTURE_OVERRIDES)
	{
		return 0;
	}

	RegisteredTextureOverride* entry = &g_textureOverrides[g_textureOverrideCount++];
	entry->id = g_nextTextureOverrideId++;
	if (g_nextTextureOverrideId <= 0)
		g_nextTextureOverrideId = 1;
	entry->descriptor = *textureOverride;
	return entry->id;
}

void PsyX_RemoveTextureOverride(int overrideId)
{
	for (int i = 0; i < g_textureOverrideCount; ++i)
	{
		if (g_textureOverrides[i].id != overrideId)
			continue;

		memmove(&g_textureOverrides[i], &g_textureOverrides[i + 1],
			sizeof(g_textureOverrides[0]) * (g_textureOverrideCount - i - 1));
		--g_textureOverrideCount;
		return;
	}
}

void PsyX_ClearTextureOverrides(void)
{
	g_textureOverrideCount = 0;
}

void PsyX_SetTextureOverridesEnabled(int enabled)
{
	g_textureOverridesEnabled = enabled != 0;
}

void PsyX_GetTextureOverrideStats(PsyXTextureOverrideStats* stats)
{
	if (!stats)
		return;

	stats->registeredCount = g_textureOverrideCount;
	stats->enabled = g_textureOverridesEnabled;
}

static int ComparableTPage(int tpage)
{
	// ABR and dither bits describe the primitive blend state, not its VRAM page.
	return tpage & 0x19F;
}

static bool IsInsideOverride(const PsyXTextureOverride& descriptor, const GrVertex* vertices, int vertexCount)
{
	for (int i = 0; i < vertexCount; ++i)
	{
		const int u = vertices[i].u;
		const int v = vertices[i].v;
		if (u < descriptor.u || v < descriptor.v ||
			u >= descriptor.u + descriptor.width || v >= descriptor.v + descriptor.height)
		{
			return false;
		}
	}

	return true;
}

static void ApplyTextureOverride(GrVertex* vertices, int vertexCount)
{
	g_automaticOverrideTexture = 0;
	g_automaticOverrideTextureWidth = 0;
	g_automaticOverrideTextureHeight = 0;
	g_automaticOverrideSourceU = 0;
	g_automaticOverrideSourceV = 0;
	g_automaticOverrideSourceWidth = 0;
	g_automaticOverrideSourceHeight = 0;

	if (!g_textureOverridesEnabled || overrideTexture != 0 || !vertices || vertexCount <= 0)
		return;

	const int page = ComparableTPage((int)vertices[0].page);
	const int clut = (int)vertices[0].clut;
	for (int i = 0; i < g_textureOverrideCount; ++i)
	{
		const PsyXTextureOverride& descriptor = g_textureOverrides[i].descriptor;
		if (ComparableTPage(descriptor.tpage) != page || descriptor.clut != clut ||
			!IsInsideOverride(descriptor, vertices, vertexCount))
		{
			continue;
		}

		for (int vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex)
		{
			const int sourceU = vertices[vertexIndex].u - descriptor.u;
			const int sourceV = vertices[vertexIndex].v - descriptor.v;
			vertices[vertexIndex].u = (u_char)(sourceU * OVERRIDE_TEXTURE_UV_SIZE / descriptor.width);
			vertices[vertexIndex].v = (u_char)(sourceV * OVERRIDE_TEXTURE_UV_SIZE / descriptor.height);
		}

		g_automaticOverrideTexture = descriptor.textureId;
		g_automaticOverrideTextureWidth = OVERRIDE_TEXTURE_UV_SIZE;
		g_automaticOverrideTextureHeight = OVERRIDE_TEXTURE_UV_SIZE;
		g_automaticOverrideSourceU = descriptor.u;
		g_automaticOverrideSourceV = descriptor.v;
		g_automaticOverrideSourceWidth = descriptor.width;
		g_automaticOverrideSourceHeight = descriptor.height;
		return;
	}
}

static void PrepareTextureOverride(short page, short clut, unsigned char* uv0, unsigned char* uv1,
	unsigned char* uv2, unsigned char* uv3, int vertexCount)
{
	GrVertex probeVertices[4] = {};
	unsigned char* uvs[4] = { uv0, uv1, uv2, uv3 };
	for (int i = 0; i < vertexCount; ++i)
	{
		probeVertices[i].u = uvs[i][0];
		probeVertices[i].v = uvs[i][1];
		probeVertices[i].page = page;
		probeVertices[i].clut = clut;
	}

	ApplyTextureOverride(probeVertices, vertexCount);
}

static void PrepareTextureOverrideRect(short page, short clut, unsigned char* uv, short width, short height)
{
	if (int(uv[0]) + width > 255) width = 255 - uv[0];
	if (int(uv[1]) + height > 255) height = 255 - uv[1];

	unsigned char uv0[] = { uv[0], uv[1] };
	unsigned char uv1[] = { uv[0], (unsigned char)(uv[1] + height) };
	unsigned char uv2[] = { (unsigned char)(uv[0] + width), (unsigned char)(uv[1] + height) };
	unsigned char uv3[] = { (unsigned char)(uv[0] + width), uv[1] };
	PrepareTextureOverride(page, clut, uv0, uv1, uv2, uv3, 4);
}

void ClearSplits()
{
	currentSplitDebugText = nullptr;
	g_automaticOverrideTexture = 0;
	g_automaticOverrideTextureWidth = 0;
	g_automaticOverrideTextureHeight = 0;
	g_automaticOverrideSourceU = 0;
	g_automaticOverrideSourceV = 0;
	g_automaticOverrideSourceWidth = 0;
	g_automaticOverrideSourceHeight = 0;
	g_vertexIndex = 0;
	g_splitIndex = 0;
	g_splits[0].texFormat = (TexFormat)0xFFFF;
}

template<class T>
void DrawEnvDimensions(T& width, T& height)
{
	if (activeDrawEnv.dfe)
	{
		width = activeDispEnv.disp.w;
		height = activeDispEnv.disp.h;
	}
	else
	{
		width = activeDrawEnv.clip.w;
		height = activeDrawEnv.clip.h;
	}
}

void DrawEnvOffset(float& ofsX, float& ofsY)
{
	if (activeDrawEnv.dfe)
	{
		// also make offset in draw dimensions range to prevent flicker
		const int x = activeDispEnv.disp.x;
		const int y = activeDispEnv.disp.y;
		ofsX = activeDrawEnv.ofs[0] - activeDispEnv.disp.x;
		ofsY = activeDrawEnv.ofs[1] - activeDispEnv.disp.y;
	}
	else
	{
		ofsX = 0.0f;
		ofsY = 0.0f;
	}
}

// remaps screen coordinates to [0..1]
// without clamping
inline void ScreenCoordsToEmulator(GrVertex* vertex, int count)
{
#if USE_PGXP
	float w, h;
	DrawEnvDimensions(w, h);

	while (count--)
	{
		vertex[count].x = vertex[count].x / w - 0.5f;
		vertex[count].y = vertex[count].y / h - 0.5f;
	}
#endif
}

void LineSwapSourceVerts(VERTTYPE*& p0, VERTTYPE*& p1, unsigned char*& c0, unsigned char*& c1)
{
	// swap line coordinates for left-to-right and up-to-bottom direction
	if ((p0[0] > p1[0]) ||
		(p0[1] > p1[1] && p0[0] == p1[0]))
	{
		VERTTYPE* tmp = p0;
		p0 = p1;
		p1 = tmp;

		unsigned char* tmpCol = c0;
		c0 = c1;
		c1 = tmpCol;
	}
}

void MakeLineArray(GrVertex* vertex, VERTTYPE* p0, VERTTYPE* p1, ushort gteidx)
{
	const VERTTYPE dx = p1[0] - p0[0];
	const VERTTYPE dy = p1[1] - p0[1];

	float ofsX, ofsY;
	DrawEnvOffset(ofsX, ofsY);

	memset(vertex, 0, sizeof(GrVertex) * 4);

	if (dx > abs((short)dy)) 
	{ // horizontal
		vertex[0].x = p0[0] + ofsX;
		vertex[0].y = p0[1] + ofsY;

		vertex[1].x = p1[0] + ofsX + 1;
		vertex[1].y = p1[1] + ofsY;

		vertex[2].x = vertex[1].x;
		vertex[2].y = vertex[1].y + 1;

		vertex[3].x = vertex[0].x;
		vertex[3].y = vertex[0].y + 1;
	}
	else 
	{ // vertical
		vertex[0].x = p0[0] + ofsX;
		vertex[0].y = p0[1] + ofsY;

		vertex[1].x = p1[0] + ofsX;
		vertex[1].y = p1[1] + ofsY + 1;

		vertex[2].x = vertex[1].x + 1;
		vertex[2].y = vertex[1].y;

		vertex[3].x = vertex[0].x + 1;
		vertex[3].y = vertex[0].y;
	} // TODO diagonal line alignment

#if USE_PGXP
	vertex[0].scr_h = vertex[1].scr_h = vertex[2].scr_h = vertex[3].scr_h = 0.0f;
#endif

	ScreenCoordsToEmulator(vertex, 4);
}

inline void ApplyVertexPGXP(GrVertex* v, VERTTYPE* p, float ofsX, float ofsY, ushort gteidx, int lookupOfs)
{
#if USE_PGXP
	uint lookup = PGXP_LOOKUP_VALUE(p[0], p[1]);

	PGXPVData vd;
	if (gteidx != 0xffff &&
		g_cfg_pgxpTextureCorrection && 
		PGXP_GetCacheData(&vd, lookup, gteidx + lookupOfs))
	{
		v->x = vd.px;
		v->y = vd.py;
		v->z = vd.pz;

		// calculate offset for our perspective matrix based on supposed GTE transformed geometry offset
		float dispW, dispH;
		DrawEnvDimensions(dispW, dispH);

		const float gteOfsX = fmodf(vd.ofx, dispW) - dispW * 0.5f;
		const float gteOfsY = fmodf(vd.ofy, dispH) - dispH * 0.5f;

		v->ofsX = (ofsX + gteOfsX) / dispW * 2.0f;
		v->ofsY = (ofsY + gteOfsY) / dispH * 2.0f;
		v->scr_h = vd.scr_h;
	}
	else
	{
		v->scr_h = 0.0f;
		v->z = 0.0f;
	}
#endif
}

void MakeVertexTriangle(GrVertex* vertex, VERTTYPE* p0, VERTTYPE* p1, VERTTYPE* p2, ushort gteidx)
{
	assert(p0);
	assert(p1);
	assert(p2);

	float ofsX, ofsY;
	DrawEnvOffset(ofsX, ofsY);

	memset(vertex, 0, sizeof(GrVertex) * 3);

	vertex[0].x = p0[0] + ofsX;
	vertex[0].y = p0[1] + ofsY;

	vertex[1].x = p1[0] + ofsX;
	vertex[1].y = p1[1] + ofsY;

	vertex[2].x = p2[0] + ofsX;
	vertex[2].y = p2[1] + ofsY;

	ApplyVertexPGXP(&vertex[0], p0, ofsX, ofsY, gteidx, -2);
	ApplyVertexPGXP(&vertex[1], p1, ofsX, ofsY, gteidx, -1);
	ApplyVertexPGXP(&vertex[2], p2, ofsX, ofsY, gteidx, 0);

	ScreenCoordsToEmulator(vertex, 3);
}

void MakeVertexQuad(GrVertex* vertex, VERTTYPE* p0, VERTTYPE* p1, VERTTYPE* p2, VERTTYPE* p3, ushort gteidx)
{
	assert(p0);
	assert(p1);
	assert(p2);
	assert(p3);

	float ofsX, ofsY;
	DrawEnvOffset(ofsX, ofsY);

	memset(vertex, 0, sizeof(GrVertex) * 4);

	vertex[0].x = p0[0] + ofsX;
	vertex[0].y = p0[1] + ofsY;

	vertex[1].x = p1[0] + ofsX;
	vertex[1].y = p1[1] + ofsY;

	vertex[2].x = p2[0] + ofsX;
	vertex[2].y = p2[1] + ofsY;

	vertex[3].x = p3[0] + ofsX;
	vertex[3].y = p3[1] + ofsY;

	ApplyVertexPGXP(&vertex[0], p0, ofsX, ofsY, gteidx, -3);
	ApplyVertexPGXP(&vertex[1], p1, ofsX, ofsY, gteidx, -2);
	ApplyVertexPGXP(&vertex[2], p2, ofsX, ofsY, gteidx, -1);
	ApplyVertexPGXP(&vertex[3], p3, ofsX, ofsY, gteidx, 0);

	ScreenCoordsToEmulator(vertex, 4);
}

void MakeVertexRect(GrVertex* vertex, VERTTYPE* p0, short w, short h, ushort gteidx)
{
	assert(p0);

	float ofsX, ofsY;
	DrawEnvOffset(ofsX, ofsY);

	memset(vertex, 0, sizeof(GrVertex) * 4);

	vertex[0].x = p0[0] + ofsX;
	vertex[0].y = p0[1] + ofsY;

	vertex[1].x = vertex[0].x;
	vertex[1].y = vertex[0].y + h;

	vertex[2].x = vertex[0].x + w;
	vertex[2].y = vertex[0].y + h;

	vertex[3].x = vertex[0].x + w;
	vertex[3].y = vertex[0].y;

#if USE_PGXP
	vertex[0].scr_h = vertex[1].scr_h = vertex[2].scr_h = vertex[3].scr_h = 0.0f;
#endif

	ScreenCoordsToEmulator(vertex, 4);
}

void MakeTexcoordQuad(GrVertex* vertex, unsigned char* uv0, unsigned char* uv1, unsigned char* uv2, unsigned char* uv3, short page, short clut, unsigned char dither)
{
	assert(uv0);
	assert(uv1);
	assert(uv2);
	assert(uv3);

	const unsigned char bright = 2;

	vertex[0].u = uv0[0];
	vertex[0].v = uv0[1];
	vertex[0].bright = bright;
	vertex[0].dither = dither;
	vertex[0].page = page;
	vertex[0].clut = clut;

	vertex[1].u = uv1[0];
	vertex[1].v = uv1[1];
	vertex[1].bright = bright;
	vertex[1].dither = dither;
	vertex[1].page = page;
	vertex[1].clut = clut;

	vertex[2].u = uv2[0];
	vertex[2].v = uv2[1];
	vertex[2].bright = bright;
	vertex[2].dither = dither;
	vertex[2].page = page;
	vertex[2].clut = clut;

	vertex[3].u = uv3[0];
	vertex[3].v = uv3[1];
	vertex[3].bright = bright;
	vertex[3].dither = dither;
	vertex[3].page = page;
	vertex[3].clut = clut;

	ApplyTextureOverride(vertex, 4);
	/*
	if (g_cfg_bilinearFiltering)
	{
		vertex[0].tcx = -1;
		vertex[0].tcy = -1;

		vertex[1].tcx = -1;
		vertex[1].tcy = -1;

		vertex[2].tcx = -1;
		vertex[2].tcy = -1;

		vertex[3].tcx = -1;
		vertex[3].tcy = -1;
	}*/
}

void MakeTexcoordTriangle(GrVertex* vertex, unsigned char* uv0, unsigned char* uv1, unsigned char* uv2, short page, short clut, unsigned char dither)
{
	assert(uv0);
	assert(uv1);
	assert(uv2);

	const unsigned char bright = 2;

	vertex[0].u = uv0[0];
	vertex[0].v = uv0[1];
	vertex[0].bright = bright;
	vertex[0].dither = dither;
	vertex[0].page = page;
	vertex[0].clut = clut;

	vertex[1].u = uv1[0];
	vertex[1].v = uv1[1];
	vertex[1].bright = bright;
	vertex[1].dither = dither;
	vertex[1].page = page;
	vertex[1].clut = clut;

	vertex[2].u = uv2[0];
	vertex[2].v = uv2[1];
	vertex[2].bright = bright;
	vertex[2].dither = dither;
	vertex[2].page = page;
	vertex[2].clut = clut;

	ApplyTextureOverride(vertex, 3);
	/*
	if (g_cfg_bilinearFiltering)
	{
		vertex[0].tcx = -1;
		vertex[0].tcy = -1;

		vertex[1].tcx = -1;
		vertex[1].tcy = -1;

		vertex[2].tcx = -1;
		vertex[2].tcy = -1;

		vertex[3].tcx = -1;
		vertex[3].tcy = -1;
	}*/
}

void MakeTexcoordRect(GrVertex* vertex, unsigned char* uv, short page, short clut, short w, short h)
{
	assert(uv);

	// sim overflow
	if (int(uv[0]) + w > 255) w = 255 - uv[0];
	if (int(uv[1]) + h > 255) h = 255 - uv[1];

	const unsigned char bright = 2;
	const unsigned char dither = 0;

	vertex[0].u = uv[0];
	vertex[0].v = uv[1];
	vertex[0].bright = bright;
	vertex[0].dither = dither;
	vertex[0].page = page;
	vertex[0].clut = clut;

	vertex[1].u = uv[0];
	vertex[1].v = uv[1] + h;
	vertex[1].bright = bright;
	vertex[1].dither = dither;
	vertex[1].page = page;
	vertex[1].clut = clut;

	vertex[2].u = uv[0] + w;
	vertex[2].v = uv[1] + h;
	vertex[2].bright = bright;
	vertex[2].dither = dither;
	vertex[2].page = page;
	vertex[2].clut = clut;

	vertex[3].u = uv[0] + w;
	vertex[3].v = uv[1];
	vertex[3].bright = bright;
	vertex[3].dither = dither;
	vertex[3].page = page;
	vertex[3].clut = clut;

	if (g_cfg_bilinearFiltering)
	{
		vertex[0].tcx = -1;
		vertex[0].tcy = -1;

		vertex[1].tcx = -1;
		vertex[1].tcy = -1;

		vertex[2].tcx = -1;
		vertex[2].tcy = -1;

		vertex[3].tcx = -1;
		vertex[3].tcy = -1;
	}

	ApplyTextureOverride(vertex, 4);
}

void MakeTexcoordLineZero(GrVertex* vertex, unsigned char dither)
{
	const unsigned char bright = 1;

	vertex[0].u = 0;
	vertex[0].v = 0;
	vertex[0].bright = bright;
	vertex[0].dither = dither;
	vertex[0].page = 0;
	vertex[0].clut = 0;

	vertex[1].u = 0;
	vertex[1].v = 0;
	vertex[1].bright = bright;
	vertex[1].dither = dither;
	vertex[1].page = 0;
	vertex[1].clut = 0;

	vertex[2].u = 0;
	vertex[2].v = 0;
	vertex[2].bright = bright;
	vertex[2].dither = dither;
	vertex[2].page = 0;
	vertex[2].clut = 0;

	vertex[3].u = 0;
	vertex[3].v = 0;
	vertex[3].bright = bright;
	vertex[3].dither = dither;
	vertex[3].page = 0;
	vertex[3].clut = 0;
}

void MakeTexcoordTriangleZero(GrVertex* vertex, unsigned char dither)
{
	const unsigned char bright = 1;

	vertex[0].u = 0;
	vertex[0].v = 0;
	vertex[0].bright = bright;
	vertex[0].dither = dither;
	vertex[0].page = 0;
	vertex[0].clut = 0;

	vertex[1].u = 0;
	vertex[1].v = 0;
	vertex[1].bright = bright;
	vertex[1].dither = dither;
	vertex[1].page = 0;
	vertex[1].clut = 0;

	vertex[2].u = 0;
	vertex[2].v = 0;
	vertex[2].bright = bright;
	vertex[2].dither = dither;
	vertex[2].page = 0;
	vertex[2].clut = 0;
}

void MakeTexcoordQuadZero(GrVertex* vertex, unsigned char dither)
{
	const unsigned char bright = 1;

	vertex[0].u = 0;
	vertex[0].v = 0;
	vertex[0].bright = bright;
	vertex[0].dither = dither;
	vertex[0].page = 0;
	vertex[0].clut = 0;

	vertex[1].u = 0;
	vertex[1].v = 0;
	vertex[1].bright = bright;
	vertex[1].dither = dither;
	vertex[1].page = 0;
	vertex[1].clut = 0;

	vertex[2].u = 0;
	vertex[2].v = 0;
	vertex[2].bright = bright;
	vertex[2].dither = dither;
	vertex[2].page = 0;
	vertex[2].clut = 0;

	vertex[3].u = 0;
	vertex[3].v = 0;
	vertex[3].bright = bright;
	vertex[3].dither = dither;
	vertex[3].page = 0;
	vertex[3].clut = 0;
}

void MakeColourNoShade(GrVertex* vertex, int n)
{
	--n;
	while (n >= 0)
	{
		vertex[n].r = 128;
		vertex[n].g = 128;
		vertex[n].b = 128;
		vertex[n].a = 255;
		--n;
	}
}

void MakeColourLine(GrVertex* vertex, bool shadeTexOn, unsigned char* col0, unsigned char* col1)
{
	if (!shadeTexOn)
	{
		MakeColourNoShade(vertex, 4);
		return;
	}
	assert(col0);
	assert(col1);

	vertex[0].r = col0[0];
	vertex[0].g = col0[1];
	vertex[0].b = col0[2];
	vertex[0].a = 255;

	vertex[1].r = col1[0];
	vertex[1].g = col1[1];
	vertex[1].b = col1[2];
	vertex[1].a = 255;

	vertex[2].r = col1[0];
	vertex[2].g = col1[1];
	vertex[2].b = col1[2];
	vertex[2].a = 255;

	vertex[3].r = col0[0];
	vertex[3].g = col0[1];
	vertex[3].b = col0[2];
	vertex[3].a = 255;
}

void MakeColourTriangle(GrVertex* vertex, bool shadeTexOn, unsigned char* col0, unsigned char* col1, unsigned char* col2)
{
	if (!shadeTexOn)
	{
		MakeColourNoShade(vertex, 3);
		return;
	}

	assert(col0);
	assert(col1);
	assert(col2);

	vertex[0].r = col0[0];
	vertex[0].g = col0[1];
	vertex[0].b = col0[2];
	vertex[0].a = 255;

	vertex[1].r = col1[0];
	vertex[1].g = col1[1];
	vertex[1].b = col1[2];
	vertex[1].a = 255;

	vertex[2].r = col2[0];
	vertex[2].g = col2[1];
	vertex[2].b = col2[2];
	vertex[2].a = 255;
}

void MakeColourQuad(GrVertex* vertex, bool shadeTexOn, unsigned char* col0, unsigned char* col1, unsigned char* col2, unsigned char* col3)
{
	if (!shadeTexOn)
	{
		MakeColourNoShade(vertex, 4);
		return;
	}

	assert(col0);
	assert(col1);
	assert(col2);
	assert(col3);

	vertex[0].r = col0[0];
	vertex[0].g = col0[1];
	vertex[0].b = col0[2];
	vertex[0].a = 255;

	vertex[1].r = col1[0];
	vertex[1].g = col1[1];
	vertex[1].b = col1[2];
	vertex[1].a = 255;

	vertex[2].r = col2[0];
	vertex[2].g = col2[1];
	vertex[2].b = col2[2];
	vertex[2].a = 255;

	vertex[3].r = col3[0];
	vertex[3].g = col3[1];
	vertex[3].b = col3[2];
	vertex[3].a = 255;
}

void TriangulateQuad()
{
	/*
	Triangulate like this:

	v0--v1
	|  / |
	| /  |
	v2--v3

	NOTE: v2 swapped with v3 during primitive parsing but it not shown here
	*/

	g_vertexBuffer[g_vertexIndex + 4] = g_vertexBuffer[g_vertexIndex + 3];

	g_vertexBuffer[g_vertexIndex + 5] = g_vertexBuffer[g_vertexIndex + 2];
	g_vertexBuffer[g_vertexIndex + 2] = g_vertexBuffer[g_vertexIndex + 3];
	g_vertexBuffer[g_vertexIndex + 3] = g_vertexBuffer[g_vertexIndex + 1];
}

//------------------------------------------------------------------------------------------------------------------------

static void AddSplit(bool semiTrans, bool textured)
{
	int tpage = activeDrawEnv.tpage;
	GPUDrawSplit& curSplit = g_splits[g_splitIndex];

	BlendMode blendMode = semiTrans ? GET_TPAGE_BLEND(tpage) : BM_NONE;
	TexFormat texFormat = GET_TPAGE_FORMAT(tpage);
	TextureID textureId = textured ? g_vramTexture : g_whiteTexture;

	if (textured && (overrideTexture != 0 || g_automaticOverrideTexture != 0))
	{
		// override texture format, zero tpage
		texFormat = TF_32_BIT_RGBA;
		textureId = overrideTexture != 0 ? overrideTexture : g_automaticOverrideTexture;
	}

	// FIXME: compare drawing environment too?
	if (curSplit.blendMode == blendMode &&
		curSplit.texFormat == texFormat &&
		curSplit.textureId == textureId &&
		curSplit.drawPrimMode == g_DrawPrimMode &&
		curSplit.drawenv.clip.x == activeDrawEnv.clip.x &&
		curSplit.drawenv.clip.y == activeDrawEnv.clip.y &&
		curSplit.drawenv.clip.w == activeDrawEnv.clip.w &&
		curSplit.drawenv.clip.h == activeDrawEnv.clip.h &&
		curSplit.drawenv.dfe == activeDrawEnv.dfe &&
		curSplit.textureOverridden == (g_automaticOverrideTexture != 0) &&
		curSplit.inspectorSourceU == (unsigned short)g_automaticOverrideSourceU &&
		curSplit.inspectorSourceV == (unsigned short)g_automaticOverrideSourceV &&
		curSplit.inspectorSourceWidth == (unsigned short)g_automaticOverrideSourceWidth &&
		curSplit.inspectorSourceHeight == (unsigned short)g_automaticOverrideSourceHeight &&
		curSplit.debugText == currentSplitDebugText)
	{
		return;
	}

	curSplit.numVerts = g_vertexIndex - curSplit.startVertex;

	if (g_splitIndex + 1 >= MAX_DRAW_SPLITS)
	{
		eprinterr("MAX_DRAW_SPLITS reached (too many blend modes, texture formats, drawEnv clip rects, dfe switches), expect rendering errors\n");
		return;
	}

	GPUDrawSplit& split = g_splits[++g_splitIndex];
	split.blendMode = blendMode;
	split.texFormat = texFormat;
	split.textureId = textureId;
	split.drawPrimMode = g_DrawPrimMode;
	split.drawenv = activeDrawEnv;
	split.dispenv = activeDispEnv;
	split.debugText = currentSplitDebugText;

	split.drawenv.tw.w = overrideTexture != 0 ? overrideTextureWidth : g_automaticOverrideTextureWidth;
	split.drawenv.tw.h = overrideTexture != 0 ? overrideTextureHeight : g_automaticOverrideTextureHeight;
	split.textureOverridden = g_automaticOverrideTexture != 0;
	split.inspectorSourceU = (unsigned short)g_automaticOverrideSourceU;
	split.inspectorSourceV = (unsigned short)g_automaticOverrideSourceV;
	split.inspectorSourceWidth = (unsigned short)g_automaticOverrideSourceWidth;
	split.inspectorSourceHeight = (unsigned short)g_automaticOverrideSourceHeight;

	split.startVertex = g_vertexIndex;
	split.numVerts = 0;
}

void DrawSplit(const GPUDrawSplit& split)
{
	if(split.debugText)
		GR_PushDebugLabel(split.debugText);

	GR_SetStencilMode(split.drawPrimMode);	// draw with mask 0x16

	GR_SetTexture(split.textureId, split.texFormat);

	if (split.texFormat == TF_32_BIT_RGBA)
	{
		// 0 = not an override, 1 = binary 0.5 cutout (compatibility default),
		// 2 = proportional when the opt-in flag is on and this draw blends.
		// Additive/subtractive and opaque draws keep the cutout so their
		// colour-driven behaviour and punch-through holes are unchanged.
		int alphaMode = 0;
		if (split.textureOverridden)
		{
			const bool proportional = g_cfg_overrideProportionalAlpha != 0 && split.blendMode == BM_AVERAGE;
			alphaMode = proportional ? 2 : 1;
		}
		GR_SetOverrideTextureSize(split.drawenv.tw.w, split.drawenv.tw.h);
		GR_SetOverrideAlphaMode(alphaMode);
	}

	const bool drawOnScreen = split.drawenv.dfe;
	GR_SetupClipMode(&split.drawenv.clip, drawOnScreen);
	GR_SetOffscreenState(&split.drawenv.clip, !drawOnScreen);

	GR_SetBlendMode(split.blendMode);

	GR_DrawTriangles(split.startVertex, split.numVerts / 3);

	if (split.debugText)
		GR_PopDebugLabel();
}

static const GPUDrawSplit* FindSplitForVertex(int vertexIndex)
{
	for (int i = 1; i <= g_splitIndex; ++i)
	{
		const GPUDrawSplit& split = g_splits[i];
		if (vertexIndex >= split.startVertex && vertexIndex < split.startVertex + split.numVerts)
			return &split;
	}
	return nullptr;
}

// Mirrors DrawSplit's overrideAlphaMode decision for the CPU picker: with the
// opt-in flag on, a blending override only treats alpha 0 as a hole.
static int OverrideProportionalSampling(const GPUDrawSplit* split)
{
	return (g_cfg_overrideProportionalAlpha != 0 && split && split->blendMode == BM_AVERAGE) ? 1 : 0;
}

static bool ProjectInspectorTriangle(int vertex, const GPUDrawSplit& split, PsyXInspectorTriangle& triangle)
{
	if (split.dispenv.disp.w <= 0 || split.dispenv.disp.h <= 0) return false;
	for (int i = 0; i < 3; ++i)
	{
		const GrVertex& v = g_vertexBuffer[vertex + i];
#if USE_PGXP
		// Mirror GR_SetOffscreenState and GTE_PERSPECTIVE_CORRECTION.
		const float aspect = (float)g_windowWidth / g_windowHeight * (240.0f / 320.0f);
		if (v.scr_h > 100.0f)
		{
			if (v.z <= 0.0f) return false;
			const float scale = cosf(0.9265f * 0.5f) / sinf(0.9265f * 0.5f) * v.scr_h / v.z;
			triangle.x[i] = ((v.x + 0.5f) * scale / aspect + v.ofsX) * 0.5f + 0.5f;
			triangle.y[i] = ((v.y + 0.5f) * scale + v.ofsY) * 0.5f + 0.5f;
		}
		else
		{
			triangle.x[i] = v.x / aspect + 0.5f;
			triangle.y[i] = v.y + 0.5f;
		}
#else
		triangle.x[i] = (float)v.x / split.dispenv.disp.w;
		triangle.y[i] = (float)v.y / split.dispenv.disp.h;
#endif
		if (!(triangle.x[i] > -10000.0f && triangle.x[i] < 10000.0f &&
			triangle.y[i] > -10000.0f && triangle.y[i] < 10000.0f)) return false;
	}
	return true;
}

static bool PointInsideTriangle(float px, float py, const PsyXInspectorTriangle& t)
{
	if ((t.x[1] - t.x[0]) * (t.y[2] - t.y[0]) == (t.y[1] - t.y[0]) * (t.x[2] - t.x[0])) return false;
	const float ab = (px - t.x[1]) * (t.y[0] - t.y[1]) - (t.x[0] - t.x[1]) * (py - t.y[1]);
	const float bc = (px - t.x[2]) * (t.y[1] - t.y[2]) - (t.x[1] - t.x[2]) * (py - t.y[2]);
	const float ca = (px - t.x[0]) * (t.y[2] - t.y[0]) - (t.x[2] - t.x[0]) * (py - t.y[0]);
	return (ab >= 0.0f && bc >= 0.0f && ca >= 0.0f) || (ab <= 0.0f && bc <= 0.0f && ca <= 0.0f);
}

// Projects the cursor back into the split's emulated display area so picking
// respects the viewport and the display clip rather than only the triangle.
static bool PointInsideDisplayArea(const GPUDrawSplit& split, float normalizedPickX, float normalizedPickY)
{
	if (split.dispenv.disp.w <= 0 || split.dispenv.disp.h <= 0)
		return false;

	const float displayX = (float)split.dispenv.disp.x + normalizedPickX * (float)split.dispenv.disp.w;
	const float displayY = (float)split.dispenv.disp.y + normalizedPickY * (float)split.dispenv.disp.h;

	return displayX >= split.dispenv.disp.x && displayX < split.dispenv.disp.x + split.dispenv.disp.w &&
		displayY >= split.dispenv.disp.y && displayY < split.dispenv.disp.y + split.dispenv.disp.h;
}

// Diagnostic: locates screen points whose override texel is cut out, so the
// cutout picking policy can be exercised deterministically instead of by aiming.
float g_inspectorCutoutBounds[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float g_inspectorCutoutPoint[2] = { 0.0f, 0.0f };
float g_inspectorCutoutMaskedPixel[2] = { 0.0f, 0.0f };
int g_inspectorCutoutRange = -1;
int g_inspectorCutoutCount = 0;
int g_inspectorCutoutMaskedCount = 0;

void PsyX_Inspector_FindCutoutSample(void)
{
	g_inspectorCutoutBounds[0] = g_inspectorCutoutBounds[1] = 0.0f;
	g_inspectorCutoutBounds[2] = g_inspectorCutoutBounds[3] = 0.0f;
	g_inspectorCutoutPoint[0] = g_inspectorCutoutPoint[1] = 0.0f;
	g_inspectorCutoutMaskedPixel[0] = g_inspectorCutoutMaskedPixel[1] = 0.0f;
	g_inspectorCutoutRange = -1;
	g_inspectorCutoutCount = 0;
	g_inspectorCutoutMaskedCount = 0;

	int found = 0;

	for (int vertex = 0; vertex + 2 < g_vertexIndex; vertex += 3)
	{
		const GPUDrawSplit* split = FindSplitForVertex(vertex);
		if (!split || !split->drawenv.dfe || !split->textureOverridden)
			continue;

		// Separately report override draws that own a cutout mask at all, so
		// "no cut-out texel here" can be told apart from "no mask on screen".
		if (HasCutoutMask(split->textureId))
		{
			if (g_inspectorCutoutMaskedCount == 0)
			{
				PsyXInspectorTriangle maskedTriangle;
				if (ProjectInspectorTriangle(vertex, *split, maskedTriangle))
				{
					g_inspectorCutoutMaskedPixel[0] = (maskedTriangle.x[0] + maskedTriangle.x[1] + maskedTriangle.x[2]) / 3.0f;
					g_inspectorCutoutMaskedPixel[1] = (maskedTriangle.y[0] + maskedTriangle.y[1] + maskedTriangle.y[2]) / 3.0f;
				}
			}
			++g_inspectorCutoutMaskedCount;
		}

		const GrVertex* vertices = &g_vertexBuffer[vertex];

		// Cut-out texels usually sit at leaf edges rather than a triangle centre,
		// so the vertices are sampled as well as the centroid.
		static const float sampleBarycentric[4][3] =
		{
			{ 1.0f, 0.0f, 0.0f },
			{ 0.0f, 1.0f, 0.0f },
			{ 0.0f, 0.0f, 1.0f },
			{ 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f },
		};

		PsyXInspectorTriangle triangle;
		int projected = 0;
		int cutOutIndex = -1;

		for (int s = 0; s < 4 && cutOutIndex < 0; ++s)
		{
			const float sampleU = (sampleBarycentric[s][0] * vertices[0].u + sampleBarycentric[s][1] * vertices[1].u +
				sampleBarycentric[s][2] * vertices[2].u) / 255.0f;
			const float sampleV = (sampleBarycentric[s][0] * vertices[0].v + sampleBarycentric[s][1] * vertices[1].v +
				sampleBarycentric[s][2] * vertices[2].v) / 255.0f;

			if (SampleCutoutMask(split->textureId, sampleU, sampleV, OverrideProportionalSampling(split)) == 0)
				cutOutIndex = s;
		}

		if (cutOutIndex < 0)
			continue;

		if (!projected)
		{
			if (!ProjectInspectorTriangle(vertex, *split, triangle))
				continue;
			projected = 1;
		}

		const float pointX = triangle.x[cutOutIndex];
		const float pointY = triangle.y[cutOutIndex];

		if (!found)
		{
			g_inspectorCutoutBounds[0] = g_inspectorCutoutBounds[2] = pointX;
			g_inspectorCutoutBounds[1] = g_inspectorCutoutBounds[3] = pointY;
			g_inspectorCutoutPoint[0] = pointX;
			g_inspectorCutoutPoint[1] = pointY;
			g_inspectorCutoutRange = g_vertexInspectorRange[vertex];
			found = 1;
		}
		else
		{
			if (pointX < g_inspectorCutoutBounds[0]) g_inspectorCutoutBounds[0] = pointX;
			if (pointX > g_inspectorCutoutBounds[2]) g_inspectorCutoutBounds[2] = pointX;
			if (pointY < g_inspectorCutoutBounds[1]) g_inspectorCutoutBounds[1] = pointY;
			if (pointY > g_inspectorCutoutBounds[3]) g_inspectorCutoutBounds[3] = pointY;
		}

		++g_inspectorCutoutCount;
	}
}

// Projected screen bounds of a registered range, in normalized window
// coordinates. Diagnostic only: lets a caller (or the debugger) locate a source
// on screen to verify that it is reachable by picking.
float g_inspectorRangeBounds[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
int g_inspectorRangeBoundsValid = 0;

void PsyX_Inspector_GetRangeBounds(int index)
{
	g_inspectorRangeBoundsValid = 0;
	g_inspectorRangeBounds[0] = g_inspectorRangeBounds[1] = 0.0f;
	g_inspectorRangeBounds[2] = g_inspectorRangeBounds[3] = 0.0f;

	if (index < 0 || index >= g_inspectorRangeCount)
		return;

	float minX = 0.0f, minY = 0.0f, maxX = 0.0f, maxY = 0.0f;
	int found = 0;

	for (int vertex = 0; vertex + 2 < g_vertexIndex; vertex += 3)
	{
		if (g_vertexInspectorRange[vertex] != index)
			continue;

		const GPUDrawSplit* split = FindSplitForVertex(vertex);
		if (!split || !split->drawenv.dfe)
			continue;

		PsyXInspectorTriangle triangle;
		if (!ProjectInspectorTriangle(vertex, *split, triangle))
			continue;

		for (int i = 0; i < 3; ++i)
		{
			if (!found)
			{
				minX = maxX = triangle.x[i];
				minY = maxY = triangle.y[i];
				found = 1;
			}
			else
			{
				if (triangle.x[i] < minX) minX = triangle.x[i];
				if (triangle.x[i] > maxX) maxX = triangle.x[i];
				if (triangle.y[i] < minY) minY = triangle.y[i];
				if (triangle.y[i] > maxY) maxY = triangle.y[i];
			}
		}
	}

	if (!found)
		return;

	g_inspectorRangeBounds[0] = minX;
	g_inspectorRangeBounds[1] = minY;
	g_inspectorRangeBounds[2] = maxX;
	g_inspectorRangeBounds[3] = maxY;
	g_inspectorRangeBoundsValid = 1;
}

// Resolves the cursor to a vertex. Shared by the real pick and by the range
// locator so both apply the same display-area, ordering-table, identified-source
// preference and cutout rules.
static int ResolvePickVertex(int windowX, int windowY, int* cutoutSampledOut)
{
	int windowWidth = 0;
	int windowHeight = 0;
	PsyX_GetScreenSize(&windowWidth, &windowHeight);
	if (windowWidth <= 0 || windowHeight <= 0)
		return -1;

	const float pickX = (float)windowX / windowWidth;
	const float pickY = (float)windowY / windowHeight;

	// The primitive walk is ordered by the ordering table (far to near), so the
	// last hit at the cursor is the nearest one. Screen-space overlays are
	// ordered near and would shadow the scene they are composited over, so an
	// identified source is preferred over an unidentified one at the same pixel
	// while draw order remains the depth test between identified sources.
	int selectedVertex = -1;
	int selectedIdentifiedVertex = -1;
	int selectedCutoutSampled = 0;
	int selectedIdentifiedCutoutSampled = 0;

	for (int vertexIndex = 0; vertexIndex + 2 < g_vertexIndex; vertexIndex += 3)
	{
		const GPUDrawSplit* split = FindSplitForVertex(vertexIndex);
		if (!split || !split->drawenv.dfe)
			continue;

		if (!PointInsideDisplayArea(*split, pickX, pickY))
			continue;

		PsyXInspectorTriangle triangle;
		if (!ProjectInspectorTriangle(vertexIndex, *split, triangle) || !PointInsideTriangle(pickX, pickY, triangle))
			continue;

		// A cutout override discards transparent texels on the GPU; with no GPU
		// feedback the same coverage is sampled on the CPU at the cursor, using
		// the shader's override UV mapping, so picking sees through the hole.
		int cutoutSampled = 0;
		if (split->textureOverridden && g_inspectorCutoutPickingEnabled)
		{
			const float determinant = (triangle.y[1] - triangle.y[2]) * (triangle.x[0] - triangle.x[2]) +
				(triangle.x[2] - triangle.x[1]) * (triangle.y[0] - triangle.y[2]);
			if (determinant != 0.0f)
			{
				const float w0 = ((triangle.y[1] - triangle.y[2]) * (pickX - triangle.x[2]) +
					(triangle.x[2] - triangle.x[1]) * (pickY - triangle.y[2])) / determinant;
				const float w1 = ((triangle.y[2] - triangle.y[0]) * (pickX - triangle.x[2]) +
					(triangle.x[0] - triangle.x[2]) * (pickY - triangle.y[2])) / determinant;
				const float w2 = 1.0f - w0 - w1;
				const GrVertex* vertices = &g_vertexBuffer[vertexIndex];

				const float cutoutU = (w0 * vertices[0].u + w1 * vertices[1].u + w2 * vertices[2].u) / 255.0f;
				const float cutoutV = (w0 * vertices[0].v + w1 * vertices[1].v + w2 * vertices[2].v) / 255.0f;
				const int coverage = SampleCutoutMask(split->textureId, cutoutU, cutoutV, OverrideProportionalSampling(split));
				if (coverage >= 0)
				{
					cutoutSampled = 1;
					if (coverage == 0)
						continue;
				}
			}
		}

		selectedVertex = vertexIndex;
		selectedCutoutSampled = cutoutSampled;

		const int range = g_vertexInspectorRange[vertexIndex];
		if (range >= 0 && range < g_inspectorRangeCount)
		{
			selectedIdentifiedVertex = vertexIndex;
			selectedIdentifiedCutoutSampled = cutoutSampled;
		}
	}

	// A registered source only outranks the nearest candidate when that candidate
	// is a screen-space overlay, i.e. carries no depth (the atmospheric haze and
	// similar full-screen effects). Real geometry keeps draw-order priority, so a
	// source that no producer claimed - a tree trunk, an effect mesh - stays
	// selectable instead of being replaced by whatever is behind it.
	if (selectedIdentifiedVertex >= 0)
	{
#if USE_PGXP
		if (g_vertexBuffer[selectedVertex].z <= 0.0f)
		{
			selectedVertex = selectedIdentifiedVertex;
			selectedCutoutSampled = selectedIdentifiedCutoutSampled;
		}
#else
		selectedVertex = selectedIdentifiedVertex;
		selectedCutoutSampled = selectedIdentifiedCutoutSampled;
#endif
	}

	if (cutoutSampledOut)
		*cutoutSampledOut = selectedCutoutSampled;

	return selectedVertex;
}

// Diagnostic: finds a screen point whose override texel is cut out and where
// the pick therefore finds nothing behind it. Without the cutout this same point
// would select the override primitive, so it demonstrates the policy changing
// the outcome rather than merely being available.
// Lets the cutout rule be switched off so its effect on a given pixel can be
// compared directly (on: transparent override texels are skipped).
int g_inspectorCutoutPickingEnabled = 1;

float g_inspectorCutoutFallThrough[2] = { 0.0f, 0.0f };
int g_inspectorCutoutFallThroughValid = 0;

void PsyX_Inspector_FindCutoutFallThrough(void)
{
	g_inspectorCutoutFallThrough[0] = g_inspectorCutoutFallThrough[1] = 0.0f;
	g_inspectorCutoutFallThroughValid = 0;

	int windowWidth = 0;
	int windowHeight = 0;
	PsyX_GetScreenSize(&windowWidth, &windowHeight);
	if (windowWidth <= 0 || windowHeight <= 0)
		return;

	static const float sampleBarycentric[4][3] =
	{
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f },
		{ 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f },
	};

	for (int vertex = 0; vertex + 2 < g_vertexIndex; vertex += 3)
	{
		const GPUDrawSplit* split = FindSplitForVertex(vertex);
		if (!split || !split->drawenv.dfe || !split->textureOverridden)
			continue;

		const GrVertex* vertices = &g_vertexBuffer[vertex];
		int cutOutIndex = -1;

		for (int s = 0; s < 4 && cutOutIndex < 0; ++s)
		{
			const float sampleU = (sampleBarycentric[s][0] * vertices[0].u + sampleBarycentric[s][1] * vertices[1].u +
				sampleBarycentric[s][2] * vertices[2].u) / 255.0f;
			const float sampleV = (sampleBarycentric[s][0] * vertices[0].v + sampleBarycentric[s][1] * vertices[1].v +
				sampleBarycentric[s][2] * vertices[2].v) / 255.0f;

			if (SampleCutoutMask(split->textureId, sampleU, sampleV, OverrideProportionalSampling(split)) == 0)
				cutOutIndex = s;
		}

		if (cutOutIndex < 0)
			continue;

		PsyXInspectorTriangle triangle;
		if (!ProjectInspectorTriangle(vertex, *split, triangle))
			continue;

		const int pixelX = (int)(triangle.x[cutOutIndex] * windowWidth);
		const int pixelY = (int)(triangle.y[cutOutIndex] * windowHeight);
		if (pixelX < 0 || pixelX >= windowWidth || pixelY < 0 || pixelY >= windowHeight)
			continue;

		int cutoutSampled = 0;
		if (ResolvePickVertex(pixelX, pixelY, &cutoutSampled) >= 0)
			continue;

		g_inspectorCutoutFallThrough[0] = triangle.x[cutOutIndex];
		g_inspectorCutoutFallThrough[1] = triangle.y[cutOutIndex];
		g_inspectorCutoutFallThroughValid = 1;
		return;
	}
}

static int g_inspectorPickCostMicros = 0;
static void ResolveInspectorPickInternal();

// Times the one-off resolution so the pick cost can be reported next to the
// frame statistics instead of being assumed negligible.
static void ResolveInspectorPick()
{
	if (!g_inspectorPickPending)
		return;

	const unsigned long long pickStart = SDL_GetPerformanceCounter();
	ResolveInspectorPickInternal();
	const unsigned long long pickEnd = SDL_GetPerformanceCounter();
	const unsigned long long frequency = SDL_GetPerformanceFrequency();

	g_inspectorPickCostMicros = frequency ? (int)((pickEnd - pickStart) * 1000000ull / frequency) : 0;
}

int PsyX_Inspector_GetPickCostMicros(void)
{
	return g_inspectorPickCostMicros;
}

// Range index behind the current selection, or -1 when the selection has no
// registered source. Lets callers query bounds/vertex counts without exposing
// the vertex-to-range table.
int PsyX_Inspector_GetSelectionRangeIndex(void)
{
	if (!g_inspectorSelection.valid)
		return -1;

	const int vertex = g_inspectorSelection.primitiveIndex * 3;
	if (vertex < 0 || vertex >= g_vertexIndex)
		return -1;

	const int range = g_vertexInspectorRange[vertex];
	return (range >= 0 && range < g_inspectorRangeCount) ? range : -1;
}

static void ResolveInspectorPickInternal()
{
	if (!g_inspectorPickPending)
		return;

	g_inspectorPickPending = 0;
	g_inspectorSelection = {};

	int selectedCutoutSampled = 0;
	const int selectedVertex = ResolvePickVertex(g_inspectorPickX, g_inspectorPickY, &selectedCutoutSampled);

	if (selectedVertex < 0)
		return;

	const GPUDrawSplit* split = FindSplitForVertex(selectedVertex);
	const GrVertex* vertices = &g_vertexBuffer[selectedVertex];
	PsyXInspectorSelection& selection = g_inspectorSelection;
	selection.valid = 1;
	selection.cutoutSampled = selectedCutoutSampled;
	selection.primitiveIndex = selectedVertex / 3;
	// The PSX semi-transparency flag lives in the primitive command, not in the
	// tpage, so record the split's blend mode for export-time alpha decisions.
	selection.semiTransparent = (split && split->blendMode != BM_NONE) ? 1 : 0;
	selection.tpage = (unsigned short)vertices[0].page;
	selection.clut = (unsigned short)vertices[0].clut;
	selection.textured = selection.tpage != 0 || selection.clut != 0;
	for (int i = 0; i < 3; ++i)
	{
		selection.u[i] = vertices[i].u;
		selection.v[i] = vertices[i].v;
	}
	if (split && split->textureOverridden)
	{
		selection.textureOverridden = 1;
		selection.sourceU = split->inspectorSourceU;
		selection.sourceV = split->inspectorSourceV;
		selection.sourceWidth = split->inspectorSourceWidth;
		selection.sourceHeight = split->inspectorSourceHeight;
	}
	else
	{
		unsigned short minimumU = selection.u[0], maximumU = selection.u[0];
		unsigned short minimumV = selection.v[0], maximumV = selection.v[0];
		for (int i = 1; i < 3; ++i)
		{
			if (selection.u[i] < minimumU) minimumU = selection.u[i];
			if (selection.u[i] > maximumU) maximumU = selection.u[i];
			if (selection.v[i] < minimumV) minimumV = selection.v[i];
			if (selection.v[i] > maximumV) maximumV = selection.v[i];
		}
		selection.sourceU = minimumU;
		selection.sourceV = minimumV;
		selection.sourceWidth = maximumU - minimumU + 1;
		selection.sourceHeight = maximumV - minimumV + 1;
	}

	const int rangeIndex = g_vertexInspectorRange[selectedVertex];
	if (rangeIndex >= 0 && rangeIndex < g_inspectorRangeCount)
	{
		strncpy(selection.provenance, g_inspectorRanges[rangeIndex].label, sizeof(selection.provenance) - 1);
		selection.provenance[sizeof(selection.provenance) - 1] = '\0';
		selection.object = g_inspectorRanges[rangeIndex].object;
	}

	// A whole-object pick replaces a component key with its parent, so the
	// selection describes the instance the part belongs to. The key format is
	// producer-owned, so the parent is taken textually at the component marker
	// and PsyCross keeps no knowledge of the catalog.
	if (g_inspectorWholeObjectPick && selection.object.key[0])
	{
		char* marker = strstr(selection.object.key, "/component:");

		if (marker != NULL)
		{
			*marker = '\0';
			selection.wholeObject = 1;
		}
	}

	// The request only applies to the pick it was issued with.
	g_inspectorWholeObjectPick = 0;
}

// Diagnostic: finds a screen pixel that actually resolves to the given range,
// by running the same resolution the cursor uses over a grid inside the range's
// projected bounds. This makes an occluded or otherwise unreachable source
// distinguishable from one that is merely hard to aim at by hand.
int g_inspectorRangePixel[2] = { 0, 0 };
int g_inspectorRangePixelValid = 0;

void PsyX_Inspector_FindRangePixel(int rangeIndex)
{
	g_inspectorRangePixel[0] = g_inspectorRangePixel[1] = 0;
	g_inspectorRangePixelValid = 0;

	if (rangeIndex < 0 || rangeIndex >= g_inspectorRangeCount)
		return;

	int windowWidth = 0;
	int windowHeight = 0;
	PsyX_GetScreenSize(&windowWidth, &windowHeight);
	if (windowWidth <= 0 || windowHeight <= 0)
		return;

	PsyX_Inspector_GetRangeBounds(rangeIndex);
	if (!g_inspectorRangeBoundsValid)
		return;

	const int minX = (int)(g_inspectorRangeBounds[0] * windowWidth);
	const int maxX = (int)(g_inspectorRangeBounds[2] * windowWidth) + 1;
	const int minY = (int)(g_inspectorRangeBounds[1] * windowHeight);
	const int maxY = (int)(g_inspectorRangeBounds[3] * windowHeight) + 1;

	const int steps = 12;

	for (int iy = 0; iy <= steps; ++iy)
	{
		for (int ix = 0; ix <= steps; ++ix)
		{
			const int pixelX = minX + (maxX - minX) * ix / steps;
			const int pixelY = minY + (maxY - minY) * iy / steps;

			int cutoutSampled = 0;
			const int vertex = ResolvePickVertex(pixelX, pixelY, &cutoutSampled);
			if (vertex < 0 || g_vertexInspectorRange[vertex] != rangeIndex)
				continue;

			g_inspectorRangePixel[0] = pixelX;
			g_inspectorRangePixel[1] = pixelY;
			g_inspectorRangePixelValid = 1;
			return;
		}
	}
}

// Diagnostic: finds the first registered component (a key carrying
// "/component:") that owns a pixel the cursor would actually resolve to, and
// reports that range and pixel. It answers "can a component be selected right
// now?" without depending on aiming or on a component being visibly exposed.
int g_inspectorComponentRange = -1;
int g_inspectorComponentPixel[2] = { 0, 0 };

void PsyX_Inspector_FindComponentPixelMatching(const char* keyFragment)
{
	if (!keyFragment)
		keyFragment = "/component:";

	g_inspectorComponentRange = -1;
	g_inspectorComponentPixel[0] = g_inspectorComponentPixel[1] = 0;

	for (int range = 0; range < g_inspectorRangeCount; ++range)
	{
		if (!strstr(g_inspectorRanges[range].object.key, keyFragment))
			continue;

		PsyX_Inspector_FindRangePixel(range);
		if (!g_inspectorRangePixelValid)
			continue;

		g_inspectorComponentRange = range;
		g_inspectorComponentPixel[0] = g_inspectorRangePixel[0];
		g_inspectorComponentPixel[1] = g_inspectorRangePixel[1];
		return;
	}
}

void PsyX_Inspector_FindComponentPixel(void)
{
	PsyX_Inspector_FindComponentPixelMatching("/component:");
}

extern int g_dbg_polygonSelected;

static void CaptureInspectorGeometry(bool picked)
{
	if (!g_inspectorSelection.valid) return;
	for (int vertex = 0; vertex + 2 < g_vertexIndex && g_inspectorTriangleCount < 4096; vertex += 3)
	{
		const int range = g_vertexInspectorRange[vertex];
		if (g_inspectorSelection.provenance[0])
		{
			if (range < 0 || range >= g_inspectorRangeCount) continue;
			if (g_inspectorSelection.object.key[0])
			{
				// A whole-object selection keeps a parent key, so its components
				// match by prefix and the whole instance is highlighted at once.
				if (g_inspectorSelection.wholeObject)
				{
					const size_t keyLength = strlen(g_inspectorSelection.object.key);
					if (keyLength == 0 || strncmp(g_inspectorRanges[range].object.key,
						g_inspectorSelection.object.key, keyLength) != 0) continue;
				}
				else if (strcmp(g_inspectorRanges[range].object.key, g_inspectorSelection.object.key) != 0)
				{
					continue;
				}
				else
				{
					g_inspectorSelection.object = g_inspectorRanges[range].object;
				}
			}
			else if (strcmp(g_inspectorRanges[range].label, g_inspectorSelection.provenance) != 0) continue;
		}
		else if (!picked || vertex / 3 != g_inspectorSelection.primitiveIndex) continue;
		const GPUDrawSplit* split = FindSplitForVertex(vertex);
		if (!split || !split->drawenv.dfe || split->dispenv.disp.w <= 0 || split->dispenv.disp.h <= 0) continue;
		PsyXInspectorTriangle& triangle = g_inspectorTriangles[g_inspectorTriangleCount];
		if (!ProjectInspectorTriangle(vertex, *split, triangle)) continue;
		const GrVertex* vertices = &g_vertexBuffer[vertex];
		triangle.tpage = (unsigned short)vertices[0].page;
		triangle.clut = (unsigned short)vertices[0].clut;
		unsigned short minU = vertices[0].u, maxU = vertices[0].u, minV = vertices[0].v, maxV = vertices[0].v;
		for (int i = 1; i < 3; ++i)
		{
			if (vertices[i].u < minU) minU = vertices[i].u;
			if (vertices[i].u > maxU) maxU = vertices[i].u;
			if (vertices[i].v < minV) minV = vertices[i].v;
			if (vertices[i].v > maxV) maxV = vertices[i].v;
		}
		triangle.u = split->textureOverridden ? split->inspectorSourceU : minU;
		triangle.v = split->textureOverridden ? split->inspectorSourceV : minV;
		triangle.width = split->textureOverridden ? split->inspectorSourceWidth : maxU - minU + 1;
		triangle.height = split->textureOverridden ? split->inspectorSourceHeight : maxV - minV + 1;
		++g_inspectorTriangleCount;
	}
}

//
// Draws all polygons after AggregatePTAG
//
void DrawAllSplits()
{
#ifdef _DEBUG
	if (g_dbg_emulatorPaused)
	{
		for (int i = 0; i < 3; i++)
		{
			GrVertex* vert = &g_vertexBuffer[g_dbg_polygonSelected + i];
			vert->r = 255;
			vert->g = 0;
			vert->b = 0;

			eprintf("==========================================\n");
			eprintf("POLYGON: %d\n", g_dbg_polygonSelected);
#if USE_PGXP
			eprintf("X: %.2f Y: %.2f\n", (float)vert->x, (float)vert->y);
			eprintf("U: %.2f V: %.2f\n", (float)vert->u, (float)vert->v);
			eprintf("TP: %d CLT: %d\n", (int)vert->page, (int)vert->clut);
#else
			eprintf("X: %d Y: %d\n", vert->x, vert->y);
			eprintf("U: %d V: %d\n", vert->u, vert->v);
			eprintf("TP: %d CLT: %d\n", vert->page, vert->clut);
#endif
			
			eprintf("==========================================\n");
		}

		PsyX_UpdateInput();
	}
#endif // _DEBUG

	// next code ideally should be called before EndScene
	PsyX_RecordRenderStats();
	GR_UpdateVertexBuffer(g_vertexBuffer, g_vertexIndex);

	for (int i = 1; i <= g_splitIndex; i++)
		DrawSplit(g_splits[i]);

	const bool inspectorPicked = g_inspectorPickPending != 0;
	ResolveInspectorPick();
	CaptureInspectorGeometry(inspectorPicked);
	g_inspectorRangeCount = 0;

	ClearSplits();
}

// forward declarations
int ParsePrimitive(P_TAG* polyTag);

void ParsePrimitivesLinkedList(u_long* p, int singlePrimitive)
{
	if (!p)
		return;

	// setup single primitive flag (needed for AddSplits)
	g_DrawPrimMode = singlePrimitive;

	if (singlePrimitive)
	{
		P_TAG* polyTag = reinterpret_cast<P_TAG*>(p);
#if USE_PGXP && USE_EXTENDED_PRIM_POINTERS
		// force PGXP off
		polyTag->pgxp_index = 0xFFFF;
#endif
		ParsePrimitive(polyTag);

		GPUDrawSplit& lastSplit = g_splits[g_splitIndex];
		lastSplit.numVerts = g_vertexIndex - lastSplit.startVertex;
	}
	else
	{
		// walk OT_TAG linked list
		for (uintptr_t basePacket = reinterpret_cast<uintptr_t>(p);; basePacket = reinterpret_cast<uintptr_t>(nextPrim(basePacket)))
		{
			const int tagLength = getlen(basePacket);
			if (tagLength > 0)
			{
				if (tagLength > 32)
				{
					eprinterr("got invalid tag length %d, code %d\n", tagLength, reinterpret_cast<P_TAG*>(basePacket)->code);
				}

				uintptr_t currentPacket = basePacket;
				const uintptr_t endPacket = basePacket + (tagLength + P_LEN) * sizeof(u_int);
				int primLength = 0;
				while (currentPacket < endPacket)
				{
					primLength = ParsePrimitive(reinterpret_cast<P_TAG*>(currentPacket));
					currentPacket += (primLength + P_LEN) * sizeof(u_int);
				}

				if (currentPacket != endPacket)
				{
					eprinterr("did not output valid primitive or ptag length is not valid (diff=%d)\n", endPacket-currentPacket);
				}
			}

			GPUDrawSplit& lastSplit = g_splits[g_splitIndex];
			lastSplit.numVerts = g_vertexIndex - lastSplit.startVertex;

			if (isendprim(basePacket))
				break;
		}
	}
}

inline int IsNull(POLY_FT3* poly)
{
	return  poly->x0 == -1 &&
		poly->y0 == -1 &&
		poly->x1 == -1 &&
		poly->y1 == -1 &&
		poly->x2 == -1 &&
		poly->y2 == -1;
}

static int ProcessFlatLines(P_TAG* polyTag)
{
#if USE_PGXP && USE_EXTENDED_PRIM_POINTERS
	const u_short gteIndex = polyTag->pgxp_index;
#else
	const u_short gteIndex = 0xFFFF;
#endif

	const bool shadeTexOn = true;
	const bool semiTrans = (polyTag->code & 2);
	const int primSubType = polyTag->code & 0x0C;

	switch (primSubType)
	{
	case 0x0:
	{
		LINE_F2* poly = (LINE_F2*)polyTag;

		AddSplit(semiTrans, false);

		VERTTYPE* p0 = &poly->x0;
		VERTTYPE* p1 = &poly->x1;
		unsigned char* c0 = &poly->r0;
		unsigned char* c1 = c0;

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		LineSwapSourceVerts(p0, p1, c0, c1);
		MakeLineArray(firstVertex, p0, p1, gteIndex);
		MakeTexcoordLineZero(firstVertex, 0);
		MakeColourLine(firstVertex, shadeTexOn, c0, c1);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 3;
	}
	case 0x8: // TODO (unused)
	{
		LINE_F3* poly = (LINE_F3*)polyTag;

		AddSplit(semiTrans, false);

		{
			VERTTYPE* p0 = &poly->x0;
			VERTTYPE* p1 = &poly->x1;
			unsigned char* c0 = &poly->r0;
			unsigned char* c1 = c0;

			GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
			LineSwapSourceVerts(p0, p1, c0, c1);
			MakeLineArray(firstVertex, p0, p1, gteIndex);
			MakeTexcoordLineZero(firstVertex, 0);
			MakeColourLine(firstVertex, shadeTexOn, c0, c1);

			TriangulateQuad();

			g_vertexIndex += 6;
#if defined(DEBUG_POLY_COUNT)
			polygon_count++;
#endif
		}

		{
			VERTTYPE* p0 = &poly->x1;
			VERTTYPE* p1 = &poly->x2;
			unsigned char* c0 = &poly->r0;
			unsigned char* c1 = c0;

			GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
			LineSwapSourceVerts(p0, p1, c0, c1);
			MakeLineArray(firstVertex, p0, p1, gteIndex);
			MakeTexcoordLineZero(firstVertex, 0);
			MakeColourLine(firstVertex, shadeTexOn, c0, c1);

			TriangulateQuad();

			g_vertexIndex += 6;
#if defined(DEBUG_POLY_COUNT)
			polygon_count++;
#endif
		}

		return 5;
	}
	case 0xc:
	{
		int i;
		LINE_F4* poly = (LINE_F4*)polyTag;

		AddSplit(semiTrans, false);

		{
			VERTTYPE* p0 = &poly->x0;
			VERTTYPE* p1 = &poly->x1;
			unsigned char* c0 = &poly->r0;
			unsigned char* c1 = c0;

			GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
			LineSwapSourceVerts(p0, p1, c0, c1);
			MakeLineArray(firstVertex, p0, p1, gteIndex);
			MakeTexcoordLineZero(firstVertex, 0);
			MakeColourLine(firstVertex, shadeTexOn, c0, c1);

			TriangulateQuad();

			g_vertexIndex += 6;
#if defined(DEBUG_POLY_COUNT)
			polygon_count++;
#endif
		}

		{
			VERTTYPE* p0 = &poly->x1;
			VERTTYPE* p1 = &poly->x2;
			unsigned char* c0 = &poly->r0;
			unsigned char* c1 = c0;

			GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
			LineSwapSourceVerts(p0, p1, c0, c1);
			MakeLineArray(firstVertex, p0, p1, gteIndex);
			MakeTexcoordLineZero(firstVertex, 0);
			MakeColourLine(firstVertex, shadeTexOn, c0, c1);

			TriangulateQuad();

			g_vertexIndex += 6;
#if defined(DEBUG_POLY_COUNT)
			polygon_count++;
#endif
		}

		{
			VERTTYPE* p0 = &poly->x2;
			VERTTYPE* p1 = &poly->x3;
			unsigned char* c0 = &poly->r0;
			unsigned char* c1 = c0;

			GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
			LineSwapSourceVerts(p0, p1, c0, c1);
			MakeLineArray(firstVertex, p0, p1, gteIndex);
			MakeTexcoordLineZero(firstVertex, 0);
			MakeColourLine(firstVertex, shadeTexOn, c0, c1);

			TriangulateQuad();

			g_vertexIndex += 6;
#if defined(DEBUG_POLY_COUNT)
			polygon_count++;
#endif
		}

		return 6;
	}
	}
	return 0;
}

static int ProcessGouraudLines(P_TAG* polyTag)
{
#if USE_PGXP && USE_EXTENDED_PRIM_POINTERS
	const u_short gteIndex = polyTag->pgxp_index;
#else
	const u_short gteIndex = 0xFFFF;
#endif

	const bool shadeTexOn = true;
	const bool semiTrans = (polyTag->code & 2);
	const int primSubType = polyTag->code & 0x0C;

	switch (primSubType)
	{
	case 0x0:
	{
		LINE_G2* poly = (LINE_G2*)polyTag;

		AddSplit(semiTrans, false);

		VERTTYPE* p0 = &poly->x0;
		VERTTYPE* p1 = &poly->x1;
		unsigned char* c0 = &poly->r0;
		unsigned char* c1 = &poly->r1;

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		LineSwapSourceVerts(p0, p1, c0, c1);
		MakeLineArray(firstVertex, p0, p1, gteIndex);
		MakeTexcoordLineZero(firstVertex, 0);
		MakeColourLine(firstVertex, shadeTexOn, c0, c1);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 4;
	}
	case 0x8:
	{
		// TODO: LINE_G3
		return 7;
	}
	case 0xC:
	{
		// TODO: LINE_G4
		return 9;
	}
	}
	return 0;
}

static int ProcessFlatPoly(P_TAG* polyTag)
{
#if USE_PGXP && USE_EXTENDED_PRIM_POINTERS
	const u_short gteIndex = polyTag->pgxp_index;
#else
	const u_short gteIndex = 0xFFFF;
#endif

	const bool shadeTexOn = (polyTag->code & 1) == 0;
	const bool semiTrans = (polyTag->code & 2);
	const int primSubType = polyTag->code & 0x0C;

	switch (primSubType)
	{
	case 0x0:
	{
		POLY_F3* poly = (POLY_F3*)polyTag;

		AddSplit(semiTrans, false);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexTriangle(firstVertex, &poly->x0, &poly->x1, &poly->x2, gteIndex);
		MakeTexcoordTriangleZero(firstVertex, 0);
		MakeColourTriangle(firstVertex, shadeTexOn, &poly->r0, &poly->r0, &poly->r0);

		g_vertexIndex += 3;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 4;
	}
	case 0x4:
	{
		POLY_FT3* poly = (POLY_FT3*)polyTag;
		activeDrawEnv.tpage = poly->tpage;

		// It is an official hack from SCE devs to not use DR_TPAGE and instead use null polygon
		if (!IsNull(poly))
		{
			PrepareTextureOverride(poly->tpage, poly->clut, &poly->u0, &poly->u1, &poly->u2, NULL, 3);
			AddSplit(semiTrans, true);

			GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
			MakeVertexTriangle(firstVertex, &poly->x0, &poly->x1, &poly->x2, gteIndex);
			MakeTexcoordTriangle(firstVertex, &poly->u0, &poly->u1, &poly->u2, poly->tpage, poly->clut, GET_TPAGE_DITHER(activeDrawEnv.tpage) || activeDrawEnv.dtd);
			MakeColourTriangle(firstVertex, shadeTexOn, &poly->r0, &poly->r0, &poly->r0);

			g_vertexIndex += 3;

#if defined(DEBUG_POLY_COUNT)
			polygon_count++;
#endif
		}
		return 7;
	}
	case 0x8:
	{
		POLY_F4* poly = (POLY_F4*)polyTag;

		AddSplit(semiTrans, false);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexQuad(firstVertex, &poly->x0, &poly->x1, &poly->x3, &poly->x2, gteIndex);
		MakeTexcoordQuadZero(firstVertex, 0);
		MakeColourQuad(firstVertex, shadeTexOn, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;
#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 5;
	}
	case 0xC:
	{
		POLY_FT4* poly = (POLY_FT4*)polyTag;
		activeDrawEnv.tpage = poly->tpage;

		PrepareTextureOverride(poly->tpage, poly->clut, &poly->u0, &poly->u1, &poly->u3, &poly->u2, 4);
		AddSplit(semiTrans, true);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexQuad(firstVertex, &poly->x0, &poly->x1, &poly->x3, &poly->x2, gteIndex);
		MakeTexcoordQuad(firstVertex, &poly->u0, &poly->u1, &poly->u3, &poly->u2, poly->tpage, poly->clut, GET_TPAGE_DITHER(activeDrawEnv.tpage) || activeDrawEnv.dtd);
		MakeColourQuad(firstVertex, shadeTexOn, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 9;
	}
	}
	return 0;
}

static int ProcessGouraudPoly(P_TAG* polyTag)
{
#if USE_PGXP && USE_EXTENDED_PRIM_POINTERS
	const u_short gteIndex = polyTag->pgxp_index;
#else
	const u_short gteIndex = 0xFFFF;
#endif

	const bool shadeTexOn = true;
	const bool semiTrans = (polyTag->code & 2);
	const int primSubType = polyTag->code & 0x0C;

	switch (primSubType)
	{
	case 0x0:
	{
		POLY_G3* poly = (POLY_G3*)polyTag;

		AddSplit(semiTrans, false);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexTriangle(firstVertex, &poly->x0, &poly->x1, &poly->x2, gteIndex);
		MakeTexcoordTriangleZero(firstVertex, 1);
		MakeColourTriangle(firstVertex, shadeTexOn, &poly->r0, &poly->r1, &poly->r2);

		g_vertexIndex += 3;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 6;
	}
	case 0x4:
	{
		POLY_GT3* poly = (POLY_GT3*)polyTag;
		activeDrawEnv.tpage = poly->tpage;

		PrepareTextureOverride(poly->tpage, poly->clut, &poly->u0, &poly->u1, &poly->u2, NULL, 3);
		AddSplit(semiTrans, true);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexTriangle(firstVertex, &poly->x0, &poly->x1, &poly->x2, gteIndex);
		MakeTexcoordTriangle(firstVertex, &poly->u0, &poly->u1, &poly->u2, poly->tpage, poly->clut, GET_TPAGE_DITHER(activeDrawEnv.tpage) || activeDrawEnv.dtd);
		MakeColourTriangle(firstVertex, shadeTexOn, &poly->r0, &poly->r1, &poly->r2);

		g_vertexIndex += 3;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 9;
	}
	case 0x8:
	{
		POLY_G4* poly = (POLY_G4*)polyTag;

		AddSplit(semiTrans, false);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexQuad(firstVertex, &poly->x0, &poly->x1, &poly->x3, &poly->x2, gteIndex);
		MakeTexcoordQuadZero(firstVertex, 1);
		MakeColourQuad(firstVertex, shadeTexOn, &poly->r0, &poly->r1, &poly->r3, &poly->r2);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 8;
	}
	case 0xC:
	{
		POLY_GT4* poly = (POLY_GT4*)polyTag;
		activeDrawEnv.tpage = poly->tpage;

		PrepareTextureOverride(poly->tpage, poly->clut, &poly->u0, &poly->u1, &poly->u3, &poly->u2, 4);
		AddSplit(semiTrans, true);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexQuad(firstVertex, &poly->x0, &poly->x1, &poly->x3, &poly->x2, gteIndex);
		MakeTexcoordQuad(firstVertex, &poly->u0, &poly->u1, &poly->u3, &poly->u2, poly->tpage, poly->clut, GET_TPAGE_DITHER(activeDrawEnv.tpage) || activeDrawEnv.dtd);
		MakeColourQuad(firstVertex, shadeTexOn, &poly->r0, &poly->r1, &poly->r3, &poly->r2);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 12;
	}
	}
	return 0;
}

static int ProcessTileAndSprt(P_TAG* polyTag)
{
#if USE_PGXP && USE_EXTENDED_PRIM_POINTERS
	const u_short gteIndex = polyTag->pgxp_index;
#else
	const u_short gteIndex = 0xFFFF;
#endif

	// NOTE: TILE does not support switching shadeTex on real PSX
	const bool shadeTexOn = (polyTag->code & 1) == 0;
	const bool semiTrans = (polyTag->code & 2);

	switch (polyTag->code & 0xFD)
	{
	case 0x60:
	{
		TILE* poly = (TILE*)polyTag;

		AddSplit(semiTrans, false);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexRect(firstVertex, &poly->x0, poly->w, poly->h, gteIndex);
		MakeTexcoordQuadZero(firstVertex, 0);
		MakeColourQuad(firstVertex, shadeTexOn, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 3;
	}
	case 0x64:
	{
		SPRT* poly = (SPRT*)polyTag;

		PrepareTextureOverrideRect(activeDrawEnv.tpage, poly->clut, &poly->u0, poly->w, poly->h);
		AddSplit(semiTrans, true);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexRect(firstVertex, &poly->x0, poly->w, poly->h, gteIndex);
		MakeTexcoordRect(firstVertex, &poly->u0, activeDrawEnv.tpage, poly->clut, poly->w, poly->h);
		MakeColourQuad(firstVertex, shadeTexOn, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 4;
	}
	case 0x68:
	{
		TILE_1* poly = (TILE_1*)polyTag;

		AddSplit(semiTrans, false);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexRect(firstVertex, &poly->x0, 1, 1, gteIndex);
		MakeTexcoordQuadZero(firstVertex, 0);
		MakeColourQuad(firstVertex, true, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 2;
	}
	case 0x70:
	{
		TILE_8* poly = (TILE_8*)polyTag;

		AddSplit(semiTrans, false);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexRect(firstVertex, &poly->x0, 8, 8, gteIndex);
		MakeTexcoordQuadZero(firstVertex, 0);
		MakeColourQuad(firstVertex, true, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 2;
	}
	case 0x74:
	{
		SPRT_8* poly = (SPRT_8*)polyTag;

		PrepareTextureOverrideRect(activeDrawEnv.tpage, poly->clut, &poly->u0, 8, 8);
		AddSplit(semiTrans, true);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexRect(firstVertex, &poly->x0, 8, 8, gteIndex);
		MakeTexcoordRect(firstVertex, &poly->u0, activeDrawEnv.tpage, poly->clut, 8, 8);
		MakeColourQuad(firstVertex, shadeTexOn, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 3;
	}
	case 0x78:
	{
		TILE_16* poly = (TILE_16*)polyTag;

		AddSplit(semiTrans, false);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexRect(firstVertex, &poly->x0, 16, 16, gteIndex);
		MakeTexcoordQuadZero(firstVertex, 0);
		MakeColourQuad(firstVertex, true, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 2;
	}
	case 0x7C:
	{
		SPRT_16* poly = (SPRT_16*)polyTag;

		PrepareTextureOverrideRect(activeDrawEnv.tpage, poly->clut, &poly->u0, 16, 16);
		AddSplit(semiTrans, true);

		GrVertex* firstVertex = &g_vertexBuffer[g_vertexIndex];
		MakeVertexRect(firstVertex, &poly->x0, 16, 16, gteIndex);
		MakeTexcoordRect(firstVertex, &poly->u0, activeDrawEnv.tpage, poly->clut, 16, 16);
		MakeColourQuad(firstVertex, shadeTexOn, &poly->r0, &poly->r0, &poly->r0, &poly->r0);

		TriangulateQuad();

		g_vertexIndex += 6;

#if defined(DEBUG_POLY_COUNT)
		polygon_count++;
#endif
		return 3;
	}
	}
	return 0;
}

static int ProcessDrawEnv(P_TAG* polyTag)
{
	const u_int* codePtr = (u_int*)&polyTag->pad0;
	int processedLongs = 0;
	for (int i = 0; i < polyTag->len; ++i)
	{
		const u_int code = codePtr[i];
		const int primSubType = code >> 24 & 0x0F;

		switch (primSubType)
		{
		case 0x1:
		{
			// DR_TPAGE
			activeDrawEnv.tpage = (code & 0x1FF);
			activeDrawEnv.dtd = (code >> 9) & 1;
			activeDrawEnv.dfe = (code >> 10) & 1;
			break;
		}
		case 0x2:
		{
			// DR_TWIN
			activeDrawEnv.tw.w = (code & 0x1F);
			activeDrawEnv.tw.h = ((code >> 5) & 0x1F);
			activeDrawEnv.tw.x = ((code >> 10) & 0x1F);
			activeDrawEnv.tw.y = ((code >> 15) & 0x1F);
			break;
		}
		case 0x3:
		{
			// DR_AREA
			activeDrawEnv.clip.x = code & 1023;
			activeDrawEnv.clip.y = (code >> 10) & 1023;
			break;
		}
		case 0x4:
		{
			// DR_AREA (second part)
			activeDrawEnv.clip.w = code & 1023;
			activeDrawEnv.clip.h = (code >> 10) & 1023;

			activeDrawEnv.clip.w -= activeDrawEnv.clip.x;
			activeDrawEnv.clip.h -= activeDrawEnv.clip.y;
			break;
		}
		case 0x5:
		{
			// DR_OFFSET
			// TODO
			activeDrawEnv.ofs[0] = code & 2047;
			activeDrawEnv.ofs[1] = (code >> 11) & 2047;
			break;
		}
		case 0x6:
		{
			eprintf("Mask setting: %08x\n", code);
			//MaskSetOR = (*cb & 1) ? 0x8000 : 0x0000;
			//MaskEvalAND = (*cb & 2) ? 0x8000 : 0x0000;
			break;
		}
		case 0:
			// proceed to next primitive tag
			return processedLongs;
		}
		++processedLongs;
	}

	return processedLongs;
}

static int ProcessPsyXPrims(P_TAG* polyTag)
{
	const int primType = polyTag->code & 0xF0;
	const int primSubType = polyTag->code & 0x0F;

	switch (primSubType)
	{
	case 0x01:
	{
		DR_PSYX_TEX* psytex = (DR_PSYX_TEX*)polyTag;
		overrideTexture = psytex->code[0] & 0xFFFFFF;
		overrideTextureWidth = psytex->code[1] & 0xFFF;
		overrideTextureHeight = psytex->code[1] >> 16 & 0xFFF;
		return 2;
	}
	case 0x02:
	{
		// [A] Psy-X custom texture packet
		DR_PSYX_DBGMARKER* psydbg = (DR_PSYX_DBGMARKER*)polyTag;
		currentSplitDebugText = psydbg->text;
		return 2;
	}
	}

	return 0;
}

// Processes primitive
// returns processed primitive primLength in longs
int ParsePrimitive(P_TAG* polyTag)
{
	const int primType = polyTag->code & 0xF0;
	const int firstVertex = g_vertexIndex;
	const int inspectorRange = FindInspectorRange(polyTag);

	int primLength = 0;

	switch (primType)
	{
	case 0x00:
	{
		const int primSubType = polyTag->code & 0x0F;
		if (primSubType == 0x0)
		{
			primLength = 3;
		}
		else if (primSubType == 0x1)
		{
			DR_MOVE* drmove = (DR_MOVE*)polyTag;

			const int y = drmove->code[3] >> 0x10 & 0xFFFF;
			const int x = drmove->code[3] & 0xFFFF;

			RECT16 rect;
			*(uint*)&rect.x = *(uint*)&drmove->code[2];
			*(uint*)&rect.w = *(uint*)&drmove->code[4];

			MoveImage(&rect, x, y);
			primLength = 5;
		}
		break;
	}
	case 0x20:
		// Flat polygons
		primLength = ProcessFlatPoly(polyTag);
		break;
	case 0x30:
		// Gouraud shaded polygons
		primLength = ProcessGouraudPoly(polyTag);
		break;
	case 0x40:
		// Flat (single colour) Lines
		primLength = ProcessFlatLines(polyTag);
		break;
	case 0x50:
		// Gouraud lines
		primLength = ProcessGouraudLines(polyTag);
		break;
	case 0x60:
	case 0x70:
		// TILE and SPRT
		primLength = ProcessTileAndSprt(polyTag);
		break;
	case 0xA0:
		// DR_LOAD
		{
			DR_LOAD* drload = (DR_LOAD*)polyTag;

			RECT16 rect;
			*(uint*)&rect.x = *(uint*)&drload->code[1];
			*(uint*)&rect.w = *(uint*)&drload->code[2];

			LoadImage(&rect, (u_long*)drload->p);
			//Emulator_UpdateVRAM();			// FIXME: should it be updated immediately?

			// FIXME: is there othercommands?
		}
		primLength = getlen(polyTag);
		break;
	case 0xB0:
		// [A] Psy-X custom primitives
		primLength = ProcessPsyXPrims(polyTag);
		break;
	case 0xE0:
		// Draw Env setup
		primLength = ProcessDrawEnv(polyTag);
		break;
	//default:
	//	eprinterr("got %0x primitive\n", primType);
	}

	if(primLength == 0)
	{
		eprinterr("Unhandled zero length %0x primitive\n", primType);
	}

	for (int vertexIndex = firstVertex; vertexIndex < g_vertexIndex; ++vertexIndex)
		g_vertexInspectorRange[vertexIndex] = inspectorRange;

	if (inspectorRange >= 0 && inspectorRange < g_inspectorRangeCount)
		g_inspectorRangeVertexCount[inspectorRange] += g_vertexIndex - firstVertex;

	return primLength;
}
