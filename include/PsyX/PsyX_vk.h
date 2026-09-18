#ifndef PSYX_VK_H
#define PSYX_VK_H

#include "PsyX_public.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Experimental native-Vulkan backend for the modern mesh scene (renderer
 * roadmap R7). It owns an SDL window and a swapchain and renders the same
 * mesh/material/light description the OpenGL modern path uses. The legacy PSX
 * renderer is not ported here; this slice exists to validate the second backend
 * for the modern scene, including presentation, resize, readback and ImGui.
 *
 * The module is inert on platforms without a delivered Vulkan loader: every
 * entry point degrades to a no-op or a failure result.
 */

#define PSYX_VK_MAX_MESHES 32
#define PSYX_VK_MAX_LIGHTS 8

typedef struct
{
	int width;
	int height;
	const char* title;
	int enableImGui;	/* 1 to create the ImGui overlay */
	int gameMode;		/* 1 when the window backs the game, not the fixture */
} PsyXVkConfig;

typedef struct
{
	int initialised;
	int width;
	int height;
	int meshCount;
	int drawCalls;
	int shadowMapSize;
	int vulkanApiVersion;	/* VK_MAKE_API_VERSION packed, 0 when unknown */
	int imguiActive;
	char deviceName[256];
} PsyXVkInfo;

/* Non-zero when the dynamic loader found a Vulkan library on this system. */
int  PsyX_Vk_IsSupported(void);

int  PsyX_Vk_Initialise(const PsyXVkConfig* config);
void PsyX_Vk_Shutdown(void);

/* Creates a mesh from the shared modern-mesh description (positions in world
   units, optional normals/UVs/colours and RGBA8 textures). Returns a handle or
   -1. Textures are copied; the caller keeps ownership of its own handles. */
int  PsyX_Vk_CreateMesh(const PsyXModernMeshDesc* desc);
void PsyX_Vk_DestroyMesh(int mesh);

/* Material textures. Vulkan owns its own handles because the OpenGL texture
   names used by the shared descriptor are not portable; RGBA8 pixels are copied
   on creation. Slots: 0 = base colour, 1 = normal, 2 = metallic/roughness,
   3 = emissive. A slot without a texture uses a neutral 1x1 default. */
int  PsyX_Vk_CreateTexture(const unsigned char* rgba, int width, int height);
void PsyX_Vk_DestroyTexture(int texture);
void PsyX_Vk_SetMeshTexture(int mesh, int slot, int texture);
void PsyX_Vk_SetMeshFactors(int mesh, float metallic, float roughness, float emissiveScale);

/* Per-instance world transform (mesh local space to world space) and tint. */
void PsyX_Vk_SetInstance(int mesh, const float worldMatrix[16], const float color[4], int visible);

/* Standard right-handed view and projection matrices (column-major). */
void PsyX_Vk_SetCamera(const float view[16], const float proj[16], const float cameraPosition[3]);
void PsyX_Vk_SetLights(const PsyXModernLightSet* lights);

/* Renders and presents one frame. Returns 0 when the window should close. */
int  PsyX_Vk_RenderFrame(void);

/* Copies the last presented frame as tightly packed RGBA8 (top-left origin).
   `rgba` must have room for width*height*4 bytes. */
int  PsyX_Vk_ReadbackRgba(unsigned char* rgba, int* width, int* height);

void PsyX_Vk_GetInfo(PsyXVkInfo* info);

/* Draws one line of text in the overlay (copied). Pass NULL to clear. */
void PsyX_Vk_SetOverlayText(const char* text);

/*
 * Emulated PSX GPU path (renderer roadmap R7b).
 *
 * This mirrors the OpenGL renderer's PSX model: a 1024x512 16-bit VRAM mirror
 * is uploaded as an R32G32_SFLOAT image (R = low byte / 255, G = high byte /
 * 255), a 256x256 RGBA table decodes PSX 5551 colours, and the ported PSX
 * shaders (vk_shaders/psx.vert, psx.frag) do the CLUT lookup. The game's GR_*
 * contract is mirrored onto these calls by PsyX_render.cpp: a frame uploads
 * one vertex buffer, then each split sets its state and draws a range.
 */

#define PSYX_VK_VRAM_WIDTH 1024
#define PSYX_VK_VRAM_HEIGHT 512

/* PSX texture formats, matching the game renderer's TexFormat order. */
#define PSYX_VK_TEX_4BIT 0
#define PSYX_VK_TEX_8BIT 1
#define PSYX_VK_TEX_16BIT 2
#define PSYX_VK_TEX_32BIT 3

/* PSX blend modes, matching the game renderer's BlendMode order. */
#define PSYX_VK_BLEND_NONE 0
#define PSYX_VK_BLEND_AVERAGE 1
#define PSYX_VK_BLEND_ADD 2
#define PSYX_VK_BLEND_SUBTRACT 3
#define PSYX_VK_BLEND_ADD_QUARTER_SOURCE 4

/* Starts a game frame: waits for the previous frame so the caller may refill
   the shared vertex buffer, then drops the queued draws. */
void PsyX_Vk_GameBeginFrame(void);

/* Replaces the whole CPU VRAM mirror (1024x512 little-endian 16-bit PSX
   pixels, row stride PSYX_VK_VRAM_WIDTH) and re-uploads it to the GPU. */
void PsyX_Vk_GameSetVram(const unsigned short* vram);

/* Column-major matrices identical to GR_Ortho2D / GR_Perspective3D. */
void PsyX_Vk_GameSetProjection2D(const float projection[16]);
void PsyX_Vk_GameSetProjection3D(const float projection[16]);

/* Copies the frame's packed GrVertex records (44 bytes, the PGXP layout) into
   the shared vertex buffer. Draws index into this upload. */
void PsyX_Vk_GameUpdateVertexBuffer(const void* vertices, int vertexCount);

/* Mirrors the per-draw GR_* state. `blendMode` is a PSYX_VK_BLEND_* value and
   `texFormat` a PSYX_VK_TEX_* value; `texture` is a PsyX_Vk_GameCreateTexture
   handle used by the 32-bit format only. */
void PsyX_Vk_GameSetBlendMode(int blendMode);
void PsyX_Vk_GameSetTexture(int texFormat, int texture);
void PsyX_Vk_GameSetOverrideTextureSize(int width, int height);
void PsyX_Vk_GameSetOverrideAlphaMode(int mode);
void PsyX_Vk_GameSetStencilMode(int drawPrimMode);
void PsyX_Vk_GameEnableDepth(int enable);
void PsyX_Vk_GameSetBilinear(int enable);

/* Window-space rectangles, top-left origin (as the caller computes them). */
void PsyX_Vk_GameSetScissor(int enable, int x, int y, int width, int height);
void PsyX_Vk_GameSetViewPort(int x, int y, int width, int height);

/* Render-to-VRAM (offscreen) target, mirroring GR_SetOffscreenState. While
   `enable` is set, subsequently queued PSX draws render into an offscreen
   buffer of `width` x `height`; once cleared, the buffer is copied into the PSX
   VRAM mirror at (x, y) so later draws in the frame sample the result, exactly
   like the OpenGL renderer's offscreen framebuffer blit. */
void PsyX_Vk_GameSetOffscreen(int enable, int x, int y, int width, int height);

/* Renders every queued offscreen group, packs the pixels into `vram` (the full
   1024x512 PSX mirror) and re-uploads the GPU VRAM image. Call once per frame
   after the draws are queued and before the frame is submitted. Returns 1 when
   at least one group was resolved. */
int  PsyX_Vk_GameResolveOffscreen(unsigned short* vram);

/* Clears the presented image (PSX framebuffer clear). */
void PsyX_Vk_GameClear(int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b);

/* Records the back-buffer -> VRAM screen-area blit rectangle (GR_StoreFrameBuffer).
   The copy itself runs once the frame has been presented, when the renderer
   consumes it through PsyX_Vk_TakeStoredFrameBuffer. */
void PsyX_Vk_GameStoreFrameBuffer(int x, int y, int width, int height);

/* Consumes a pending PsyX_Vk_GameStoreFrameBuffer request after the presented
   frame is available. On success it returns 1 and fills:
     - rgba: pointer into the backend-owned readback buffer, valid until the
       next frame is submitted (top-left origin, `stride` bytes per row);
     - stride/srcWidth/srcHeight: the presented image layout;
     - bgra: 1 when the byte order is B,G,R,A (B8G8R8A8), 0 for R,G,B,A;
     - x/y/width/height: the PSX VRAM destination rectangle.
   Returns 0 when nothing is pending or the readback is unavailable. */
int  PsyX_Vk_TakeStoredFrameBuffer(const unsigned char** rgba, int* stride,
	int* srcWidth, int* srcHeight, int* bgra, int* x, int* y, int* width, int* height);

/* Queues a triangle list drawing `triangles` triangles from the uploaded
   vertex buffer, starting at vertex `firstVertex`. Returns queued triangles. */
int PsyX_Vk_GameDrawTriangles(int firstVertex, int triangles);

/* Game-owned RGBA textures (hi-res fonts, HD texture overrides). Returns a
   handle > 0, or 0 when the texture could not be created. */
int  PsyX_Vk_GameCreateTexture(const unsigned char* rgba, int width, int height, int mipmapped);
void PsyX_Vk_GameDestroyTexture(int texture);

/* Submits the queued draws into the main render pass and presents. */
void PsyX_Vk_GameEndFrame(void);
void PsyX_Vk_GameResetDevice(void);

/* 1 when the Vulkan backend is the active game renderer. */
int  PsyX_Vk_GameIsActive(void);

/* SDL window owned by the Vulkan backend (for input, screenshots, ImGui). */
struct SDL_Window* PsyX_Vk_GetSDLWindow(void);

/* Phase-1 acceptance test: writes synthetic VRAM patterns, renders PSX quads
   through the Vulkan PSX pipeline and checks the readback pixels against the
   values the OpenGL shader maths produces. Returns 1 on success and writes a
   short human-readable report. */
int PsyX_Vk_GameSelfTest(char* report, int reportSize);

#ifdef __cplusplus
}
#endif

#endif // PSYX_VK_H
