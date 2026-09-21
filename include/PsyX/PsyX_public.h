#ifndef EMULATOR_PUBLIC_H
#define EMULATOR_PUBLIC_H

#include <SDL_events.h>
#include <SDL_video.h>

#define CONTROLLER_MAP_FLAG_AXIS		0x4000
#define CONTROLLER_MAP_FLAG_INVERSE		0x8000
#define PSYX_INPUT_CAPTURE_KEYBOARD	0x1
#define PSYX_INPUT_CAPTURE_GAMEPAD	0x2

typedef struct
{
	int id;

	int kc_square, kc_circle, kc_triangle, kc_cross;

	int kc_l1, kc_l2, kc_l3;
	int kc_r1, kc_r2, kc_r3;

	int kc_start, kc_select;

	int kc_dpad_left, kc_dpad_right, kc_dpad_up, kc_dpad_down;
} PsyXKeyboardMapping;

typedef struct
{
	int id;

	// you can bind axis by adding CONTROLLER_MAP_AXIS_FLAG
	int gc_square, gc_circle, gc_triangle, gc_cross;

	int gc_l1, gc_l2, gc_l3;
	int gc_r1, gc_r2, gc_r3;

	int gc_start, gc_select;

	int gc_dpad_left, gc_dpad_right, gc_dpad_up, gc_dpad_down;

	int gc_axis_left_x, gc_axis_left_y;
	int gc_axis_right_x, gc_axis_right_y;
} PsyXControllerMapping;

typedef void(*GameDebugKeysHandlerFunc)(int nKey, char down);
typedef void(*GameDebugMouseHandlerFunc)(int x, int y, int dx, int dy);
typedef void(*GameOnTextInputHandler)(const char* buf);
typedef int(*PsyXSDLEventHandlerFunc)(const SDL_Event* event);
typedef void(*PsyXRenderOverlayHandlerFunc)(void);

typedef struct
{
	int vertexCount;
	int drawSplitCount;
} PsyXRenderStats;

/* A replacement for one rectangular, CLUT-qualified PSX texture region.
   Texture coordinates are remapped to the complete RGBA image at draw time;
   the original VRAM contents remain present for the normal fallback path. */
typedef struct
{
	unsigned short tpage;
	unsigned short clut;
	unsigned short u;
	unsigned short v;
	unsigned short width;
	unsigned short height;
	unsigned int textureId;
} PsyXTextureOverride;

typedef struct
{
	int registeredCount;
	int enabled;
} PsyXTextureOverrideStats;

#define PSYX_INSPECTOR_LABEL_LENGTH 128

/* A primitive selected from the completed PSX draw stream. The source texture
   rectangle contains original PSX UV coordinates when an RGBA override was
   used, so tools can still identify and export the original resource. */
typedef struct
{
	char key[128]; /* Producer-owned identity, independent of LOD/display label. */
	char modelName[64];
	int modelIndex;
	int vertexCount;
	int polygonCount;
	int position[3];
} PsyXInspectorObject;

typedef struct
{
	int valid;
	int primitiveIndex;
	int textured;
	int textureOverridden;
	int cutoutSampled; /* override cutout coverage was consulted for this pick */
	int wholeObject; /* the key is the parent of the picked component, not the part */
	int semiTransparent; /* the picked primitive is drawn with PSX semi-transparency */
	unsigned short tpage;
	unsigned short clut;
	unsigned short u[3];
	unsigned short v[3];
	unsigned short sourceU;
	unsigned short sourceV;
	unsigned short sourceWidth;
	unsigned short sourceHeight;
	char provenance[PSYX_INSPECTOR_LABEL_LENGTH];
	PsyXInspectorObject object;
} PsyXInspectorSelection;

/* Normalized window coordinates for a diagnostic, non-depth-tested overlay.
 * Geometry is refreshed every frame for labelled draw sources. */
typedef struct
{
	float x[3], y[3];
	unsigned short tpage, clut, u, v, width, height;
} PsyXInspectorTriangle;

//------------------------------------------------------------------------

#if defined(_LANGUAGE_C_PLUS_PLUS)||defined(__cplusplus)||defined(c_plusplus)
extern "C" {
#endif

/* Mapped inputs */
extern PsyXControllerMapping		g_cfg_controllerMapping;
extern PsyXKeyboardMapping			g_cfg_keyboardMapping;
extern int							g_cfg_controllerToSlotMapping[2];

/* Game inputs */
extern GameOnTextInputHandler		g_cfg_gameOnTextInput;
extern int							g_cfg_inputCapture;

/* Graphics configuration */
extern int							g_cfg_swapInterval;
extern int							g_cfg_pgxpZBuffer;
extern int							g_cfg_bilinearFiltering;
extern int							g_cfg_pgxpTextureCorrection;
/* Opt-in: honour override PNG alpha proportionally on BM_AVERAGE draws.
   Off keeps the binary 0.5 cutout that existing mods were authored against. */
extern int							g_cfg_overrideProportionalAlpha;

/* Debug inputs */
extern GameDebugKeysHandlerFunc		g_dbg_gameDebugKeys;
extern GameDebugMouseHandlerFunc	g_dbg_gameDebugMouse;

/* Usually called at the beginning of main function */
extern void PsyX_Initialise(char* windowName, int screenWidth, int screenHeight, int fullscreen);

/* Cleans all resources and closes open instances */
extern void PsyX_Shutdown(void);

/* Returns the screen size dimensions */
extern void PsyX_GetScreenSize(int* screenWidth, int* screenHeight);

/* Sets mouse cursor position */
extern void PsyX_SetCursorPosition(int x, int y);

/* Sets mouse relative movement */
extern void PsyX_SetCursorRelative(int enable);

/* Returns the SDL window owned by PsyCross. Do not destroy it. */
extern SDL_Window* PsyX_GetSDLWindow(void);

/* Applies a window mode at runtime and resets the render device so the viewport
   and the game's screen-size consumers follow it: `fullscreen` selects desktop
   fullscreen (the size arguments are ignored), otherwise the window is resized
   to width x height (best effort; read the applied size back). Pass
   outWidth/outHeight (either may be NULL) to read back the size SDL actually
   applied, which is what callers must persist. Returns 1 when the window mode
   was reached, 0 when there is no window or SDL refused the fullscreen switch,
   leaving the previous mode in place. */
extern int PsyX_ApplyWindowMode(int fullscreen, int width, int height, int* outWidth, int* outHeight);

/* Hooks are invoked on the render thread. An event handler may return non-zero
   to consume keyboard, mouse, or text input before game debug callbacks. */
extern void PsyX_SetSDLEventHandler(PsyXSDLEventHandlerFunc handler);
extern void PsyX_SetRenderOverlayHandler(PsyXRenderOverlayHandlerFunc handler);
extern void PsyX_SetInputCapture(int captureFlags);

/* Returns aggregate primitive data for the current rendered frame. */
extern void PsyX_GetRenderStats(PsyXRenderStats* stats);

/* Saves the current window contents to SCREENSHOT.BMP in the working
   directory. Used by scripted debug captures; unavailable on web/Android. */
extern void PsyX_TakeScreenshot(void);

/* RGBA texture and texture-region override support. Texture IDs are owned by
   the caller and must be destroyed with PsyX_DestroyRGBATexture when unused. */
extern unsigned int PsyX_CreateRGBATexture(int width, int height, const unsigned char* rgbaPixels);
extern void PsyX_DestroyRGBATexture(unsigned int textureId);

/* Backend-aware texture accessor for developer overlays (Dear ImGui). The
   OpenGL backend returns the GL texture name; the Vulkan backend returns the
   ImGui descriptor set that binds the texture, as an integer. 0 means the
   texture cannot be drawn by the overlay. The handle stays owned by PsyCross
   and must not be destroyed by the caller. */
extern unsigned long long PsyX_GetOverlayTextureId(unsigned int textureId);
/* Real pixel size of a PsyX_CreateRGBATexture texture (0 when unknown). */
extern void PsyX_GetRGBATextureSize(unsigned int textureId, int* width, int* height);
extern int PsyX_RegisterTextureOverride(const PsyXTextureOverride* textureOverride);
extern void PsyX_RemoveTextureOverride(int overrideId);
extern void PsyX_ClearTextureOverrides(void);
extern void PsyX_SetTextureOverridesEnabled(int enabled);
extern void PsyX_GetTextureOverrideStats(PsyXTextureOverrideStats* stats);

/* Inspector support. A mouse pick is resolved after the next completed draw
   stream. Producers may associate a primitive-memory range with a copied,
   human-readable label; ranges are frame-local and never affect rendering. */
extern void PsyX_Inspector_RequestPick(int windowX, int windowY);
/* Applies to the next resolved pick only: the selection keeps the parent of the
   picked component instead of the part, and highlights every part at once. */
extern void PsyX_Inspector_SetWholeObjectPick(int enabled);
extern int PsyX_Inspector_GetSelection(PsyXInspectorSelection* selection);
extern int PsyX_Inspector_GetTriangle(int index, PsyXInspectorTriangle* triangle);
extern void PsyX_Inspector_ClearSelection(void);
extern void PsyX_Inspector_RegisterPrimitiveRange(const void* begin, const void* end, const char* label);
extern void PsyX_Inspector_RegisterObjectRange(const void* begin, const void* end, const char* label, const PsyXInspectorObject* object);
/* Reports a frame-local registered range and how many parsed vertices it owns.
 * A zero vertex count marks a source that is registered but unreachable by
 * picking, so the caller can report it instead of failing silently. */
extern int PsyX_Inspector_GetRangeInfo(int index, char* label, int labelCapacity, int* vertexCount);
/* CPU microseconds spent resolving the most recent pick (0 when none ran). */
extern int PsyX_Inspector_GetPickCostMicros(void);
/* Diagnostic: normalised window bounds of a registered range, filled into
 * g_inspectorRangeBounds[0..3] (minX, minY, maxX, maxY) with
 * g_inspectorRangeBoundsValid set when the range owns visible vertices. */
extern float g_inspectorRangeBounds[4];
extern int g_inspectorRangeBoundsValid;
extern void PsyX_Inspector_GetRangeBounds(int index);
/* Range index behind the current selection, or -1 when it has no source. */
extern int PsyX_Inspector_GetSelectionRangeIndex(void);
/* Diagnostic: scans the frame for override primitives whose texel under the
 * triangle centroid is cut out, filling g_inspectorCutoutPoint (normalised),
 * g_inspectorCutoutBounds, g_inspectorCutoutRange (the owner of the first such
 * point) and g_inspectorCutoutCount. */
extern float g_inspectorCutoutBounds[4];
extern float g_inspectorCutoutPoint[2];
extern float g_inspectorCutoutMaskedPixel[2];
extern int g_inspectorCutoutRange;
extern int g_inspectorCutoutCount;
extern int g_inspectorCutoutMaskedCount;
extern void PsyX_Inspector_FindCutoutSample(void);
/* Diagnostic: finds a pixel that resolves to the given range, filling
 * g_inspectorRangePixel[2] and g_inspectorRangePixelValid. Distinguishes a
 * source that is occluded/unreachable from one that is merely hard to aim. */
extern int g_inspectorRangePixel[2];
extern int g_inspectorRangePixelValid;
extern void PsyX_Inspector_FindRangePixel(int rangeIndex);
/* Diagnostic: first component source that owns a pickable pixel right now.
 * Fills g_inspectorComponentRange (-1 when none) and g_inspectorComponentPixel. */
extern int g_inspectorComponentRange;
extern int g_inspectorComponentPixel[2];
extern void PsyX_Inspector_FindComponentPixel(void);
/* Same, restricted to component keys containing keyFragment (for example
 * "/component:wheel:"); NULL/empty means any component. */
extern void PsyX_Inspector_FindComponentPixelMatching(const char* keyFragment);
/* Diagnostic: point whose override texel is cut out and where the pick finds
 * nothing behind, i.e. the cutout changed the outcome. */
extern int g_inspectorCutoutPickingEnabled;
extern float g_inspectorCutoutFallThrough[2];
extern int g_inspectorCutoutFallThroughValid;
extern void PsyX_Inspector_FindCutoutFallThrough(void);

/* Experimental modern (non-PSX) scene path. A persistent, unlit GPU mesh that
   shares the legacy camera projection and depth buffer, so a modern mesh and
   legacy geometry mutually occlude. This is renderer roadmap R2's seam; it is
   intentionally minimal and does not read PSX primitive streams.
   - positions: 3 floats per vertex, in the same world units as game geometry.
   - indices: optional 3 indices per triangle; NULL for a non-indexed mesh.
   - colors: optional 4 unsigned bytes per vertex (RGBA); NULL for white.
   - viewMatrix: column-major world->camera transform in game units, matching
     the GTE view transform used by the legacy renderer (see draw.c). */
extern int  PsyX_ModernMesh_Create(const float* positions, int vertexCount,
                                   const unsigned short* indices, int indexCount,
                                   const unsigned char* colors);

/* Extended mesh description for the imported static asset path (roadmap R3):
   optional normals/UVs and one base-colour RGBA texture. Attributes stay in
   world units; the texture handle is owned by the caller and must outlive the
   mesh (destroy it with PsyX_DestroyRGBATexture after the mesh). */
typedef struct
{
	const float* positions;			/* 3 floats/vertex, required */
	const float* normals;			/* 3 floats/vertex, optional */
	const float* uvs;				/* 2 floats/vertex, optional */
	const unsigned char* colors;	/* 4 bytes/vertex, optional */
	int vertexCount;
	const unsigned short* indices;	/* optional */
	int indexCount;
	unsigned int baseColorTexture;	/* PsyX_CreateRGBATexture handle, 0 = none */
	unsigned int normalTexture;		/* tangent-space normal map, 0 = none */
	unsigned int metallicRoughnessTexture; /* G=roughness, B=metallic, 0 = none */
	unsigned int emissiveTexture;	/* emissive map, 0 = none */
	const float* baseColorFactor;	/* 4 floats, optional (default white) */
	const float* emissiveFactor;	/* 3 floats, optional (default black) */
	float metallicFactor;			/* default 1 */
	float roughnessFactor;			/* default 1 */
} PsyXModernMeshDesc;

extern int  PsyX_ModernMesh_CreateEx(const PsyXModernMeshDesc* desc);

/* Forward light set for the experimental modern path. `type` 0 is a
   directional light (world-space direction towards the light), 1 is a point
   light (world-space position, linear falloff over `range`). */
#define PSYX_MODERN_MAX_LIGHTS 8

typedef struct
{
	int type;
	float position[3];
	float direction[3];
	float color[3];
	float intensity;
	float range;
} PsyXModernLight;

typedef struct
{
	int count;
	PsyXModernLight lights[PSYX_MODERN_MAX_LIGHTS];
	float ambient[3];
	float exposure;
	int shadowsEnabled;
	int aoEnabled;
	float shadowCenter[3];	/* world-space centre of the shadow volume */
	float shadowExtent;		/* half-size of the orthographic shadow volume */
	/* Legacy-geometry light receptivity (roadmap: legacy-lighting-receptivity).
	   When non-zero, the already-rendered legacy scene is additionally lit by
	   the modern light set in the composite pass: a diffuse sun term (light 0
	   when it is directional) scaled by this factor. The legacy shading itself
	   is preserved; 0 keeps the shipped look. */
	float legacyLightingScale;
} PsyXModernLightSet;

extern void PsyX_ModernMesh_SetLights(const PsyXModernLightSet* lights);

/* Camera state for the experimental modern path, supplied by the game once the
   frame's camera is final (after the legacy scene camera is built and before
   PsyX_EndScene). `viewRotation` is the rotation-only world->camera matrix
   (same fixed-point-derived transform the instances use, no translation) and
   `cameraPosition` is the world-space camera position. It is used to project
   the modern shadow map onto pixels of the already-rendered legacy scene, so
   legacy scenery receives shadows cast by modern meshes, and to reconstruct
   the world position/normal used by legacy lighting receptivity. */
extern void PsyX_ModernMesh_SetCamera(const float viewRotation[16],
                                      const float cameraPosition[3]);

/* Developer diagnostic for the legacy shadow projection pass:
   0 = normal tint, 1 = show the sampled scene depth, 2 = show whether each
   pixel falls inside the shadow volume (green) or not (red). */
extern void PsyX_ModernMesh_SetShadowDebug(int mode);

extern void PsyX_ModernMesh_Destroy(int mesh);
extern void PsyX_ModernMesh_SetInstance(int mesh, const float viewMatrix[16],
                                        const float color[4], int visible);

/* World transform of an instance (mesh local space to world space). The shadow
   map pass renders casters through it, so an instance must supply it whenever
   its view matrix places the mesh anywhere other than the world origin. */
extern void PsyX_ModernMesh_SetInstanceWorld(int mesh, const float worldMatrix[16]);
extern void PsyX_ModernMesh_SetEnabled(int enabled);
extern int  PsyX_ModernMesh_GetEnabled(void);

typedef struct
{
	int meshCount;			/* live meshes */
	int visibleInstances;	/* instances drawn last frame */
	int vertexCount;		/* vertices submitted last frame */
	int drawCalls;			/* GL draw calls last frame */
	int lastFrameMicros;	/* CPU submission cost last frame */
	int depthShared;		/* legacy depth buffer was available */
	int legacyShadowPass;	/* shadow projection ran over the legacy scene */
	int legacyLightPass;	/* lighting receptivity ran over the legacy scene */
} PsyXModernMeshStats;

extern void PsyX_ModernMesh_GetStats(PsyXModernMeshStats* stats);

/* Usually called after ClearOTag/ClearOTagR */
extern char PsyX_BeginScene(void);

/* Usually called after DrawOTag/DrawOTagEnv */
extern void PsyX_EndScene(void);

/* Explicitly updates emulator input loop */
extern void PsyX_UpdateInput(void);

/* Returns keyboard mapping index */
extern int PsyX_LookupKeyboardMapping(const char* str, int default_value);

/* Returns controller mapping index */
extern int PsyX_LookupGameControllerMapping(const char* str, int default_value);

/* Screen size of emulated PSX viewport with widescreen offsets */
extern void PsyX_GetPSXWidescreenMappedViewport(struct _RECT16* rect);

/* Waits for timer */
extern void PsyX_WaitForTimestep(int count);

/* Changes swap interval state */
extern void PsyX_EnableSwapInterval(int enable);

/* Changes swap interval interval interval */
extern void PsyX_SetSwapInterval(int interval);

/* Render backend selection. OpenGL is the default; the Vulkan backend is
   optional and needs the experimental PsyX_Vk_* slice to be available. Set
   this before PsyX_Initialise. */
#define PSYX_BACKEND_OPENGL 0
#define PSYX_BACKEND_VULKAN 1

extern void PsyX_SetRenderBackend(int backend);
extern int  PsyX_GetRenderBackend(void);

/* Invokes the registered post-frame overlay handler (if any) so a backend can
   contribute its widgets to a frame it does not itself drive. */
extern void PsyX_InvokeRenderOverlayHandler(void);

#if defined(_LANGUAGE_C_PLUS_PLUS)||defined(__cplusplus)||defined(c_plusplus)
}
#endif

#endif
