/*
 * Native Vulkan backend for the experimental modern scene (renderer roadmap
 * R7). See PsyX_vk.h for the public contract.
 *
 * Design notes:
 * - The Vulkan loader is discovered at runtime through SDL, so the build needs
 *   the Vulkan headers only (no import library). On macOS SDL resolves
 *   MoltenVK, which is why this file has no platform-specific surface code.
 * - Single frame in flight with a fence wait per frame keeps the
 *   synchronisation simple and correct for a developer tool.
 * - Instance world transforms travel through push constants; the scene uniform
 *   buffer, shadow map and material textures share one descriptor set per mesh.
 */

#include "PsyX/PsyX_vk.h"

#if !defined(PSX) && !defined(__ANDROID__) && !defined(__EMSCRIPTEN__)

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "../platform.h"
#include "../gpu/PsyX_GPU.h"

#include "PsyX_Vk_Shaders.h"

#include "PsyX/PsyX_public.h"
#include "PsyX/PsyX_render.h"

// The in-game modern path shares the projection captured by GR_Perspective3D.
#include "PsyX_ModernMesh.h"
#include "psx/gtereg.h"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <new>

#if !defined(PSYX_VK_DISABLE_IMGUI)
#	define PSYX_VK_IMGUI 1
#	include "imgui.h"
#	include "imgui_impl_sdl2.h"
#	include "imgui_impl_vulkan.h"
#endif

// ---------------------------------------------------------------------------
// Dynamic function loading

#define PSYX_VK_FUNCTIONS(X) \
	X(vkCreateInstance) \
	X(vkDestroyInstance) \
	X(vkEnumerateInstanceExtensionProperties) \
	X(vkEnumerateInstanceLayerProperties) \
	X(vkEnumerateInstanceVersion) \
	X(vkEnumeratePhysicalDevices) \
	X(vkGetPhysicalDeviceProperties) \
	X(vkGetPhysicalDeviceQueueFamilyProperties) \
	X(vkGetPhysicalDeviceMemoryProperties) \
	X(vkGetPhysicalDeviceFormatProperties) \
	X(vkDestroySurfaceKHR) \
	X(vkGetPhysicalDeviceSurfaceSupportKHR) \
	X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
	X(vkGetPhysicalDeviceSurfaceFormatsKHR) \
	X(vkGetPhysicalDeviceSurfacePresentModesKHR) \
	X(vkCreateDevice) \
	X(vkDestroyDevice) \
	X(vkGetDeviceProcAddr) \
	X(vkGetDeviceQueue) \
	X(vkDeviceWaitIdle) \
	X(vkQueueWaitIdle) \
	X(vkQueueSubmit) \
	X(vkQueuePresentKHR) \
	X(vkCreateSwapchainKHR) \
	X(vkDestroySwapchainKHR) \
	X(vkGetSwapchainImagesKHR) \
	X(vkAcquireNextImageKHR) \
	X(vkCreateFence) \
	X(vkDestroyFence) \
	X(vkWaitForFences) \
	X(vkResetFences) \
	X(vkCreateSemaphore) \
	X(vkDestroySemaphore) \
	X(vkCreateBuffer) \
	X(vkDestroyBuffer) \
	X(vkGetBufferMemoryRequirements) \
	X(vkBindBufferMemory) \
	X(vkMapMemory) \
	X(vkUnmapMemory) \
	X(vkAllocateMemory) \
	X(vkFreeMemory) \
	X(vkGetImageMemoryRequirements) \
	X(vkBindImageMemory) \
	X(vkCreateImage) \
	X(vkDestroyImage) \
	X(vkCreateImageView) \
	X(vkDestroyImageView) \
	X(vkCreateSampler) \
	X(vkDestroySampler) \
	X(vkCreateRenderPass) \
	X(vkDestroyRenderPass) \
	X(vkCreateFramebuffer) \
	X(vkDestroyFramebuffer) \
	X(vkCreateShaderModule) \
	X(vkDestroyShaderModule) \
	X(vkCreatePipelineLayout) \
	X(vkDestroyPipelineLayout) \
	X(vkCreateGraphicsPipelines) \
	X(vkDestroyPipeline) \
	X(vkCreateDescriptorSetLayout) \
	X(vkDestroyDescriptorSetLayout) \
	X(vkCreateDescriptorPool) \
	X(vkDestroyDescriptorPool) \
	X(vkAllocateDescriptorSets) \
	X(vkResetDescriptorPool) \
	X(vkUpdateDescriptorSets) \
	X(vkCreateCommandPool) \
	X(vkDestroyCommandPool) \
	X(vkAllocateCommandBuffers) \
	X(vkFreeCommandBuffers) \
	X(vkResetCommandBuffer) \
	X(vkBeginCommandBuffer) \
	X(vkEndCommandBuffer) \
	X(vkCmdBeginRenderPass) \
	X(vkCmdEndRenderPass) \
	X(vkCmdBindPipeline) \
	X(vkCmdBindDescriptorSets) \
	X(vkCmdBindVertexBuffers) \
	X(vkCmdBindIndexBuffer) \
	X(vkCmdPushConstants) \
	X(vkCmdDraw) \
	X(vkCmdDrawIndexed) \
	X(vkCmdSetViewport) \
	X(vkCmdSetScissor) \
	X(vkCmdSetBlendConstants) \
	X(vkCmdPipelineBarrier) \
	X(vkCmdCopyBufferToImage) \
	X(vkCmdCopyImageToBuffer) \
	X(vkCmdCopyImage) \
	X(vkCmdBlitImage) \
	X(vkCmdClearColorImage) \
	X(vkCmdClearAttachments)

#define PSYX_VK_DECLARE_FN(name) static PFN_##name name = nullptr;
PSYX_VK_FUNCTIONS(PSYX_VK_DECLARE_FN)
#undef PSYX_VK_DECLARE_FN

static PFN_vkGetInstanceProcAddr g_gipa = nullptr;
static int g_loaderReady = 0;

// ---------------------------------------------------------------------------
// State

static const int kShadowSize = 2048;

typedef struct
{
	float pos[3];
	float color[4];
	float normal[3];
	float uv[2];
} VkVertex;

typedef struct
{
	float posRange[4];
	float dirType[4];
	float color[4];
} VkLightStd140;

typedef struct
{
	float view[16];
	float proj[16];
	float shadowMatrix[16];
	float cameraPos[4];
	float ambientExposure[4];
	float shadowParams[4];
	float lightInfo[4];
	VkLightStd140 lights[PSYX_VK_MAX_LIGHTS];
} VkSceneUbo;

#define PSYX_VK_MAX_TEXTURES 64

typedef struct
{
	int used;
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
} VkTexture;

typedef struct
{
	int used;
	int vertexCount;
	int indexCount;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexMemory;
	VkDescriptorSet descriptorSet;
	int textureSlots[4];		// handles into the shared texture table, -1 = default
	float factors[3];			// metallic, roughness, emissive scale
	float world[16];
	float color[4];
	int visible;
} VkMesh;

static VkTexture g_textures[PSYX_VK_MAX_TEXTURES];

// ---------------------------------------------------------------------------
// In-game modern mesh (renderer roadmap R7b)
//
// The OpenGL modern path's Vulkan equivalent: the same persistent, world
// anchored meshes drawn into the game's own frame so modern geometry and the
// emulated PSX scene share one depth buffer. Material texture handles are the
// game's PsyX_CreateRGBATexture handles, which the Vulkan backend stores in its
// PSX texture table, not the fixture's separate VkTexture table.

#define PSYX_VK_GAME_MODERN_MAX_MESHES 16

typedef struct
{
	int used;
	int vertexCount;
	int indexCount;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexMemory;
	VkDescriptorSet descriptorSet;
	int textureSlots[4];		// 1-based PSX texture handles, 0 = neutral
	float factors[2];		// metallic, roughness
	float emissive[3];		// emissive factor
	float world[16];
	float color[4];
	int visible;
} VkGameMesh;

// Uniform block shared by the modern mesh shaders and the shadow composite.
// The layout must match ModernUBO in vk_shaders/psx_modern.vert,
// psx_modern.frag and psx_composite.frag.
typedef struct
{
	float proj[16];
	float projInverse[16];
	float shadowMatrix[16];
	float cameraViewInverse[16];
	float cameraRotation[16];
	float xyScale[4];		// x, y, z (1/128), unused
	float shadowParams[4];		// x = enabled, y = texel, z = strength, w = ao
	float lightInfo[4];		// x = count, y = srgb output, z = shadow debug
	float ambientExposure[4];	// rgb = ambient, w = exposure
	float cameraPos[4];
	float viewport[4];		// width, height
	VkLightStd140 lights[PSYX_VK_MAX_LIGHTS];
} VkGameModernUbo;

// ---------------------------------------------------------------------------
// Emulated PSX GPU path (renderer roadmap R7b)
//
// The packed vertex mirrors PsyX_render.h's GrVertex with PGXP enabled; the
// layout is asserted below so a change on either side breaks the build instead
// of the picture.

// Defined in PsyX_render.cpp with C linkage; the PSX self-test exercises the
// VRAM TGA export (F10) so the Vulkan build cannot silently regress to a
// header-only file. Declared here rather than including PsyX_render.h to keep
// the backend independent of the OpenGL renderer header.
extern "C" void GR_SaveVRAM(const char* outputFileName, int x, int y, int width, int height, int bReadFromFrameBuffer);

typedef struct
{
	float x, y, page, clut;
	float z, scr_h, ofsX, ofsY;
	unsigned char u, v, bright, dither;
	unsigned char r, g, b, a;
	signed char tcx, tcy, p0, p1;
} VkPsxVertex;

typedef char VkPsxVertexLayoutCheck[(sizeof(VkPsxVertex) == 44) ? 1 : -1];

#define PSYX_VK_PSX_MAX_DRAWS 4096
#define PSYX_VK_PSX_BLEND_COUNT 5
#define PSYX_VK_PSX_VRAM_BYTES (PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT * 8)
// MAX_VERTEX_BUFFER_SIZE (65536) bounds one flush, as it does in OpenGL; the
// Vulkan backend keeps several flushes so a deferred frame can still read the
// vertices of every earlier flush. 44 is sizeof(VkPsxVertex) (asserted below).
#define PSYX_VK_PSX_MAX_VERTICES 65536
#define PSYX_VK_PSX_VERTEX_FLUSHES 4
#define PSYX_VK_PSX_VERTEX_CAPACITY (PSYX_VK_PSX_MAX_VERTICES * PSYX_VK_PSX_VERTEX_FLUSHES * 44)
#define PSYX_VK_PSX_MAX_TEXTURES 512
// The game rewrites VRAM between draw flushes (the overhead map streams tiles
// through sixteen VRAM slots every frame). The draws are recorded at frame end,
// so those writes are replayed in order instead of uploading the memory once;
// without that, every tile samples the last batch's slot contents.
#define PSYX_VK_PSX_MAX_VRAM_UPLOADS 8192
#define PSYX_VK_PSX_VRAM_UPLOAD_BYTES (8 * 1024 * 1024)
// GR_SetOffscreenState groups per frame (the game reaches for a render-to-VRAM
// target from DR_ENV dfe=0 draws such as the Tanner shadow).
#define PSYX_VK_PSX_MAX_OFFSCREEN_GROUPS 16

typedef struct
{
	int texFormat;
	int bilinearFilter;
	float texelSize[2];
	int overrideAlphaMode;
	int blendMode;
	int depthTest;
	int scissorEnable;
	int scissor[4];			// x, y, width, height, top-left origin
	int viewport[4];		// x, y, width, height, top-left origin
	VkDescriptorSet textureSet;	// 32-bit game texture, or VK_NULL_HANDLE
	uint32_t firstVertex;
	uint32_t vertexCount;
	int frame;			// g_vk.psx.frameIndex at queue time
	int offscreen;			// queued while GR_SetOffscreenState(enable=1)
	int stencilMode;		// 1 = PSX mask-bit set, 0 = mask-bit test
	int srgbEncode;			// 1 when this draw targets an sRGB attachment
	uint32_t vramGeneration;	// VRAM writes visible to this draw
} VkPsxDraw;

// One VRAM rectangle rewritten between draw flushes. `generation` orders the
// write against the deferred draws: a draw sees every write with a smaller
// generation. The pixels are RG32F in `vramUploadStaging`, matching the VRAM
// image, so the replay is a plain buffer-to-image copy.
typedef struct
{
	uint32_t generation;
	int x, y, w, h;
	VkDeviceSize offset;
} VkPsxVramUpload;

// One GR_SetOffscreenState(enable=1 .. enable=0) run: the PSX draws render into
// an offscreen image and are copied back into VRAM at `rect`. The projection
// active at queue time is captured so the offscreen pass reproduces exactly
// what the immediate-mode OpenGL renderer would have drawn.
typedef struct
{
	int frame;			// frameIndex whose draw list holds the run
	int rect[4];
	float matrix[32];
} VkPsxOffscreenGroup;

typedef struct
{
	int used;
	int width;
	int height;
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
	VkSampler sampler;
	VkDescriptorSet set;
} VkPsxTexture;

typedef struct
{
	int ready;
	int failed;

	VkDescriptorSetLayout setLayout;
	VkPipelineLayout layout;
	VkPipeline pipelines[PSYX_VK_PSX_BLEND_COUNT];
	// Mask-bit variants: the PSX primitive flag (drawPrimMode / DrawPrim) both
	// writes the stencil mask (enable=1) and, for every other draw, only passes
	// where the mask is clear (enable=0). Stencil state is baked into the
	// pipeline, so each blend mode needs a second pipeline for the set case.
	VkPipeline pipelinesStencilWrite[PSYX_VK_PSX_BLEND_COUNT];
	int stencilSupported;		// main depth attachment carries an stencil aspect
	VkPipeline pipelineNoDepth;
	VkPipeline pipelineNoDepthStencilWrite;
	VkDescriptorSet set;

	// Offscreen (render-to-VRAM) target, mirroring GR_SetOffscreenState. The
	// dedicated pipelines are needed because a pipeline is bound to the render
	// pass it was created with, and the offscreen pass has no depth attachment.
	VkRenderPass offscreenRenderPass;
	VkPipeline offscreenPipelines[PSYX_VK_PSX_BLEND_COUNT];
	VkImage offscreenImage;
	VkDeviceMemory offscreenMemory;
	VkImageView offscreenView;
	VkFramebuffer offscreenFramebuffer;
	int offscreenWidth;
	int offscreenHeight;
	VkBuffer offscreenReadback;
	VkDeviceMemory offscreenReadbackMemory;
	void* offscreenReadbackMapped;
	VkDeviceSize offscreenReadbackSize;
	int frameIndex;			// g_vk frame the queued draw list belongs to
	int offscreenActive;
	int offscreenRect[4];
	int offscreenGroupPendingDraw;
	float offscreenGroupMatrix[32];
	VkPsxOffscreenGroup offscreenGroups[PSYX_VK_PSX_MAX_OFFSCREEN_GROUPS];
	int offscreenGroupCount;

	VkSampler sampler;		// VRAM: nearest, wrapping
	VkSampler lutSampler;		// RG8 table: nearest, clamped (the shader offsets
					// the coordinate slightly negative, like GL's)

	VkImage vramImage;
	VkDeviceMemory vramMemory;
	VkImageView vramView;
	VkBuffer vramStaging;
	VkDeviceMemory vramStagingMemory;
	void* vramStagingMapped;

	VkImage lutImage;
	VkDeviceMemory lutMemory;
	VkImageView lutView;
	VkBuffer lutStaging;
	VkDeviceMemory lutStagingMemory;
	void* lutStagingMapped;

	VkBuffer ubo;
	VkDeviceMemory uboMemory;
	float* uboMapped;		// mat4 Projection, mat4 Projection3D

	VkBuffer vertexBuffer;
	VkDeviceMemory vertexMemory;
	unsigned char* vertexMapped;
	VkDeviceSize vertexCapacity;
	VkDeviceSize vertexSize;
	uint32_t vertexCount;
	// The game flushes more than once in some frames (the overhead map's
	// 16-tile batches, for one), and the Vulkan draws are recorded at frame
	// end. Each flush therefore has to keep its vertices, so uploads append
	// and every draw is offset by the base of the upload it came from.
	uint32_t vertexUploadBase;	// first vertex of the most recent upload
	uint32_t vertexUploadedTotal;	// vertices kept for the current frame

	// VRAM writes issued between draw flushes, replayed in order while the
	// deferred draw list is recorded.
	uint32_t vramUploadGeneration;
	VkBuffer vramUploadStaging;
	VkDeviceMemory vramUploadStagingMemory;
	void* vramUploadStagingMapped;
	VkDeviceSize vramUploadStagingCapacity;
	VkDeviceSize vramUploadStagingUsed;
	int vramUploadCount;
	int vramUploadsApplied;
	VkPsxVramUpload vramUploads[PSYX_VK_PSX_MAX_VRAM_UPLOADS];

	VkPsxDraw draws[PSYX_VK_PSX_MAX_DRAWS];
	int drawCount;

	/*
	 * Mirror of the GR_* state the game sets between draws. The OpenGL renderer
	 * keeps the equivalent in GL objects and uniforms; here it is plain state
	 * captured when each draw is queued.
	 */
	int stTexFormat;
	int stTexture;			// 1-based game texture handle, 0 = dummy
	int stOverrideWidth;
	int stOverrideHeight;
	int stOverrideAlphaMode;
	int stBlendMode;
	int stDepth;
	int stBilinear;
	int stStencilMode;
	int lastDraws;			// draws recorded for the last on-screen frame
	int lastStencilDraws;		// of those, the mask-bit (stencil write) draws
	int stScissorEnable;
	int stScissor[4];
	int stViewport[4];

	int clearRequested;
	float clearColor[3];

	// Pending back-buffer -> VRAM blit rect (GR_StoreFrameBuffer). Consumed
	// through PsyX_Vk_TakeStoredFrameBuffer once the presented frame is done.
	int frameBufferRect[4];
	int frameBufferPending;

	VkDescriptorSet dummySet;	// white texture, used by 4/8/16-bit draws
	VkPsxTexture textures[PSYX_VK_PSX_MAX_TEXTURES];

	// Developer-overlay (Dear ImGui) bridge for one game texture at a time. The
	// ImGui Vulkan backend draws textures through descriptor sets from its own
	// pool, not GL names, so a preview asks for one and it is cached here.
	int overlayTextureSlot;
	unsigned long long overlayTextureId;
} VkPsxState;

static struct
{
	int initialised;
	int gameMode;			// owns the game window (not the developer fixture)
	SDL_Window* window;
	int width;
	int height;
	int windowWidth;
	int windowHeight;

	VkInstance instance;
	uint32_t apiVersion;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice;
	VkPhysicalDeviceMemoryProperties memoryProperties;
	uint32_t queueFamily;
	VkDevice device;
	VkQueue queue;

	VkSwapchainKHR swapchain;
	VkFormat swapchainFormat;
	int srgbOutput;
	uint32_t swapchainImageCount;
	VkImage swapchainImages[8];
	VkImageView swapchainViews[8];
	VkFramebuffer framebuffers[8];
	int framebufferCount;

	// Which swapchain images already hold a rendered frame. The main pass
	// preserves the previous contents (like OpenGL, which only clears when the
	// draw environment asks for it), so a brand-new image must be cleared once
	// before it can be loaded.
	int swapchainImageDrawn[8];

	// The main pass is closed and reopened around replayed VRAM writes (a
	// transfer cannot be recorded inside a render pass). The reopen uses the
	// modern pass, which loads the same colour/depth attachments.
	VkRenderPassBeginInfo mainPassBegin;
	uint32_t mainPassImageIndex;
	int mainPassOpen;

	VkImage depthImage;
	VkDeviceMemory depthMemory;
	VkImageView depthView;
	VkFormat depthStencilFormat;	// main pass depth attachment (stencil-capable when possible)

	VkRenderPass mainRenderPass;
	VkRenderPass shadowRenderPass;
	VkPipelineLayout pipelineLayout;
	VkPipelineLayout shadowPipelineLayout;
	VkPipeline pbrPipeline;
	VkPipeline shadowPipeline;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;

	VkImage shadowImage;
	VkDeviceMemory shadowMemory;
	VkImageView shadowView;
	VkSampler shadowSampler;
	VkFramebuffer shadowFramebuffer;

	VkBuffer uboBuffer;
	VkDeviceMemory uboMemory;
	VkSceneUbo* uboMapped;

	VkBuffer readbackBuffer;
	VkDeviceMemory readbackMemory;
	unsigned char* readbackMapped;
	VkDeviceSize readbackSize;

	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
	VkFence frameFence;
	VkSemaphore imageAvailable;
	VkSemaphore renderFinished[8];

	VkSampler textureSampler;
	int dummyTextures[4];

	VkPsxState psx;

	VkMesh meshes[PSYX_VK_MAX_MESHES];
	int meshCount;
	int resizePending;
	int frameIndex;
	int lastDrawCalls;

	PsyXModernLightSet lights;

	// In-game modern mesh state. The render pass loads the legacy colour and
	// depth written by the main pass so modern meshes share the PSX depth
	// buffer; the fixture's render pass clears instead.
	VkRenderPass modernRenderPass;
	VkFramebuffer modernFramebuffers[8];
	int modernFramebufferCount;
	VkPipeline gameModernPipeline;
	VkPipeline gameCompositePipeline;
	VkBuffer gameModernUboBuffer;
	VkDeviceMemory gameModernUboMemory;
	VkGameModernUbo* gameModernUboMapped;
	VkDescriptorSet gameCompositeSet;
	VkGameMesh gameMeshes[PSYX_VK_GAME_MODERN_MAX_MESHES];
	int gameModernMeshCount;
	VkImage sceneDepthImage;
	VkDeviceMemory sceneDepthMemory;
	VkImageView sceneDepthView;
	VkSampler sceneDepthSampler;
	float modernCameraRotation[16];
	float modernCameraPosition[3];
	int modernCameraValid;
	int gameModernEnabled;
	int gameModernShadowDebug;
	PsyXModernMeshStats gameModernStats;

	int imguiActive;
	char overlayText[512];
	double lastFps;

	PsyXVkInfo info;
} g_vk;

static int g_supported = -1;

// Shared helpers defined with their owners further down.
static void FillMeshVertices(const PsyXModernMeshDesc* desc, VkVertex* vertices);
static void BuildShadowMatrix(float out[16]);

// PsyCross logging is only wired up once the game has started, so the Vulkan
// initialisation stages also go to a small side log for developer runs.
static void VkStage(const char* stage, ...)
{
	FILE* file = fopen("psyx_vk.log", "a");
	if (!file)
		return;

	va_list args;
	va_start(args, stage);
	vfprintf(file, stage, args);
	va_end(args);
	fprintf(file, "\n");
	fclose(file);
}

// ---------------------------------------------------------------------------
// Small helpers

static const char* VkResultName(VkResult result)
{
	switch (result)
	{
	case VK_SUCCESS: return "VK_SUCCESS";
	case VK_NOT_READY: return "VK_NOT_READY";
	case VK_TIMEOUT: return "VK_TIMEOUT";
	case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
	case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
	case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
	case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
	case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
	case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
	case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
	case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
	case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
	case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
	default: return "VK_ERROR_OTHER";
	}
}

static int VkOk(VkResult result, const char* what)
{
	if (result == VK_SUCCESS)
		return 1;

	eprinterr("PsyX Vulkan: %s failed (%s)\n", what, VkResultName(result));
	return 0;
}

static void MulMatrix4(const float a[16], const float b[16], float out[16])
{
	for (int c = 0; c < 4; c++)
	{
		for (int r = 0; r < 4; r++)
		{
			float sum = 0.0f;
			for (int k = 0; k < 4; k++)
				sum += a[k * 4 + r] * b[c * 4 + k];
			out[c * 4 + r] = sum;
		}
	}
}

static void ImageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
	VkImageLayout oldLayout, VkImageLayout newLayout,
	VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
	VkAccessFlags srcAccess, VkAccessFlags dstAccess)
{
	VkImageMemoryBarrier barrier;
	memset(&barrier, 0, sizeof(barrier));
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = aspect;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = srcAccess;
	barrier.dstAccessMask = dstAccess;

	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, NULL, 0, NULL, 1, &barrier);
}

static uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties)
{
	for (uint32_t i = 0; i < g_vk.memoryProperties.memoryTypeCount; i++)
	{
		if ((typeBits & (1u << i)) &&
			(g_vk.memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	return 0xFFFFFFFFu;
}

static int CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
	VkBuffer* buffer, VkDeviceMemory* memory, void** mapped)
{
	VkBufferCreateInfo info;
	memset(&info, 0, sizeof(info));
	info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size = size;
	info.usage = usage;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (!VkOk(vkCreateBuffer(g_vk.device, &info, NULL, buffer), "vkCreateBuffer"))
		return 0;

	VkMemoryRequirements requirements;
	vkGetBufferMemoryRequirements(g_vk.device, *buffer, &requirements);

	VkMemoryAllocateInfo allocate;
	memset(&allocate, 0, sizeof(allocate));
	allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.allocationSize = requirements.size;
	allocate.memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits, properties);
	if (allocate.memoryTypeIndex == 0xFFFFFFFFu)
	{
		eprinterr("PsyX Vulkan: no memory type for buffer\n");
		return 0;
	}

	if (!VkOk(vkAllocateMemory(g_vk.device, &allocate, NULL, memory), "vkAllocateMemory(buffer)"))
		return 0;
	if (!VkOk(vkBindBufferMemory(g_vk.device, *buffer, *memory, 0), "vkBindBufferMemory"))
		return 0;

	if (mapped)
	{
		if (!VkOk(vkMapMemory(g_vk.device, *memory, 0, size, 0, mapped), "vkMapMemory"))
		{
			*mapped = NULL;
			return 0;
		}
	}
	return 1;
}

static int CreateImage2D(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
	VkImage* image, VkDeviceMemory* memory)
{
	VkImageCreateInfo info;
	memset(&info, 0, sizeof(info));
	info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType = VK_IMAGE_TYPE_2D;
	info.format = format;
	info.extent.width = width;
	info.extent.height = height;
	info.extent.depth = 1;
	info.mipLevels = 1;
	info.arrayLayers = 1;
	info.samples = VK_SAMPLE_COUNT_1_BIT;
	info.tiling = VK_IMAGE_TILING_OPTIMAL;
	info.usage = usage;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	if (!VkOk(vkCreateImage(g_vk.device, &info, NULL, image), "vkCreateImage"))
		return 0;

	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(g_vk.device, *image, &requirements);

	VkMemoryAllocateInfo allocate;
	memset(&allocate, 0, sizeof(allocate));
	allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.allocationSize = requirements.size;
	allocate.memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (allocate.memoryTypeIndex == 0xFFFFFFFFu)
	{
		eprinterr("PsyX Vulkan: no memory type for image\n");
		return 0;
	}

	if (!VkOk(vkAllocateMemory(g_vk.device, &allocate, NULL, memory), "vkAllocateMemory(image)"))
		return 0;
	if (!VkOk(vkBindImageMemory(g_vk.device, *image, *memory, 0), "vkBindImageMemory"))
		return 0;

	return 1;
}

static int CreateImageView2D(VkImage image, VkFormat format, VkImageAspectFlags aspect, VkImageView* view)
{
	VkImageViewCreateInfo info;
	memset(&info, 0, sizeof(info));
	info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	info.image = image;
	info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	info.format = format;
	info.subresourceRange.aspectMask = aspect;
	info.subresourceRange.levelCount = 1;
	info.subresourceRange.layerCount = 1;

	return VkOk(vkCreateImageView(g_vk.device, &info, NULL, view), "vkCreateImageView");
}

static void ImageBarrierLevels(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
	int baseLevel, int levelCount, VkImageLayout oldLayout, VkImageLayout newLayout,
	VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
	VkAccessFlags srcAccess, VkAccessFlags dstAccess)
{
	VkImageMemoryBarrier barrier;
	memset(&barrier, 0, sizeof(barrier));
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = aspect;
	barrier.subresourceRange.baseMipLevel = (uint32_t)baseLevel;
	barrier.subresourceRange.levelCount = (uint32_t)levelCount;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = srcAccess;
	barrier.dstAccessMask = dstAccess;

	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, NULL, 0, NULL, 1, &barrier);
}

static int CreateImage2DLevels(uint32_t width, uint32_t height, int levels, VkFormat format, VkImageUsageFlags usage,
	VkImage* image, VkDeviceMemory* memory)
{
	VkImageCreateInfo info;
	memset(&info, 0, sizeof(info));
	info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType = VK_IMAGE_TYPE_2D;
	info.format = format;
	info.extent.width = width;
	info.extent.height = height;
	info.extent.depth = 1;
	info.mipLevels = (uint32_t)(levels > 0 ? levels : 1);
	info.arrayLayers = 1;
	info.samples = VK_SAMPLE_COUNT_1_BIT;
	info.tiling = VK_IMAGE_TILING_OPTIMAL;
	info.usage = usage;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	if (!VkOk(vkCreateImage(g_vk.device, &info, NULL, image), "vkCreateImage(levels)"))
		return 0;

	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(g_vk.device, *image, &requirements);

	VkMemoryAllocateInfo allocate;
	memset(&allocate, 0, sizeof(allocate));
	allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.allocationSize = requirements.size;
	allocate.memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (allocate.memoryTypeIndex == 0xFFFFFFFFu)
	{
		eprinterr("PsyX Vulkan: no memory type for image\n");
		return 0;
	}

	if (!VkOk(vkAllocateMemory(g_vk.device, &allocate, NULL, memory), "vkAllocateMemory(image)"))
		return 0;
	if (!VkOk(vkBindImageMemory(g_vk.device, *image, *memory, 0), "vkBindImageMemory"))
		return 0;

	return 1;
}

static int CreateImageView2DLevels(VkImage image, VkFormat format, int levels, VkImageView* view)
{
	VkImageViewCreateInfo info;
	memset(&info, 0, sizeof(info));
	info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	info.image = image;
	info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	info.format = format;
	info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	info.subresourceRange.levelCount = (uint32_t)(levels > 0 ? levels : 1);
	info.subresourceRange.layerCount = 1;

	return VkOk(vkCreateImageView(g_vk.device, &info, NULL, view), "vkCreateImageView(levels)");
}

static VkCommandBuffer BeginOneShot(void)
{
	VkCommandBufferAllocateInfo allocate;
	memset(&allocate, 0, sizeof(allocate));
	allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate.commandPool = g_vk.commandPool;
	allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate.commandBufferCount = 1;

	VkCommandBuffer cmd = VK_NULL_HANDLE;
	if (!VkOk(vkAllocateCommandBuffers(g_vk.device, &allocate, &cmd), "vkAllocateCommandBuffers(one-shot)"))
		return VK_NULL_HANDLE;

	VkCommandBufferBeginInfo begin;
	memset(&begin, 0, sizeof(begin));
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &begin);
	return cmd;
}

static void EndOneShot(VkCommandBuffer cmd)
{
	vkEndCommandBuffer(cmd);

	VkSubmitInfo submit;
	memset(&submit, 0, sizeof(submit));
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	vkQueueSubmit(g_vk.queue, 1, &submit, VK_NULL_HANDLE);
	vkQueueWaitIdle(g_vk.queue);
	vkFreeCommandBuffers(g_vk.device, g_vk.commandPool, 1, &cmd);
}

// ---------------------------------------------------------------------------
// Loader

int PsyX_Vk_IsSupported(void)
{
	if (g_supported >= 0)
		return g_supported;

	g_supported = 0;

	if (SDL_WasInit(SDL_INIT_VIDEO) == 0 && SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
		return 0;
	if (SDL_Vulkan_LoadLibrary(NULL) != 0)
	{
		eprinterr("PsyX Vulkan: no Vulkan loader (%s)\n", SDL_GetError());
		return 0;
	}

	g_gipa = (PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr();
	if (!g_gipa)
		return 0;

	g_loaderReady = 1;
	g_supported = 1;
	return g_supported;
}

static void LoadInstanceFunctions()
{
#define PSYX_VK_LOAD_FN(name) name = (PFN_##name)g_gipa(g_vk.instance, #name);
	PSYX_VK_FUNCTIONS(PSYX_VK_LOAD_FN)
#undef PSYX_VK_LOAD_FN
}

// Global commands must be resolved with a NULL instance before the instance
// exists; resolving them afterwards is also valid but too late for creation.
static void LoadGlobalFunctions()
{
	vkCreateInstance = (PFN_vkCreateInstance)g_gipa(NULL, "vkCreateInstance");
	vkEnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)g_gipa(NULL, "vkEnumerateInstanceExtensionProperties");
	vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)g_gipa(NULL, "vkEnumerateInstanceLayerProperties");
	vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)g_gipa(NULL, "vkEnumerateInstanceVersion");
}

// ---------------------------------------------------------------------------
// Swapchain and render targets

static VkFormat PickSurfaceFormat(VkFormat* outFormat, int* outSrgb)
{
	uint32_t count = 0;
	if (!VkOk(vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.physicalDevice, g_vk.surface, &count, NULL), "surface formats") || count == 0)
		return VK_FORMAT_UNDEFINED;

	VkSurfaceFormatKHR formats[64];
	if (count > 64)
		count = 64;
	vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.physicalDevice, g_vk.surface, &count, formats);

	const VkFormat preferred[4] =
	{
		VK_FORMAT_B8G8R8A8_SRGB,
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_FORMAT_B8G8R8A8_UNORM,
		VK_FORMAT_R8G8B8A8_UNORM,
	};

	for (int p = 0; p < 4; p++)
	{
		// PSX blending operates on display-referred values, as in GL with
		// FRAMEBUFFER_SRGB disabled. Shader inverse gamma cannot undo the
		// linear destination read performed by an sRGB blend attachment.
		const int candidate = g_vk.gameMode ? (p + 2) % 4 : p;
		for (uint32_t i = 0; i < count; i++)
		{
			if (formats[i].format == preferred[candidate] &&
				formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				*outFormat = formats[i].format;
				*outSrgb = (candidate < 2) ? 1 : 0;
				return formats[i].format;
			}
		}
	}

	*outFormat = formats[0].format;
	*outSrgb = 0;
	return formats[0].format;
}

static void DestroySwapchainResources(void)
{
	for (int i = 0; i < g_vk.framebufferCount; i++)
	{
		if (g_vk.framebuffers[i])
			vkDestroyFramebuffer(g_vk.device, g_vk.framebuffers[i], NULL);
		g_vk.framebuffers[i] = VK_NULL_HANDLE;
	}
	g_vk.framebufferCount = 0;

	for (int i = 0; i < g_vk.modernFramebufferCount; i++)
	{
		if (g_vk.modernFramebuffers[i])
			vkDestroyFramebuffer(g_vk.device, g_vk.modernFramebuffers[i], NULL);
		g_vk.modernFramebuffers[i] = VK_NULL_HANDLE;
	}
	g_vk.modernFramebufferCount = 0;

	for (uint32_t i = 0; i < g_vk.swapchainImageCount; i++)
	{
		if (g_vk.swapchainViews[i])
			vkDestroyImageView(g_vk.device, g_vk.swapchainViews[i], NULL);
		g_vk.swapchainViews[i] = VK_NULL_HANDLE;
	}

	if (g_vk.sceneDepthView)
	{
		vkDestroyImageView(g_vk.device, g_vk.sceneDepthView, NULL);
		g_vk.sceneDepthView = VK_NULL_HANDLE;
	}
	if (g_vk.sceneDepthImage)
	{
		vkDestroyImage(g_vk.device, g_vk.sceneDepthImage, NULL);
		g_vk.sceneDepthImage = VK_NULL_HANDLE;
	}
	if (g_vk.sceneDepthMemory)
	{
		vkFreeMemory(g_vk.device, g_vk.sceneDepthMemory, NULL);
		g_vk.sceneDepthMemory = VK_NULL_HANDLE;
	}

	if (g_vk.depthView)
	{
		vkDestroyImageView(g_vk.device, g_vk.depthView, NULL);
		g_vk.depthView = VK_NULL_HANDLE;
	}
	if (g_vk.depthImage)
	{
		vkDestroyImage(g_vk.device, g_vk.depthImage, NULL);
		g_vk.depthImage = VK_NULL_HANDLE;
	}
	if (g_vk.depthMemory)
	{
		vkFreeMemory(g_vk.device, g_vk.depthMemory, NULL);
		g_vk.depthMemory = VK_NULL_HANDLE;
	}
}

// The surface format must be known before the render passes are created,
// because the main pass declares its colour attachment with that format.
static int QuerySwapchainFormat(void)
{
	VkFormat format = PickSurfaceFormat(&g_vk.swapchainFormat, &g_vk.srgbOutput);
	if (format == VK_FORMAT_UNDEFINED)
		return 0;

	g_vk.swapchainFormat = format;
	return 1;
}

// Picks the main-pass depth attachment. D24_UNORM_S8_UINT is preferred because
// the emulated PSX GPU needs stencil for the primitive mask bit
// (GR_SetStencilMode); D32_SFLOAT is the fallback when the driver refuses a
// combined format, in which case the mask bit degrades to a no-op.
static VkFormat PickDepthStencilFormat(int* stencilSupported)
{
	VkFormat best = VK_FORMAT_UNDEFINED;
	const VkFormat candidates[] =
	{
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D16_UNORM_S8_UINT,
	};

	for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++)
	{
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(g_vk.physicalDevice, candidates[i], &properties);
		if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0)
		{
			best = candidates[i];
			break;
		}
	}

	if (best != VK_FORMAT_UNDEFINED)
	{
		*stencilSupported = 1;
		return best;
	}

	VkFormatProperties properties;
	vkGetPhysicalDeviceFormatProperties(g_vk.physicalDevice, VK_FORMAT_D32_SFLOAT, &properties);
	if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0)
	{
		*stencilSupported = 0;
		return VK_FORMAT_D32_SFLOAT;
	}

	*stencilSupported = 0;
	return VK_FORMAT_D16_UNORM;
}

static int CreateSwapchain(void)
{
	VkSurfaceCapabilitiesKHR capabilities;
	if (!VkOk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk.physicalDevice, g_vk.surface, &capabilities), "surface capabilities"))
		return 0;

	uint32_t width = (uint32_t)g_vk.windowWidth;
	uint32_t height = (uint32_t)g_vk.windowHeight;
	if (capabilities.currentExtent.width != 0xFFFFFFFFu)
	{
		width = capabilities.currentExtent.width;
		height = capabilities.currentExtent.height;
	}
	if (width == 0 || height == 0)
		return 0;

	if (width < capabilities.minImageExtent.width) width = capabilities.minImageExtent.width;
	if (height < capabilities.minImageExtent.height) height = capabilities.minImageExtent.height;
	if (width > capabilities.maxImageExtent.width) width = capabilities.maxImageExtent.width;
	if (height > capabilities.maxImageExtent.height) height = capabilities.maxImageExtent.height;

	uint32_t imageCount = capabilities.minImageCount + 1;
	if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
		imageCount = capabilities.maxImageCount;
	if (imageCount > 8)
		imageCount = 8;

	// Normally already negotiated during initialisation (the render passes
	// need it); fall back to querying here so a resize still works.
	if (g_vk.swapchainFormat == VK_FORMAT_UNDEFINED && !QuerySwapchainFormat())
		return 0;

	VkFormat format = g_vk.swapchainFormat;

	VkSwapchainCreateInfoKHR info;
	memset(&info, 0, sizeof(info));
	info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	info.surface = g_vk.surface;
	info.minImageCount = imageCount;
	info.imageFormat = format;
	info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	info.imageExtent.width = width;
	info.imageExtent.height = height;
	info.imageArrayLayers = 1;
	info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	info.preTransform = capabilities.currentTransform;
	info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
	// Developer/benchmark override. The shipped default stays FIFO (vsync),
	// which caps the reported rate at the display refresh and would make a
	// backend throughput comparison meaningless. PSYX_VK_PRESENT_MODE=mailbox
	// (or immediate) picks an uncapped mode when the surface supports it.
	if (const char* requestedMode = getenv("PSYX_VK_PRESENT_MODE"))
	{
		VkPresentModeKHR wanted = VK_PRESENT_MODE_FIFO_KHR;
		if (!strcmp(requestedMode, "mailbox"))
			wanted = VK_PRESENT_MODE_MAILBOX_KHR;
		else if (!strcmp(requestedMode, "immediate"))
			wanted = VK_PRESENT_MODE_IMMEDIATE_KHR;

		if (wanted != VK_PRESENT_MODE_FIFO_KHR)
		{
			uint32_t modeCount = 0;
			vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.physicalDevice, g_vk.surface, &modeCount, NULL);
			VkPresentModeKHR modes[16];
			if (modeCount > 16)
				modeCount = 16;
			if (modeCount > 0 &&
				vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.physicalDevice, g_vk.surface, &modeCount, modes) == VK_SUCCESS)
			{
				for (uint32_t m = 0; m < modeCount; m++)
				{
					if (modes[m] == wanted)
					{
						info.presentMode = wanted;
						break;
					}
				}
			}
		}
	}
	info.clipped = VK_TRUE;
	info.oldSwapchain = g_vk.swapchain;

	g_vk.swapchainFormat = format;

	if (!VkOk(vkCreateSwapchainKHR(g_vk.device, &info, NULL, &g_vk.swapchain), "vkCreateSwapchainKHR"))
		return 0;

	uint32_t actualImageCount = 0;
	if (!VkOk(vkGetSwapchainImagesKHR(g_vk.device, g_vk.swapchain, &actualImageCount, NULL), "swapchain image count"))
		return 0;
	if (actualImageCount > 8)
		actualImageCount = 8;
	g_vk.swapchainImageCount = actualImageCount;
	vkGetSwapchainImagesKHR(g_vk.device, g_vk.swapchain, &g_vk.swapchainImageCount, g_vk.swapchainImages);

	for (uint32_t i = 0; i < g_vk.swapchainImageCount; i++)
	{
		if (!CreateImageView2D(g_vk.swapchainImages[i], format, VK_IMAGE_ASPECT_COLOR_BIT, &g_vk.swapchainViews[i]))
			return 0;
	}

	// Depth buffer, shared by every framebuffer (single frame in flight). The
	// format was fixed by CreateRenderPasses; recompute only if unset.
	VkFormat depthFormat = g_vk.depthStencilFormat;
	if (depthFormat == VK_FORMAT_UNDEFINED)
		depthFormat = PickDepthStencilFormat(&g_vk.psx.stencilSupported);
	g_vk.depthStencilFormat = depthFormat;

	VkImageAspectFlags depthAspect = (depthFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
		depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT || depthFormat == VK_FORMAT_D16_UNORM_S8_UINT)
		? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
		: VK_IMAGE_ASPECT_DEPTH_BIT;

	// TRANSFER_SRC lets the in-game modern path copy the legacy scene depth out
	// for the shadow composite.
	if (!CreateImage2D(width, height, depthFormat,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		&g_vk.depthImage, &g_vk.depthMemory))
		return 0;
	if (!CreateImageView2D(g_vk.depthImage, depthFormat, depthAspect, &g_vk.depthView))
		return 0;

	for (uint32_t i = 0; i < g_vk.swapchainImageCount; i++)
	{
		VkImageView attachments[2] = { g_vk.swapchainViews[i], g_vk.depthView };

		VkFramebufferCreateInfo framebuffer;
		memset(&framebuffer, 0, sizeof(framebuffer));
		framebuffer.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer.renderPass = g_vk.mainRenderPass;
		framebuffer.attachmentCount = 2;
		framebuffer.pAttachments = attachments;
		framebuffer.width = width;
		framebuffer.height = height;
		framebuffer.layers = 1;

		if (!VkOk(vkCreateFramebuffer(g_vk.device, &framebuffer, NULL, &g_vk.framebuffers[i]), "vkCreateFramebuffer"))
			return 0;
	}
	g_vk.framebufferCount = (int)g_vk.swapchainImageCount;

	// The modern pass uses the same views with a load render pass.
	for (uint32_t i = 0; i < g_vk.swapchainImageCount; i++)
	{
		VkImageView attachments[2] = { g_vk.swapchainViews[i], g_vk.depthView };

		VkFramebufferCreateInfo framebuffer;
		memset(&framebuffer, 0, sizeof(framebuffer));
		framebuffer.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer.renderPass = g_vk.modernRenderPass;
		framebuffer.attachmentCount = 2;
		framebuffer.pAttachments = attachments;
		framebuffer.width = width;
		framebuffer.height = height;
		framebuffer.layers = 1;

		if (!VkOk(vkCreateFramebuffer(g_vk.device, &framebuffer, NULL, &g_vk.modernFramebuffers[i]), "vkCreateFramebuffer(modern)"))
			return 0;
	}
	g_vk.modernFramebufferCount = (int)g_vk.swapchainImageCount;

	// Fresh images have undefined contents, so force a clear on first use.
	memset(g_vk.swapchainImageDrawn, 0, sizeof(g_vk.swapchainImageDrawn));

	// Sampleable copy of the legacy scene depth for the shadow composite.
	if (!CreateImage2D(width, height, depthFormat,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		&g_vk.sceneDepthImage, &g_vk.sceneDepthMemory))
		return 0;
	if (!CreateImageView2D(g_vk.sceneDepthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, &g_vk.sceneDepthView))
		return 0;

	// Readback buffer for the last frame.
	if (g_vk.readbackBuffer)
	{
		vkDestroyBuffer(g_vk.device, g_vk.readbackBuffer, NULL);
		vkFreeMemory(g_vk.device, g_vk.readbackMemory, NULL);
		g_vk.readbackBuffer = VK_NULL_HANDLE;
		g_vk.readbackMemory = VK_NULL_HANDLE;
		g_vk.readbackMapped = NULL;
	}

	g_vk.readbackSize = (VkDeviceSize)width * height * 4;
	if (!CreateBuffer(g_vk.readbackSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&g_vk.readbackBuffer, &g_vk.readbackMemory, (void**)&g_vk.readbackMapped))
		return 0;

	g_vk.width = (int)width;
	g_vk.height = (int)height;
	g_vk.info.width = (int)width;
	g_vk.info.height = (int)height;

	eprintinfo("PsyX Vulkan: swapchain %dx%d, %d images, %s format\n", (int)width, (int)height,
		(int)g_vk.swapchainImageCount, g_vk.srgbOutput ? "sRGB" : "linear");

	return 1;
}

static void DestroySwapchain(void)
{
	if (g_vk.swapchain)
	{
		vkDestroySwapchainKHR(g_vk.device, g_vk.swapchain, NULL);
		g_vk.swapchain = VK_NULL_HANDLE;
		g_vk.swapchainImageCount = 0;
	}
}

// Full swapchain recreation for a resize or an out-of-date acquire/present.
// DestroySwapchainResources() must run first: recreating the framebuffers while
// the previous ones still reference the old swapchain views and depth image
// leaks both and leaves dangling attachments. The device is idle before any of
// it so nothing in flight still uses the images being released.
static int RecreateSwapchain(void)
{
	vkDeviceWaitIdle(g_vk.device);
	DestroySwapchainResources();
	DestroySwapchain();
	const int ok = CreateSwapchain();
	if (ok)
	{
		char line[128];
		snprintf(line, sizeof(line), "swapchain recreated %dx%d images=%d",
			g_vk.width, g_vk.height, g_vk.swapchainImageCount);
		VkStage(line);
	}
	return ok;
}

// ---------------------------------------------------------------------------
// Pipelines and descriptors

static VkShaderModule CreateShaderModuleFromSpirv(const unsigned int* words, unsigned int wordCount)
{
	VkShaderModuleCreateInfo info;
	memset(&info, 0, sizeof(info));
	info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	info.codeSize = (size_t)wordCount * 4;
	info.pCode = words;

	VkShaderModule module = VK_NULL_HANDLE;
	if (!VkOk(vkCreateShaderModule(g_vk.device, &info, NULL, &module), "vkCreateShaderModule"))
		return VK_NULL_HANDLE;
	return module;
}

static int CreatePipelines(void)
{
	VkDescriptorSetLayoutBinding bindings[6];
	memset(bindings, 0, sizeof(bindings));

	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	for (int i = 1; i < 6; i++)
	{
		bindings[i].binding = (uint32_t)i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo;
	memset(&layoutInfo, 0, sizeof(layoutInfo));
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 6;
	layoutInfo.pBindings = bindings;

	if (!VkOk(vkCreateDescriptorSetLayout(g_vk.device, &layoutInfo, NULL, &g_vk.descriptorSetLayout), "vkCreateDescriptorSetLayout"))
		return 0;

	VkPushConstantRange pushRange;
	memset(&pushRange, 0, sizeof(pushRange));
	pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	// mat4 world + vec4 color + vec4 factors + vec4 emissive factor. The
	// fixture shader only reads the first 96 bytes of the block.
	pushRange.size = 112;

	VkPipelineLayoutCreateInfo pipelineLayout;
	memset(&pipelineLayout, 0, sizeof(pipelineLayout));
	pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayout.setLayoutCount = 1;
	pipelineLayout.pSetLayouts = &g_vk.descriptorSetLayout;
	pipelineLayout.pushConstantRangeCount = 1;
	pipelineLayout.pPushConstantRanges = &pushRange;

	if (!VkOk(vkCreatePipelineLayout(g_vk.device, &pipelineLayout, NULL, &g_vk.pipelineLayout), "vkCreatePipelineLayout(main)"))
		return 0;

	VkShaderModule vertex = CreateShaderModuleFromSpirv(fixture_vert_spv, fixture_vert_spv_size);
	VkShaderModule fragment = CreateShaderModuleFromSpirv(fixture_frag_spv, fixture_frag_spv_size);
	if (!vertex || !fragment)
		return 0;

	VkPipelineShaderStageCreateInfo stages[2];
	memset(stages, 0, sizeof(stages));
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vertex;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fragment;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding;
	memset(&binding, 0, sizeof(binding));
	binding.binding = 0;
	binding.stride = sizeof(VkVertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributes[4];
	memset(attributes, 0, sizeof(attributes));
	attributes[0].location = 0; attributes[0].binding = 0; attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT; attributes[0].offset = 0;
	attributes[1].location = 1; attributes[1].binding = 0; attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; attributes[1].offset = 12;
	attributes[2].location = 2; attributes[2].binding = 0; attributes[2].format = VK_FORMAT_R32G32B32_SFLOAT; attributes[2].offset = 28;
	attributes[3].location = 3; attributes[3].binding = 0; attributes[3].format = VK_FORMAT_R32G32_SFLOAT; attributes[3].offset = 40;

	VkPipelineVertexInputStateCreateInfo vertexInput;
	memset(&vertexInput, 0, sizeof(vertexInput));
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 4;
	vertexInput.pVertexAttributeDescriptions = attributes;

	VkPipelineInputAssemblyStateCreateInfo assembly;
	memset(&assembly, 0, sizeof(assembly));
	assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewport;
	memset(&viewport, 0, sizeof(viewport));
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = 1;
	viewport.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo raster;
	memset(&raster, 0, sizeof(raster));
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisample;
	memset(&multisample, 0, sizeof(multisample));
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depthStencil;
	memset(&depthStencil, 0, sizeof(depthStencil));
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

	VkPipelineColorBlendAttachmentState blendAttachment;
	memset(&blendAttachment, 0, sizeof(blendAttachment));
	blendAttachment.blendEnable = VK_FALSE;
	blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo blend;
	memset(&blend, 0, sizeof(blend));
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 1;
	blend.pAttachments = &blendAttachment;

	VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamic;
	memset(&dynamic, 0, sizeof(dynamic));
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dynamicStates;

	VkGraphicsPipelineCreateInfo pipeline;
	memset(&pipeline, 0, sizeof(pipeline));
	pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline.stageCount = 2;
	pipeline.pStages = stages;
	pipeline.pVertexInputState = &vertexInput;
	pipeline.pInputAssemblyState = &assembly;
	pipeline.pViewportState = &viewport;
	pipeline.pRasterizationState = &raster;
	pipeline.pMultisampleState = &multisample;
	pipeline.pDepthStencilState = &depthStencil;
	pipeline.pColorBlendState = &blend;
	pipeline.pDynamicState = &dynamic;
	pipeline.layout = g_vk.pipelineLayout;
	pipeline.renderPass = g_vk.mainRenderPass;
	pipeline.subpass = 0;

	if (!VkOk(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &pipeline, NULL, &g_vk.pbrPipeline), "vkCreateGraphicsPipelines(pbr)"))
		return 0;

	vkDestroyShaderModule(g_vk.device, vertex, NULL);
	vkDestroyShaderModule(g_vk.device, fragment, NULL);

	// Shadow pipeline: push constants only.
	VkPushConstantRange shadowPush;
	memset(&shadowPush, 0, sizeof(shadowPush));
	shadowPush.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	shadowPush.offset = 0;
	shadowPush.size = 64;

	VkPipelineLayoutCreateInfo shadowLayout;
	memset(&shadowLayout, 0, sizeof(shadowLayout));
	shadowLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	shadowLayout.pushConstantRangeCount = 1;
	shadowLayout.pPushConstantRanges = &shadowPush;

	if (!VkOk(vkCreatePipelineLayout(g_vk.device, &shadowLayout, NULL, &g_vk.shadowPipelineLayout), "vkCreatePipelineLayout(shadow)"))
		return 0;

	VkShaderModule shadowVertex = CreateShaderModuleFromSpirv(shadow_vert_spv, shadow_vert_spv_size);
	VkShaderModule shadowFragment = CreateShaderModuleFromSpirv(shadow_frag_spv, shadow_frag_spv_size);
	if (!shadowVertex || !shadowFragment)
		return 0;

	VkPipelineShaderStageCreateInfo shadowStages[2];
	memset(shadowStages, 0, sizeof(shadowStages));
	shadowStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shadowStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	shadowStages[0].module = shadowVertex;
	shadowStages[0].pName = "main";
	shadowStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shadowStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shadowStages[1].module = shadowFragment;
	shadowStages[1].pName = "main";

	VkVertexInputBindingDescription shadowBinding;
	memset(&shadowBinding, 0, sizeof(shadowBinding));
	shadowBinding.binding = 0;
	shadowBinding.stride = sizeof(VkVertex);
	shadowBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription shadowAttribute;
	memset(&shadowAttribute, 0, sizeof(shadowAttribute));
	shadowAttribute.location = 0;
	shadowAttribute.binding = 0;
	shadowAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	shadowAttribute.offset = 0;

	VkPipelineVertexInputStateCreateInfo shadowVertexInput;
	memset(&shadowVertexInput, 0, sizeof(shadowVertexInput));
	shadowVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	shadowVertexInput.vertexBindingDescriptionCount = 1;
	shadowVertexInput.pVertexBindingDescriptions = &shadowBinding;
	shadowVertexInput.vertexAttributeDescriptionCount = 1;
	shadowVertexInput.pVertexAttributeDescriptions = &shadowAttribute;

	VkPipelineRasterizationStateCreateInfo shadowRaster;
	memset(&shadowRaster, 0, sizeof(shadowRaster));
	shadowRaster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	shadowRaster.polygonMode = VK_POLYGON_MODE_FILL;
	shadowRaster.cullMode = VK_CULL_MODE_NONE;
	shadowRaster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	shadowRaster.lineWidth = 1.0f;

	VkPipelineDepthStencilStateCreateInfo shadowDepth;
	memset(&shadowDepth, 0, sizeof(shadowDepth));
	shadowDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	shadowDepth.depthTestEnable = VK_TRUE;
	shadowDepth.depthWriteEnable = VK_TRUE;
	shadowDepth.depthCompareOp = VK_COMPARE_OP_LESS;

	VkPipelineColorBlendStateCreateInfo shadowBlend;
	memset(&shadowBlend, 0, sizeof(shadowBlend));
	shadowBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

	memset(&pipeline, 0, sizeof(pipeline));
	pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline.stageCount = 2;
	pipeline.pStages = shadowStages;
	pipeline.pVertexInputState = &shadowVertexInput;
	pipeline.pInputAssemblyState = &assembly;
	pipeline.pViewportState = &viewport;
	pipeline.pRasterizationState = &shadowRaster;
	pipeline.pMultisampleState = &multisample;
	pipeline.pDepthStencilState = &shadowDepth;
	pipeline.pColorBlendState = &shadowBlend;
	pipeline.pDynamicState = &dynamic;
	pipeline.layout = g_vk.shadowPipelineLayout;
	pipeline.renderPass = g_vk.shadowRenderPass;

	if (!VkOk(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &pipeline, NULL, &g_vk.shadowPipeline), "vkCreateGraphicsPipelines(shadow)"))
		return 0;

	vkDestroyShaderModule(g_vk.device, shadowVertex, NULL);
	vkDestroyShaderModule(g_vk.device, shadowFragment, NULL);

	VkDescriptorPoolSize poolSizes[2];
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	// Meshes get one set each; the PSX path takes one set for the dummy
	// (VRAM-decoded) case plus one per game RGBA texture. The in-game modern
	// path takes one set per mesh plus one for the shadow composite.
	const int psxSetCount = PSYX_VK_PSX_MAX_TEXTURES + 1;
	const int gameSetCount = PSYX_VK_GAME_MODERN_MAX_MESHES + 1;
	const int totalSets = PSYX_VK_MAX_MESHES + 4 + psxSetCount + gameSetCount;
	poolSizes[0].descriptorCount = totalSets;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = (PSYX_VK_MAX_MESHES + 4) * 5 + psxSetCount * 3 + PSYX_VK_GAME_MODERN_MAX_MESHES * 5 + 2;

	VkDescriptorPoolCreateInfo pool;
	memset(&pool, 0, sizeof(pool));
	pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool.maxSets = totalSets;
	pool.poolSizeCount = 2;
	pool.pPoolSizes = poolSizes;

	if (!VkOk(vkCreateDescriptorPool(g_vk.device, &pool, NULL, &g_vk.descriptorPool), "vkCreateDescriptorPool"))
		return 0;

	return 1;
}

// Pipelines for the in-game modern mesh path. Both draw into the load render
// pass so they see the legacy scene colour and share its depth buffer.
static int CreateGameModernPipelines(void)
{
	if (!g_vk.modernRenderPass)
		return 0;

	VkVertexInputBindingDescription binding;
	memset(&binding, 0, sizeof(binding));
	binding.binding = 0;
	binding.stride = sizeof(VkVertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributes[4];
	memset(attributes, 0, sizeof(attributes));
	attributes[0].location = 0; attributes[0].binding = 0; attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT; attributes[0].offset = 0;
	attributes[1].location = 1; attributes[1].binding = 0; attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; attributes[1].offset = 12;
	attributes[2].location = 2; attributes[2].binding = 0; attributes[2].format = VK_FORMAT_R32G32B32_SFLOAT; attributes[2].offset = 28;
	attributes[3].location = 3; attributes[3].binding = 0; attributes[3].format = VK_FORMAT_R32G32_SFLOAT; attributes[3].offset = 40;

	VkPipelineVertexInputStateCreateInfo vertexInput;
	memset(&vertexInput, 0, sizeof(vertexInput));
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 4;
	vertexInput.pVertexAttributeDescriptions = attributes;

	VkPipelineInputAssemblyStateCreateInfo assembly;
	memset(&assembly, 0, sizeof(assembly));
	assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewport;
	memset(&viewport, 0, sizeof(viewport));
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = 1;
	viewport.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo raster;
	memset(&raster, 0, sizeof(raster));
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisample;
	memset(&multisample, 0, sizeof(multisample));
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamic;
	memset(&dynamic, 0, sizeof(dynamic));
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dynamicStates;

	// PBR pass: opaque, shares the legacy depth (LEQUAL, writes).
	{
		VkShaderModule vertex = CreateShaderModuleFromSpirv(psx_modern_vert_spv, psx_modern_vert_spv_size);
		VkShaderModule fragment = CreateShaderModuleFromSpirv(psx_modern_frag_spv, psx_modern_frag_spv_size);
		if (!vertex || !fragment)
			return 0;

		VkPipelineShaderStageCreateInfo stages[2];
		memset(stages, 0, sizeof(stages));
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertex;
		stages[0].pName = "main";
		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragment;
		stages[1].pName = "main";

		VkPipelineDepthStencilStateCreateInfo depthStencil;
		memset(&depthStencil, 0, sizeof(depthStencil));
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

		VkPipelineColorBlendAttachmentState blendAttachment;
		memset(&blendAttachment, 0, sizeof(blendAttachment));
		blendAttachment.blendEnable = VK_FALSE;
		blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blend;
		memset(&blend, 0, sizeof(blend));
		blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blend.attachmentCount = 1;
		blend.pAttachments = &blendAttachment;

		VkGraphicsPipelineCreateInfo pipeline;
		memset(&pipeline, 0, sizeof(pipeline));
		pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline.stageCount = 2;
		pipeline.pStages = stages;
		pipeline.pVertexInputState = &vertexInput;
		pipeline.pInputAssemblyState = &assembly;
		pipeline.pViewportState = &viewport;
		pipeline.pRasterizationState = &raster;
		pipeline.pMultisampleState = &multisample;
		pipeline.pDepthStencilState = &depthStencil;
		pipeline.pColorBlendState = &blend;
		pipeline.pDynamicState = &dynamic;
		pipeline.layout = g_vk.pipelineLayout;
		pipeline.renderPass = g_vk.modernRenderPass;
		pipeline.subpass = 0;

		const int ok = VkOk(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &pipeline, NULL, &g_vk.gameModernPipeline),
			"vkCreateGraphicsPipelines(game modern)");

		vkDestroyShaderModule(g_vk.device, vertex, NULL);
		vkDestroyShaderModule(g_vk.device, fragment, NULL);

		if (!ok)
			return 0;
	}

	// Shadow composite: fullscreen triangle over the legacy colour, multiplied
	// by the shadow term (the OpenGL path's GL_DST_COLOR/GL_ZERO blend).
	{
		VkShaderModule vertex = CreateShaderModuleFromSpirv(fullscreen_vert_spv, fullscreen_vert_spv_size);
		VkShaderModule fragment = CreateShaderModuleFromSpirv(psx_composite_frag_spv, psx_composite_frag_spv_size);
		if (!vertex || !fragment)
			return 0;

		VkPipelineShaderStageCreateInfo stages[2];
		memset(stages, 0, sizeof(stages));
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertex;
		stages[0].pName = "main";
		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragment;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo emptyVertexInput;
		memset(&emptyVertexInput, 0, sizeof(emptyVertexInput));
		emptyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		VkPipelineDepthStencilStateCreateInfo depthStencil;
		memset(&depthStencil, 0, sizeof(depthStencil));
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState blendAttachment;
		memset(&blendAttachment, 0, sizeof(blendAttachment));
		blendAttachment.blendEnable = VK_TRUE;
		blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
		blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blend;
		memset(&blend, 0, sizeof(blend));
		blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blend.attachmentCount = 1;
		blend.pAttachments = &blendAttachment;

		VkGraphicsPipelineCreateInfo pipeline;
		memset(&pipeline, 0, sizeof(pipeline));
		pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline.stageCount = 2;
		pipeline.pStages = stages;
		pipeline.pVertexInputState = &emptyVertexInput;
		pipeline.pInputAssemblyState = &assembly;
		pipeline.pViewportState = &viewport;
		pipeline.pRasterizationState = &raster;
		pipeline.pMultisampleState = &multisample;
		pipeline.pDepthStencilState = &depthStencil;
		pipeline.pColorBlendState = &blend;
		pipeline.pDynamicState = &dynamic;
		pipeline.layout = g_vk.pipelineLayout;
		pipeline.renderPass = g_vk.modernRenderPass;
		pipeline.subpass = 0;

		const int ok = VkOk(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &pipeline, NULL, &g_vk.gameCompositePipeline),
			"vkCreateGraphicsPipelines(game composite)");

		vkDestroyShaderModule(g_vk.device, vertex, NULL);
		vkDestroyShaderModule(g_vk.device, fragment, NULL);

		if (!ok)
			return 0;
	}

	return 1;
}

static int CreateGameModernResources(void)
{
	if (!CreateBuffer(sizeof(VkGameModernUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&g_vk.gameModernUboBuffer, &g_vk.gameModernUboMemory, (void**)&g_vk.gameModernUboMapped))
		return 0;

	memset(g_vk.gameModernUboMapped, 0, sizeof(VkGameModernUbo));
	return CreateGameModernPipelines();
}

// ---------------------------------------------------------------------------
// Render passes

static int CreateRenderPasses(void)
{
	// Main pass: swapchain colour + depth.
	VkAttachmentDescription attachments[2];
	memset(attachments, 0, sizeof(attachments));

	attachments[0].format = g_vk.swapchainFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	// The main pass preserves the previous frame unless the game asked for a
	// clear (GR_Clear), mirroring the OpenGL renderer which only clears when
	// activeDrawEnv.isbg is set. Loading art and other code that relies on the
	// framebuffer persisting therefore survive. The first use of a swapchain
	// image is cleared explicitly because its contents are undefined.
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkFormat depthFormat = PickDepthStencilFormat(&g_vk.psx.stencilSupported);
	g_vk.depthStencilFormat = depthFormat;

	attachments[1].format = depthFormat;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	// The depth and mask bit have to survive the pass being closed and reopened
	// when a deferred VRAM write is replayed mid-frame, so they are stored rather
	// than discarded. They are still cleared at the start of every frame.
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	// The PSX mask bit lives in this attachment's stencil aspect; it is cleared
	// every frame so the mask never leaks between frames.
	attachments[1].stencilLoadOp = (g_vk.psx.stencilSupported) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[1].stencilStoreOp = (g_vk.psx.stencilSupported) ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorReference;
	memset(&colorReference, 0, sizeof(colorReference));
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthReference;
	memset(&depthReference, 0, sizeof(depthReference));
	depthReference.attachment = 1;
	depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass;
	memset(&subpass, 0, sizeof(subpass));
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorReference;
	subpass.pDepthStencilAttachment = &depthReference;

	VkSubpassDependency dependency;
	memset(&dependency, 0, sizeof(dependency));
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
		VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependency.dstStageMask = dependency.srcStageMask;
	dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
		VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo renderPass;
	memset(&renderPass, 0, sizeof(renderPass));
	renderPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPass.attachmentCount = 2;
	renderPass.pAttachments = attachments;
	renderPass.subpassCount = 1;
	renderPass.pSubpasses = &subpass;
	renderPass.dependencyCount = 1;
	renderPass.pDependencies = &dependency;

	if (!VkOk(vkCreateRenderPass(g_vk.device, &renderPass, NULL, &g_vk.mainRenderPass), "vkCreateRenderPass(main)"))
		return 0;

	VkStage("main pass depth-stencil format %d stencil=%d", (int)depthFormat, g_vk.psx.stencilSupported);

	// Modern pass: the in-game modern meshes and the shadow composite draw
	// after the legacy scene into the same colour/depth images, so this pass
	// loads instead of clearing. The attachment formats and references match
	// the main pass, so the same attachment views are used.
	VkAttachmentDescription modernAttachments[2];
	memset(modernAttachments, 0, sizeof(modernAttachments));
	modernAttachments[0] = attachments[0];
	modernAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	modernAttachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	modernAttachments[1] = attachments[1];
	modernAttachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	modernAttachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	modernAttachments[1].stencilStoreOp = attachments[1].stencilStoreOp;
	modernAttachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	// Both passes make prior colour/depth writes available to attachment loads.
	// Dependencies must match too for pipelines to be render-pass compatible.
	VkSubpassDependency modernDependency = dependency;

	memset(&renderPass, 0, sizeof(renderPass));
	renderPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPass.attachmentCount = 2;
	renderPass.pAttachments = modernAttachments;
	renderPass.subpassCount = 1;
	renderPass.pSubpasses = &subpass;
	renderPass.dependencyCount = 1;
	renderPass.pDependencies = &modernDependency;

	if (!VkOk(vkCreateRenderPass(g_vk.device, &renderPass, NULL, &g_vk.modernRenderPass), "vkCreateRenderPass(modern)"))
		return 0;

	// Shadow pass: depth only, sampled afterwards.
	VkAttachmentDescription shadowAttachment;
	memset(&shadowAttachment, 0, sizeof(shadowAttachment));
	shadowAttachment.format = VK_FORMAT_D32_SFLOAT;
	shadowAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	shadowAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	shadowAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	shadowAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	shadowAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	shadowAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	shadowAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// D32 only exists when the device supports it; otherwise use the first
	// supported depth format. The shadow map always uses its own image, so the
	// format can differ from the swapchain depth buffer.
	VkFormatProperties formatProperties;
	vkGetPhysicalDeviceFormatProperties(g_vk.physicalDevice, VK_FORMAT_D32_SFLOAT, &formatProperties);
	if ((formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0)
	{
		shadowAttachment.format = VK_FORMAT_D16_UNORM;
	}

	VkAttachmentReference shadowDepthReference;
	memset(&shadowDepthReference, 0, sizeof(shadowDepthReference));
	shadowDepthReference.attachment = 0;
	shadowDepthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription shadowSubpass;
	memset(&shadowSubpass, 0, sizeof(shadowSubpass));
	shadowSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	shadowSubpass.pDepthStencilAttachment = &shadowDepthReference;

	VkSubpassDependency shadowDependency;
	memset(&shadowDependency, 0, sizeof(shadowDependency));
	shadowDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	shadowDependency.dstSubpass = 0;
	shadowDependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	shadowDependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	shadowDependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	shadowDependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	memset(&renderPass, 0, sizeof(renderPass));
	renderPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPass.attachmentCount = 1;
	renderPass.pAttachments = &shadowAttachment;
	renderPass.subpassCount = 1;
	renderPass.pSubpasses = &shadowSubpass;
	renderPass.dependencyCount = 1;
	renderPass.pDependencies = &shadowDependency;

	if (!VkOk(vkCreateRenderPass(g_vk.device, &renderPass, NULL, &g_vk.shadowRenderPass), "vkCreateRenderPass(shadow)"))
		return 0;

	return 1;
}

// ---------------------------------------------------------------------------
// Emulated PSX GPU path (renderer roadmap R7b, phase 1)
//
// Mirrors the OpenGL renderer's PSX model. The CPU VRAM mirror is uploaded as
// an R32G32_SFLOAT image (R = low byte, G = high byte of every little-endian
// 16-bit PSX pixel), a 256x256 RGBA table decodes PSX 5551 colours, and the
// ported psx.vert/psx.frag do the CLUT, texture-window, dither and bilinear
// work inside the shader. knowledge/roadmap/done/vulkan-game-renderer.md holds
// the phased plan that maps the game's GR_* contract onto this.

// Mirrors GR_InitRG8LUT (PsyX_render.cpp). The table turns a PSX 5551 word,
// addressed as its two VRAM bytes, into RGBA8. Kept in sync by hand so this
// backend does not include the OpenGL renderer's header; the self-test below
// fails if the two ever disagree.
static void BuildRG8Lut(unsigned char* lut)
{
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			const unsigned short c = (unsigned short)((y << 8) | x);
			unsigned char* pixel = lut + (y * 256 + x) * 4;
			pixel[0] = (unsigned char)((c & 31)) << 3;
			pixel[1] = (unsigned char)((c >> 5) & 31) << 3;
			pixel[2] = (unsigned char)((c >> 10) & 31) << 3;
			pixel[3] = (unsigned char)((c >> 15) & 1) << 7;
		}
	}
}

static int CreatePsxPipeline(int blendMode, int depthEnable, int stencilMode, VkRenderPass renderPass, VkPipeline* pipeline)
{
	VkShaderModule vertex = CreateShaderModuleFromSpirv(psx_vert_spv, psx_vert_spv_size);
	VkShaderModule fragment = CreateShaderModuleFromSpirv(psx_frag_spv, psx_frag_spv_size);
	if (!vertex || !fragment)
		return 0;

	VkPipelineShaderStageCreateInfo stages[2];
	memset(stages, 0, sizeof(stages));
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vertex;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fragment;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding;
	memset(&binding, 0, sizeof(binding));
	binding.binding = 0;
	binding.stride = sizeof(VkPsxVertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributes[5];
	memset(attributes, 0, sizeof(attributes));
	attributes[0].location = 0; attributes[0].binding = 0; attributes[0].format = VK_FORMAT_R32G32B32A32_SFLOAT; attributes[0].offset = 0;
	attributes[1].location = 1; attributes[1].binding = 0; attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; attributes[1].offset = 16;
	attributes[2].location = 2; attributes[2].binding = 0; attributes[2].format = VK_FORMAT_R8G8B8A8_UINT; attributes[2].offset = 32;
	attributes[3].location = 3; attributes[3].binding = 0; attributes[3].format = VK_FORMAT_R8G8B8A8_UNORM; attributes[3].offset = 36;
	attributes[4].location = 4; attributes[4].binding = 0; attributes[4].format = VK_FORMAT_R8G8B8A8_SINT; attributes[4].offset = 40;

	VkPipelineVertexInputStateCreateInfo vertexInput;
	memset(&vertexInput, 0, sizeof(vertexInput));
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 5;
	vertexInput.pVertexAttributeDescriptions = attributes;

	VkPipelineInputAssemblyStateCreateInfo assembly;
	memset(&assembly, 0, sizeof(assembly));
	assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewport;
	memset(&viewport, 0, sizeof(viewport));
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = 1;
	viewport.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo raster;
	memset(&raster, 0, sizeof(raster));
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisample;
	memset(&multisample, 0, sizeof(multisample));
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depthStencil;
	memset(&depthStencil, 0, sizeof(depthStencil));
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = depthEnable ? VK_TRUE : VK_FALSE;
	// Both GL and Vulkan disable depth writes when the depth test is disabled.
	const VkBool32 passHasDepth = (renderPass == g_vk.mainRenderPass) ? VK_TRUE : VK_FALSE;
	depthStencil.depthWriteEnable = depthEnable && passHasDepth;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

	// Mirror GR_SetStencilMode exactly: GL's 0x10 is the comparison mask,
	// not the write mask. Mask-set draws store reference 1 with all bits writable.
	// Other draws reject that value and keep stencil on a passing fragment.
	// The offscreen pass has
	// no depth-stencil attachment, so stencil stays disabled there.
	if (stencilMode >= 0 && g_vk.psx.stencilSupported)
	{
		depthStencil.stencilTestEnable = VK_TRUE;

		VkStencilOpState& front = depthStencil.front;
		VkStencilOpState& back = depthStencil.back;
		front.compareMask = 0x10;
		front.writeMask = 0xFF;
		front.reference = 1;
		if (stencilMode)
		{
			front.compareOp = VK_COMPARE_OP_ALWAYS;
			front.passOp = VK_STENCIL_OP_REPLACE;
			front.failOp = VK_STENCIL_OP_REPLACE;
			front.depthFailOp = VK_STENCIL_OP_REPLACE;
		}
		else
		{
			front.compareMask = 0xFF;
			front.compareOp = VK_COMPARE_OP_NOT_EQUAL;
			front.passOp = VK_STENCIL_OP_KEEP;
			front.failOp = VK_STENCIL_OP_REPLACE;
			front.depthFailOp = VK_STENCIL_OP_KEEP;
		}
		back = front;
	}

	VkPipelineColorBlendAttachmentState blendAttachment;
	memset(&blendAttachment, 0, sizeof(blendAttachment));
	blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	// The PSX blend modes map straight onto GL's: Vulkan has no reverse
	// subtract for alpha, so only the colour channel reverses.
	switch (blendMode)
	{
	case 0:	// BM_NONE
		blendAttachment.blendEnable = VK_FALSE;
		break;
	case 1:	// BM_AVERAGE
		blendAttachment.blendEnable = VK_TRUE;
		blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		break;
	case 2:	// BM_ADD
		blendAttachment.blendEnable = VK_TRUE;
		blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		break;
	case 3:	// BM_SUBTRACT
		blendAttachment.blendEnable = VK_TRUE;
		blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachment.colorBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT;
		blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		break;
	default:	// BM_ADD_QUATER_SOURCE
		blendAttachment.blendEnable = VK_TRUE;
		blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
		blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
		blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		break;
	}

	VkPipelineColorBlendStateCreateInfo blend;
	memset(&blend, 0, sizeof(blend));
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 1;
	blend.pAttachments = &blendAttachment;

	VkDynamicState dynamicStates[3] =
	{
		VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS
	};
	VkPipelineDynamicStateCreateInfo dynamic;
	memset(&dynamic, 0, sizeof(dynamic));
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 3;
	dynamic.pDynamicStates = dynamicStates;

	VkGraphicsPipelineCreateInfo pipelineInfo;
	memset(&pipelineInfo, 0, sizeof(pipelineInfo));
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = stages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &assembly;
	pipelineInfo.pViewportState = &viewport;
	pipelineInfo.pRasterizationState = &raster;
	pipelineInfo.pMultisampleState = &multisample;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &blend;
	pipelineInfo.pDynamicState = &dynamic;
	pipelineInfo.layout = g_vk.psx.layout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;

	const int ok = VkOk(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, pipeline),
		"vkCreateGraphicsPipelines(psx)");

	vkDestroyShaderModule(g_vk.device, vertex, NULL);
	vkDestroyShaderModule(g_vk.device, fragment, NULL);
	return ok;
}

// Allocates and writes a PSX descriptor set. The matrices, VRAM mirror and
// RG8 table are shared by every set; only the RGBA slot (the game's 32-bit
// textures) differs.
static int PsxAllocateSet(VkDescriptorSet* set, VkImageView rgbaView, VkSampler rgbaSampler)
{
	VkPsxState* psx = &g_vk.psx;

	VkDescriptorSetAllocateInfo setAllocate;
	memset(&setAllocate, 0, sizeof(setAllocate));
	setAllocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	setAllocate.descriptorPool = g_vk.descriptorPool;
	setAllocate.descriptorSetCount = 1;
	setAllocate.pSetLayouts = &psx->setLayout;
	if (!VkOk(vkAllocateDescriptorSets(g_vk.device, &setAllocate, set), "vkAllocateDescriptorSets(psx)"))
		return 0;

	VkDescriptorBufferInfo uboInfo;
	memset(&uboInfo, 0, sizeof(uboInfo));
	uboInfo.buffer = psx->ubo;
	uboInfo.range = VK_WHOLE_SIZE;

	VkDescriptorImageInfo vramInfo;
	memset(&vramInfo, 0, sizeof(vramInfo));
	vramInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vramInfo.imageView = psx->vramView;
	vramInfo.sampler = psx->sampler;

	VkDescriptorImageInfo lutInfo;
	memset(&lutInfo, 0, sizeof(lutInfo));
	lutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	lutInfo.imageView = psx->lutView;
	lutInfo.sampler = psx->lutSampler;

	VkDescriptorImageInfo rgbaInfo;
	memset(&rgbaInfo, 0, sizeof(rgbaInfo));
	rgbaInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	rgbaInfo.imageView = rgbaView;
	rgbaInfo.sampler = rgbaSampler;

	VkWriteDescriptorSet writes[4];
	memset(writes, 0, sizeof(writes));
	for (int i = 0; i < 4; i++)
	{
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = *set;
		writes[i].dstBinding = (uint32_t)i;
		writes[i].descriptorCount = 1;
	}
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[0].pBufferInfo = &uboInfo;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[1].pImageInfo = &vramInfo;
	writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[2].pImageInfo = &lutInfo;
	writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[3].pImageInfo = &rgbaInfo;
	vkUpdateDescriptorSets(g_vk.device, 4, writes, 0, NULL);

	return 1;
}

static int PsxWriteTextureSet(VkPsxTexture* tex)
{
	return PsxAllocateSet(&tex->set, tex->view, tex->sampler);
}

// Colour-only pass for the GR_SetOffscreenState render target. GL renders the
// offscreen FBO with no depth attachment, so this mirrors it; the final layout
// is TRANSFER_SRC because the result is copied back into VRAM right away.
static int CreatePsxOffscreenRenderPass(void)
{
	VkAttachmentDescription attachment;
	memset(&attachment, 0, sizeof(attachment));
	attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
	attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

	VkAttachmentReference colorReference;
	memset(&colorReference, 0, sizeof(colorReference));
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass;
	memset(&subpass, 0, sizeof(subpass));
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorReference;

	VkSubpassDependency dependency;
	memset(&dependency, 0, sizeof(dependency));
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo renderPass;
	memset(&renderPass, 0, sizeof(renderPass));
	renderPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPass.attachmentCount = 1;
	renderPass.pAttachments = &attachment;
	renderPass.subpassCount = 1;
	renderPass.pSubpasses = &subpass;
	renderPass.dependencyCount = 1;
	renderPass.pDependencies = &dependency;

	return VkOk(vkCreateRenderPass(g_vk.device, &renderPass, NULL, &g_vk.psx.offscreenRenderPass),
		"vkCreateRenderPass(psx offscreen)");
}

static int CreatePsxResources(void)
{
	VkPsxState* psx = &g_vk.psx;
	// CreateRenderPasses already selected the depth/stencil format. Clearing
	// this capability here silently created every PSX pipeline without stencil.
	const int stencilSupported = psx->stencilSupported;
	memset(psx, 0, sizeof(*psx));
	psx->stencilSupported = stencilSupported;

	// Descriptor set: 0 = matrices, 1 = VRAM, 2 = RG8 LUT, 3 = RGBA texture.
	VkDescriptorSetLayoutBinding bindings[4];
	memset(bindings, 0, sizeof(bindings));
	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	for (int i = 1; i < 4; i++)
	{
		bindings[i].binding = (uint32_t)i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo;
	memset(&layoutInfo, 0, sizeof(layoutInfo));
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 4;
	layoutInfo.pBindings = bindings;

	if (!VkOk(vkCreateDescriptorSetLayout(g_vk.device, &layoutInfo, NULL, &psx->setLayout), "vkCreateDescriptorSetLayout(psx)"))
		return 0;

	VkPushConstantRange pushRange;
	memset(&pushRange, 0, sizeof(pushRange));
	pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	pushRange.offset = 0;
	pushRange.size = 32;	// texFormat, bilinearFilter, texelSize, overrideAlphaMode, srgbEncode

	VkPipelineLayoutCreateInfo pipelineLayout;
	memset(&pipelineLayout, 0, sizeof(pipelineLayout));
	pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayout.setLayoutCount = 1;
	pipelineLayout.pSetLayouts = &psx->setLayout;
	pipelineLayout.pushConstantRangeCount = 1;
	pipelineLayout.pPushConstantRanges = &pushRange;

	if (!VkOk(vkCreatePipelineLayout(g_vk.device, &pipelineLayout, NULL, &psx->layout), "vkCreatePipelineLayout(psx)"))
		return 0;

	for (int mode = 0; mode < PSYX_VK_PSX_BLEND_COUNT; mode++)
	{
		// BM_NONE draws depth-tested (PGXP-Z); the other modes depth-test off.
		if (!CreatePsxPipeline(mode, mode == 0 ? 1 : 0, 0, g_vk.mainRenderPass, &psx->pipelines[mode]))
			return 0;
		// The mask-set draw still writes the stencil bit; its depth config
		// matches the test variant so the mask draw's depth result is identical.
		if (!CreatePsxPipeline(mode, mode == 0 ? 1 : 0, 1, g_vk.mainRenderPass, &psx->pipelinesStencilWrite[mode]))
			return 0;
	}
	if (!CreatePsxPipeline(0, 0, 0, g_vk.mainRenderPass, &psx->pipelineNoDepth))
		return 0;
	if (!CreatePsxPipeline(0, 0, 1, g_vk.mainRenderPass, &psx->pipelineNoDepthStencilWrite))
		return 0;

	// The per-frame draw list belongs to frameIndex, so the queue maps one
	// frame to one render pass, mirroring the double-buffered display lists.
	psx->frameIndex = g_vk.frameIndex;

	// Offscreen pass and its depth-less pipelines.
	if (!CreatePsxOffscreenRenderPass())
		return 0;
	for (int mode = 0; mode < PSYX_VK_PSX_BLEND_COUNT; mode++)
	{
		if (!CreatePsxPipeline(mode, 0, -1, psx->offscreenRenderPass, &psx->offscreenPipelines[mode]))
			return 0;
	}

	// Nearest sampling with wrapping addresses, like PSX VRAM.
	VkSamplerCreateInfo samplerInfo;
	memset(&samplerInfo, 0, sizeof(samplerInfo));
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_NEAREST;
	samplerInfo.minFilter = VK_FILTER_NEAREST;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.maxLod = 0.25f;

	if (!VkOk(vkCreateSampler(g_vk.device, &samplerInfo, NULL, &psx->sampler), "vkCreateSampler(psx)"))
		return 0;

	// The RG8 lookup subtracts a fraction of a texel from the coordinate, so it
	// must clamp: GL sets GL_CLAMP_TO_EDGE on the same table. With wrapping, a
	// zero high byte would sample the last row instead of the first.
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	if (!VkOk(vkCreateSampler(g_vk.device, &samplerInfo, NULL, &psx->lutSampler), "vkCreateSampler(psx lut)"))
		return 0;

	// VRAM: RG32F because the shader samples the two bytes independently.
	if (!CreateImage2D(PSYX_VK_VRAM_WIDTH, PSYX_VK_VRAM_HEIGHT, VK_FORMAT_R32G32_SFLOAT,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &psx->vramImage, &psx->vramMemory))
		return 0;
	if (!CreateImageView2D(psx->vramImage, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, &psx->vramView))
		return 0;
	if (!CreateBuffer((VkDeviceSize)PSYX_VK_PSX_VRAM_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&psx->vramStaging, &psx->vramStagingMemory, &psx->vramStagingMapped))
		return 0;

	// RG8 decode table.
	if (!CreateImage2D(256, 256, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &psx->lutImage, &psx->lutMemory))
		return 0;
	if (!CreateImageView2D(psx->lutImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &psx->lutView))
		return 0;
	if (!CreateBuffer(256 * 256 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&psx->lutStaging, &psx->lutStagingMemory, &psx->lutStagingMapped))
		return 0;

	if (!CreateBuffer(32 * sizeof(float), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&psx->ubo, &psx->uboMemory, (void**)&psx->uboMapped))
		return 0;
	// Identity defaults so a draw before any GR_Ortho2D still lands on screen.
	psx->uboMapped[0] = psx->uboMapped[5] = psx->uboMapped[10] = psx->uboMapped[15] = 1.0f;
	psx->uboMapped[16] = psx->uboMapped[21] = psx->uboMapped[26] = psx->uboMapped[31] = 1.0f;

	if (!CreateBuffer(PSYX_VK_PSX_VERTEX_CAPACITY, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&psx->vertexBuffer, &psx->vertexMemory, (void**)&psx->vertexMapped))
		return 0;
	psx->vertexCapacity = PSYX_VK_PSX_VERTEX_CAPACITY;

	// Staging for the VRAM rectangles replayed between deferred draws.
	if (!CreateBuffer(PSYX_VK_PSX_VRAM_UPLOAD_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&psx->vramUploadStaging, &psx->vramUploadStagingMemory, &psx->vramUploadStagingMapped))
		return 0;
	psx->vramUploadStagingCapacity = PSYX_VK_PSX_VRAM_UPLOAD_BYTES;

	// Upload the LUT once; it only depends on the PSX colour format.
	BuildRG8Lut((unsigned char*)psx->lutStagingMapped);
	{
		VkCommandBuffer cmd = BeginOneShot();
		ImageBarrier(cmd, psx->lutImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
		VkBufferImageCopy region;
		memset(&region, 0, sizeof(region));
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.layerCount = 1;
		region.imageExtent.width = 256;
		region.imageExtent.height = 256;
		region.imageExtent.depth = 1;
		vkCmdCopyBufferToImage(cmd, psx->lutStaging, psx->lutImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		ImageBarrier(cmd, psx->lutImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
		EndOneShot(cmd);
	}

	// VRAM starts zeroed and shader-readable.
	{
		VkCommandBuffer cmd = BeginOneShot();
		ImageBarrier(cmd, psx->vramImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
		VkClearColorValue clearValue;
		memset(&clearValue, 0, sizeof(clearValue));
		VkImageSubresourceRange range;
		memset(&range, 0, sizeof(range));
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.levelCount = 1;
		range.layerCount = 1;
		vkCmdClearColorImage(cmd, psx->vramImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
		ImageBarrier(cmd, psx->vramImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
		EndOneShot(cmd);
	}

	// The RGBA slot reuses the shared white texture: 4/8/16-bit draws decode
	// from VRAM and never sample it.
	const int white = (g_vk.dummyTextures[0] >= 0) ? g_vk.dummyTextures[0] : 0;
	if (!PsxAllocateSet(&psx->dummySet, g_textures[white].view, g_vk.textureSampler))
		return 0;

	// State the game expects before it has set anything.
	psx->stTexFormat = PSYX_VK_TEX_16BIT;
	psx->stBlendMode = 0;
	psx->stDepth = 1;
	psx->stBilinear = g_cfg_bilinearFiltering ? 1 : 0;
	psx->stScissor[2] = g_vk.width;
	psx->stScissor[3] = g_vk.height;
	psx->stViewport[2] = g_vk.width;
	psx->stViewport[3] = g_vk.height;

	psx->ready = 1;
	return 1;
}

static void DestroyPsxResources(void)
{
	VkPsxState* psx = &g_vk.psx;

	for (int i = 0; i < PSYX_VK_PSX_MAX_TEXTURES; i++)
	{
		VkPsxTexture* tex = &psx->textures[i];
		if (!tex->used)
			continue;
		if (tex->sampler) vkDestroySampler(g_vk.device, tex->sampler, NULL);
		if (tex->view) vkDestroyImageView(g_vk.device, tex->view, NULL);
		if (tex->image) vkDestroyImage(g_vk.device, tex->image, NULL);
		if (tex->memory) vkFreeMemory(g_vk.device, tex->memory, NULL);
		tex->used = 0;
	}

	if (psx->vramStaging) vkDestroyBuffer(g_vk.device, psx->vramStaging, NULL);
	if (psx->vramStagingMemory) vkFreeMemory(g_vk.device, psx->vramStagingMemory, NULL);
	if (psx->lutStaging) vkDestroyBuffer(g_vk.device, psx->lutStaging, NULL);
	if (psx->lutStagingMemory) vkFreeMemory(g_vk.device, psx->lutStagingMemory, NULL);
	if (psx->vramView) vkDestroyImageView(g_vk.device, psx->vramView, NULL);
	if (psx->vramImage) vkDestroyImage(g_vk.device, psx->vramImage, NULL);
	if (psx->vramMemory) vkFreeMemory(g_vk.device, psx->vramMemory, NULL);
	if (psx->lutView) vkDestroyImageView(g_vk.device, psx->lutView, NULL);
	if (psx->lutImage) vkDestroyImage(g_vk.device, psx->lutImage, NULL);
	if (psx->lutMemory) vkFreeMemory(g_vk.device, psx->lutMemory, NULL);
	if (psx->ubo) vkDestroyBuffer(g_vk.device, psx->ubo, NULL);
	if (psx->uboMemory) vkFreeMemory(g_vk.device, psx->uboMemory, NULL);
	if (psx->vertexBuffer) vkDestroyBuffer(g_vk.device, psx->vertexBuffer, NULL);
	if (psx->vertexMemory) vkFreeMemory(g_vk.device, psx->vertexMemory, NULL);
	if (psx->vramUploadStaging) vkDestroyBuffer(g_vk.device, psx->vramUploadStaging, NULL);
	if (psx->vramUploadStagingMemory) vkFreeMemory(g_vk.device, psx->vramUploadStagingMemory, NULL);
	for (int i = 0; i < PSYX_VK_PSX_BLEND_COUNT; i++)
	{
		if (psx->pipelines[i]) vkDestroyPipeline(g_vk.device, psx->pipelines[i], NULL);
		if (psx->pipelinesStencilWrite[i]) vkDestroyPipeline(g_vk.device, psx->pipelinesStencilWrite[i], NULL);
	}
	if (psx->pipelineNoDepth) vkDestroyPipeline(g_vk.device, psx->pipelineNoDepth, NULL);
	if (psx->pipelineNoDepthStencilWrite) vkDestroyPipeline(g_vk.device, psx->pipelineNoDepthStencilWrite, NULL);
	for (int i = 0; i < PSYX_VK_PSX_BLEND_COUNT; i++)
	{
		if (psx->offscreenPipelines[i]) vkDestroyPipeline(g_vk.device, psx->offscreenPipelines[i], NULL);
	}
	if (psx->offscreenFramebuffer) vkDestroyFramebuffer(g_vk.device, psx->offscreenFramebuffer, NULL);
	if (psx->offscreenView) vkDestroyImageView(g_vk.device, psx->offscreenView, NULL);
	if (psx->offscreenImage) vkDestroyImage(g_vk.device, psx->offscreenImage, NULL);
	if (psx->offscreenMemory) vkFreeMemory(g_vk.device, psx->offscreenMemory, NULL);
	if (psx->offscreenReadback) vkDestroyBuffer(g_vk.device, psx->offscreenReadback, NULL);
	if (psx->offscreenReadbackMemory) vkFreeMemory(g_vk.device, psx->offscreenReadbackMemory, NULL);
	if (psx->offscreenRenderPass) vkDestroyRenderPass(g_vk.device, psx->offscreenRenderPass, NULL);
	if (psx->sampler) vkDestroySampler(g_vk.device, psx->sampler, NULL);
	if (psx->lutSampler) vkDestroySampler(g_vk.device, psx->lutSampler, NULL);
	if (psx->layout) vkDestroyPipelineLayout(g_vk.device, psx->layout, NULL);
	if (psx->setLayout) vkDestroyDescriptorSetLayout(g_vk.device, psx->setLayout, NULL);

	memset(psx, 0, sizeof(*psx));
}

void PsyX_Vk_GameBeginFrame(void)
{
	if (!g_vk.initialised || !g_vk.psx.ready)
		return;

	// The caller refills the shared vertex buffer next, so make sure the GPU is
	// done with the previous frame before those writes land.
	if (g_vk.frameFence)
		vkWaitForFences(g_vk.device, 1, &g_vk.frameFence, VK_TRUE, UINT64_MAX);

	g_vk.psx.vertexSize = 0;
	g_vk.psx.vertexCount = 0;
	g_vk.psx.vertexUploadBase = 0;
	g_vk.psx.vertexUploadedTotal = 0;
	g_vk.psx.drawCount = 0;
	// VRAM writes queued before this frame have already been replayed (or are
	// irrelevant now); each frame starts from the image the GPU holds.
	g_vk.psx.vramUploadGeneration = 0;
	g_vk.psx.vramUploadCount = 0;
	g_vk.psx.vramUploadsApplied = 0;
	g_vk.psx.vramUploadStagingUsed = 0;
	// The game queues the frame N draws during frame N+1's DrawSync, so the
	// list being built is one frame ahead of the one now presented.
	g_vk.psx.frameIndex = g_vk.frameIndex + 1;
	g_vk.psx.offscreenActive = 0;
	g_vk.psx.offscreenGroupPendingDraw = 0;
	g_vk.psx.offscreenGroupCount = 0;
}

static void UploadVramMirror(const unsigned short* vram)
{
	// GL uploads the same mirror as GL_RG / GL_UNSIGNED_BYTE into an RG32F
	// texture, so R = low byte / 255 and G = high byte / 255.
	float* dst = (float*)g_vk.psx.vramStagingMapped;
	for (int i = 0; i < PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT; i++)
	{
		const unsigned short word = vram[i];
		dst[i * 2 + 0] = (float)(word & 0xFF) * (1.0f / 255.0f);
		dst[i * 2 + 1] = (float)((word >> 8) & 0xFF) * (1.0f / 255.0f);
	}

	VkCommandBuffer cmd = BeginOneShot();
	ImageBarrier(cmd, g_vk.psx.vramImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);

	VkBufferImageCopy region;
	memset(&region, 0, sizeof(region));
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent.width = PSYX_VK_VRAM_WIDTH;
	region.imageExtent.height = PSYX_VK_VRAM_HEIGHT;
	region.imageExtent.depth = 1;
	vkCmdCopyBufferToImage(cmd, g_vk.psx.vramStaging, g_vk.psx.vramImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	ImageBarrier(cmd, g_vk.psx.vramImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	EndOneShot(cmd);
}

void PsyX_Vk_GameSetVram(const unsigned short* vram)
{
	if (!g_vk.psx.ready || !vram)
		return;

	UploadVramMirror(vram);
}

// Queues one VRAM rectangle for replay inside the deferred draw list. Called
// for the DS_LoadImage path, which the overhead map uses to stream its tiles
// through a handful of VRAM slots between draw flushes. The pixels are copied
// now because the CPU mirror keeps changing.
void PsyX_Vk_GameCopyVRAM(const unsigned short* src, int srcStride,
	int x, int y, int w, int h, int dstX, int dstY)
{
	VkPsxState* psx = &g_vk.psx;
	if (!psx->ready || !src || !psx->vramUploadStagingMapped || w <= 0 || h <= 0)
		return;

	if (x < 0 || y < 0 || dstX < 0 || dstY < 0 ||
		x + w > PSYX_VK_VRAM_WIDTH || y + h > PSYX_VK_VRAM_HEIGHT ||
		dstX + w > PSYX_VK_VRAM_WIDTH || dstY + h > PSYX_VK_VRAM_HEIGHT)
	{
		eprintwarn("PsyX Vulkan: VRAM upload out of range (%d,%d %dx%d -> %d,%d)\n",
			x, y, w, h, dstX, dstY);
		return;
	}

	const VkDeviceSize bytes = (VkDeviceSize)w * h * 8;
	if (psx->vramUploadCount >= PSYX_VK_PSX_MAX_VRAM_UPLOADS ||
		psx->vramUploadStagingUsed + bytes > psx->vramUploadStagingCapacity)
	{
		eprintwarn("PsyX Vulkan: VRAM upload queue full, dropping %dx%d at %d,%d\n",
			w, h, dstX, dstY);
		return;
	}

	float* dst = (float*)((unsigned char*)psx->vramUploadStagingMapped + psx->vramUploadStagingUsed);
	for (int row = 0; row < h; row++)
	{
		const unsigned short* s = src + (size_t)(y + row) * srcStride + x;
		for (int col = 0; col < w; col++)
		{
			const unsigned short word = s[col];
			*dst++ = (float)(word & 0xFF) * (1.0f / 255.0f);
			*dst++ = (float)((word >> 8) & 0xFF) * (1.0f / 255.0f);
		}
	}

	VkPsxVramUpload* up = &psx->vramUploads[psx->vramUploadCount++];
	up->generation = psx->vramUploadGeneration++;
	up->x = dstX;
	up->y = dstY;
	up->w = w;
	up->h = h;
	up->offset = psx->vramUploadStagingUsed;
	psx->vramUploadStagingUsed += bytes;
}

void PsyX_Vk_GameSetProjection2D(const float projection[16])
{
	if (!g_vk.psx.ready || !projection)
		return;
	memcpy(g_vk.psx.uboMapped, projection, 16 * sizeof(float));
}

void PsyX_Vk_GameSetProjection3D(const float projection[16])
{
	if (!g_vk.psx.ready || !projection)
		return;
	memcpy(g_vk.psx.uboMapped + 16, projection, 16 * sizeof(float));
}

void PsyX_Vk_GameUpdateVertexBuffer(const void* vertices, int vertexCount)
{
	VkPsxState* psx = &g_vk.psx;
	if (!psx->ready || !vertices || vertexCount <= 0)
		return;

	if (vertexCount > PSYX_VK_PSX_MAX_VERTICES)
	{
		eprintwarn("PsyX Vulkan: PSX vertex flush too large (%d vertices)\n", vertexCount);
		vertexCount = PSYX_VK_PSX_MAX_VERTICES;
	}

	const uint32_t capacityVertices = (uint32_t)(psx->vertexCapacity / sizeof(VkPsxVertex));

	// The OpenGL renderer draws each flush immediately, so it can overwrite one
	// buffer every time. Here the draws are recorded at frame end, so an
	// overwrite would leave the earlier draws pointing at unrelated vertices
	// (the overhead map's tile batches all disappeared). Append instead, and
	// only wrap when a frame really exceeds the capacity.
	if (psx->vertexUploadedTotal + (uint32_t)vertexCount > capacityVertices)
	{
		eprintwarn("PsyX Vulkan: PSX vertex buffer full (%u + %d), reusing from the start\n",
			psx->vertexUploadedTotal, vertexCount);
		psx->vertexUploadedTotal = 0;
	}

	psx->vertexUploadBase = psx->vertexUploadedTotal;
	psx->vertexUploadedTotal += (uint32_t)vertexCount;

	const VkDeviceSize bytes = (VkDeviceSize)vertexCount * sizeof(VkPsxVertex);
	memcpy(psx->vertexMapped + (size_t)psx->vertexUploadBase * sizeof(VkPsxVertex), vertices, (size_t)bytes);
	psx->vertexSize = (VkDeviceSize)psx->vertexUploadedTotal * sizeof(VkPsxVertex);
	psx->vertexCount = psx->vertexUploadedTotal;
}

void PsyX_Vk_GameSetBlendMode(int blendMode)
{
	g_vk.psx.stBlendMode = (blendMode >= 0 && blendMode < PSYX_VK_PSX_BLEND_COUNT) ? blendMode : 0;
}

void PsyX_Vk_GameSetTexture(int texFormat, int texture)
{
	VkPsxState* psx = &g_vk.psx;
	psx->stTexFormat = (texFormat >= 0) ? texFormat : PSYX_VK_TEX_16BIT;
	psx->stTexture = texture;
}

void PsyX_Vk_GameSetOverrideTextureSize(int width, int height)
{
	VkPsxState* psx = &g_vk.psx;
	psx->stOverrideWidth = width > 0 ? width : 0;
	psx->stOverrideHeight = height > 0 ? height : 0;
}

void PsyX_Vk_GameSetOverrideAlphaMode(int mode)
{
	g_vk.psx.stOverrideAlphaMode = mode;
}

void PsyX_Vk_GameSetStencilMode(int drawPrimMode)
{
	g_vk.psx.stStencilMode = drawPrimMode;
}

void PsyX_Vk_GameEnableDepth(int enable)
{
	g_vk.psx.stDepth = enable ? 1 : 0;
}

void PsyX_Vk_GameSetBilinear(int enable)
{
	g_vk.psx.stBilinear = enable ? 1 : 0;
}

void PsyX_Vk_GameSetScissor(int enable, int x, int y, int width, int height)
{
	VkPsxState* psx = &g_vk.psx;
	psx->stScissorEnable = enable ? 1 : 0;
	psx->stScissor[0] = x;
	psx->stScissor[1] = y;
	psx->stScissor[2] = width;
	psx->stScissor[3] = height;
}

void PsyX_Vk_GameSetViewPort(int x, int y, int width, int height)
{
	VkPsxState* psx = &g_vk.psx;
	psx->stViewport[0] = x;
	psx->stViewport[1] = y;
	psx->stViewport[2] = width;
	psx->stViewport[3] = height;
}

void PsyX_Vk_GameSetOffscreen(int enable, int x, int y, int width, int height)
{
	VkPsxState* psx = &g_vk.psx;
	if (!psx->ready)
		return;

	if (enable)
	{
		// GR_SetOffscreenState returns early on a repeated enable, so back-to-
		// back offscreen splits keep sharing the first rectangle and image.
		// The game calls GR_SetOffscreenState when building the display list
		// (DrawSync time is too late), so groups are tracked per frame index
		// instead of by a monotonic draw counter: group i is the run queued in
		// frame i, which is the run the renderer submits for frame i.
		if (psx->offscreenActive)
			return;

		psx->offscreenActive = 1;
		psx->offscreenGroupPendingDraw = psx->frameIndex;
		psx->offscreenRect[0] = x;
		psx->offscreenRect[1] = y;
		psx->offscreenRect[2] = width;
		psx->offscreenRect[3] = height;
		// The projection active at that moment is the offscreen one set by the
		// caller's GR_Ortho2D.
		memcpy(psx->offscreenGroupMatrix, psx->uboMapped, sizeof(psx->offscreenGroupMatrix));
		return;
	}

	if (!psx->offscreenActive)
		return;

	psx->offscreenActive = 0;

	const int frame = psx->offscreenGroupPendingDraw;
	if (psx->offscreenGroupCount >= PSYX_VK_PSX_MAX_OFFSCREEN_GROUPS)
	{
		eprintwarn("PsyX Vulkan: too many offscreen groups, dropping one\n");
		return;
	}

	VkPsxOffscreenGroup* group = &psx->offscreenGroups[psx->offscreenGroupCount++];
	group->frame = frame;
	memcpy(group->rect, psx->offscreenRect, sizeof(group->rect));
	memcpy(group->matrix, psx->offscreenGroupMatrix, sizeof(group->matrix));
}

void PsyX_Vk_GameStoreFrameBuffer(int x, int y, int width, int height)
{
	VkPsxState* psx = &g_vk.psx;
	psx->frameBufferRect[0] = x;
	psx->frameBufferRect[1] = y;
	psx->frameBufferRect[2] = width;
	psx->frameBufferRect[3] = height;
	psx->frameBufferPending = 1;
}

int PsyX_Vk_TakeStoredFrameBuffer(const unsigned char** rgba, int* stride,
	int* srcWidth, int* srcHeight, int* bgra, int* x, int* y, int* width, int* height)
{
	VkPsxState* psx = &g_vk.psx;
	if (!g_vk.initialised || !psx->frameBufferPending || !g_vk.readbackMapped)
		return 0;

	// The readback buffer was filled when the previous frame was submitted; the
	// fence was waited at the start of this frame, so the pixels are complete.
	psx->frameBufferPending = 0;

	if (rgba) *rgba = g_vk.readbackMapped;
	if (stride) *stride = g_vk.width * 4;
	if (srcWidth) *srcWidth = g_vk.width;
	if (srcHeight) *srcHeight = g_vk.height;
	if (bgra)
		*bgra = (g_vk.swapchainFormat == VK_FORMAT_B8G8R8A8_SRGB ||
				 g_vk.swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM) ? 1 : 0;
	if (x) *x = psx->frameBufferRect[0];
	if (y) *y = psx->frameBufferRect[1];
	if (width) *width = psx->frameBufferRect[2];
	if (height) *height = psx->frameBufferRect[3];
	return 1;
}

void PsyX_Vk_GameClear(int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b)
{
	// GR_Clear only ever clears the whole on-screen buffer (the rect is applied
	// by the PSX clip environment through the scissor); the Vulkan pass clear
	// does the same, so the queued draws are what is left to do here.
	(void)x; (void)y; (void)width; (void)height; (void)r; (void)g; (void)b;
	g_vk.psx.clearRequested = 1;
	g_vk.psx.clearColor[0] = r / 255.0f;
	g_vk.psx.clearColor[1] = g / 255.0f;
	g_vk.psx.clearColor[2] = b / 255.0f;
}

int PsyX_Vk_GameDrawTriangles(int firstVertex, int triangles)
{
	VkPsxState* psx = &g_vk.psx;
	if (!psx->ready || triangles <= 0)
		return 0;
	if (psx->drawCount >= PSYX_VK_PSX_MAX_DRAWS)
	{
		eprintwarn("PsyX Vulkan: PSX draw list full, dropping draw\n");
		return 0;
	}
	const uint32_t uploadBase = psx->vertexUploadBase;
	const uint32_t uploadCount = psx->vertexUploadedTotal - uploadBase;

	if (firstVertex < 0 || (uint32_t)(firstVertex + triangles * 3) > uploadCount)
	{
		eprintwarn("PsyX Vulkan: PSX draw out of range (%d + %d triangles, %u vertices)\n",
			firstVertex, triangles, uploadCount);
		return 0;
	}

	VkPsxDraw* draw = &psx->draws[psx->drawCount++];
	memset(draw, 0, sizeof(*draw));
	draw->texFormat = psx->stTexFormat;
	draw->bilinearFilter = psx->stBilinear;
	draw->texelSize[0] = psx->stOverrideWidth > 0 ? 1.0f / (float)psx->stOverrideWidth : 1.0f / 256.0f;
	draw->texelSize[1] = psx->stOverrideHeight > 0 ? 1.0f / (float)psx->stOverrideHeight : 1.0f / 256.0f;
	draw->overrideAlphaMode = psx->stOverrideAlphaMode;
	// The PSX shader emits display-referred (encoded) values. When the draw
	// targets an sRGB attachment the hardware encodes on store, so the shader
	// must inverse-encode to cancel it and match the OpenGL image exactly.
	// Offscreen targets are UNORM, so those draws leave srgbEncode at 0.
	draw->srgbEncode = (psx->offscreenActive || !g_vk.srgbOutput) ? 0 : 1;
	draw->blendMode = psx->stBlendMode;
	draw->depthTest = psx->stDepth;
	draw->scissorEnable = psx->stScissorEnable;
	memcpy(draw->scissor, psx->stScissor, sizeof(draw->scissor));
	memcpy(draw->viewport, psx->stViewport, sizeof(draw->viewport));
	draw->textureSet = (psx->stTexture > 0 && psx->stTexture <= PSYX_VK_PSX_MAX_TEXTURES &&
		psx->textures[psx->stTexture - 1].used)
		? psx->textures[psx->stTexture - 1].set
		: VK_NULL_HANDLE;
	draw->firstVertex = uploadBase + (uint32_t)firstVertex;	draw->vertexCount = (uint32_t)(triangles * 3);
	draw->frame = psx->frameIndex;
	draw->offscreen = psx->offscreenActive;
	draw->stencilMode = psx->stStencilMode;
	draw->vramGeneration = psx->vramUploadGeneration;

	return triangles;
}

int PsyX_Vk_GameCreateTexture(const unsigned char* rgba, int width, int height, int mipmapped)
{
	VkPsxState* psx = &g_vk.psx;
	if (!psx->ready || !rgba || width <= 0 || height <= 0)
		return 0;

	int slot = -1;
	for (int i = 0; i < PSYX_VK_PSX_MAX_TEXTURES; i++)
	{
		if (!psx->textures[i].used)
		{
			slot = i;
			break;
		}
	}
	if (slot < 0)
	{
		eprinterr("PsyX Vulkan: out of PSX game textures\n");
		return 0;
	}

	VkPsxTexture* tex = &psx->textures[slot];
	memset(tex, 0, sizeof(*tex));

	// Same filtering the OpenGL renderer applies to game RGBA textures.
	int levels = 1;
	if (mipmapped)
	{
		int w = width;
		int h = height;
		while (w > 1 || h > 1)
		{
			w = w > 1 ? w / 2 : 1;
			h = h > 1 ? h / 2 : 1;
			levels++;
		}
	}

	if (!CreateImage2DLevels(width, height, levels, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		&tex->image, &tex->memory))
		return 0;
	if (!CreateImageView2DLevels(tex->image, VK_FORMAT_R8G8B8A8_UNORM, levels, &tex->view))
	{
		vkDestroyImage(g_vk.device, tex->image, NULL);
		vkFreeMemory(g_vk.device, tex->memory, NULL);
		memset(tex, 0, sizeof(*tex));
		return 0;
	}

	const VkDeviceSize stagingSize = (VkDeviceSize)width * height * 4;
	VkBuffer staging = VK_NULL_HANDLE;
	VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
	void* stagingMapped = NULL;
	if (!CreateBuffer(stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&staging, &stagingMemory, &stagingMapped))
	{
		vkDestroyImageView(g_vk.device, tex->view, NULL);
		vkDestroyImage(g_vk.device, tex->image, NULL);
		vkFreeMemory(g_vk.device, tex->memory, NULL);
		memset(tex, 0, sizeof(*tex));
		return 0;
	}
	memcpy(stagingMapped, rgba, (size_t)stagingSize);

	VkCommandBuffer cmd = BeginOneShot();
	ImageBarrier(cmd, tex->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);

	VkBufferImageCopy region;
	memset(&region, 0, sizeof(region));
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent.width = (uint32_t)width;
	region.imageExtent.height = (uint32_t)height;
	region.imageExtent.depth = 1;
	vkCmdCopyBufferToImage(cmd, staging, tex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	// Mirror GR_CreateRGBATextureMipmapped's glGenerateMipmap.
	//
	// Barriers are per mip level. A single image can hold different layouts in
	// different mips, but a whole-image barrier would force one layout on every
	// level, which is what previously produced oldLayout-01197 and 00221: the
	// source mip was left TRANSFER_SRC while the destination mip was written as
	// TRANSFER_DST. Mip 0 is moved DST -> SRC once up front; each iteration then
	// writes the new level as UNDEFINED -> DST (a generated mip has no prior
	// contents) and immediately leaves it TRANSFER_SRC so it feeds the next
	// blit.
	if (levels > 1)
	{
		ImageBarrierLevels(cmd, tex->image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	}

	for (int level = 1; level < levels; level++)
	{
		const int srcW = width >> (level - 1);
		const int srcH = height >> (level - 1);
		int dstW = srcW > 1 ? srcW / 2 : 1;
		int dstH = srcH > 1 ? srcH / 2 : 1;

		ImageBarrierLevels(cmd, tex->image, VK_IMAGE_ASPECT_COLOR_BIT, level, 1,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, VK_ACCESS_TRANSFER_WRITE_BIT);

		VkImageBlit blit;
		memset(&blit, 0, sizeof(blit));
		blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel = (uint32_t)(level - 1);
		blit.srcSubresource.layerCount = 1;
		blit.srcOffsets[1].x = srcW;
		blit.srcOffsets[1].y = srcH;
		blit.srcOffsets[1].z = 1;
		blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.mipLevel = (uint32_t)level;
		blit.dstSubresource.layerCount = 1;
		blit.dstOffsets[1].x = dstW;
		blit.dstOffsets[1].y = dstH;
		blit.dstOffsets[1].z = 1;
		vkCmdBlitImage(cmd, tex->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &blit, VK_FILTER_LINEAR);

		ImageBarrierLevels(cmd, tex->image, VK_IMAGE_ASPECT_COLOR_BIT, level, 1,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	}

	// With no mip chain mip 0 is still TRANSFER_DST; otherwise every level is
	// TRANSFER_SRC after generation. The transition covers every level (the
	// single-level ImageBarrier helper would leave mips 1..N as TRANSFER_SRC,
	// which the sampler then rejects with VUID-vkCmdDraw-None-09600).
	ImageBarrierLevels(cmd, tex->image, VK_IMAGE_ASPECT_COLOR_BIT, 0, levels,
		(levels > 1) ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
	EndOneShot(cmd);

	vkDestroyBuffer(g_vk.device, staging, NULL);
	vkFreeMemory(g_vk.device, stagingMemory, NULL);

	// Filtering matches GR_CreateRGBATexture: bilinear when the config asks for
	// it, clamped addresses either way.
	VkSamplerCreateInfo samplerInfo;
	memset(&samplerInfo, 0, sizeof(samplerInfo));
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = g_cfg_bilinearFiltering ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
	samplerInfo.minFilter = samplerInfo.magFilter;
	samplerInfo.mipmapMode = (mipmapped && g_cfg_bilinearFiltering)
		? VK_SAMPLER_MIPMAP_MODE_LINEAR
		: VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.maxLod = (float)levels;
	if (!VkOk(vkCreateSampler(g_vk.device, &samplerInfo, NULL, &tex->sampler), "vkCreateSampler(psx game texture)"))
	{
		vkDestroyImageView(g_vk.device, tex->view, NULL);
		vkDestroyImage(g_vk.device, tex->image, NULL);
		vkFreeMemory(g_vk.device, tex->memory, NULL);
		memset(tex, 0, sizeof(*tex));
		return 0;
	}

	if (!PsxWriteTextureSet(tex))
	{
		vkDestroySampler(g_vk.device, tex->sampler, NULL);
		vkDestroyImageView(g_vk.device, tex->view, NULL);
		vkDestroyImage(g_vk.device, tex->image, NULL);
		vkFreeMemory(g_vk.device, tex->memory, NULL);
		memset(tex, 0, sizeof(*tex));
		return 0;
	}

	tex->used = 1;
	tex->width = width;
	tex->height = height;
	return slot + 1;	// 0 is reserved for "no texture"
}

void PsyX_Vk_GameDestroyTexture(int texture)
{
	VkPsxState* psx = &g_vk.psx;
	if (texture <= 0 || texture > PSYX_VK_PSX_MAX_TEXTURES)
		return;

	VkPsxTexture* tex = &psx->textures[texture - 1];
	if (!tex->used)
		return;

	// Drop the overlay bridge first: its descriptor set references this view.
	if (psx->overlayTextureSlot == texture)
	{
#ifdef PSYX_VK_IMGUI
		if (psx->overlayTextureId)
			ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)psx->overlayTextureId);
#endif
		psx->overlayTextureId = 0;
		psx->overlayTextureSlot = 0;
	}

	if (tex->sampler) vkDestroySampler(g_vk.device, tex->sampler, NULL);
	if (tex->view) vkDestroyImageView(g_vk.device, tex->view, NULL);
	if (tex->image) vkDestroyImage(g_vk.device, tex->image, NULL);
	if (tex->memory) vkFreeMemory(g_vk.device, tex->memory, NULL);
	memset(tex, 0, sizeof(*tex));
}

unsigned long long PsyX_Vk_GameGetOverlayTextureId(int texture)
{
#ifdef PSYX_VK_IMGUI
	VkPsxState* psx = &g_vk.psx;
	if (!g_vk.initialised || !g_vk.imguiActive)
		return 0;
	if (texture <= 0 || texture > PSYX_VK_PSX_MAX_TEXTURES || !psx->textures[texture - 1].used)
		return 0;

	// One preview is visible at a time and the ImGui pool is small, so the
	// bridge keeps exactly one user descriptor set alive.
	if (psx->overlayTextureSlot == texture && psx->overlayTextureId != 0)
		return psx->overlayTextureId;

	if (psx->overlayTextureId)
	{
		ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)psx->overlayTextureId);
		psx->overlayTextureId = 0;
		psx->overlayTextureSlot = 0;
	}

	VkPsxTexture* tex = &psx->textures[texture - 1];
	VkDescriptorSet set = ImGui_ImplVulkan_AddTexture(tex->sampler, tex->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	if (set == VK_NULL_HANDLE)
		return 0;

	psx->overlayTextureId = (unsigned long long)set;
	psx->overlayTextureSlot = texture;
	return psx->overlayTextureId;
#else
	(void)texture;
	return 0;
#endif
}

void PsyX_Vk_GameGetTextureSize(int texture, int* width, int* height)
{
	VkPsxState* psx = &g_vk.psx;
	const VkPsxTexture* tex = (texture > 0 && texture <= PSYX_VK_PSX_MAX_TEXTURES)
		? &psx->textures[texture - 1]
		: NULL;
	if (width) *width = (tex && tex->used) ? tex->width : 0;
	if (height) *height = (tex && tex->used) ? tex->height : 0;
}

// ---------------------------------------------------------------------------
// In-game modern mesh (renderer roadmap R7b)
//
// Mirrors PsyX_ModernMesh.cpp's OpenGL path on the Vulkan backend: persistent
// meshes drawn after the legacy scene into the shared colour/depth attachments,
// the same PBR lighting set, the directional shadow map (casters) and the
// legacy shadow composite (receivers). The game-side API is unchanged; only the
// storage and draw calls differ.

// Cofactor inverse of a column-major 4x4. Returns 0 when singular. Kept local
// so the Vulkan backend does not depend on the OpenGL renderer's internals.
static int InvertMatrix4(const float m[16], float out[16])
{
	float inv[16];

	inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
		+ m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
		- m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
		+ m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
		- m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
		- m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
		+ m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
		- m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
		+ m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
		+ m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
	inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
		- m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
		+ m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
		- m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
	inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
		- m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
	inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
		+ m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
		- m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
		+ m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

	float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (det > -1e-12f && det < 1e-12f)
		return 0;

	det = 1.0f / det;
	for (int i = 0; i < 16; i++)
		out[i] = inv[i] * det;
	return 1;
}

// Game texture handle accessor: 1-based PSX texture slots, 0 = neutral default.
static const VkPsxTexture* GameTextureForHandle(int handle)
{
	if (handle <= 0 || handle > PSYX_VK_PSX_MAX_TEXTURES)
		return NULL;
	const VkPsxTexture* tex = &g_vk.psx.textures[handle - 1];
	return tex->used ? tex : NULL;
}

static void GameDefaultMaterialView(int slot, VkImageView* viewOut, VkSampler* samplerOut)
{
	// 0 = base colour, 1 = normal, 2 = metallic/roughness, 3 = emissive.
	// White is the identity multiply for base/mr/emissive; the flat normal map
	// reconstructs the geometric normal.
	static const int defaults[4] = { 0, 1, 0, 0 };
	const int dummy = g_vk.dummyTextures[defaults[slot]];
	if (dummy >= 0 && g_textures[dummy].used)
	{
		*viewOut = g_textures[dummy].view;
		*samplerOut = g_vk.textureSampler;
	}
	else
	{
		*viewOut = VK_NULL_HANDLE;
		*samplerOut = VK_NULL_HANDLE;
	}
}

static void UpdateGameMeshDescriptorSet(VkGameMesh* mesh)
{
	if (!mesh->descriptorSet || !g_vk.gameModernUboBuffer)
		return;

	VkDescriptorBufferInfo bufferInfo;
	memset(&bufferInfo, 0, sizeof(bufferInfo));
	bufferInfo.buffer = g_vk.gameModernUboBuffer;
	bufferInfo.offset = 0;
	bufferInfo.range = sizeof(VkGameModernUbo);

	VkDescriptorImageInfo shadowInfo;
	memset(&shadowInfo, 0, sizeof(shadowInfo));
	shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	shadowInfo.imageView = g_vk.shadowView;
	shadowInfo.sampler = g_vk.shadowSampler;

	VkDescriptorImageInfo materialInfos[4];
	memset(materialInfos, 0, sizeof(materialInfos));
	for (int i = 0; i < 4; i++)
	{
		VkImageView view = VK_NULL_HANDLE;
		VkSampler sampler = VK_NULL_HANDLE;
		GameDefaultMaterialView(i, &view, &sampler);

		const VkPsxTexture* texture = GameTextureForHandle(mesh->textureSlots[i]);
		if (texture)
		{
			view = texture->view;
			sampler = texture->sampler;
		}

		materialInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		materialInfos[i].imageView = view;
		materialInfos[i].sampler = sampler;
	}

	VkWriteDescriptorSet writes[6];
	memset(writes, 0, sizeof(writes));

	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = mesh->descriptorSet;
	writes[0].dstBinding = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[0].pBufferInfo = &bufferInfo;

	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = mesh->descriptorSet;
	writes[1].dstBinding = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[1].pImageInfo = &shadowInfo;

	for (int i = 0; i < 4; i++)
	{
		writes[2 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[2 + i].dstSet = mesh->descriptorSet;
		writes[2 + i].dstBinding = (uint32_t)(2 + i);
		writes[2 + i].descriptorCount = 1;
		writes[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[2 + i].pImageInfo = &materialInfos[i];
	}

	vkUpdateDescriptorSets(g_vk.device, 6, writes, 0, NULL);
}

static void UpdateGameCompositeDescriptorSet(void)
{
	if (!g_vk.gameCompositeSet || !g_vk.gameModernUboBuffer || !g_vk.sceneDepthView)
		return;

	VkDescriptorBufferInfo bufferInfo;
	memset(&bufferInfo, 0, sizeof(bufferInfo));
	bufferInfo.buffer = g_vk.gameModernUboBuffer;
	bufferInfo.offset = 0;
	bufferInfo.range = sizeof(VkGameModernUbo);

	VkDescriptorImageInfo shadowInfo;
	memset(&shadowInfo, 0, sizeof(shadowInfo));
	shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	shadowInfo.imageView = g_vk.shadowView;
	shadowInfo.sampler = g_vk.shadowSampler;

	VkDescriptorImageInfo depthInfo;
	memset(&depthInfo, 0, sizeof(depthInfo));
	depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	depthInfo.imageView = g_vk.sceneDepthView;
	depthInfo.sampler = g_vk.sceneDepthSampler;

	VkWriteDescriptorSet writes[3];
	memset(writes, 0, sizeof(writes));

	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = g_vk.gameCompositeSet;
	writes[0].dstBinding = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[0].pBufferInfo = &bufferInfo;

	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = g_vk.gameCompositeSet;
	writes[1].dstBinding = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[1].pImageInfo = &shadowInfo;

	writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet = g_vk.gameCompositeSet;
	writes[2].dstBinding = 2;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[2].pImageInfo = &depthInfo;

	vkUpdateDescriptorSets(g_vk.device, 3, writes, 0, NULL);
}

int PsyX_Vk_GameModernMeshCreate(const PsyXModernMeshDesc* desc)
{
	if (!g_vk.initialised || !g_vk.psx.ready || !desc || !desc->positions || desc->vertexCount <= 0)
		return -1;
	if (g_vk.gameModernMeshCount >= PSYX_VK_GAME_MODERN_MAX_MESHES)
		return -1;

	int slot = -1;
	for (int i = 0; i < PSYX_VK_GAME_MODERN_MAX_MESHES; i++)
	{
		if (!g_vk.gameMeshes[i].used)
		{
			slot = i;
			break;
		}
	}
	if (slot < 0)
		return -1;

	VkGameMesh* mesh = &g_vk.gameMeshes[slot];
	const VkDescriptorSet descriptorSet = mesh->descriptorSet;
	memset(mesh, 0, sizeof(*mesh));
	mesh->descriptorSet = descriptorSet;

	VkVertex* vertices = new VkVertex[desc->vertexCount];
	FillMeshVertices(desc, vertices);

	const VkDeviceSize vertexBytes = (VkDeviceSize)desc->vertexCount * sizeof(VkVertex);
	void* vertexMapped = NULL;
	if (!CreateBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&mesh->vertexBuffer, &mesh->vertexMemory, &vertexMapped))
	{
		delete[] vertices;
		return -1;
	}
	memcpy(vertexMapped, vertices, (size_t)vertexBytes);
	delete[] vertices;
	mesh->vertexCount = desc->vertexCount;

	if (desc->indices && desc->indexCount > 0)
	{
		uint32_t* indices = new uint32_t[desc->indexCount];
		for (int i = 0; i < desc->indexCount; i++)
			indices[i] = desc->indices[i];

		const VkDeviceSize indexBytes = (VkDeviceSize)desc->indexCount * sizeof(uint32_t);
		void* indexMapped = NULL;
		if (!CreateBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&mesh->indexBuffer, &mesh->indexMemory, &indexMapped))
		{
			delete[] indices;
			return -1;
		}
		memcpy(indexMapped, indices, (size_t)indexBytes);
		delete[] indices;
		mesh->indexCount = desc->indexCount;
	}

	// Texture handles are the game's PsyX_CreateRGBATexture handles; the
	// descriptor set resolves them against the PSX texture table at draw time.
	mesh->textureSlots[0] = (int)desc->baseColorTexture;
	mesh->textureSlots[1] = (int)desc->normalTexture;
	mesh->textureSlots[2] = (int)desc->metallicRoughnessTexture;
	mesh->textureSlots[3] = (int)desc->emissiveTexture;

	mesh->factors[0] = desc->metallicFactor;
	mesh->factors[1] = desc->roughnessFactor;
	mesh->emissive[0] = desc->emissiveFactor ? desc->emissiveFactor[0] : 0.0f;
	mesh->emissive[1] = desc->emissiveFactor ? desc->emissiveFactor[1] : 0.0f;
	mesh->emissive[2] = desc->emissiveFactor ? desc->emissiveFactor[2] : 0.0f;

	mesh->world[0] = mesh->world[5] = mesh->world[10] = mesh->world[15] = 1.0f;
	mesh->color[0] = mesh->color[1] = mesh->color[2] = mesh->color[3] = 1.0f;
	mesh->visible = 1;
	mesh->used = 1;
	g_vk.gameModernMeshCount++;

	UpdateGameMeshDescriptorSet(mesh);
	return slot;
}

void PsyX_Vk_GameModernMeshDestroy(int mesh)
{
	if (mesh < 0 || mesh >= PSYX_VK_GAME_MODERN_MAX_MESHES || !g_vk.gameMeshes[mesh].used)
		return;

	VkGameMesh* m = &g_vk.gameMeshes[mesh];
	vkDeviceWaitIdle(g_vk.device);

	const VkDescriptorSet descriptorSet = m->descriptorSet;
	if (m->vertexBuffer) vkDestroyBuffer(g_vk.device, m->vertexBuffer, NULL);
	if (m->vertexMemory) vkFreeMemory(g_vk.device, m->vertexMemory, NULL);
	if (m->indexBuffer) vkDestroyBuffer(g_vk.device, m->indexBuffer, NULL);
	if (m->indexMemory) vkFreeMemory(g_vk.device, m->indexMemory, NULL);

	memset(m, 0, sizeof(*m));
	m->descriptorSet = descriptorSet;
	g_vk.gameModernMeshCount--;
}

void PsyX_Vk_GameModernMeshSetInstance(int mesh, const float viewMatrix[16],
	const float color[4], int visible)
{
	if (mesh < 0 || mesh >= PSYX_VK_GAME_MODERN_MAX_MESHES || !g_vk.gameMeshes[mesh].used)
		return;

	// The view matrix is derived in the shader from the world matrix and the
	// frame camera, so only the tint and visibility are stored here.
	(void)viewMatrix;
	if (color)
		memcpy(g_vk.gameMeshes[mesh].color, color, sizeof(g_vk.gameMeshes[mesh].color));
	g_vk.gameMeshes[mesh].visible = visible != 0;
}

void PsyX_Vk_GameModernMeshSetInstanceWorld(int mesh, const float worldMatrix[16])
{
	if (mesh < 0 || mesh >= PSYX_VK_GAME_MODERN_MAX_MESHES || !g_vk.gameMeshes[mesh].used)
		return;

	if (worldMatrix)
		memcpy(g_vk.gameMeshes[mesh].world, worldMatrix, sizeof(g_vk.gameMeshes[mesh].world));
}

void PsyX_Vk_GameModernMeshSetLights(const PsyXModernLightSet* lights)
{
	if (lights)
		g_vk.lights = *lights;
}

void PsyX_Vk_GameModernMeshSetCamera(const float viewRotation[16], const float cameraPosition[3])
{
	if (viewRotation)
		memcpy(g_vk.modernCameraRotation, viewRotation, sizeof(g_vk.modernCameraRotation));
	if (cameraPosition)
		memcpy(g_vk.modernCameraPosition, cameraPosition, sizeof(g_vk.modernCameraPosition));

	g_vk.modernCameraValid = (viewRotation != NULL && cameraPosition != NULL);
}

void PsyX_Vk_GameModernMeshSetShadowDebug(int mode)
{
	g_vk.gameModernShadowDebug = mode;
}

void PsyX_Vk_GameModernMeshSetEnabled(int enabled)
{
	g_vk.gameModernEnabled = enabled != 0;
}

int PsyX_Vk_GameModernMeshGetEnabled(void)
{
	return g_vk.gameModernEnabled;
}

void PsyX_Vk_GameModernMeshGetStats(PsyXModernMeshStats* stats)
{
	if (stats)
		*stats = g_vk.gameModernStats;
}

void PsyX_Vk_GameModernMeshShutdown(void)
{
	for (int i = 0; i < PSYX_VK_GAME_MODERN_MAX_MESHES; i++)
	{
		if (g_vk.gameMeshes[i].used)
			PsyX_Vk_GameModernMeshDestroy(i);
	}
}

// Builds the frame's modern UBO: the legacy Projection3D captured by
// GR_Perspective3D, the GTE screen scales, the frame camera, the light set and
// the shadow volume.
static void UpdateGameModernUbo(void)
{
	VkGameModernUbo* ubo = g_vk.gameModernUboMapped;
	if (!ubo)
		return;

	memset(ubo, 0, sizeof(*ubo));

	static const float identity[16] =
	{
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f,
	};

	const float* projection = g_psyxModernProjectionValid ? g_psyxModernProjection : identity;
	memcpy(ubo->proj, projection, sizeof(ubo->proj));
	if (!InvertMatrix4(ubo->proj, ubo->projInverse))
		memset(ubo->projInverse, 0, sizeof(ubo->projInverse));

	// The OpenGL modern path's device-space encoding: the GTE screen distance
	// over the display size, and camera z over the same fixed-point factor.
	float displayWidth = 320.0f;
	float displayHeight = 240.0f;
	if (activeDispEnv.disp.w > 0 && activeDispEnv.disp.h > 0)
	{
		displayWidth = (float)activeDispEnv.disp.w;
		displayHeight = (float)activeDispEnv.disp.h;
	}
	ubo->xyScale[0] = (float)C2_H / (128.0f * displayWidth);
	ubo->xyScale[1] = (float)C2_H / (128.0f * displayHeight);
	ubo->xyScale[2] = 1.0f / 128.0f;

	memcpy(ubo->cameraRotation, g_vk.modernCameraRotation, sizeof(ubo->cameraRotation));
	ubo->cameraPos[0] = g_vk.modernCameraPosition[0];
	ubo->cameraPos[1] = g_vk.modernCameraPosition[1];
	ubo->cameraPos[2] = g_vk.modernCameraPosition[2];

	// Full camera view (rotation + translation) inverted for the composite's
	// view-space -> world-space reconstruction, exactly like the OpenGL path.
	{
		float view[16];
		memcpy(view, g_vk.modernCameraRotation, sizeof(view));
		const float cx = g_vk.modernCameraPosition[0];
		const float cy = g_vk.modernCameraPosition[1];
		const float cz = g_vk.modernCameraPosition[2];
		view[12] = -(view[0] * cx + view[4] * cy + view[8] * cz);
		view[13] = -(view[1] * cx + view[5] * cy + view[9] * cz);
		view[14] = -(view[2] * cx + view[6] * cy + view[10] * cz);
		view[3] = view[7] = view[11] = 0.0f;
		view[15] = 1.0f;
		if (!InvertMatrix4(view, ubo->cameraViewInverse))
			memset(ubo->cameraViewInverse, 0, sizeof(ubo->cameraViewInverse));
	}

	BuildShadowMatrix(ubo->shadowMatrix);

	ubo->shadowParams[0] = g_vk.lights.shadowsEnabled ? 1.0f : 0.0f;
	ubo->shadowParams[1] = 1.0f / (float)kShadowSize;
	ubo->shadowParams[2] = 0.45f;
	ubo->shadowParams[3] = g_vk.lights.aoEnabled ? 1.0f : 0.0f;

	int lightCount = g_vk.lights.count;
	if (lightCount < 0) lightCount = 0;
	if (lightCount > PSYX_VK_MAX_LIGHTS) lightCount = PSYX_VK_MAX_LIGHTS;

	ubo->lightInfo[0] = (float)lightCount;
	ubo->lightInfo[1] = g_vk.srgbOutput ? 1.0f : 0.0f;
	ubo->lightInfo[2] = (float)g_vk.gameModernShadowDebug;

	ubo->ambientExposure[0] = g_vk.lights.ambient[0];
	ubo->ambientExposure[1] = g_vk.lights.ambient[1];
	ubo->ambientExposure[2] = g_vk.lights.ambient[2];
	ubo->ambientExposure[3] = g_vk.lights.exposure > 0.0f ? g_vk.lights.exposure : 1.0f;

	ubo->viewport[0] = (float)g_vk.width;
	ubo->viewport[1] = (float)g_vk.height;

	for (int i = 0; i < lightCount; i++)
	{
		const PsyXModernLight* light = &g_vk.lights.lights[i];
		ubo->lights[i].posRange[0] = light->position[0];
		ubo->lights[i].posRange[1] = light->position[1];
		ubo->lights[i].posRange[2] = light->position[2];
		ubo->lights[i].posRange[3] = light->range > 0.0f ? light->range : 1000.0f;
		ubo->lights[i].dirType[0] = light->direction[0];
		ubo->lights[i].dirType[1] = light->direction[1];
		ubo->lights[i].dirType[2] = light->direction[2];
		ubo->lights[i].dirType[3] = (float)light->type;
		ubo->lights[i].color[0] = light->color[0] * light->intensity;
		ubo->lights[i].color[1] = light->color[1] * light->intensity;
		ubo->lights[i].color[2] = light->color[2] * light->intensity;
	}
}

// Records the modern shadow casters into the already-open shadow pass. Casters
// use their world matrix, so an instance casts where it stands.
static void RecordGameModernShadowPass(VkCommandBuffer cmd)
{
	if (!g_vk.gameModernEnabled || !g_vk.lights.shadowsEnabled || !g_vk.gameModernUboMapped)
		return;

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.shadowPipeline);

	for (int i = 0; i < PSYX_VK_GAME_MODERN_MAX_MESHES; i++)
	{
		VkGameMesh* mesh = &g_vk.gameMeshes[i];
		if (!mesh->used || !mesh->visible || mesh->vertexCount == 0)
			continue;

		float lightWorld[16];
		MulMatrix4(g_vk.gameModernUboMapped->shadowMatrix, mesh->world, lightWorld);

		vkCmdPushConstants(cmd, g_vk.shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, lightWorld);

		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vertexBuffer, &offset);
		if (mesh->indexCount > 0)
		{
			vkCmdBindIndexBuffer(cmd, mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmd, (uint32_t)mesh->indexCount, 1, 0, 0, 0);
		}
		else
		{
			vkCmdDraw(cmd, (uint32_t)mesh->vertexCount, 1, 0, 0);
		}
	}
}

// Copies the legacy scene depth (written by the main pass) into a sampleable
// image so the composite can reconstruct world positions for shaded pixels.
static void RecordGameModernSceneDepthCopy(VkCommandBuffer cmd)
{
	// Layout transitions cover both aspects unless separateDepthStencilLayouts
	// is enabled. The copy and sampling view still access depth alone.
	const VkImageAspectFlags aspect = g_vk.psx.stencilSupported
		? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
		: VK_IMAGE_ASPECT_DEPTH_BIT;

	ImageBarrier(cmd, g_vk.depthImage, aspect,
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	// This frame overwrites the entire depth copy, so discard its old contents.
	// Synchronize any preceding composite reads before writing the new copy.
	ImageBarrier(cmd, g_vk.sceneDepthImage, aspect,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

	VkImageCopy copy;
	memset(&copy, 0, sizeof(copy));
	copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	copy.srcSubresource.layerCount = 1;
	copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	copy.dstSubresource.layerCount = 1;
	copy.extent.width = (uint32_t)g_vk.width;
	copy.extent.height = (uint32_t)g_vk.height;
	copy.extent.depth = 1;
	vkCmdCopyImage(cmd, g_vk.depthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		g_vk.sceneDepthImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

	ImageBarrier(cmd, g_vk.depthImage, aspect,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

	ImageBarrier(cmd, g_vk.sceneDepthImage, aspect,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

// Draws the modern meshes into the current (load) pass. Returns the number of
// draw calls and fills the shared stats.
static int RecordGameModernMeshes(VkCommandBuffer cmd)
{
	PsyXModernMeshStats* stats = &g_vk.gameModernStats;
	const int legacyShadowPass = stats->legacyShadowPass;
	memset(stats, 0, sizeof(*stats));
	stats->legacyShadowPass = legacyShadowPass;
	stats->meshCount = g_vk.gameModernMeshCount;
	stats->depthShared = 1;

	if (!g_vk.gameModernEnabled || g_vk.gameModernMeshCount == 0 || !g_vk.gameModernPipeline)
		return 0;

	const Uint64 start = SDL_GetPerformanceCounter();
	int draws = 0;

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.gameModernPipeline);

	for (int i = 0; i < PSYX_VK_GAME_MODERN_MAX_MESHES; i++)
	{
		VkGameMesh* mesh = &g_vk.gameMeshes[i];
		if (!mesh->used || !mesh->visible || mesh->vertexCount == 0)
			continue;

		float push[28];
		memcpy(push, mesh->world, sizeof(mesh->world));
		memcpy(push + 16, mesh->color, sizeof(mesh->color));
		push[20] = mesh->factors[0];
		push[21] = mesh->factors[1];
		push[22] = 0.0f;
		push[23] = 0.0f;
		push[24] = mesh->emissive[0];
		push[25] = mesh->emissive[1];
		push[26] = mesh->emissive[2];
		push[27] = 0.0f;

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			g_vk.pipelineLayout, 0, 1, &mesh->descriptorSet, 0, NULL);
		vkCmdPushConstants(cmd, g_vk.pipelineLayout,
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), push);

		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vertexBuffer, &offset);
		if (mesh->indexCount > 0)
		{
			vkCmdBindIndexBuffer(cmd, mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmd, (uint32_t)mesh->indexCount, 1, 0, 0, 0);
		}
		else
		{
			vkCmdDraw(cmd, (uint32_t)mesh->vertexCount, 1, 0, 0);
		}

		stats->visibleInstances++;
		stats->vertexCount += mesh->vertexCount;
		stats->drawCalls++;
		draws++;
	}

	const Uint64 end = SDL_GetPerformanceCounter();
	const Uint64 frequency = SDL_GetPerformanceFrequency();
	stats->lastFrameMicros = frequency ? (int)((end - start) * 1000000ull / frequency) : 0;

	return draws;
}

void PsyX_Vk_GameEndFrame(void)
{
	if (!g_vk.initialised || !g_vk.psx.ready)
		return;

	PsyX_Vk_RenderFrame();
}

void PsyX_Vk_GameResetDevice(void)
{
	// RenderFrame re-creates the swapchain when the drawable size changes, so
	// a resize needs no explicit reset here.
}

int PsyX_Vk_GameIsActive(void)
{
	return g_vk.initialised && g_vk.gameMode;
}

SDL_Window* PsyX_Vk_GetSDLWindow(void)
{
	return g_vk.window;
}

// Replays the VRAM writes a draw must see. A transfer cannot be recorded
// inside a render pass, so the pass is closed and immediately reopened against
// the loaded attachments; the draw state set per draw is unaffected.
static void ApplyVramUploadsUpTo(VkCommandBuffer cmd, uint32_t generation, int *passClosed)
{
	VkPsxState* psx = &g_vk.psx;

	while (psx->vramUploadsApplied < psx->vramUploadCount &&
		psx->vramUploads[psx->vramUploadsApplied].generation < generation)
	{
		const VkPsxVramUpload* up = &psx->vramUploads[psx->vramUploadsApplied];

		if (g_vk.mainPassOpen)
		{
			vkCmdEndRenderPass(cmd);
			g_vk.mainPassOpen = 0;
			*passClosed = 1;
		}

		ImageBarrier(cmd, psx->vramImage, VK_IMAGE_ASPECT_COLOR_BIT,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

		VkBufferImageCopy region;
		memset(&region, 0, sizeof(region));
		region.bufferOffset = up->offset;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.layerCount = 1;
		region.imageOffset.x = up->x;
		region.imageOffset.y = up->y;
		region.imageExtent.width = (uint32_t)up->w;
		region.imageExtent.height = (uint32_t)up->h;
		region.imageExtent.depth = 1;
		vkCmdCopyBufferToImage(cmd, psx->vramUploadStaging, psx->vramImage,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		ImageBarrier(cmd, psx->vramImage, VK_IMAGE_ASPECT_COLOR_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

		psx->vramUploadsApplied++;
	}
}

static void ResumeMainPass(VkCommandBuffer cmd)
{
	if (g_vk.mainPassOpen || !g_vk.mainPassBegin.renderPass)
		return;

	VkRenderPassBeginInfo resume = g_vk.mainPassBegin;
	resume.renderPass = g_vk.modernRenderPass;	// loads colour and depth
	resume.framebuffer = g_vk.modernFramebuffers[g_vk.mainPassImageIndex];
	vkCmdBeginRenderPass(cmd, &resume, VK_SUBPASS_CONTENTS_INLINE);
	g_vk.mainPassOpen = 1;
}

// Records the queued PSX draws into `cmd`. A draw belongs to a single display
// list (frame index); `frame` selects which one. `offscreen` selects the draws
// queued inside a GR_SetOffscreenState render-to-VRAM run, and the main pass
// records the complement so those draws are not also drawn on screen.
static void RecordPsxDraws(VkCommandBuffer cmd, int offscreen, int frame,
	int targetWidth, int targetHeight)
{
	VkPsxState* psx = &g_vk.psx;
	// A frame can carry VRAM writes without any draw (level loading); those
	// still have to reach the image.
	if (!psx->ready || (psx->drawCount == 0 && (offscreen != 0 || psx->vramUploadCount == 0)))
		return;

	const int endDraw = psx->drawCount;

	// One vertex buffer per frame, as the game uploads it once.
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &psx->vertexBuffer, &offset);

	// GL uses GL_CONSTANT_ALPHA with glBlendColor(..., 0.5f).
	const float blendConstants[4] = { 0.25f, 0.25f, 0.25f, 0.5f };
	vkCmdSetBlendConstants(cmd, blendConstants);

	VkDescriptorSet boundSet = VK_NULL_HANDLE;
	int passClosed = 0;

	int stencilDraws = 0;
	int recordedDraws = 0;

	for (int i = 0; i < endDraw; i++)
	{
		const VkPsxDraw* draw = &psx->draws[i];
		if (draw->frame != frame)
			continue;
		if ((draw->offscreen ? 1 : 0) != offscreen)
			continue;

		recordedDraws++;
		if (!offscreen && draw->stencilMode)
			stencilDraws++;

		// The game rewrites VRAM between flushes; only the writes that happened
		// before this draw may be visible to it.
		if (!offscreen)
		{
			ApplyVramUploadsUpTo(cmd, draw->vramGeneration, &passClosed);
			ResumeMainPass(cmd);
		}

		const VkDescriptorSet set = draw->textureSet ? draw->textureSet : psx->dummySet;
		const VkPipeline pipeline = offscreen
			? psx->offscreenPipelines[draw->blendMode]
			: ((draw->blendMode == 0 && !draw->depthTest)
				? (draw->stencilMode ? psx->pipelineNoDepthStencilWrite : psx->pipelineNoDepth)
				: (draw->stencilMode ? psx->pipelinesStencilWrite[draw->blendMode] : psx->pipelines[draw->blendMode]));

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		if (set != boundSet)
		{
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
				psx->layout, 0, 1, &set, 0, NULL);
			boundSet = set;
		}

		// GR_SetupClipMode / GR_SetViewPort work in top-left window space; the
		// viewport flips the y axis to match the OpenGL conventions the game
		// relies on.
		// psx.vert already converts GL clip space to Vulkan, so the viewport is
		// right-side up; only the GL bottom-left origin has to be mapped to
		// Vulkan's top-left one.
		VkViewport viewport;
		viewport.x = (float)draw->viewport[0];
		viewport.width = (float)(draw->viewport[2] > 0 ? draw->viewport[2] : targetWidth);
		viewport.height = (float)(draw->viewport[3] > 0 ? draw->viewport[3] : targetHeight);
		viewport.y = (float)targetHeight - ((float)draw->viewport[1] + viewport.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		VkRect2D scissor;
		scissor.offset.x = draw->scissorEnable ? draw->scissor[0] : 0;
		// GR_SetupClipMode hands over the same bottom-left rectangle glScissor
		// receives, so map it to Vulkan's top-left origin like the viewport.
		scissor.offset.y = draw->scissorEnable
			? (int)targetHeight - (draw->scissor[1] + draw->scissor[3])
			: 0;
		scissor.extent.width = (uint32_t)(draw->scissorEnable ? draw->scissor[2] : targetWidth);
		scissor.extent.height = (uint32_t)(draw->scissorEnable ? draw->scissor[3] : targetHeight);
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		struct
		{
			int texFormat;
			int bilinearFilter;
			float texelSize[2];
			int overrideAlphaMode;
			int srgbEncode;
		} constants;
		constants.texFormat = draw->texFormat;
		constants.bilinearFilter = draw->bilinearFilter;
		constants.texelSize[0] = draw->texelSize[0];
		constants.texelSize[1] = draw->texelSize[1];
		constants.overrideAlphaMode = draw->overrideAlphaMode;
		constants.srgbEncode = draw->srgbEncode;

		vkCmdPushConstants(cmd, psx->layout,
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants), &constants);

		vkCmdDraw(cmd, draw->vertexCount, 1, draw->firstVertex, 0);
	}

	if (!offscreen)
	{
		psx->lastDraws = recordedDraws;
		psx->lastStencilDraws = stencilDraws;

		// Writes queued after the last draw still have to land in the image so
		// the next frame starts from the memory the game expects.
		ApplyVramUploadsUpTo(cmd, psx->vramUploadGeneration + 1, &passClosed);
	}
}

// Creates or resizes the render-to-VRAM target so its attachment matches the
// GR_SetOffscreenState rectangle (the GL renderer resizes its offscreen texture
// the same way). The result is read back through a host-visible buffer.
static int EnsureOffscreenTarget(int width, int height)
{
	VkPsxState* psx = &g_vk.psx;

	if (width <= 0 || height <= 0)
		return 0;
	if (width > PSYX_VK_VRAM_WIDTH || height > PSYX_VK_VRAM_HEIGHT)
		return 0;

	if (psx->offscreenImage && psx->offscreenWidth == width && psx->offscreenHeight == height)
		return 1;

	if (psx->offscreenFramebuffer) { vkDestroyFramebuffer(g_vk.device, psx->offscreenFramebuffer, NULL); psx->offscreenFramebuffer = VK_NULL_HANDLE; }
	if (psx->offscreenView) { vkDestroyImageView(g_vk.device, psx->offscreenView, NULL); psx->offscreenView = VK_NULL_HANDLE; }
	if (psx->offscreenImage) { vkDestroyImage(g_vk.device, psx->offscreenImage, NULL); psx->offscreenImage = VK_NULL_HANDLE; }
	if (psx->offscreenMemory) { vkFreeMemory(g_vk.device, psx->offscreenMemory, NULL); psx->offscreenMemory = VK_NULL_HANDLE; }
	if (psx->offscreenReadback) { vkDestroyBuffer(g_vk.device, psx->offscreenReadback, NULL); psx->offscreenReadback = VK_NULL_HANDLE; }
	if (psx->offscreenReadbackMemory) { vkFreeMemory(g_vk.device, psx->offscreenReadbackMemory, NULL); psx->offscreenReadbackMemory = VK_NULL_HANDLE; }
	psx->offscreenReadbackMapped = NULL;
	psx->offscreenReadbackSize = 0;
	psx->offscreenWidth = 0;
	psx->offscreenHeight = 0;

	if (!CreateImage2D((uint32_t)width, (uint32_t)height, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		&psx->offscreenImage, &psx->offscreenMemory))
		return 0;
	if (!CreateImageView2D(psx->offscreenImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &psx->offscreenView))
		return 0;

	VkFramebufferCreateInfo framebuffer;
	memset(&framebuffer, 0, sizeof(framebuffer));
	framebuffer.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebuffer.renderPass = psx->offscreenRenderPass;
	framebuffer.attachmentCount = 1;
	framebuffer.pAttachments = &psx->offscreenView;
	framebuffer.width = (uint32_t)width;
	framebuffer.height = (uint32_t)height;
	framebuffer.layers = 1;
	if (!VkOk(vkCreateFramebuffer(g_vk.device, &framebuffer, NULL, &psx->offscreenFramebuffer), "vkCreateFramebuffer(psx offscreen)"))
		return 0;

	const VkDeviceSize size = (VkDeviceSize)width * height * 4;
	if (!CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&psx->offscreenReadback, &psx->offscreenReadbackMemory, &psx->offscreenReadbackMapped))
		return 0;
	psx->offscreenReadbackSize = size;

	psx->offscreenWidth = width;
	psx->offscreenHeight = height;

	{
		char line[128];
		snprintf(line, sizeof(line), "psx: offscreen target %dx%d", width, height);
		VkStage(line);
	}
	return 1;
}

// Renders one GR_SetOffscreenState group into the offscreen target, copies the
// pixels back and packs them into the PSX VRAM mirror at the group rectangle.
static void ResolveOffscreenGroup(const VkPsxOffscreenGroup* group, unsigned short* vram)
{
	VkPsxState* psx = &g_vk.psx;
	const int w = group->rect[2];
	const int h = group->rect[3];

	if (group->rect[0] < 0 || group->rect[1] < 0 ||
		group->rect[0] + w > PSYX_VK_VRAM_WIDTH ||
		group->rect[1] + h > PSYX_VK_VRAM_HEIGHT)
	{
		eprintwarn("PsyX Vulkan: offscreen rect outside VRAM, skipping\n");
		return;
	}
	if (!EnsureOffscreenTarget(w, h) || !psx->offscreenReadbackMapped)
		return;

	// The projection active when the group was queued (GR_SetOffscreenState's
	// GR_Ortho2D) is restored for this pass only; the main frame sees whatever
	// the game left behind, exactly like the immediate-mode OpenGL renderer.
	float saved[32];
	memcpy(saved, psx->uboMapped, sizeof(saved));

	VkCommandBuffer cmd = BeginOneShot();
	if (cmd == VK_NULL_HANDLE)
		return;

	memcpy(psx->uboMapped, group->matrix, sizeof(saved));

	VkClearValue clear;
	memset(&clear, 0, sizeof(clear));
	clear.color.float32[0] = 0.5f;
	clear.color.float32[1] = 0.5f;
	clear.color.float32[2] = 0.5f;
	clear.color.float32[3] = 0.0f;

	VkRenderPassBeginInfo begin;
	memset(&begin, 0, sizeof(begin));
	begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin.renderPass = psx->offscreenRenderPass;
	begin.framebuffer = psx->offscreenFramebuffer;
	begin.renderArea.extent.width = (uint32_t)w;
	begin.renderArea.extent.height = (uint32_t)h;
	begin.clearValueCount = 1;
	begin.pClearValues = &clear;

	vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport;
	memset(&viewport, 0, sizeof(viewport));
	viewport.width = (float)w;
	viewport.height = (float)h;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	VkRect2D scissor;
	memset(&scissor, 0, sizeof(scissor));
	scissor.extent.width = (uint32_t)w;
	scissor.extent.height = (uint32_t)h;
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	RecordPsxDraws(cmd, 1, group->frame, w, h);

	vkCmdEndRenderPass(cmd);

	VkBufferImageCopy copy;
	memset(&copy, 0, sizeof(copy));
	copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent.width = (uint32_t)w;
	copy.imageExtent.height = (uint32_t)h;
	copy.imageExtent.depth = 1;
	vkCmdCopyImageToBuffer(cmd, psx->offscreenImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		psx->offscreenReadback, 1, &copy);

	EndOneShot(cmd);	// submits and waits, so the pixels below are complete

	memcpy(psx->uboMapped, saved, sizeof(saved));

	// The Vulkan render target is top-down like the presented image, so the
	// rows map straight onto VRAM (the GL path flips its bottom-up FBO).
	const unsigned char* src = (const unsigned char*)psx->offscreenReadbackMapped;
	for (int row = 0; row < h; row++)
	{
		const unsigned char* s = src + (size_t)row * (size_t)w * 4;
		unsigned short* d = vram + (size_t)(group->rect[1] + row) * PSYX_VK_VRAM_WIDTH + group->rect[0];
		for (int col = 0; col < w; col++)
		{
			const int r = s[0] >> 3;
			const int g = s[1] >> 3;
			const int b = s[2] >> 3;
			// PSX 5551 is "transparent unless the word is non-zero", matching
			// the texture decode (master.alpha = w != 0 ? 1 : 0).
			const int a = (r | g | b) ? 1 : 0;
			d[col] = (unsigned short)(r | (g << 5) | (b << 10) | (a << 15));
			s += 4;
		}
	}
}

int PsyX_Vk_GameResolveOffscreen(unsigned short* vram)
{
	VkPsxState* psx = &g_vk.psx;
	if (!g_vk.initialised || !psx->ready || !vram || psx->offscreenGroupCount == 0)
		return 0;

	for (int i = 0; i < psx->offscreenGroupCount; i++)
		ResolveOffscreenGroup(&psx->offscreenGroups[i], vram);

	psx->offscreenGroupCount = 0;

	// The mirror changed: re-upload it so the on-screen draws that sample the
	// offscreen region see it during this frame, and so the frame's mirror
	// upload does not overwrite it with stale data.
	UploadVramMirror(vram);
	return 1;
}

static void ReportAppend(char* report, int size, const char* text)
{
	if (!report || size <= 0)
		return;
	const size_t length = strlen(report);
	if (length + 1 >= (size_t)size)
		return;
	strncat(report, text, (size_t)size - length - 1);
}

static void PsxBuildOrtho(float left, float right, float bottom, float top, float znear, float zfar, float* out)
{
	memset(out, 0, 16 * sizeof(float));
	out[0] = 2.0f / (right - left);
	out[5] = 2.0f / (top - bottom);
	out[10] = -2.0f / (zfar - znear);
	out[12] = -(right + left) / (right - left);
	out[13] = -(top + bottom) / (top - bottom);
	out[14] = -(zfar + znear) / (zfar - znear);
	out[15] = 1.0f;
}

static void PsxFillQuad(float width, float height, float page, float clut, float u, float v, VkPsxVertex* vertices)
{
	const float xs[4] = { 0.0f, width, width, 0.0f };
	const float ys[4] = { 0.0f, 0.0f, height, height };
	const int order[6] = { 0, 1, 2, 0, 2, 3 };

	for (int i = 0; i < 6; i++)
	{
		VkPsxVertex* vertex = &vertices[i];
		memset(vertex, 0, sizeof(*vertex));
		vertex->x = xs[order[i]];
		vertex->y = ys[order[i]];
		vertex->page = page;
		vertex->clut = clut;
		vertex->u = (unsigned char)u;
		vertex->v = (unsigned char)v;
		vertex->bright = 1;	// multiplies the vertex colour by 1.0
		vertex->r = vertex->g = vertex->b = vertex->a = 255;
	}
}

static int PsxCheckPixel(const unsigned char* rgba, int width, int height, int x, int y,
	const float expected[4], int tolerance, const char* label, char* report, int reportSize)
{
	const unsigned char* pixel = rgba + ((size_t)y * (size_t)width + (size_t)x) * 4;
	int worst = 0;
	for (int c = 0; c < 4; c++)
	{
		const int want = (int)(expected[c] * 255.0f + 0.5f);
		int diff = (int)pixel[c] - want;
		if (diff < 0)
			diff = -diff;
		if (diff > worst)
			worst = diff;
	}

	char line[256];
	snprintf(line, sizeof(line), "%s: got (%u,%u,%u,%u) worst=%d %s\n", label,
		pixel[0], pixel[1], pixel[2], pixel[3], worst, worst <= tolerance ? "ok" : "FAIL");
	ReportAppend(report, reportSize, line);
	return worst <= tolerance;
}

int PsyX_Vk_GameSelfTest(char* report, int reportSize)
{
	if (report && reportSize > 0)
		report[0] = 0;

	if (!g_vk.initialised || !g_vk.psx.ready)
	{
		ReportAppend(report, reportSize, "psx path unavailable\n");
		return 0;
	}

	const int width = g_vk.width;
	const int height = g_vk.height;

	unsigned short* vram = new unsigned short[PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT];
	if (!vram)
		return 0;

	float ortho[16];
	PsxBuildOrtho(0.0f, (float)width, (float)height, 0.0f, -1.0f, 1.0f, ortho);
	PsyX_Vk_GameSetProjection2D(ortho);

	const int tolerance = 3;
	int failures = 0;

	unsigned char* rgba = new unsigned char[(size_t)width * height * 4];
	if (!rgba)
	{
		delete[] vram;
		return 0;
	}

	// Case 1: 16-bit direct colour. VRAM (0,0) = 0x001F, pure red in PSX 5551,
	// so the shader must decode it to R = 31 << 3 = 248 with alpha 1.
	memset(vram, 0, sizeof(unsigned short) * PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT);
	vram[0] = 0x001F;

	VkPsxVertex quad[6];
	PsxFillQuad((float)width, (float)height, 0.0f, 0.0f, 0.0f, 0.0f, quad);

	PsyX_Vk_GameBeginFrame();
	PsyX_Vk_GameSetVram(vram);
	PsyX_Vk_GameUpdateVertexBuffer(quad, 6);
	PsyX_Vk_GameSetBlendMode(PSYX_VK_BLEND_NONE);
	PsyX_Vk_GameSetTexture(PSYX_VK_TEX_16BIT, 0);
	PsyX_Vk_GameEnableDepth(0);
	PsyX_Vk_GameSetBilinear(0);
	PsyX_Vk_GameSetViewPort(0, 0, width, height);
	PsyX_Vk_GameSetScissor(0, 0, 0, width, height);
	PsyX_Vk_GameDrawTriangles(0, 2);
	PsyX_Vk_RenderFrame();

	int readWidth = 0;
	int readHeight = 0;
	if (!PsyX_Vk_ReadbackRgba(rgba, &readWidth, &readHeight))
	{
		ReportAppend(report, reportSize, "readback failed\n");
		delete[] rgba;
		delete[] vram;
		return 0;
	}

	{
		// The readback is the presented swapchain. The PSX shader emits
		// display-referred values; on an sRGB attachment it inverse-encodes so
		// the hardware store re-encodes back to the original value. The pixel
		// therefore reads as the raw PSX value on both backends.
		const float expected[4] = { 248.0f / 255.0f, 0.0f, 0.0f, 1.0f };
		if (!PsxCheckPixel(rgba, readWidth, readHeight, readWidth / 2, readHeight / 2, expected, tolerance,
			"16-bit 0x001F", report, reportSize))
			failures++;
	}

	// Case 2: 4-bit CLUT. Every nibble of the texture word is 5 (so whichever
	// nibble the shader picks is 5) and CLUT entry 5 is blue while the rest is
	// green: matching blue proves the nibble extraction, CLUT row addressing and
	// the RG8 table all resolve to the right entry.
	memset(vram, 0, sizeof(unsigned short) * PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT);
	for (int x = 0; x < 16; x++)
		vram[(8 * PSYX_VK_VRAM_WIDTH) + x] = 0x03E0;	// green
	vram[(8 * PSYX_VK_VRAM_WIDTH) + 5] = 0x7C00;		// blue
	for (int x = 0; x < 64; x++)
		vram[x] = 0x5555;

	PsxFillQuad((float)width, (float)height, 0.0f, 512.0f, 0.0f, 0.0f, quad);

	PsyX_Vk_GameBeginFrame();
	PsyX_Vk_GameSetVram(vram);
	PsyX_Vk_GameUpdateVertexBuffer(quad, 6);
	PsyX_Vk_GameSetBlendMode(PSYX_VK_BLEND_NONE);
	PsyX_Vk_GameSetTexture(PSYX_VK_TEX_4BIT, 0);
	PsyX_Vk_GameEnableDepth(0);
	PsyX_Vk_GameSetBilinear(0);
	PsyX_Vk_GameSetViewPort(0, 0, width, height);
	PsyX_Vk_GameSetScissor(0, 0, 0, width, height);
	PsyX_Vk_GameDrawTriangles(0, 2);

	if (!PsyX_Vk_RenderFrame() || !PsyX_Vk_ReadbackRgba(rgba, &readWidth, &readHeight))
	{
		ReportAppend(report, reportSize, "case 2 readback failed\n");
		failures++;
	}
	else
	{
		// Same reasoning as case 1: the presented pixel is the display-referred
		// PSX value, independent of the attachment colour space.
		const float expected[4] = { 0.0f, 0.0f, 248.0f / 255.0f, 1.0f };
		if (!PsxCheckPixel(rgba, readWidth, readHeight, readWidth / 2, readHeight / 2, expected, tolerance,
			"4-bit CLUT entry 5", report, reportSize))
			failures++;
	}

	// Case 3: offscreen (render-to-VRAM) target. A 32x32 quad sampling the red
	// word at VRAM (0,0) renders into the GR_SetOffscreenState target and must
	// land in the mirror at (64,64), which proves PsyX_Vk_GameResolveOffscreen's
	// render + readback + pack path the game's Tanner shadow relies on.
	memset(vram, 0, sizeof(unsigned short) * PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT);
	vram[0] = 0x001F;	// red

	PsxFillQuad(32.0f, 32.0f, 0.0f, 0.0f, 0.0f, 0.0f, quad);

	PsyX_Vk_GameBeginFrame();
	PsyX_Vk_GameSetVram(vram);
	PsyX_Vk_GameUpdateVertexBuffer(quad, 6);
	PsyX_Vk_GameSetBlendMode(PSYX_VK_BLEND_NONE);
	PsyX_Vk_GameSetTexture(PSYX_VK_TEX_16BIT, 0);
	PsyX_Vk_GameEnableDepth(0);
	PsyX_Vk_GameSetBilinear(0);
	PsyX_Vk_GameSetViewPort(0, 0, 32, 32);
	PsyX_Vk_GameSetScissor(0, 0, 0, 32, 32);

	float offscreenOrtho[16];
	PsxBuildOrtho(0.0f, 32.0f, 32.0f, 0.0f, -1.0f, 1.0f, offscreenOrtho);
	PsyX_Vk_GameSetProjection2D(offscreenOrtho);

	PsyX_Vk_GameSetOffscreen(1, 64, 64, 32, 32);
	PsyX_Vk_GameDrawTriangles(0, 2);
	PsyX_Vk_GameSetOffscreen(0, 64, 64, 32, 32);	// group->frame = frameIndex

	if (!PsyX_Vk_GameResolveOffscreen(vram))
	{
		ReportAppend(report, reportSize, "offscreen resolve: no group resolved FAIL\n");
		failures++;
	}
	else
	{
		// Alpha is derived from "non-zero word", exactly like the texture
		// decode, so the expected word carries the opaque bit. The corners are
		// checked too so a partial quad fails even if the center happens to
		// match.
		const unsigned short expected = 0x801F;
		const int samples[3][2] = { { 8, 8 }, { 16, 16 }, { 30, 30 } };
		int offscreenFailures = 0;
		for (int s = 0; s < 3; s++)
		{
			const unsigned short packed = vram[(64 + samples[s][1]) * PSYX_VK_VRAM_WIDTH + (64 + samples[s][0])];
			if (packed != expected)
				offscreenFailures++;
		}

		char line[128];
		snprintf(line, sizeof(line), "offscreen 64,64 32x32: samples %s\n",
			offscreenFailures == 0 ? "ok" : "FAIL");
		ReportAppend(report, reportSize, line);
		if (offscreenFailures != 0)
			failures++;
	}

	// Exercise the game bridge, not just the backend's textured fast path.
	// Empty VRAM must not make a POLY_F/G primitive disappear or change colour.
	const int savedBackend = PsyX_GetRenderBackend();
	PsyX_SetRenderBackend(PSYX_BACKEND_VULKAN);
	PsyX_Vk_GameSetProjection2D(ortho);
	memset(vram, 0, sizeof(unsigned short) * PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT);
	PsyX_Vk_GameSetVram(vram);
	const float white = 248.0f / 255.0f;
	const float alpha = 127.0f / 255.0f;
	auto beginUiFrame = [&]() {
		PsyX_Vk_GameBeginFrame();
		PsyX_Vk_GameSetViewPort(0, 0, width, height);
		PsyX_Vk_GameSetScissor(0, 0, 0, width, height);
		PsyX_Vk_GameEnableDepth(0);
		PsyX_Vk_GameSetStencilMode(0);
		GR_SetTexture(g_whiteTexture, TF_16_BIT);
	};
	auto drawUiQuad = [&](int r, int g, int b, int blend, float quadWidth) {
		PsxFillQuad(quadWidth, (float)height, 0, 0, 0, 0, quad);
		for (int i = 0; i < 6; i++) {
			quad[i].r = (unsigned char)r;
			quad[i].g = (unsigned char)g;
			quad[i].b = (unsigned char)b;
		}
		PsyX_Vk_GameUpdateVertexBuffer(quad, 6);
		PsyX_Vk_GameSetBlendMode(blend);
		PsyX_Vk_GameDrawTriangles(0, 2);
	};
	auto checkUiFrame = [&](const char* label, int x, const float* expected) {
		// Public presentation readback deliberately reports opaque alpha.
		float presented[4] = { expected[0], expected[1], expected[2], 1.0f };
		if (!PsyX_Vk_RenderFrame() || !PsyX_Vk_ReadbackRgba(rgba, &readWidth, &readHeight)) {
			ReportAppend(report, reportSize, "UI readback FAIL\n");
			failures++;
		} else if (!PsxCheckPixel(rgba, readWidth, readHeight, x, height / 2,
			presented, tolerance, label, report, reportSize)) {
			failures++;
		}
	};
	for (int format = 0; format < 3; format++) {
		beginUiFrame();
		GR_SetTexture(g_whiteTexture, (TexFormat)format);
		drawUiQuad(255, 255, 255, 0, (float)width);
		const float expected[4] = { white, white, white, alpha };
		checkUiFrame("untextured white / empty VRAM", width / 2, expected);
	}

	// All PSX blend modes must blend in display space, like OpenGL. An sRGB
	// attachment gives different answers even if the shader inverse-encodes.
	for (int mode = 0; mode < PSYX_VK_PSX_BLEND_COUNT; mode++) {
		beginUiFrame();
		drawUiQuad(64, 64, 64, 0, (float)width);
		drawUiQuad(128, 0, 0, mode, (float)width);
		const float dst = white * 64.0f / 255.0f;
		const float src = white * 128.0f / 255.0f;
		float expected[4] = { src, 0, 0, alpha };
		if (mode == 1) {
			expected[0] = src * alpha + dst * (1 - alpha);
			expected[1] = expected[2] = dst * (1 - alpha);
		} else if (mode >= 2) {
			expected[0] = mode == 3 ? 0 : dst + src * (mode == 4 ? 0.5f : 1.0f);
			expected[1] = expected[2] = dst;
			expected[3] = alpha * (mode == 4 ? 1.5f : 2.0f);
		}
		char label[64];
		snprintf(label, sizeof(label), "untextured blend mode %d", mode);
		checkUiFrame(label, width / 2, expected);
	}

	// DrawPrim protection must survive multiple VRAM transfers/pass restarts.
	// Later non-DrawPrim draws must still render outside the protected region.
	beginUiFrame();
	PsyX_Vk_GameSetStencilMode(1);
	drawUiQuad(255, 255, 255, 0, width * 0.5f);
	PsyX_Vk_GameSetStencilMode(0);
	for (int split = 0; split < 2; split++) {
		PsyX_Vk_GameCopyVRAM(vram, PSYX_VK_VRAM_WIDTH, 0, 0, 1, 1, 0, 0);
		drawUiQuad(255, 0, 0, 0, (float)width);
	}
	const float protectedPixel[4] = { white, white, white, alpha };
	checkUiFrame("stencil survives two pass restarts", width / 4, protectedPixel);
	const float outsidePixel[4] = { white, 0, 0, 1.0f };
	if (!PsxCheckPixel(rgba, readWidth, readHeight, width * 3 / 4, height / 2,
		outsidePixel, tolerance, "stencil outside mask", report, reportSize))
		failures++;

	// Seed one frame, then update only its left half for several presentations.
	// This checks the acquired-image sequence, not guaranteed swapchain rotation.
	beginUiFrame();
	drawUiQuad(255, 255, 255, 0, (float)width);
	checkUiFrame("persistent framebuffer seed", width * 3 / 4, protectedPixel);
	for (uint32_t frame = 0; frame <= g_vk.swapchainImageCount; frame++) {
		beginUiFrame();
		drawUiQuad(0, 0, 0, 0, width * 0.5f);
		checkUiFrame("partial frame retains preceding image", width * 3 / 4, protectedPixel);
	}
	PsyX_SetRenderBackend(savedBackend);
	delete[] rgba;
	delete[] vram;

	// The VRAM TGA export (F10 / GR_SaveVRAM) must write the full pixel
	// payload on the Vulkan build. It used to be compiled out unless USE_OPENGL,
	// leaving a 18-byte header-only file.
	{
		const char* path = "vk_fixture_vram.tga";
		remove(path);
		GR_SaveVRAM(path, 0, 0, PSYX_VK_VRAM_WIDTH, PSYX_VK_VRAM_HEIGHT, 0);

		long size = 0;
		FILE* file = fopen(path, "rb");
		if (file)
		{
			fseek(file, 0, SEEK_END);
			size = ftell(file);
			fclose(file);
		}

		const long expected = 18 + (long)PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT * 2;
		char line[128];
		snprintf(line, sizeof(line), "vram export %ld/%ld bytes: %s\n", size, expected,
			size == expected ? "ok" : "FAIL");
		ReportAppend(report, reportSize, line);
		if (size != expected)
			failures++;
	}

	{
		char line[128];
		snprintf(line, sizeof(line), "psx self-test: %s\n", failures == 0 ? "PASS" : "FAIL");
		ReportAppend(report, reportSize, line);
	}

	// Leave no queued draws behind for the next frame.
	g_vk.psx.drawCount = 0;
	g_vk.psx.vertexSize = 0;

	return failures == 0;
}

// ---------------------------------------------------------------------------
// Textures

static int UploadTexture(const unsigned char* pixels, int width, int height)
{
	if (!pixels || width <= 0 || height <= 0)
		return -1;

	int slot = -1;
	for (int i = 0; i < PSYX_VK_MAX_TEXTURES; i++)
	{
		if (!g_textures[i].used)
		{
			slot = i;
			break;
		}
	}
	if (slot < 0)
		return -1;

	VkTexture* texture = &g_textures[slot];
	memset(texture, 0, sizeof(*texture));

	if (!CreateImage2D((uint32_t)width, (uint32_t)height, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &texture->image, &texture->memory))
		return -1;

	const VkDeviceSize size = (VkDeviceSize)width * height * 4;
	VkBuffer staging = VK_NULL_HANDLE;
	VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
	void* mapped = NULL;
	if (!CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&staging, &stagingMemory, &mapped))
		return -1;

	memcpy(mapped, pixels, (size_t)size);

	VkCommandBuffer cmd = BeginOneShot();
	ImageBarrier(cmd, texture->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);

	VkBufferImageCopy region;
	memset(&region, 0, sizeof(region));
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent.width = (uint32_t)width;
	region.imageExtent.height = (uint32_t)height;
	region.imageExtent.depth = 1;
	vkCmdCopyBufferToImage(cmd, staging, texture->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	ImageBarrier(cmd, texture->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

	EndOneShot(cmd);

	vkDestroyBuffer(g_vk.device, staging, NULL);
	vkFreeMemory(g_vk.device, stagingMemory, NULL);

	if (!CreateImageView2D(texture->image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &texture->view))
		return -1;

	texture->used = 1;
	return slot;
}

// ---------------------------------------------------------------------------
// Textures and mesh creation

static void UpdateDescriptorSetForMesh(VkMesh* mesh);

static int CreateDummyTextures(void)
{
	const unsigned char white[4] = { 255, 255, 255, 255 };
	const unsigned char black[4] = { 0, 0, 0, 255 };
	const unsigned char flatNormal[4] = { 128, 128, 255, 255 };
	const unsigned char neutralMr[4] = { 255, 255, 0, 255 };	// g = roughness 1, b = metallic 0

	const unsigned char* sources[4] = { white, flatNormal, neutralMr, black };
	for (int i = 0; i < 4; i++)
	{
		g_vk.dummyTextures[i] = UploadTexture(sources[i], 1, 1);
		if (g_vk.dummyTextures[i] < 0)
			return 0;
	}
	return 1;
}

int PsyX_Vk_CreateTexture(const unsigned char* rgba, int width, int height)
{
	if (!g_vk.initialised)
		return -1;
	return UploadTexture(rgba, width, height);
}

void PsyX_Vk_DestroyTexture(int texture)
{
	if (texture < 0 || texture >= PSYX_VK_MAX_TEXTURES || !g_textures[texture].used)
		return;

	vkDeviceWaitIdle(g_vk.device);

	// Drop mesh references first so no descriptor keeps a destroyed view.
	for (int i = 0; i < PSYX_VK_MAX_MESHES; i++)
	{
		if (!g_vk.meshes[i].used)
			continue;

		int changed = 0;
		for (int s = 0; s < 4; s++)
		{
			if (g_vk.meshes[i].textureSlots[s] == texture)
			{
				g_vk.meshes[i].textureSlots[s] = -1;
				changed = 1;
			}
		}
		if (changed)
			UpdateDescriptorSetForMesh(&g_vk.meshes[i]);
	}

	VkTexture* t = &g_textures[texture];
	if (t->view) vkDestroyImageView(g_vk.device, t->view, NULL);
	if (t->image) vkDestroyImage(g_vk.device, t->image, NULL);
	if (t->memory) vkFreeMemory(g_vk.device, t->memory, NULL);
	memset(t, 0, sizeof(*t));
}

void PsyX_Vk_SetMeshTexture(int mesh, int slot, int texture)
{
	if (mesh < 0 || mesh >= PSYX_VK_MAX_MESHES || !g_vk.meshes[mesh].used)
		return;
	if (slot < 0 || slot > 3)
		return;

	g_vk.meshes[mesh].textureSlots[slot] = texture;

	if (g_vk.meshes[mesh].descriptorSet)
		UpdateDescriptorSetForMesh(&g_vk.meshes[mesh]);
}

void PsyX_Vk_SetMeshFactors(int mesh, float metallic, float roughness, float emissiveScale)
{
	if (mesh < 0 || mesh >= PSYX_VK_MAX_MESHES || !g_vk.meshes[mesh].used)
		return;

	g_vk.meshes[mesh].factors[0] = metallic;
	g_vk.meshes[mesh].factors[1] = roughness;
	g_vk.meshes[mesh].factors[2] = emissiveScale;
}

// Normalises a shared modern-mesh description into the pipeline's fixed
// vertex layout. The base colour factor is baked into the vertex colour so the
// shaders only need the per-instance tint.
static void FillMeshVertices(const PsyXModernMeshDesc* desc, VkVertex* vertices)
{
	for (int i = 0; i < desc->vertexCount; i++)
	{
		vertices[i].pos[0] = desc->positions[i * 3 + 0];
		vertices[i].pos[1] = desc->positions[i * 3 + 1];
		vertices[i].pos[2] = desc->positions[i * 3 + 2];

		vertices[i].color[0] = vertices[i].color[1] = vertices[i].color[2] = vertices[i].color[3] = 1.0f;
		if (desc->colors)
		{
			for (int c = 0; c < 4; c++)
				vertices[i].color[c] = (float)desc->colors[i * 4 + c] / 255.0f;
		}

		vertices[i].normal[0] = 0.0f;
		vertices[i].normal[1] = 1.0f;
		vertices[i].normal[2] = 0.0f;
		if (desc->normals)
		{
			vertices[i].normal[0] = desc->normals[i * 3 + 0];
			vertices[i].normal[1] = desc->normals[i * 3 + 1];
			vertices[i].normal[2] = desc->normals[i * 3 + 2];
		}

		vertices[i].uv[0] = 0.0f;
		vertices[i].uv[1] = 0.0f;
		if (desc->uvs)
		{
			vertices[i].uv[0] = desc->uvs[i * 2 + 0];
			vertices[i].uv[1] = desc->uvs[i * 2 + 1];
		}

		if (desc->baseColorFactor)
		{
			for (int c = 0; c < 4; c++)
				vertices[i].color[c] *= desc->baseColorFactor[c];
		}
	}
}

int PsyX_Vk_CreateMesh(const PsyXModernMeshDesc* desc)
{
	if (!g_vk.initialised || !desc || !desc->positions || desc->vertexCount <= 0)
		return -1;
	if (g_vk.meshCount >= PSYX_VK_MAX_MESHES)
		return -1;

	int slot = -1;
	for (int i = 0; i < PSYX_VK_MAX_MESHES; i++)
	{
		if (!g_vk.meshes[i].used)
		{
			slot = i;
			break;
		}
	}
	if (slot < 0)
		return -1;

	VkMesh* mesh = &g_vk.meshes[slot];
	// The descriptor set is allocated once at initialisation; keep it across
	// the reset.
	const VkDescriptorSet descriptorSet = mesh->descriptorSet;
	memset(mesh, 0, sizeof(*mesh));
	mesh->descriptorSet = descriptorSet;

	VkVertex* vertices = new VkVertex[desc->vertexCount];
	FillMeshVertices(desc, vertices);

	const VkDeviceSize vertexBytes = (VkDeviceSize)desc->vertexCount * sizeof(VkVertex);
	void* vertexMapped = NULL;
	if (!CreateBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&mesh->vertexBuffer, &mesh->vertexMemory, &vertexMapped))
	{
		delete[] vertices;
		return -1;
	}
	memcpy(vertexMapped, vertices, (size_t)vertexBytes);
	delete[] vertices;
	mesh->vertexCount = desc->vertexCount;

	if (desc->indices && desc->indexCount > 0)
	{
		uint32_t* indices = new uint32_t[desc->indexCount];
		for (int i = 0; i < desc->indexCount; i++)
			indices[i] = desc->indices[i];

		const VkDeviceSize indexBytes = (VkDeviceSize)desc->indexCount * sizeof(uint32_t);
		void* indexMapped = NULL;
		if (!CreateBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&mesh->indexBuffer, &mesh->indexMemory, &indexMapped))
		{
			delete[] indices;
			return -1;
		}
		memcpy(indexMapped, indices, (size_t)indexBytes);
		delete[] indices;
		mesh->indexCount = desc->indexCount;
	}

	// Textures are Vulkan-owned and assigned through PsyX_Vk_SetMeshTexture;
	// the OpenGL handles in the shared descriptor are not portable.
	for (int i = 0; i < 4; i++)
		mesh->textureSlots[i] = -1;

	mesh->factors[0] = desc->metallicFactor;
	mesh->factors[1] = desc->roughnessFactor;
	mesh->factors[2] = desc->emissiveFactor ? desc->emissiveFactor[0] : 0.0f;
	if (mesh->factors[2] < 0.0f) mesh->factors[2] = 0.0f;

	mesh->world[0] = mesh->world[5] = mesh->world[10] = mesh->world[15] = 1.0f;
	mesh->color[0] = mesh->color[1] = mesh->color[2] = mesh->color[3] = 1.0f;
	mesh->visible = 1;

	mesh->used = 1;
	g_vk.meshCount++;
	g_vk.info.meshCount = g_vk.meshCount;

	UpdateDescriptorSetForMesh(mesh);
	return slot;
}

void PsyX_Vk_DestroyMesh(int mesh)
{
	if (mesh < 0 || mesh >= PSYX_VK_MAX_MESHES || !g_vk.meshes[mesh].used)
		return;

	VkMesh* m = &g_vk.meshes[mesh];
	vkDeviceWaitIdle(g_vk.device);

	if (m->vertexBuffer) vkDestroyBuffer(g_vk.device, m->vertexBuffer, NULL);
	if (m->vertexMemory) vkFreeMemory(g_vk.device, m->vertexMemory, NULL);
	if (m->indexBuffer) vkDestroyBuffer(g_vk.device, m->indexBuffer, NULL);
	if (m->indexMemory) vkFreeMemory(g_vk.device, m->indexMemory, NULL);

	// Textures live in the shared table; release them through
	// PsyX_Vk_DestroyTexture so any other mesh reference is cleared too.

	memset(m, 0, sizeof(*m));
	g_vk.meshCount--;
	g_vk.info.meshCount = g_vk.meshCount;
}

void PsyX_Vk_SetInstance(int mesh, const float worldMatrix[16], const float color[4], int visible)
{
	if (mesh < 0 || mesh >= PSYX_VK_MAX_MESHES || !g_vk.meshes[mesh].used)
		return;

	if (worldMatrix)
		memcpy(g_vk.meshes[mesh].world, worldMatrix, sizeof(g_vk.meshes[mesh].world));
	if (color)
		memcpy(g_vk.meshes[mesh].color, color, sizeof(g_vk.meshes[mesh].color));
	g_vk.meshes[mesh].visible = visible != 0;
}

void PsyX_Vk_SetCamera(const float view[16], const float proj[16], const float cameraPosition[3])
{
	if (!g_vk.uboMapped)
		return;

	if (view)
		memcpy(g_vk.uboMapped->view, view, sizeof(g_vk.uboMapped->view));
	if (proj)
		memcpy(g_vk.uboMapped->proj, proj, sizeof(g_vk.uboMapped->proj));
	if (cameraPosition)
	{
		g_vk.uboMapped->cameraPos[0] = cameraPosition[0];
		g_vk.uboMapped->cameraPos[1] = cameraPosition[1];
		g_vk.uboMapped->cameraPos[2] = cameraPosition[2];
		g_vk.uboMapped->cameraPos[3] = 1.0f;
	}
}

void PsyX_Vk_SetLights(const PsyXModernLightSet* lights)
{
	if (lights)
		g_vk.lights = *lights;
}

void PsyX_Vk_SetOverlayText(const char* text)
{
	if (!text)
	{
		g_vk.overlayText[0] = '\0';
		return;
	}
	strncpy(g_vk.overlayText, text, sizeof(g_vk.overlayText) - 1);
	g_vk.overlayText[sizeof(g_vk.overlayText) - 1] = '\0';
}

// ---------------------------------------------------------------------------
// Frame rendering

static void BuildShadowMatrix(float out[16]);

static void UpdateSceneUbo(void)
{
	VkSceneUbo* ubo = g_vk.uboMapped;
	if (!ubo)
		return;

	ubo->cameraPos[0] = ubo->cameraPos[0];
	ubo->cameraPos[3] = 1.0f;

	const float ambient = g_vk.lights.ambient[0];
	ubo->ambientExposure[0] = g_vk.lights.ambient[0];
	ubo->ambientExposure[1] = g_vk.lights.ambient[1];
	ubo->ambientExposure[2] = g_vk.lights.ambient[2];
	ubo->ambientExposure[3] = g_vk.lights.exposure > 0.0f ? g_vk.lights.exposure : 1.0f;
	(void)ambient;

	ubo->shadowParams[0] = g_vk.lights.shadowsEnabled ? 1.0f : 0.0f;
	ubo->shadowParams[1] = 1.0f / (float)kShadowSize;
	ubo->shadowParams[2] = 0.45f;
	ubo->shadowParams[3] = g_vk.lights.aoEnabled ? 1.0f : 0.0f;

	ubo->lightInfo[0] = (float)g_vk.lights.count;
	ubo->lightInfo[1] = g_vk.srgbOutput ? 1.0f : 0.0f;

	int lightCount = g_vk.lights.count;
	if (lightCount < 0) lightCount = 0;
	if (lightCount > PSYX_VK_MAX_LIGHTS) lightCount = PSYX_VK_MAX_LIGHTS;

	for (int i = 0; i < PSYX_VK_MAX_LIGHTS; i++)
	{
		VkLightStd140* out = &ubo->lights[i];
		memset(out, 0, sizeof(*out));

		if (i >= lightCount)
			continue;

		const PsyXModernLight* light = &g_vk.lights.lights[i];
		out->posRange[0] = light->position[0];
		out->posRange[1] = light->position[1];
		out->posRange[2] = light->position[2];
		out->posRange[3] = light->range > 0.0f ? light->range : 1000.0f;

		out->dirType[0] = light->direction[0];
		out->dirType[1] = light->direction[1];
		out->dirType[2] = light->direction[2];
		out->dirType[3] = (float)light->type;

		out->color[0] = light->color[0] * light->intensity;
		out->color[1] = light->color[1] * light->intensity;
		out->color[2] = light->color[2] * light->intensity;
	}

	BuildShadowMatrix(ubo->shadowMatrix);
}

// Orthographic shadow volume around the light set's shadow centre, with
// Vulkan's [0,1] depth convention. Shared by the fixture UBO and the in-game
// modern UBO so both shadow paths match the OpenGL construction.
static void BuildShadowMatrix(float out[16])
{
	int lightCount = g_vk.lights.count;
	if (lightCount < 0) lightCount = 0;
	if (lightCount > PSYX_VK_MAX_LIGHTS) lightCount = PSYX_VK_MAX_LIGHTS;

	if (lightCount == 0)
	{
		memset(out, 0, 16 * sizeof(float));
		return;
	}

	const float extent = g_vk.lights.shadowExtent > 0.0f ? g_vk.lights.shadowExtent : 4000.0f;

	const PsyXModernLight* sun = NULL;
	for (int i = 0; i < lightCount; i++)
	{
		if (g_vk.lights.lights[i].type == 0)
		{
			sun = &g_vk.lights.lights[i];
			break;
		}
	}

	float sunDir[3] = { 0.35f, 0.8f, 0.45f };
	if (sun)
	{
		sunDir[0] = sun->direction[0];
		sunDir[1] = sun->direction[1];
		sunDir[2] = sun->direction[2];
	}
	float length = sqrtf(sunDir[0] * sunDir[0] + sunDir[1] * sunDir[1] + sunDir[2] * sunDir[2]);
	length = length > 1e-5f ? length : 1.0f;
	sunDir[0] /= length; sunDir[1] /= length; sunDir[2] /= length;

	const float* c = g_vk.lights.shadowCenter;
	const float eye[3] = { c[0] + sunDir[0] * extent * 2.0f, c[1] + sunDir[1] * extent * 2.0f, c[2] + sunDir[2] * extent * 2.0f };

	// Reuse the OpenGL path's look-at/ortho construction (column-major).
	float f[3] = { c[0] - eye[0], c[1] - eye[1], c[2] - eye[2] };
	float fl = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
	fl = fl > 1e-5f ? fl : 1.0f;
	f[0] /= fl; f[1] /= fl; f[2] /= fl;

	const int vertical = (sunDir[1] > 0.95f || sunDir[1] < -0.95f);
	const float up[3] = { 0.0f, vertical ? 0.0f : 1.0f, vertical ? 1.0f : 0.0f };

	float s[3] = { f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0] };
	float sl = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
	sl = sl > 1e-5f ? sl : 1.0f;
	s[0] /= sl; s[1] /= sl; s[2] /= sl;

	const float u[3] = { s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0] };

	float lightView[16];
	lightView[0] = s[0]; lightView[4] = s[1]; lightView[8] = s[2]; lightView[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
	lightView[1] = u[0]; lightView[5] = u[1]; lightView[9] = u[2]; lightView[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
	lightView[2] = -f[0]; lightView[6] = -f[1]; lightView[10] = -f[2]; lightView[14] = (f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]);
	lightView[3] = 0.0f; lightView[7] = 0.0f; lightView[11] = 0.0f; lightView[15] = 1.0f;

	// Orthographic projection with Vulkan's [0,1] depth convention.
	float lightProj[16];
	const float nearZ = 0.05f;
	const float farZ = extent * 4.0f;
	memset(lightProj, 0, sizeof(lightProj));
	lightProj[0] = 2.0f / (2.0f * extent);
	lightProj[5] = 2.0f / (2.0f * extent);
	lightProj[10] = 1.0f / (nearZ - farZ);
	lightProj[14] = nearZ / (nearZ - farZ);
	lightProj[15] = 1.0f;

	MulMatrix4(lightProj, lightView, out);
}

static void UpdateDescriptorSetForMesh(VkMesh* mesh)
{
	VkDescriptorBufferInfo bufferInfo;
	memset(&bufferInfo, 0, sizeof(bufferInfo));
	bufferInfo.buffer = g_vk.uboBuffer;
	bufferInfo.offset = 0;
	bufferInfo.range = sizeof(VkSceneUbo);

	VkDescriptorImageInfo shadowInfo;
	memset(&shadowInfo, 0, sizeof(shadowInfo));
	shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	shadowInfo.imageView = g_vk.shadowView;
	shadowInfo.sampler = g_vk.shadowSampler;

	VkImageView defaultViews[4];
	for (int i = 0; i < 4; i++)
	{
		const int dummy = g_vk.dummyTextures[i];
		defaultViews[i] = (dummy >= 0 && g_textures[dummy].used) ? g_textures[dummy].view : VK_NULL_HANDLE;
	}

	VkDescriptorImageInfo materialInfos[4];
	memset(materialInfos, 0, sizeof(materialInfos));
	for (int i = 0; i < 4; i++)
	{
		VkImageView view = defaultViews[i];
		const int slot = mesh->textureSlots[i];
		if (slot >= 0 && slot < PSYX_VK_MAX_TEXTURES && g_textures[slot].used)
			view = g_textures[slot].view;

		materialInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		materialInfos[i].imageView = view;
		materialInfos[i].sampler = g_vk.textureSampler;
	}

	VkWriteDescriptorSet writes[6];
	memset(writes, 0, sizeof(writes));

	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = mesh->descriptorSet;
	writes[0].dstBinding = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[0].pBufferInfo = &bufferInfo;

	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = mesh->descriptorSet;
	writes[1].dstBinding = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[1].pImageInfo = &shadowInfo;

	for (int i = 0; i < 4; i++)
	{
		writes[2 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[2 + i].dstSet = mesh->descriptorSet;
		writes[2 + i].dstBinding = (uint32_t)(2 + i);
		writes[2 + i].descriptorCount = 1;
		writes[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[2 + i].pImageInfo = &materialInfos[i];
	}

	vkUpdateDescriptorSets(g_vk.device, 6, writes, 0, NULL);
}

int PsyX_Vk_RenderFrame(void)
{
	if (!g_vk.initialised)
		return 0;

	if (g_vk.gameMode)
	{
		// The game owns the SDL event pump (keyboard, pad, resize, quit); the
		// backend only has to notice a new drawable size. Consuming events here
		// would starve the game's input handling.
		int w = 0, h = 0;
		SDL_Vulkan_GetDrawableSize(g_vk.window, &w, &h);
		if (w > 0 && h > 0 && (w != g_vk.windowWidth || h != g_vk.windowHeight))
		{
			g_vk.windowWidth = w;
			g_vk.windowHeight = h;
			g_vk.resizePending = 1;
		}
	}
	else
	{
		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT)
				return 0;
			if (event.type == SDL_WINDOWEVENT)
			{
				if (event.window.event == SDL_WINDOWEVENT_CLOSE)
					return 0;
				if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
					event.window.event == SDL_WINDOWEVENT_RESIZED)
				{
					int w = 0, h = 0;
					SDL_Vulkan_GetDrawableSize(g_vk.window, &w, &h);
					if (w > 0 && h > 0)
					{
						g_vk.windowWidth = w;
						g_vk.windowHeight = h;
						g_vk.resizePending = 1;
					}
				}
			}

			if (g_vk.imguiActive)
				ImGui_ImplSDL2_ProcessEvent(&event);
		}

		// Loader-independent quit path.
		const Uint8* keys = SDL_GetKeyboardState(NULL);
		if (keys && keys[SDL_SCANCODE_ESCAPE])
			return 0;
	}

	if (g_vk.resizePending)
	{
		g_vk.resizePending = 0;
		if (!RecreateSwapchain())
			return 1;	// try again next frame
	}

	if (!VkOk(vkWaitForFences(g_vk.device, 1, &g_vk.frameFence, VK_TRUE, UINT64_MAX), "vkWaitForFences"))
		return 0;
	vkResetFences(g_vk.device, 1, &g_vk.frameFence);

	uint32_t imageIndex = 0;
	VkResult acquire = vkAcquireNextImageKHR(g_vk.device, g_vk.swapchain, UINT64_MAX,
		g_vk.imageAvailable, VK_NULL_HANDLE, &imageIndex);
	if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
	{
		RecreateSwapchain();
		return 1;
	}
	if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
	{
		eprinterr("PsyX Vulkan: acquire failed (%s)\n", VkResultName(acquire));
		return 0;
	}
	if (imageIndex >= g_vk.swapchainImageCount)
		return 1;

	UpdateSceneUbo();
	if (g_vk.gameMode)
		UpdateGameModernUbo();

	// Command buffer.
	vkResetCommandBuffer(g_vk.commandBuffer, 0);
	VkCommandBufferBeginInfo begin;
	memset(&begin, 0, sizeof(begin));
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(g_vk.commandBuffer, &begin);

	// Shadow pass.
	ImageBarrier(g_vk.commandBuffer, g_vk.shadowImage, VK_IMAGE_ASPECT_DEPTH_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

	VkClearValue shadowClear;
	memset(&shadowClear, 0, sizeof(shadowClear));
	shadowClear.depthStencil.depth = 1.0f;

	VkRenderPassBeginInfo shadowBegin;
	memset(&shadowBegin, 0, sizeof(shadowBegin));
	shadowBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	shadowBegin.renderPass = g_vk.shadowRenderPass;
	shadowBegin.framebuffer = g_vk.shadowFramebuffer;
	shadowBegin.renderArea.extent.width = (uint32_t)kShadowSize;
	shadowBegin.renderArea.extent.height = (uint32_t)kShadowSize;
	shadowBegin.clearValueCount = 1;
	shadowBegin.pClearValues = &shadowClear;

	vkCmdBeginRenderPass(g_vk.commandBuffer, &shadowBegin, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport shadowViewport;
	memset(&shadowViewport, 0, sizeof(shadowViewport));
	shadowViewport.width = (float)kShadowSize;
	shadowViewport.height = (float)kShadowSize;
	shadowViewport.maxDepth = 1.0f;
	vkCmdSetViewport(g_vk.commandBuffer, 0, 1, &shadowViewport);

	VkRect2D shadowScissor;
	memset(&shadowScissor, 0, sizeof(shadowScissor));
	shadowScissor.extent.width = (uint32_t)kShadowSize;
	shadowScissor.extent.height = (uint32_t)kShadowSize;
	vkCmdSetScissor(g_vk.commandBuffer, 0, 1, &shadowScissor);

	int drawCalls = 0;
	if (g_vk.lights.shadowsEnabled)
	{
		vkCmdBindPipeline(g_vk.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.shadowPipeline);

		for (int i = 0; i < PSYX_VK_MAX_MESHES; i++)
		{
			// Fixture meshes only; in game mode this table is empty and the
			// in-game modern meshes are recorded below instead.
			VkMesh* mesh = &g_vk.meshes[i];
			if (!mesh->used || !mesh->visible || mesh->vertexCount == 0)
				continue;

			float lightWorld[16];
			MulMatrix4(g_vk.uboMapped->shadowMatrix, mesh->world, lightWorld);

			vkCmdPushConstants(g_vk.commandBuffer, g_vk.shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, lightWorld);

			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(g_vk.commandBuffer, 0, 1, &mesh->vertexBuffer, &offset);
			if (mesh->indexCount > 0)
			{
				vkCmdBindIndexBuffer(g_vk.commandBuffer, mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdDrawIndexed(g_vk.commandBuffer, (uint32_t)mesh->indexCount, 1, 0, 0, 0);
			}
			else
			{
				vkCmdDraw(g_vk.commandBuffer, (uint32_t)mesh->vertexCount, 1, 0, 0);
			}
			drawCalls++;
		}

		if (g_vk.gameMode)
			RecordGameModernShadowPass(g_vk.commandBuffer);
	}

	// The render pass moves the shadow map to SHADER_READ_ONLY_OPTIMAL.
	vkCmdEndRenderPass(g_vk.commandBuffer);
	if (g_vk.frameIndex == 0)
	{
		char line[128];
		snprintf(line, sizeof(line), "frame: shadow pass recorded draws=%d", drawCalls);
		VkStage(line);
	}

	// Main pass. The colour attachment preserves the previous frame; the clear
	// colour only matters when the game asked for a clear (GR_Clear, i.e. the
	// draw environment had isbg set) or when this swapchain image has never
	// been drawn, because then its contents are undefined. A fresh image is
	// first moved to the PRESENT_SRC layout the pass declares.
	const int firstUse = !g_vk.swapchainImageDrawn[imageIndex];
	if (firstUse)
	{
		ImageBarrier(g_vk.commandBuffer, g_vk.swapchainImages[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
	}

	VkClearValue clears[2];
	memset(clears, 0, sizeof(clears));
	int clearColor = 0;
	if (g_vk.gameMode)
	{
		clears[0].color.float32[0] = g_vk.psx.clearColor[0];
		clears[0].color.float32[1] = g_vk.psx.clearColor[1];
		clears[0].color.float32[2] = g_vk.psx.clearColor[2];
		clears[0].color.float32[3] = 1.0f;
		clearColor = g_vk.psx.clearRequested || firstUse;
		g_vk.psx.clearRequested = 0;
	}
	else
	{
		// The fixture window has no draw environment and always clears.
		clears[0].color.float32[0] = 0.42f;
		clears[0].color.float32[1] = 0.58f;
		clears[0].color.float32[2] = 0.78f;
		clears[0].color.float32[3] = 1.0f;
		clearColor = 1;
	}
	clears[1].depthStencil.depth = 1.0f;
	clears[1].depthStencil.stencil = 0;	// PSX mask bit starts clear every frame

	VkRenderPassBeginInfo mainBegin;
	memset(&mainBegin, 0, sizeof(mainBegin));
	mainBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	mainBegin.renderPass = g_vk.mainRenderPass;
	mainBegin.framebuffer = g_vk.framebuffers[imageIndex];
	mainBegin.renderArea.extent.width = (uint32_t)g_vk.width;
	mainBegin.renderArea.extent.height = (uint32_t)g_vk.height;
	mainBegin.clearValueCount = 2;
	mainBegin.pClearValues = clears;

	vkCmdBeginRenderPass(g_vk.commandBuffer, &mainBegin, VK_SUBPASS_CONTENTS_INLINE);

	if (clearColor)
	{
		// The pass loads, so the colour attachment is cleared explicitly. The
		// depth/stencil attachment still uses the render-pass clear above.
		VkClearAttachment colorClear;
		memset(&colorClear, 0, sizeof(colorClear));
		colorClear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		colorClear.colorAttachment = 0;
		colorClear.clearValue = clears[0];

		VkClearRect colorRect;
		memset(&colorRect, 0, sizeof(colorRect));
		colorRect.rect = mainBegin.renderArea;
		colorRect.layerCount = 1;

		vkCmdClearAttachments(g_vk.commandBuffer, 1, &colorClear, 1, &colorRect);
	}

	g_vk.swapchainImageDrawn[imageIndex] = 1;

	VkViewport viewport;
	memset(&viewport, 0, sizeof(viewport));
	viewport.width = (float)g_vk.width;
	viewport.height = (float)g_vk.height;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(g_vk.commandBuffer, 0, 1, &viewport);

	VkRect2D scissor;
	memset(&scissor, 0, sizeof(scissor));
	scissor.extent.width = (uint32_t)g_vk.width;
	scissor.extent.height = (uint32_t)g_vk.height;
	vkCmdSetScissor(g_vk.commandBuffer, 0, 1, &scissor);

	// The emulated PSX game image goes down first; the modern meshes and the
	// overlay draw on top of it. Offscreen (render-to-VRAM) draws are resolved
	// into the VRAM mirror before the frame, so they are skipped here.
	g_vk.mainPassBegin = mainBegin;
	g_vk.mainPassImageIndex = imageIndex;
	g_vk.mainPassOpen = 1;
	RecordPsxDraws(g_vk.commandBuffer, 0, g_vk.psx.frameIndex, g_vk.width, g_vk.height);
	drawCalls += g_vk.psx.drawCount;

	if (g_vk.gameMode)
	{
		// In-game modern path. The legacy scene is complete in the main pass;
		// end it, copy the scene depth when the shadow composite is active, and
		// continue in the load pass so modern meshes share the legacy depth
		// buffer. The developer overlay is contributed below, in this pass.
		if (g_vk.mainPassOpen)
		{
			vkCmdEndRenderPass(g_vk.commandBuffer);
			g_vk.mainPassOpen = 0;
		}

		const int modernShadows = g_vk.gameModernEnabled && g_vk.lights.shadowsEnabled &&
			g_vk.modernCameraValid && g_vk.sceneDepthImage != VK_NULL_HANDLE;
		if (modernShadows)
			RecordGameModernSceneDepthCopy(g_vk.commandBuffer);

		VkRenderPassBeginInfo modernBegin = mainBegin;
		modernBegin.renderPass = g_vk.modernRenderPass;
		modernBegin.framebuffer = g_vk.modernFramebuffers[imageIndex];
		modernBegin.clearValueCount = 0;
		modernBegin.pClearValues = NULL;
		vkCmdBeginRenderPass(g_vk.commandBuffer, &modernBegin, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(g_vk.commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(g_vk.commandBuffer, 0, 1, &scissor);

		if (modernShadows)
		{
			UpdateGameCompositeDescriptorSet();
			vkCmdBindPipeline(g_vk.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.gameCompositePipeline);
			vkCmdBindDescriptorSets(g_vk.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				g_vk.pipelineLayout, 0, 1, &g_vk.gameCompositeSet, 0, NULL);
			vkCmdDraw(g_vk.commandBuffer, 3, 1, 0, 0);
			g_vk.gameModernStats.legacyShadowPass = 1;
		}

		drawCalls += RecordGameModernMeshes(g_vk.commandBuffer);
	}
	else
	{
		// A replayed VRAM write may have closed the pass; the gallery draws in
		// the same loaded pass.
		ResumeMainPass(g_vk.commandBuffer);

		vkCmdBindPipeline(g_vk.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.pbrPipeline);

		for (int i = 0; i < PSYX_VK_MAX_MESHES; i++)
		{
			VkMesh* mesh = &g_vk.meshes[i];
			if (!mesh->used || !mesh->visible || mesh->vertexCount == 0 || !mesh->descriptorSet)
				continue;

			float push[24];
			memcpy(push, mesh->world, sizeof(mesh->world));
			memcpy(push + 16, mesh->color, sizeof(mesh->color));
			push[20] = mesh->factors[0];
			push[21] = mesh->factors[1];
			push[22] = mesh->factors[2];
			push[23] = 0.0f;

			vkCmdBindDescriptorSets(g_vk.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				g_vk.pipelineLayout, 0, 1, &mesh->descriptorSet, 0, NULL);
			vkCmdPushConstants(g_vk.commandBuffer, g_vk.pipelineLayout,
				VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 96, push);

			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(g_vk.commandBuffer, 0, 1, &mesh->vertexBuffer, &offset);
			if (mesh->indexCount > 0)
			{
				vkCmdBindIndexBuffer(g_vk.commandBuffer, mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdDrawIndexed(g_vk.commandBuffer, (uint32_t)mesh->indexCount, 1, 0, 0, 0);
			}
			else
			{
				vkCmdDraw(g_vk.commandBuffer, (uint32_t)mesh->vertexCount, 1, 0, 0);
			}
		}
	}

	if (g_vk.frameIndex == 0)
	{
		char line[128];
		snprintf(line, sizeof(line), "frame: main pass recorded draws=%d stencil=%d",
			g_vk.psx.lastDraws, g_vk.psx.lastStencilDraws);
		VkStage(line);
	}

	// Overlay.
	if (g_vk.imguiActive)
	{
		if (g_vk.frameIndex == 0)
			VkStage("imgui: newframe");
		ImGui_ImplSDL2_NewFrame();
		// The renderer hook builds the font atlas on the first frame.
		ImGui_ImplVulkan_NewFrame();
		ImGui::NewFrame();
		if (g_vk.gameMode)
		{
			// The game contributes its own windows (the developer graphics
			// panel) into the frame the backend owns.
			PsyX_InvokeRenderOverlayHandler();
		}
		else
		{
			ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_Always);
			ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_Always);
			ImGui::Begin("Vulkan fixture", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::TextUnformatted(g_vk.overlayText);
			ImGui::Separator();
			ImGui::Text("API %u.%u.%u  %s", VK_VERSION_MAJOR(g_vk.apiVersion), VK_VERSION_MINOR(g_vk.apiVersion), VK_VERSION_PATCH(g_vk.apiVersion), g_vk.info.deviceName);
			ImGui::Text("%.1f FPS  %d meshes  %d draws", g_vk.lastFps, g_vk.meshCount, drawCalls);
			ImGui::End();
		}
		if (g_vk.frameIndex == 0)
			VkStage("imgui: widgets");
		ImGui::Render();
		if (g_vk.frameIndex == 0)
			VkStage("imgui: render");
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), g_vk.commandBuffer, VK_NULL_HANDLE);
		if (g_vk.frameIndex == 0)
			VkStage("imgui: renderdrawdata");
	}

	vkCmdEndRenderPass(g_vk.commandBuffer);

	// Readback copy.
	ImageBarrier(g_vk.commandBuffer, g_vk.swapchainImages[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	VkBufferImageCopy copy;
	memset(&copy, 0, sizeof(copy));
	copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent.width = (uint32_t)g_vk.width;
	copy.imageExtent.height = (uint32_t)g_vk.height;
	copy.imageExtent.depth = 1;
	vkCmdCopyImageToBuffer(g_vk.commandBuffer, g_vk.swapchainImages[imageIndex],
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_vk.readbackBuffer, 1, &copy);

	ImageBarrier(g_vk.commandBuffer, g_vk.swapchainImages[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		VK_ACCESS_TRANSFER_READ_BIT, 0);

	if (g_vk.frameIndex == 0)
		VkStage("frame: imgui recorded");

	vkEndCommandBuffer(g_vk.commandBuffer);

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo submit;
	memset(&submit, 0, sizeof(submit));
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &g_vk.imageAvailable;
	submit.pWaitDstStageMask = &waitStage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &g_vk.commandBuffer;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &g_vk.renderFinished[imageIndex];

	if (!VkOk(vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.frameFence), "vkQueueSubmit"))
		return 0;

	VkPresentInfoKHR present;
	memset(&present, 0, sizeof(present));
	present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present.waitSemaphoreCount = 1;
	present.pWaitSemaphores = &g_vk.renderFinished[imageIndex];
	present.swapchainCount = 1;
	present.pSwapchains = &g_vk.swapchain;
	present.pImageIndices = &imageIndex;

	VkResult presentResult = vkQueuePresentKHR(g_vk.queue, &present);
	if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
	{
		g_vk.resizePending = 1;
	}
	else if (presentResult != VK_SUCCESS)
	{
		eprinterr("PsyX Vulkan: present failed (%s)\n", VkResultName(presentResult));
		return 0;
	}

	if (g_vk.frameIndex == 0)
		VkStage("frame: presented");

	g_vk.lastDrawCalls = drawCalls;
	g_vk.frameIndex++;
	g_vk.info.drawCalls = drawCalls;

	// Simple FPS counter.
	{
		static Uint32 lastTicks = 0;
		Uint32 now = SDL_GetTicks();
		if (lastTicks != 0)
		{
			const float delta = (float)(now - lastTicks) * 0.001f;
			if (delta > 0.0f)
				g_vk.lastFps = g_vk.lastFps * 0.9 + (1.0 / delta) * 0.1;
		}
		lastTicks = now;

		// Publish the smoothed rate so the game can log a real measurement
		// without the ImGui overlay.
		g_vk.info.fps = g_vk.lastFps;
		g_vk.info.frameTimeMs = (g_vk.lastFps > 0.0) ? (1000.0 / g_vk.lastFps) : 0.0;
	}

	// Periodic performance sample. The counter has warmed up after a couple of
	// seconds, so the first entry lands once the rate is meaningful. `drawCalls`
	// is the recorded PSX draw count for this frame; `vertexCount` comes from
	// the emulated-GPU stats the panel also reads.
	if (g_vk.lastFps > 0.0 && (g_vk.frameIndex % 120) == 0)
	{
		PsyXRenderStats psxStats;
		PsyX_GetRenderStats(&psxStats);
		char line[192];
		snprintf(line, sizeof(line),
			"perf: frame=%llu fps=%.1f frame_ms=%.2f draws=%d vertices=%d",
			(unsigned long long)g_vk.frameIndex, g_vk.lastFps, g_vk.info.frameTimeMs,
			drawCalls, psxStats.vertexCount);
		VkStage(line);
	}

	return 1;
}

int PsyX_Vk_ReadbackRgba(unsigned char* rgba, int* width, int* height)
{
	if (!g_vk.initialised || !rgba || !g_vk.readbackMapped)
		return 0;

	vkDeviceWaitIdle(g_vk.device);

	const int w = g_vk.width;
	const int h = g_vk.height;
	for (int y = 0; y < h; y++)
	{
		// Vulkan stores row 0 at the top of the presented image; keep that
		// order so consumers (SDL surfaces, BMP writers) get a natural
		// top-down RGBA image.
		const unsigned char* src = g_vk.readbackMapped + (size_t)y * w * 4;
		unsigned char* dst = rgba + (size_t)y * w * 4;

		if (g_vk.swapchainFormat == VK_FORMAT_B8G8R8A8_SRGB || g_vk.swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM)
		{
			for (int x = 0; x < w; x++)
			{
				dst[x * 4 + 0] = src[x * 4 + 2];
				dst[x * 4 + 1] = src[x * 4 + 1];
				dst[x * 4 + 2] = src[x * 4 + 0];
				dst[x * 4 + 3] = 255;
			}
		}
		else
		{
			for (int x = 0; x < w; x++)
			{
				dst[x * 4 + 0] = src[x * 4 + 0];
				dst[x * 4 + 1] = src[x * 4 + 1];
				dst[x * 4 + 2] = src[x * 4 + 2];
				dst[x * 4 + 3] = 255;
			}
		}
	}

	if (width) *width = w;
	if (height) *height = h;
	return 1;
}

void PsyX_Vk_GetInfo(PsyXVkInfo* info)
{
	if (info)
		*info = g_vk.info;
}

// ---------------------------------------------------------------------------
// Lifecycle

static int CreateInstance()
{
	LoadGlobalFunctions();
	if (!vkCreateInstance)
	{
		eprinterr("PsyX Vulkan: vkCreateInstance not resolvable\n");
		return 0;
	}

	uint32_t loaderVersion = VK_API_VERSION_1_0;
	if (vkEnumerateInstanceVersion)
		vkEnumerateInstanceVersion(&loaderVersion);

	// Request the newest API the loader supports, clamped to the headers.
	uint32_t requested = VK_API_VERSION_1_4;
	if (loaderVersion < requested)
		requested = loaderVersion;
	if (requested < VK_API_VERSION_1_1)
		requested = VK_API_VERSION_1_1;
	g_vk.apiVersion = requested;

	unsigned int extensionCount = 0;
	if (!SDL_Vulkan_GetInstanceExtensions(g_vk.window, &extensionCount, NULL))
	{
		eprinterr("PsyX Vulkan: SDL_Vulkan_GetInstanceExtensions failed (%s)\n", SDL_GetError());
		return 0;
	}

	const char* extensions[8];
	if (extensionCount > 8)
		extensionCount = 8;
	SDL_Vulkan_GetInstanceExtensions(g_vk.window, &extensionCount, extensions);

	const char* layerNames[1] = { "VK_LAYER_KHRONOS_validation" };
	uint32_t layerCount = 0;
	vkEnumerateInstanceLayerProperties(&layerCount, NULL);
	VkLayerProperties availableLayers[64];
	uint32_t useLayers = 0;
	if (layerCount > 0 && layerCount <= 64)
	{
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);
		for (uint32_t i = 0; i < layerCount; i++)
		{
			if (strcmp(availableLayers[i].layerName, layerNames[0]) == 0)
			{
				useLayers = 1;
				break;
			}
		}
	}

	VkApplicationInfo application;
	memset(&application, 0, sizeof(application));
	application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	application.pApplicationName = "REDRIVER2-Plus Vulkan fixture";
	application.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	application.pEngineName = "PsyCross";
	application.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	application.apiVersion = requested;

	VkInstanceCreateInfo createInfo;
	memset(&createInfo, 0, sizeof(createInfo));
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &application;
	createInfo.enabledExtensionCount = extensionCount;
	createInfo.ppEnabledExtensionNames = extensions;
	createInfo.enabledLayerCount = useLayers;
	createInfo.ppEnabledLayerNames = useLayers ? layerNames : NULL;

	if (!VkOk(vkCreateInstance(&createInfo, NULL, &g_vk.instance), "vkCreateInstance"))
		return 0;

	LoadInstanceFunctions();

	if (useLayers)
		eprintinfo("PsyX Vulkan: validation layer enabled\n");

	return 1;
}

static int PickPhysicalDevice()
{
	uint32_t count = 0;
	if (!VkOk(vkEnumeratePhysicalDevices(g_vk.instance, &count, NULL), "vkEnumeratePhysicalDevices") || count == 0)
	{
		eprinterr("PsyX Vulkan: no physical devices\n");
		return 0;
	}

	VkPhysicalDevice devices[16];
	if (count > 16)
		count = 16;
	vkEnumeratePhysicalDevices(g_vk.instance, &count, devices);

	int best = -1;
	int bestScore = -1;

	for (uint32_t i = 0; i < count; i++)
	{
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(devices[i], &properties);

		uint32_t familyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &familyCount, NULL);
		if (familyCount == 0)
			continue;

		VkQueueFamilyProperties families[32];
		if (familyCount > 32)
			familyCount = 32;
		vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &familyCount, families);

		int graphicsFamily = -1;
		for (uint32_t f = 0; f < familyCount; f++)
		{
			VkBool32 present = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR(devices[i], f, g_vk.surface, &present);
			if ((families[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present)
			{
				graphicsFamily = (int)f;
				break;
			}
		}
		if (graphicsFamily < 0)
			continue;

		int score = (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 100 : 10;
		score += (int)(properties.apiVersion >> 22);

		if (score > bestScore)
		{
			bestScore = score;
			best = (int)i;
			g_vk.queueFamily = (uint32_t)graphicsFamily;
		}
	}

	if (best < 0)
	{
		eprinterr("PsyX Vulkan: no device with graphics+present queue\n");
		return 0;
	}

	g_vk.physicalDevice = devices[best];
	vkGetPhysicalDeviceMemoryProperties(g_vk.physicalDevice, &g_vk.memoryProperties);

	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(g_vk.physicalDevice, &properties);
	strncpy(g_vk.info.deviceName, properties.deviceName, sizeof(g_vk.info.deviceName) - 1);

	eprintinfo("PsyX Vulkan: device '%s' API %u.%u.%u\n", properties.deviceName,
		VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion), VK_VERSION_PATCH(properties.apiVersion));

	return 1;
}

static int CreateDevice()
{
	const char* extensions[1] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	float priority = 1.0f;
	VkDeviceQueueCreateInfo queueInfo;
	memset(&queueInfo, 0, sizeof(queueInfo));
	queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueInfo.queueFamilyIndex = g_vk.queueFamily;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = &priority;

	VkDeviceCreateInfo createInfo;
	memset(&createInfo, 0, sizeof(createInfo));
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.queueCreateInfoCount = 1;
	createInfo.pQueueCreateInfos = &queueInfo;
	createInfo.enabledExtensionCount = 1;
	createInfo.ppEnabledExtensionNames = extensions;

	if (!VkOk(vkCreateDevice(g_vk.physicalDevice, &createInfo, NULL, &g_vk.device), "vkCreateDevice"))
		return 0;

	vkGetDeviceQueue(g_vk.device, g_vk.queueFamily, 0, &g_vk.queue);
	return 1;
}

static int CreateShadowResources(void)
{
	if (!CreateImage2D((uint32_t)kShadowSize, (uint32_t)kShadowSize, VK_FORMAT_D32_SFLOAT,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		&g_vk.shadowImage, &g_vk.shadowMemory))
		return 0;

	if (!CreateImageView2D(g_vk.shadowImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, &g_vk.shadowView))
		return 0;

	VkSamplerCreateInfo sampler;
	memset(&sampler, 0, sizeof(sampler));
	sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler.magFilter = VK_FILTER_NEAREST;
	sampler.minFilter = VK_FILTER_NEAREST;
	sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler.maxLod = 1.0f;

	if (!VkOk(vkCreateSampler(g_vk.device, &sampler, NULL, &g_vk.shadowSampler), "vkCreateSampler(shadow)"))
		return 0;

	VkFramebufferCreateInfo framebuffer;
	memset(&framebuffer, 0, sizeof(framebuffer));
	framebuffer.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebuffer.renderPass = g_vk.shadowRenderPass;
	framebuffer.attachmentCount = 1;
	framebuffer.pAttachments = &g_vk.shadowView;
	framebuffer.width = (uint32_t)kShadowSize;
	framebuffer.height = (uint32_t)kShadowSize;
	framebuffer.layers = 1;

	if (!VkOk(vkCreateFramebuffer(g_vk.device, &framebuffer, NULL, &g_vk.shadowFramebuffer), "vkCreateFramebuffer(shadow)"))
		return 0;

	// Texture sampler for materials.
	memset(&sampler, 0, sizeof(sampler));
	sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler.magFilter = VK_FILTER_LINEAR;
	sampler.minFilter = VK_FILTER_LINEAR;
	sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.maxLod = 1.0f;

	if (!VkOk(vkCreateSampler(g_vk.device, &sampler, NULL, &g_vk.textureSampler), "vkCreateSampler(material)"))
		return 0;

	// Scene-depth copy sampler for the modern shadow composite.
	memset(&sampler, 0, sizeof(sampler));
	sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler.magFilter = VK_FILTER_NEAREST;
	sampler.minFilter = VK_FILTER_NEAREST;
	sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler.maxLod = 1.0f;

	if (!VkOk(vkCreateSampler(g_vk.device, &sampler, NULL, &g_vk.sceneDepthSampler), "vkCreateSampler(scene depth)"))
		return 0;

	g_vk.info.shadowMapSize = kShadowSize;
	return 1;
}

static int CreateSceneResources(void)
{
	if (!CreateBuffer(sizeof(VkSceneUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&g_vk.uboBuffer, &g_vk.uboMemory, (void**)&g_vk.uboMapped))
		return 0;

	memset(g_vk.uboMapped, 0, sizeof(VkSceneUbo));
	// Identity defaults.
	g_vk.uboMapped->view[0] = g_vk.uboMapped->view[5] = g_vk.uboMapped->view[10] = g_vk.uboMapped->view[15] = 1.0f;
	g_vk.uboMapped->proj[0] = g_vk.uboMapped->proj[5] = g_vk.uboMapped->proj[10] = g_vk.uboMapped->proj[15] = 1.0f;
	g_vk.uboMapped->ambientExposure[3] = 1.0f;
	g_vk.uboMapped->lightInfo[1] = (float)(g_vk.srgbOutput ? 1 : 0);

	return 1;
}

int PsyX_Vk_Initialise(const PsyXVkConfig* config)
{
	VkStage("initialise: enter");
	if (g_vk.initialised)
		return 1;
	if (!PsyX_Vk_IsSupported())
		return 0;

	const int width = config && config->width > 0 ? config->width : 1280;
	const int height = config && config->height > 0 ? config->height : 720;
	const char* title = config && config->title ? config->title : "REDRIVER2 - Vulkan fixture";
	g_vk.gameMode = (config && config->gameMode) ? 1 : 0;

	if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
	{
		eprinterr("PsyX Vulkan: SDL video init failed (%s)\n", SDL_GetError());
		return 0;
	}

	g_vk.window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
	if (!g_vk.window)
	{
		eprinterr("PsyX Vulkan: SDL_CreateWindow failed (%s)\n", SDL_GetError());
		return 0;
	}

	SDL_Vulkan_GetDrawableSize(g_vk.window, &g_vk.windowWidth, &g_vk.windowHeight);
	VkStage("initialise: window");

	if (!CreateInstance())
		return 0;
	VkStage("initialise: instance");

	if (!SDL_Vulkan_CreateSurface(g_vk.window, g_vk.instance, &g_vk.surface))
	{
		eprinterr("PsyX Vulkan: SDL_Vulkan_CreateSurface failed (%s)\n", SDL_GetError());
		return 0;
	}
	VkStage("initialise: surface");

	if (!PickPhysicalDevice())
		return 0;
	VkStage("initialise: physical device");

	if (!CreateDevice())
		return 0;
	VkStage("initialise: device");

	VkCommandPoolCreateInfo poolInfo;
	memset(&poolInfo, 0, sizeof(poolInfo));
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = g_vk.queueFamily;
	if (!VkOk(vkCreateCommandPool(g_vk.device, &poolInfo, NULL, &g_vk.commandPool), "vkCreateCommandPool"))
		return 0;

	VkCommandBufferAllocateInfo allocate;
	memset(&allocate, 0, sizeof(allocate));
	allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate.commandPool = g_vk.commandPool;
	allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate.commandBufferCount = 1;
	if (!VkOk(vkAllocateCommandBuffers(g_vk.device, &allocate, &g_vk.commandBuffer), "vkAllocateCommandBuffers"))
		return 0;

	VkFenceCreateInfo fenceInfo;
	memset(&fenceInfo, 0, sizeof(fenceInfo));
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	if (!VkOk(vkCreateFence(g_vk.device, &fenceInfo, NULL, &g_vk.frameFence), "vkCreateFence"))
		return 0;

	VkSemaphoreCreateInfo semaphoreInfo;
	memset(&semaphoreInfo, 0, sizeof(semaphoreInfo));
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	if (!VkOk(vkCreateSemaphore(g_vk.device, &semaphoreInfo, NULL, &g_vk.imageAvailable), "vkCreateSemaphore"))
		return 0;
	for (int i = 0; i < 8; i++)
	{
		if (!VkOk(vkCreateSemaphore(g_vk.device, &semaphoreInfo, NULL, &g_vk.renderFinished[i]), "vkCreateSemaphore"))
			return 0;
	}

	// Negotiate the surface format first: the main render pass declares its
	// colour attachment with it, and a VK_FORMAT_UNDEFINED attachment
	// silently discards every draw recorded into that pass.
	if (!QuerySwapchainFormat())
		return 0;
	VkStage("initialise: surface format");

	if (!CreateRenderPasses())
		return 0;
	VkStage("initialise: render passes");

	if (!CreateShadowResources())
		return 0;
	VkStage("initialise: shadow resources");

	if (!CreatePipelines())
		return 0;
	VkStage("initialise: pipelines");

	// The in-game modern path is additive: a failure disables it but leaves the
	// PSX renderer (and the fixture) intact.
	if (CreateGameModernResources())
		VkStage("initialise: game modern resources");
	else
		VkStage("initialise: game modern resources failed");

	// The swapchain depends on the main render pass.
	if (!CreateSwapchain())
		return 0;
	VkStage("initialise: swapchain");

	if (!CreateSceneResources())
		return 0;
	VkStage("initialise: scene resources");

	if (!CreateDummyTextures())
		return 0;
	VkStage("initialise: dummy textures");

	// The emulated PSX path is additive: the modern scene must keep working if
	// it cannot be created, so a failure only disables that path.
	if (CreatePsxResources())
		VkStage("initialise: psx resources");
	else
	{
		VkStage("initialise: psx resources failed");
		g_vk.psx.failed = 1;
		DestroyPsxResources();
	}

	// Allocate and wire one descriptor set per mesh slot.
	VkDescriptorSetLayout layouts[PSYX_VK_MAX_MESHES];
	for (int i = 0; i < PSYX_VK_MAX_MESHES; i++)
		layouts[i] = g_vk.descriptorSetLayout;

	VkDescriptorSetAllocateInfo setAllocate;
	memset(&setAllocate, 0, sizeof(setAllocate));
	setAllocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	setAllocate.descriptorPool = g_vk.descriptorPool;
	setAllocate.descriptorSetCount = PSYX_VK_MAX_MESHES;
	setAllocate.pSetLayouts = layouts;
	VkDescriptorSet sets[PSYX_VK_MAX_MESHES];
	if (!VkOk(vkAllocateDescriptorSets(g_vk.device, &setAllocate, sets), "vkAllocateDescriptorSets"))
		return 0;
	for (int i = 0; i < PSYX_VK_MAX_MESHES; i++)
		g_vk.meshes[i].descriptorSet = sets[i];

	// In-game modern mesh sets: one per mesh slot plus the shadow composite.
	{
		VkDescriptorSetLayout layouts[PSYX_VK_GAME_MODERN_MAX_MESHES + 1];
		for (int i = 0; i <= PSYX_VK_GAME_MODERN_MAX_MESHES; i++)
			layouts[i] = g_vk.descriptorSetLayout;

		VkDescriptorSetAllocateInfo gameAllocate;
		memset(&gameAllocate, 0, sizeof(gameAllocate));
		gameAllocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		gameAllocate.descriptorPool = g_vk.descriptorPool;
		gameAllocate.descriptorSetCount = PSYX_VK_GAME_MODERN_MAX_MESHES + 1;
		gameAllocate.pSetLayouts = layouts;

		VkDescriptorSet gameSets[PSYX_VK_GAME_MODERN_MAX_MESHES + 1];
		if (VkOk(vkAllocateDescriptorSets(g_vk.device, &gameAllocate, gameSets), "vkAllocateDescriptorSets(game modern)"))
		{
			for (int i = 0; i < PSYX_VK_GAME_MODERN_MAX_MESHES; i++)
				g_vk.gameMeshes[i].descriptorSet = gameSets[i];
			g_vk.gameCompositeSet = gameSets[PSYX_VK_GAME_MODERN_MAX_MESHES];
		}
	}

	VkStage("initialise: descriptors");

	g_vk.initialised = 1;
	g_vk.info.initialised = 1;
	g_vk.info.width = g_vk.width;
	g_vk.info.height = g_vk.height;
	g_vk.info.vulkanApiVersion = (int)g_vk.apiVersion;
	g_vk.info.imguiActive = 0;
	VkStage("initialise: core ready");

#ifdef PSYX_VK_IMGUI
	if (!config || config->enableImGui)
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGui_ImplSDL2_InitForVulkan(g_vk.window);

		// With IMGUI_IMPL_VULKAN_NO_PROTOTYPES the backend resolves every
		// Vulkan entry point through this loader and refuses to initialise if
		// one is missing, so log the failing name.
		const bool functionsLoaded = ImGui_ImplVulkan_LoadFunctions(g_vk.apiVersion,
			[](const char* functionName, void* userData) -> PFN_vkVoidFunction
			{
				(void)userData;
				PFN_vkVoidFunction function = g_gipa(g_vk.instance, functionName);
				if (!function)
					function = g_gipa(NULL, functionName);
				if (!function)
				{
					char line[192];
					snprintf(line, sizeof(line), "imgui loader: missing %s", functionName);
					VkStage(line);
				}
				return function;
			}, NULL);
		VkStage(functionsLoaded ? "imgui loader: ok" : "imgui loader: failed");

		ImGui_ImplVulkan_InitInfo initInfo;
		memset(&initInfo, 0, sizeof(initInfo));
		initInfo.ApiVersion = g_vk.apiVersion;
		initInfo.Instance = g_vk.instance;
		initInfo.PhysicalDevice = g_vk.physicalDevice;
		initInfo.Device = g_vk.device;
		initInfo.QueueFamily = g_vk.queueFamily;
		initInfo.Queue = g_vk.queue;
		initInfo.DescriptorPoolSize = 8;
		// In game mode the overlay is recorded inside the modern load pass (the
		// PSX pass ends before it), so ImGui's pipeline must target that pass.
		initInfo.RenderPass = g_vk.gameMode ? g_vk.modernRenderPass : g_vk.mainRenderPass;
		initInfo.MinImageCount = g_vk.swapchainImageCount < 2 ? 2 : g_vk.swapchainImageCount;
		initInfo.ImageCount = g_vk.swapchainImageCount;
		initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

		if (ImGui_ImplVulkan_Init(&initInfo))
		{
			g_vk.imguiActive = 1;
			g_vk.info.imguiActive = 1;
			VkStage("initialise: imgui ready");
		}
		else
		{
			VkStage("initialise: imgui failed");
		}
	}
#endif

	VkStage("initialise: done");
	return 1;
}

void PsyX_Vk_Shutdown(void)
{
	if (!g_vk.initialised)
		return;

	vkDeviceWaitIdle(g_vk.device);

#ifdef PSYX_VK_IMGUI
	if (g_vk.imguiActive)
	{
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
		g_vk.imguiActive = 0;
	}
#endif

	DestroyPsxResources();

	for (int i = 0; i < PSYX_VK_MAX_MESHES; i++)
	{
		if (g_vk.meshes[i].used)
			PsyX_Vk_DestroyMesh(i);
	}

	PsyX_Vk_GameModernMeshShutdown();

	if (g_vk.gameModernUboBuffer) vkDestroyBuffer(g_vk.device, g_vk.gameModernUboBuffer, NULL);
	if (g_vk.gameModernUboMemory) vkFreeMemory(g_vk.device, g_vk.gameModernUboMemory, NULL);
	if (g_vk.gameModernPipeline) vkDestroyPipeline(g_vk.device, g_vk.gameModernPipeline, NULL);
	if (g_vk.gameCompositePipeline) vkDestroyPipeline(g_vk.device, g_vk.gameCompositePipeline, NULL);
	if (g_vk.sceneDepthSampler) vkDestroySampler(g_vk.device, g_vk.sceneDepthSampler, NULL);
	if (g_vk.modernRenderPass) vkDestroyRenderPass(g_vk.device, g_vk.modernRenderPass, NULL);

	if (g_vk.uboBuffer) vkDestroyBuffer(g_vk.device, g_vk.uboBuffer, NULL);
	if (g_vk.uboMemory) vkFreeMemory(g_vk.device, g_vk.uboMemory, NULL);

	for (int i = 0; i < PSYX_VK_MAX_TEXTURES; i++)
	{
		if (!g_textures[i].used)
			continue;
		if (g_textures[i].view) vkDestroyImageView(g_vk.device, g_textures[i].view, NULL);
		if (g_textures[i].image) vkDestroyImage(g_vk.device, g_textures[i].image, NULL);
		if (g_textures[i].memory) vkFreeMemory(g_vk.device, g_textures[i].memory, NULL);
	}

	if (g_vk.textureSampler) vkDestroySampler(g_vk.device, g_vk.textureSampler, NULL);
	if (g_vk.shadowSampler) vkDestroySampler(g_vk.device, g_vk.shadowSampler, NULL);
	if (g_vk.shadowFramebuffer) vkDestroyFramebuffer(g_vk.device, g_vk.shadowFramebuffer, NULL);
	if (g_vk.shadowView) vkDestroyImageView(g_vk.device, g_vk.shadowView, NULL);
	if (g_vk.shadowImage) vkDestroyImage(g_vk.device, g_vk.shadowImage, NULL);
	if (g_vk.shadowMemory) vkFreeMemory(g_vk.device, g_vk.shadowMemory, NULL);

	if (g_vk.readbackBuffer) vkDestroyBuffer(g_vk.device, g_vk.readbackBuffer, NULL);
	if (g_vk.readbackMemory) vkFreeMemory(g_vk.device, g_vk.readbackMemory, NULL);
	g_vk.readbackMapped = NULL;

	if (g_vk.pbrPipeline) vkDestroyPipeline(g_vk.device, g_vk.pbrPipeline, NULL);
	if (g_vk.shadowPipeline) vkDestroyPipeline(g_vk.device, g_vk.shadowPipeline, NULL);
	if (g_vk.pipelineLayout) vkDestroyPipelineLayout(g_vk.device, g_vk.pipelineLayout, NULL);
	if (g_vk.shadowPipelineLayout) vkDestroyPipelineLayout(g_vk.device, g_vk.shadowPipelineLayout, NULL);
	if (g_vk.descriptorPool) vkDestroyDescriptorPool(g_vk.device, g_vk.descriptorPool, NULL);
	if (g_vk.descriptorSetLayout) vkDestroyDescriptorSetLayout(g_vk.device, g_vk.descriptorSetLayout, NULL);
	if (g_vk.mainRenderPass) vkDestroyRenderPass(g_vk.device, g_vk.mainRenderPass, NULL);
	if (g_vk.shadowRenderPass) vkDestroyRenderPass(g_vk.device, g_vk.shadowRenderPass, NULL);

	DestroySwapchainResources();
	DestroySwapchain();

	if (g_vk.frameFence) vkDestroyFence(g_vk.device, g_vk.frameFence, NULL);
	if (g_vk.imageAvailable) vkDestroySemaphore(g_vk.device, g_vk.imageAvailable, NULL);
	for (int i = 0; i < 8; i++)
	{
		if (g_vk.renderFinished[i])
			vkDestroySemaphore(g_vk.device, g_vk.renderFinished[i], NULL);
	}
	if (g_vk.commandPool) vkDestroyCommandPool(g_vk.device, g_vk.commandPool, NULL);

	if (g_vk.device) vkDestroyDevice(g_vk.device, NULL);
	if (g_vk.surface) vkDestroySurfaceKHR(g_vk.instance, g_vk.surface, NULL);
	if (g_vk.instance) vkDestroyInstance(g_vk.instance, NULL);

	if (g_vk.window)
	{
		SDL_DestroyWindow(g_vk.window);
		g_vk.window = NULL;
	}

	memset(&g_vk, 0, sizeof(g_vk));
	g_vk.info.initialised = 0;
}

#else // platform without Vulkan

int PsyX_Vk_IsSupported(void) { return 0; }
int PsyX_Vk_Initialise(const PsyXVkConfig* config) { (void)config; return 0; }
void PsyX_Vk_Shutdown(void) {}
int PsyX_Vk_CreateMesh(const PsyXModernMeshDesc* desc) { (void)desc; return -1; }
void PsyX_Vk_DestroyMesh(int mesh) { (void)mesh; }
void PsyX_Vk_SetInstance(int mesh, const float worldMatrix[16], const float color[4], int visible)
{
	(void)mesh; (void)worldMatrix; (void)color; (void)visible;
}
void PsyX_Vk_SetCamera(const float view[16], const float proj[16], const float cameraPosition[3])
{
	(void)view; (void)proj; (void)cameraPosition;
}
void PsyX_Vk_SetLights(const PsyXModernLightSet* lights) { (void)lights; }
int PsyX_Vk_RenderFrame(void) { return 0; }
int PsyX_Vk_ReadbackRgba(unsigned char* rgba, int* width, int* height)
{
	(void)rgba; (void)width; (void)height; return 0;
}
void PsyX_Vk_GetInfo(PsyXVkInfo* info) { if (info) memset(info, 0, sizeof(*info)); }
void PsyX_Vk_SetOverlayText(const char* text) { (void)text; }
void PsyX_Vk_GameBeginFrame(void) {}
void PsyX_Vk_GameSetVram(const unsigned short* vram) { (void)vram; }
void PsyX_Vk_GameSetProjection2D(const float projection[16]) { (void)projection; }
void PsyX_Vk_GameSetProjection3D(const float projection[16]) { (void)projection; }
void PsyX_Vk_GameUpdateVertexBuffer(const void* vertices, int vertexCount) { (void)vertices; (void)vertexCount; }
void PsyX_Vk_GameSetBlendMode(int blendMode) { (void)blendMode; }
void PsyX_Vk_GameSetTexture(int texFormat, int texture) { (void)texFormat; (void)texture; }
void PsyX_Vk_GameSetOverrideTextureSize(int width, int height) { (void)width; (void)height; }
void PsyX_Vk_GameSetOverrideAlphaMode(int mode) { (void)mode; }
void PsyX_Vk_GameSetStencilMode(int drawPrimMode) { (void)drawPrimMode; }
void PsyX_Vk_GameEnableDepth(int enable) { (void)enable; }
void PsyX_Vk_GameSetBilinear(int enable) { (void)enable; }
void PsyX_Vk_GameSetScissor(int enable, int x, int y, int width, int height)
{
	(void)enable; (void)x; (void)y; (void)width; (void)height;
}
void PsyX_Vk_GameSetViewPort(int x, int y, int width, int height) { (void)x; (void)y; (void)width; (void)height; }
void PsyX_Vk_GameSetOffscreen(int enable, int x, int y, int width, int height)
{
	(void)enable; (void)x; (void)y; (void)width; (void)height;
}
int PsyX_Vk_GameResolveOffscreen(unsigned short* vram) { (void)vram; return 0; }
void PsyX_Vk_GameStoreFrameBuffer(int x, int y, int width, int height) { (void)x; (void)y; (void)width; (void)height; }
int PsyX_Vk_TakeStoredFrameBuffer(const unsigned char** rgba, int* stride,
	int* srcWidth, int* srcHeight, int* bgra, int* x, int* y, int* width, int* height)
{
	(void)rgba; (void)stride; (void)srcWidth; (void)srcHeight; (void)bgra;
	(void)x; (void)y; (void)width; (void)height;
	return 0;
}
void PsyX_Vk_GameClear(int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b)
{
	(void)x; (void)y; (void)width; (void)height; (void)r; (void)g; (void)b;
}
int PsyX_Vk_GameDrawTriangles(int firstVertex, int triangles) { (void)firstVertex; (void)triangles; return 0; }
int PsyX_Vk_GameCreateTexture(const unsigned char* rgba, int width, int height, int mipmapped)
{
	(void)rgba; (void)width; (void)height; (void)mipmapped;
	return 0;
}
void PsyX_Vk_GameDestroyTexture(int texture) { (void)texture; }
unsigned long long PsyX_Vk_GameGetOverlayTextureId(int texture) { (void)texture; return 0; }
void PsyX_Vk_GameGetTextureSize(int texture, int* width, int* height)
{
	(void)texture;
	if (width) *width = 0;
	if (height) *height = 0;
}
int PsyX_Vk_GameModernMeshCreate(const PsyXModernMeshDesc* desc) { (void)desc; return -1; }
void PsyX_Vk_GameModernMeshDestroy(int mesh) { (void)mesh; }
void PsyX_Vk_GameModernMeshSetInstance(int mesh, const float viewMatrix[16],
	const float color[4], int visible)
{
	(void)mesh; (void)viewMatrix; (void)color; (void)visible;
}
void PsyX_Vk_GameModernMeshSetInstanceWorld(int mesh, const float worldMatrix[16])
{
	(void)mesh; (void)worldMatrix;
}
void PsyX_Vk_GameModernMeshSetLights(const PsyXModernLightSet* lights) { (void)lights; }
void PsyX_Vk_GameModernMeshSetCamera(const float viewRotation[16], const float cameraPosition[3])
{
	(void)viewRotation; (void)cameraPosition;
}
void PsyX_Vk_GameModernMeshSetShadowDebug(int mode) { (void)mode; }
void PsyX_Vk_GameModernMeshSetEnabled(int enabled) { (void)enabled; }
int  PsyX_Vk_GameModernMeshGetEnabled(void) { return 0; }
void PsyX_Vk_GameModernMeshGetStats(PsyXModernMeshStats* stats)
{
	if (stats) memset(stats, 0, sizeof(*stats));
}
void PsyX_Vk_GameModernMeshShutdown(void) {}
void PsyX_Vk_GameEndFrame(void) {}
void PsyX_Vk_GameResetDevice(void) {}
int PsyX_Vk_GameIsActive(void) { return 0; }
SDL_Window* PsyX_Vk_GetSDLWindow(void) { return NULL; }
int PsyX_Vk_GameSelfTest(char* report, int reportSize)
{
	if (report && reportSize > 0)
		report[0] = 0;
	return 0;
}

#endif // platform without Vulkan
