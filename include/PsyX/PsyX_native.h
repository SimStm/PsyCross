#ifndef PSYX_NATIVE_H
#define PSYX_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* R01 native world contributor. All calls belong to the render thread. No
 * Vulkan objects or borrowed game pointers cross this C boundary. */
#define PSYX_NATIVE_VERSION 6u
#define PSYX_NATIVE_MAX_MESHES 512u
#define PSYX_NATIVE_MAX_INSTANCES 4096u
#define PSYX_NATIVE_MAX_BYTES (32u * 1024u * 1024u)
#define PSYX_NATIVE_MAX_MATERIALS 256u
#define PSYX_NATIVE_MAX_MATERIAL_BYTES (64u * 1024u * 1024u)

typedef enum PsyXNativeResult
{
	PSYX_NATIVE_OK = 0,
	PSYX_NATIVE_PENDING = 1,
	PSYX_NATIVE_UNSUPPORTED = 2,
	PSYX_NATIVE_INVALID = 3,
	PSYX_NATIVE_STALE = 4,
	PSYX_NATIVE_OUT_OF_BUDGET = 5,
	PSYX_NATIVE_UPLOAD_FAILED = 6,
	PSYX_NATIVE_NO_SNAPSHOT = 7,
	PSYX_NATIVE_NO_BOUNDARY = 8
} PsyXNativeResult;

typedef struct PsyXNativeMeshHandle
{
	uint32_t slot;
	uint32_t generation; /* zero is invalid; slot alone is never an identity */
} PsyXNativeMeshHandle;

typedef struct PsyXNativeMaterialHandle
{
	uint32_t slot;
	uint32_t generation;
} PsyXNativeMaterialHandle;

/* NEAREST/LINEAR sample level zero; TRILINEAR owns a complete mip chain. */
typedef enum PsyXNativeFilter { PSYX_NATIVE_NEAREST = 0, PSYX_NATIVE_LINEAR = 1, PSYX_NATIVE_TRILINEAR = 2 } PsyXNativeFilter;
typedef enum PsyXNativeCull { PSYX_NATIVE_CULL_BACK = 0, PSYX_NATIVE_CULL_NONE = 1 } PsyXNativeCull;
typedef struct PsyXNativeMaterialDesc
{
	uint32_t size;
	uint32_t version;
	const uint8_t* rgba; /* copied sRGB artwork; tightly packed width*height*4 */
	uint32_t width;
	uint32_t height;
	uint64_t byteCount; /* actual supplied span; must match dimensions exactly */
	PsyXNativeFilter filter;
	float alphaCutoff; /* opaque/cutout slice; alpha does not imply blending */
	PsyXNativeCull cull; /* explicit two-sided cutouts; no optional GPU feature */
} PsyXNativeMaterialDesc;

typedef struct PsyXNativeVertex
{
	float position[3];
	float normal[3];
	float uv[2];
	float color[4]; /* linear RGB, opaque alpha=1 in this unlit slice */
} PsyXNativeVertex;

typedef struct PsyXNativeMeshDesc
{
	uint32_t size;
	uint32_t version;
	const PsyXNativeVertex* vertices;
	const uint32_t* indices;
	uint32_t vertexCount;
	uint32_t indexCount; /* triangle list; copied before returning */
	float boundsMin[3];
	float boundsMax[3];
	PsyXNativeMaterialHandle material; /* generation=0 selects analytic untextured */
} PsyXNativeMeshDesc;

typedef enum PsyXNativeLayer { PSYX_NATIVE_WORLD = 0, PSYX_NATIVE_BACKDROP = 1 } PsyXNativeLayer;
/* Per-instance selection uses immutable descriptors on the same owned image.
 * TRILINEAR requires a complete material mip chain; default inherits material. */
typedef enum PsyXNativeSampling
{
	PSYX_NATIVE_USE_MATERIAL_FILTER = 0,
	PSYX_NATIVE_SAMPLE_NEAREST = 1,
	PSYX_NATIVE_SAMPLE_LINEAR = 2,
	PSYX_NATIVE_SAMPLE_TRILINEAR = 3
} PsyXNativeSampling;
typedef struct PsyXNativeInstance
{
	PsyXNativeMeshHandle mesh;
	float world[16]; /* column-major affine local-to-world */
	float tint[4]; /* linear RGB, alpha=1 */
	uint64_t identity;
	PsyXNativeLayer layer; /* backdrop ignores view translation, draws first with no depth test/write */
	PsyXNativeSampling sampling;
} PsyXNativeInstance;

typedef struct PsyXNativeView
{
	float view[16];
	float projection[16]; /* RH -Z; Vulkan depth 0..1; positive viewport/Y flip */
	float cameraPosition[3];
	float nearPlane;
	float farPlane;
} PsyXNativeView;

typedef struct PsyXNativeSnapshot
{
	uint32_t size;
	uint32_t version;
	uint64_t sceneGeneration;
	uint64_t simulationTick;
	PsyXNativeView view;
	const PsyXNativeInstance* instances;
	uint32_t instanceCount;
} PsyXNativeSnapshot;

typedef struct PsyXNativeStats
{
	uint32_t size;
	uint32_t version;
	int requested;
	int effective;
	PsyXNativeResult reason;
	uint32_t residentMeshes;
	uint32_t pendingMeshes;
	uint32_t retiringMeshes;
	uint32_t nativeDraws;
	uint32_t legacyWorldDraws;
	uint32_t legacyOverlayDraws;
	uint32_t suppressedWorldDraws;
	uint32_t rejectedCalls;
	uint64_t ownedBytes;
	uint64_t sceneGeneration;
	uint64_t submittedSerial;
	uint64_t completedSerial;
	uint64_t simulationTick;
	uint32_t residentMaterials;
	uint32_t pendingMaterials;
	uint32_t retiringMaterials;
	uint64_t ownedMaterialBytes;
	uint32_t nativeBackdropDraws; /* subset of nativeDraws, excluded from world depth/lighting */
	uint64_t effectiveFrames; /* cumulative submissions since backend initialization */
	uint64_t fallbackFrames; /* requested native mode but whole legacy frame submitted */
	uint64_t materialUploads; /* successful owned material image uploads, excludes analytic white */
} PsyXNativeStats;

/* Pending is a successful owned CPU create, with a valid handle. GPU upload
 * happens after the existing frame fence; query failures explicitly. */
PsyXNativeResult PsyX_Native_CreateMesh(const PsyXNativeMeshDesc* desc, PsyXNativeMeshHandle* handle);
PsyXNativeResult PsyX_Native_DestroyMesh(PsyXNativeMeshHandle handle);
PsyXNativeResult PsyX_Native_CreateMaterial(const PsyXNativeMaterialDesc* desc, PsyXNativeMaterialHandle* handle);
PsyXNativeResult PsyX_Native_DestroyMaterial(PsyXNativeMaterialHandle handle);
PsyXNativeResult PsyX_Native_Publish(const PsyXNativeSnapshot* snapshot);
void PsyX_Native_SetRequested(int requested);
uint64_t PsyX_Native_ResetScene(void); /* invalidates all handles immediately */
uint64_t PsyX_Native_GetSceneGeneration(void);
void PsyX_Native_GetStats(PsyXNativeStats* stats);
const char* PsyX_Native_ResultName(PsyXNativeResult result);

#ifdef __cplusplus
}
#endif
#endif
