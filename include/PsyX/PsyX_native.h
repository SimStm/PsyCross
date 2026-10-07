#ifndef PSYX_NATIVE_H
#define PSYX_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* R01 native world contributor. All calls belong to the render thread. No
 * Vulkan objects or borrowed game pointers cross this C boundary. */
#define PSYX_NATIVE_VERSION 1u
#define PSYX_NATIVE_MAX_MESHES 32u
#define PSYX_NATIVE_MAX_INSTANCES 64u
#define PSYX_NATIVE_MAX_BYTES (8u * 1024u * 1024u)

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
} PsyXNativeMeshDesc;

typedef struct PsyXNativeInstance
{
	PsyXNativeMeshHandle mesh;
	float world[16]; /* column-major affine local-to-world */
	float tint[4]; /* linear RGB, alpha=1 */
	uint64_t identity;
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
} PsyXNativeStats;

/* Pending is a successful owned CPU create, with a valid handle. GPU upload
 * happens after the existing frame fence; query failures explicitly. */
PsyXNativeResult PsyX_Native_CreateMesh(const PsyXNativeMeshDesc* desc, PsyXNativeMeshHandle* handle);
PsyXNativeResult PsyX_Native_DestroyMesh(PsyXNativeMeshHandle handle);
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
