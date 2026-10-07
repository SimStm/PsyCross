#include "PsyX_NativeScene.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

namespace
{
bool Finite(const float* values, unsigned int count)
{
	for (unsigned int i = 0; i < count; ++i)
		if (!std::isfinite(values[i])) return false;
	return true;
}

bool Affine(const float matrix[16])
{
	return Finite(matrix, 16) && matrix[3] == 0.0f && matrix[7] == 0.0f &&
		matrix[11] == 0.0f && matrix[15] == 1.0f;
}

bool OpaqueColor(const float color[4])
{
	return Finite(color, 4) && color[0] >= 0 && color[0] <= 1 &&
		color[1] >= 0 && color[1] <= 1 && color[2] >= 0 && color[2] <= 1 && color[3] == 1;
}

bool ValidMesh(const PsyXNativeMeshDesc& desc)
{
	if (desc.depthLayer > PSYX_NATIVE_MAX_DEPTH_LAYER) return false;
	if (!desc.vertices || !desc.indices || desc.vertexCount < 3 || desc.indexCount < 3 ||
		desc.indexCount % 3 || !Finite(desc.boundsMin, 3) || !Finite(desc.boundsMax, 3)) return false;
	for (unsigned int axis = 0; axis < 3; ++axis)
		if (desc.boundsMin[axis] > desc.boundsMax[axis]) return false;
	for (uint32_t i = 0; i < desc.indexCount; ++i)
		if (desc.indices[i] >= desc.vertexCount) return false;
	for (uint32_t i = 0; i < desc.vertexCount; ++i)
	{
		const PsyXNativeVertex& v = desc.vertices[i];
		if (!Finite(v.position, 3) || !Finite(v.normal, 3) || !Finite(v.uv, 2) ||
			!OpaqueColor(v.color)) return false;
		for (unsigned int axis = 0; axis < 3; ++axis)
			if (v.position[axis] < desc.boundsMin[axis] || v.position[axis] > desc.boundsMax[axis]) return false;
	}
	return true;
}
}

uint64_t PsyXNativeScene::Mesh::Bytes() const
{
	return uint64_t(vertices.size()) * sizeof(PsyXNativeVertex) + uint64_t(indices.size()) * sizeof(uint32_t);
}

PsyXNativeScene::PsyXNativeScene() : snapshotPending(false), requested(false), worldExpected(false), producerReason(PSYX_NATIVE_NO_SNAPSHOT), generation(1), ownedBytes(0), ownedMaterialBytes(0), rejected(0), generationExhausted(false)
{
	memset(instances, 0, sizeof(instances));
	memset(&snapshot, 0, sizeof(snapshot));
	snapshot.instances = instances;
}

PsyXNativeResult PsyXNativeScene::Reject(PsyXNativeResult reason)
{
	if (rejected != std::numeric_limits<uint32_t>::max()) ++rejected;
	return reason;
}

PsyXNativeResult PsyXNativeScene::Create(const PsyXNativeMeshDesc* desc, PsyXNativeMeshHandle* handle)
{
	if (handle) memset(handle, 0, sizeof(*handle));
	if (!desc || !handle || desc->size != sizeof(*desc) || desc->version != PSYX_NATIVE_VERSION)
		return Reject(PSYX_NATIVE_INVALID);
	if (generationExhausted) return Reject(PSYX_NATIVE_OUT_OF_BUDGET);
	// Check widened byte arithmetic before traversing caller spans or allocating.
	const uint64_t bytes = uint64_t(desc->vertexCount) * sizeof(PsyXNativeVertex) + uint64_t(desc->indexCount) * sizeof(uint32_t);
	if (bytes > PSYX_NATIVE_MAX_BYTES || ownedBytes > PSYX_NATIVE_MAX_BYTES - bytes)
		return Reject(PSYX_NATIVE_OUT_OF_BUDGET);
	if (!ValidMesh(*desc)) return Reject(PSYX_NATIVE_INVALID);
	if (desc->material.generation && !IsMaterialLive(desc->material)) return Reject(PSYX_NATIVE_STALE);
	if (!desc->material.generation && desc->material.slot) return Reject(PSYX_NATIVE_INVALID);
	uint32_t slot = PSYX_NATIVE_MAX_MESHES;
	for (uint32_t i = 0; i < PSYX_NATIVE_MAX_MESHES; ++i)
		if (meshes[i].state == Empty) { slot = i; break; }
	if (slot == PSYX_NATIVE_MAX_MESHES) return Reject(PSYX_NATIVE_OUT_OF_BUDGET);

	// Build in temporary owned vectors. A failed allocation cannot publish half
	// a mesh or charge the budget; exceptions never escape the C API.
	try
	{
		std::vector<PsyXNativeVertex> vertices(desc->vertices, desc->vertices + desc->vertexCount);
		std::vector<uint32_t> indices(desc->indices, desc->indices + desc->indexCount);
		Mesh& mesh = meshes[slot];
		mesh.vertices.swap(vertices);
		mesh.indices.swap(indices);
		++mesh.generation;
		mesh.lastSubmission = 0;
		mesh.material = desc->material;
		mesh.depthLayer = desc->depthLayer;
		mesh.state = Pending;
		ownedBytes += bytes;
		handle->slot = slot;
		handle->generation = mesh.generation;
	}
	catch (const std::bad_alloc&) { return Reject(PSYX_NATIVE_OUT_OF_BUDGET); }
	return PSYX_NATIVE_PENDING;
}

bool PsyXNativeScene::IsLive(PsyXNativeMeshHandle handle) const
{
	if (!handle.generation || handle.slot >= PSYX_NATIVE_MAX_MESHES) return false;
	const Mesh& mesh = meshes[handle.slot];
	return mesh.generation == handle.generation &&
		(mesh.state == Pending || mesh.state == Ready || mesh.state == Failed);
}

PsyXNativeResult PsyXNativeScene::Destroy(PsyXNativeMeshHandle handle)
{
	if (!IsLive(handle)) return Reject(PSYX_NATIVE_STALE);
	meshes[handle.slot].state = Retiring;
	// An already published instance may refer to the invalidated handle. The
	// recorder revalidates every reference rather than retaining a slot pointer.
	return PSYX_NATIVE_OK;
}

PsyXNativeResult PsyXNativeScene::Publish(const PsyXNativeSnapshot* source)
{
	if (!source || source->size != sizeof(*source) || source->version != PSYX_NATIVE_VERSION)
		return Reject(PSYX_NATIVE_INVALID);
	if (generationExhausted || source->sceneGeneration != generation) return Reject(PSYX_NATIVE_STALE);
	if (source->instanceCount > PSYX_NATIVE_MAX_INSTANCES) return Reject(PSYX_NATIVE_OUT_OF_BUDGET);
	if (!source->instances || !source->instanceCount || !Affine(source->view.view) ||
		!Finite(source->view.projection, 16) || !Finite(source->view.cameraPosition, 3) ||
		!std::isfinite(source->view.nearPlane) || !std::isfinite(source->view.farPlane) ||
		source->view.nearPlane <= 0 || source->view.farPlane <= source->view.nearPlane)
		return Reject(PSYX_NATIVE_INVALID);
	for (uint32_t i = 0; i < source->instanceCount; ++i)
	{
		if (!IsLive(source->instances[i].mesh)) return Reject(PSYX_NATIVE_STALE);
		const PsyXNativeMaterialHandle material = meshes[source->instances[i].mesh.slot].material;
		if (material.generation && !IsMaterialLive(material)) return Reject(PSYX_NATIVE_STALE);
		if (!Affine(source->instances[i].world) || !OpaqueColor(source->instances[i].tint) ||
			(source->instances[i].layer != PSYX_NATIVE_WORLD && source->instances[i].layer != PSYX_NATIVE_BACKDROP) ||
			unsigned(source->instances[i].sampling)>PSYX_NATIVE_SAMPLE_TRILINEAR)
			return Reject(PSYX_NATIVE_INVALID);
		if (material.generation && source->instances[i].sampling==PSYX_NATIVE_SAMPLE_TRILINEAR)
		{
			const std::vector<PsyXNativeMip::Level>& levels=materials[material.slot].levels;
			if (levels.empty() || levels.back().width!=1 || levels.back().height!=1)
				return Reject(PSYX_NATIVE_UNSUPPORTED);
		}
	}
	// Commit only after every record passed. A rejected publish leaves the
	// previous valid snapshot intact; recording still verifies mesh liveness.
	memmove(instances, source->instances, source->instanceCount * sizeof(*instances));
	snapshot = *source;
	snapshot.instances = instances;
	snapshotPending = true;
	worldExpected = true;
	producerReason = PSYX_NATIVE_OK;
	return PSYX_NATIVE_OK;
}

void PsyXNativeScene::SetFrameStatus(PsyXNativeResult reason)
{
	worldExpected = true;
	producerReason = reason;
	if (reason != PSYX_NATIVE_OK) { snapshotPending = false; snapshot.instanceCount = 0; }
}

PsyXNativeFrameState PsyXNativeScene::FrameState(PsyXNativeResult reason, bool worldBoundary) const
{
	if (!requested) return PSYX_NATIVE_FRAME_OFF;
	if (!worldExpected && !worldBoundary && !snapshotPending) return PSYX_NATIVE_FRAME_AUXILIARY;
	if (reason == PSYX_NATIVE_OK) return PSYX_NATIVE_FRAME_READY;
	return reason == PSYX_NATIVE_PENDING ? PSYX_NATIVE_FRAME_LOADING : PSYX_NATIVE_FRAME_FAILED;
}

void PsyXNativeScene::SetRequested(int value)
{
	requested = value != 0;
	if (!requested) ConsumeSnapshot();
}

uint64_t PsyXNativeScene::Reset()
{
	// Never wrap to an earlier scene identity. At exhaustion all future creates
	// remain unavailable; this is preferable to accepting stale references.
	if (generation != std::numeric_limits<uint64_t>::max()) ++generation;
	else generationExhausted = true;
	ConsumeSnapshot();
	for (uint32_t i = 0; i < PSYX_NATIVE_MAX_MESHES; ++i)
		if (meshes[i].state == Pending || meshes[i].state == Ready || meshes[i].state == Failed)
			meshes[i].state = Retiring;
	for (uint32_t i = 0; i < PSYX_NATIVE_MAX_MATERIALS; ++i)
		if (materials[i].state == Pending || materials[i].state == Ready || materials[i].state == Failed)
			materials[i].state = Retiring;
	return generation;
}

bool PsyXNativeScene::CanReclaim(uint32_t slot, uint64_t completed) const
{
	return slot < PSYX_NATIVE_MAX_MESHES && meshes[slot].state == Retiring && meshes[slot].lastSubmission <= completed;
}

void PsyXNativeScene::Reclaim(uint32_t slot)
{
	Mesh& mesh = meshes[slot];
	ownedBytes -= mesh.Bytes();
	std::vector<PsyXNativeVertex>().swap(mesh.vertices);
	std::vector<uint32_t>().swap(mesh.indices);
	mesh.lastSubmission = 0;
	mesh.material = PsyXNativeMaterialHandle();
	mesh.depthLayer = 0;
	mesh.state = mesh.generation == std::numeric_limits<uint32_t>::max() ? Exhausted : Empty;
}

void PsyXNativeScene::ConsumeSnapshot()
{
	snapshotPending = false;
	snapshot.instanceCount = 0;
	worldExpected = false;
	producerReason = PSYX_NATIVE_NO_SNAPSHOT;
}

void PsyXNativeScene::GetStats(PsyXNativeStats* out) const
{
	memset(out, 0, sizeof(*out));
	out->size = sizeof(*out);
	out->version = PSYX_NATIVE_VERSION;
	out->requested = requested;
	out->reason = requested ? PSYX_NATIVE_NO_SNAPSHOT : PSYX_NATIVE_OK;
	out->rejectedCalls = rejected;
	out->ownedBytes = ownedBytes;
	out->sceneGeneration = generation;
	out->simulationTick = snapshot.simulationTick;
	out->ownedMaterialBytes = ownedMaterialBytes;
	for (uint32_t i = 0; i < PSYX_NATIVE_MAX_MATERIALS; ++i)
	{
		if (materials[i].state == Ready) ++out->residentMaterials;
		if (materials[i].state == Pending) ++out->pendingMaterials;
		if (materials[i].state == Retiring) ++out->retiringMaterials;
	}
	for (uint32_t i = 0; i < PSYX_NATIVE_MAX_MESHES; ++i)
	{
		if (meshes[i].state == Ready) ++out->residentMeshes;
		if (meshes[i].state == Pending) ++out->pendingMeshes;
		if (meshes[i].state == Retiring) ++out->retiringMeshes;
	}
}

PsyXNativeResult PsyXNativeScene::CreateMaterial(const PsyXNativeMaterialDesc* desc, PsyXNativeMaterialHandle* handle)
{
	if (handle) memset(handle, 0, sizeof(*handle));
	if (!desc || !handle || desc->size != sizeof(*desc) || desc->version != PSYX_NATIVE_VERSION ||
		!desc->rgba || !desc->width || !desc->height || desc->width > 4096 || desc->height > 4096 ||
		(desc->filter != PSYX_NATIVE_NEAREST && desc->filter != PSYX_NATIVE_LINEAR && desc->filter != PSYX_NATIVE_TRILINEAR) ||
		(desc->cull != PSYX_NATIVE_CULL_BACK && desc->cull != PSYX_NATIVE_CULL_NONE) ||
		!std::isfinite(desc->alphaCutoff) || desc->alphaCutoff < 0 || desc->alphaCutoff > 1)
		return Reject(PSYX_NATIVE_INVALID);
	if (generationExhausted) return Reject(PSYX_NATIVE_OUT_OF_BUDGET);
	if (desc->byteCount != uint64_t(desc->width) * desc->height * 4) return Reject(PSYX_NATIVE_INVALID);
	uint32_t slot = PSYX_NATIVE_MAX_MATERIALS;
	for (uint32_t i = 0; i < PSYX_NATIVE_MAX_MATERIALS; ++i)
		if (materials[i].state == Empty) { slot = i; break; }
	if (slot == PSYX_NATIVE_MAX_MATERIALS) return Reject(PSYX_NATIVE_OUT_OF_BUDGET);
	try
	{
		std::vector<PsyXNativeMip::Level> levels;
		const uint64_t bytes=PsyXNativeMip::Layout(desc->width,desc->height,desc->filter==PSYX_NATIVE_TRILINEAR,levels);
		if (bytes > PSYX_NATIVE_MAX_MATERIAL_BYTES || ownedMaterialBytes > PSYX_NATIVE_MAX_MATERIAL_BYTES - bytes)
			return Reject(PSYX_NATIVE_OUT_OF_BUDGET);
		std::vector<uint8_t> pixels;
		PsyXNativeMip::Build(desc->rgba,levels,bytes,desc->alphaCutoff,pixels);
		Material& material = materials[slot];
		material.rgba.swap(pixels);
		material.levels.swap(levels);
		material.width = desc->width; material.height = desc->height;
		material.filter = desc->filter; material.alphaCutoff = desc->alphaCutoff;
		material.cull = desc->cull;
		++material.generation;
		material.lastSubmission = 0; material.state = Pending;
		ownedMaterialBytes += bytes;
		handle->slot = slot; handle->generation = material.generation;
	}
	catch (const std::bad_alloc&) { return Reject(PSYX_NATIVE_OUT_OF_BUDGET); }
	return PSYX_NATIVE_PENDING;
}
bool PsyXNativeScene::IsMaterialLive(PsyXNativeMaterialHandle handle) const
{
	if (!handle.generation || handle.slot >= PSYX_NATIVE_MAX_MATERIALS) return false;
	const Material& material = materials[handle.slot];
	return material.generation == handle.generation &&
		(material.state == Pending || material.state == Ready || material.state == Failed);
}
PsyXNativeResult PsyXNativeScene::DestroyMaterial(PsyXNativeMaterialHandle handle)
{
	if (!IsMaterialLive(handle)) return Reject(PSYX_NATIVE_STALE);
	materials[handle.slot].state = Retiring;
	return PSYX_NATIVE_OK;
}
bool PsyXNativeScene::CanReclaimMaterial(uint32_t slot, uint64_t completed) const
{
	return slot < PSYX_NATIVE_MAX_MATERIALS && materials[slot].state == Retiring && materials[slot].lastSubmission <= completed;
}
void PsyXNativeScene::ReclaimMaterial(uint32_t slot)
{
	Material& material = materials[slot];
	ownedMaterialBytes -= material.rgba.size();
	std::vector<uint8_t>().swap(material.rgba);
	std::vector<PsyXNativeMip::Level>().swap(material.levels);
	material.lastSubmission = 0; material.width = material.height = 0;
	material.state = material.generation == std::numeric_limits<uint32_t>::max() ? Exhausted : Empty;
}

const char* PsyX_Native_ResultName(PsyXNativeResult result)
{
	switch (result)
	{
	case PSYX_NATIVE_OK: return "ready";
	case PSYX_NATIVE_PENDING: return "loading native sources or GPU resources";
	case PSYX_NATIVE_UNSUPPORTED: return "unsupported native backend, mode, source or GPU capability";
	case PSYX_NATIVE_INVALID: return "invalid descriptor";
	case PSYX_NATIVE_STALE: return "stale handle or scene";
	case PSYX_NATIVE_OUT_OF_BUDGET: return "resource budget exceeded";
	case PSYX_NATIVE_UPLOAD_FAILED: return "GPU upload failed";
	case PSYX_NATIVE_NO_SNAPSHOT: return "missing native scene publication this world frame";
	case PSYX_NATIVE_NO_BOUNDARY: return "no tested world/overlay boundary";
	default: return "unknown result";
	}
}
