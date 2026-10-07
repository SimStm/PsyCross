#ifndef PSYX_NATIVE_SCENE_INTERNAL_H
#define PSYX_NATIVE_SCENE_INTERNAL_H

#include "PsyX/PsyX_native.h"
#include <vector>

/* CPU ownership and validation are independent of Vulkan. The backend alone
 * advances Ready/Retiring after uploads and completion of its submission fence. */
class PsyXNativeScene
{
public:
	enum State { Empty, Pending, Ready, Failed, Retiring, Exhausted };
	struct Mesh
	{
		State state;
		uint32_t generation;
		uint64_t lastSubmission;
		std::vector<PsyXNativeVertex> vertices;
		std::vector<uint32_t> indices;
		PsyXNativeMaterialHandle material;
		Mesh() : state(Empty), generation(0), lastSubmission(0), material() {}
		uint64_t Bytes() const;
	};
	struct Material
	{
		State state;
		uint32_t generation;
		uint64_t lastSubmission;
		uint32_t width, height;
		PsyXNativeFilter filter;
		float alphaCutoff;
		PsyXNativeCull cull;
		std::vector<uint8_t> rgba;
		Material() : state(Empty), generation(0), lastSubmission(0), width(0), height(0), filter(PSYX_NATIVE_NEAREST), alphaCutoff(0.5f), cull(PSYX_NATIVE_CULL_BACK) {}
	};

	PsyXNativeScene();
	PsyXNativeResult Create(const PsyXNativeMeshDesc* desc, PsyXNativeMeshHandle* handle);
	PsyXNativeResult Destroy(PsyXNativeMeshHandle handle);
	PsyXNativeResult CreateMaterial(const PsyXNativeMaterialDesc* desc, PsyXNativeMaterialHandle* handle);
	PsyXNativeResult DestroyMaterial(PsyXNativeMaterialHandle handle);
	bool IsMaterialLive(PsyXNativeMaterialHandle handle) const;
	bool CanReclaimMaterial(uint32_t slot, uint64_t completed) const;
	void ReclaimMaterial(uint32_t slot);
	PsyXNativeResult Publish(const PsyXNativeSnapshot* snapshot);
	uint64_t Reset();
	void SetRequested(int requested);
	bool IsLive(PsyXNativeMeshHandle handle) const;
	bool CanReclaim(uint32_t slot, uint64_t completed) const;
	void Reclaim(uint32_t slot); /* only after backend destroys retired GPU objects */
	void ConsumeSnapshot();
	void GetStats(PsyXNativeStats* out) const;
	PsyXNativeResult Reject(PsyXNativeResult reason);

	Mesh meshes[PSYX_NATIVE_MAX_MESHES];
	Material materials[PSYX_NATIVE_MAX_MATERIALS];
	PsyXNativeInstance instances[PSYX_NATIVE_MAX_INSTANCES];
	PsyXNativeSnapshot snapshot;
	bool snapshotPending;
	bool requested;
	uint64_t generation;
	uint64_t ownedBytes;
	uint64_t ownedMaterialBytes;
	uint32_t rejected;
	bool generationExhausted;
};

#endif
