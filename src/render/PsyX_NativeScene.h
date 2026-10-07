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
		Mesh() : state(Empty), generation(0), lastSubmission(0) {}
		uint64_t Bytes() const;
	};

	PsyXNativeScene();
	PsyXNativeResult Create(const PsyXNativeMeshDesc* desc, PsyXNativeMeshHandle* handle);
	PsyXNativeResult Destroy(PsyXNativeMeshHandle handle);
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
	PsyXNativeInstance instances[PSYX_NATIVE_MAX_INSTANCES];
	PsyXNativeSnapshot snapshot;
	bool snapshotPending;
	bool requested;
	uint64_t generation;
	uint64_t ownedBytes;
	uint32_t rejected;
	bool generationExhausted;
};

#endif
