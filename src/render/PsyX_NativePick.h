#ifndef PSYX_NATIVE_PICK_INTERNAL_H
#define PSYX_NATIVE_PICK_INTERNAL_H

#include "PsyX_NativeScene.h"
#include <cstring>

/* Bounded frame-owned ID lookup. Capture precedes ConsumeSnapshot; completion
 * follows the presenter's fence. A later publish cannot change the picked key.
 * Kept independent of Vulkan so reset/supersession/stale rules can be tested. */
class PsyXNativePick
{
public:
	PsyXNativePick() : serial(0), pending(false), recorded(false), inFlight(false),
		windowX(0), windowY(0), windowWidth(0), windowHeight(0), count(0), captured() { Cancel(); }
	PsyXNativeResult Request(int x, int y, int width, int height)
	{
		if (x < 0 || y < 0 || width <= 0 || height <= 0 || x >= width || y >= height) return PSYX_NATIVE_INVALID;
		if (serial == UINT64_MAX) return PSYX_NATIVE_OUT_OF_BUDGET;
		++serial;
		result = Empty(PSYX_NATIVE_PENDING); result.requestSerial = serial;
		windowX=x; windowY=y; windowWidth=width; windowHeight=height; pending=true;
		return PSYX_NATIVE_OK;
	}
	void Cancel()
	{
		pending=false; recorded=false;
		result=Empty(PSYX_NATIVE_NO_SNAPSHOT);
	}
	bool Pending() const { return pending; }
	void Fail(PsyXNativeResult reason)
	{
		if (!pending) return;
		pending=false; result=Empty(reason); result.requestSerial=serial; recorded=false;
	}
	bool Capture(const PsyXNativeScene& scene, int width, int height, int drawableWidth, int drawableHeight,
		uint32_t& pixelX, uint32_t& pixelY)
	{
		if (!pending) return false;
		if (width!=windowWidth || height!=windowHeight || drawableWidth<=0 || drawableHeight<=0)
		{ Fail(PSYX_NATIVE_STALE); return false; }
		pixelX=uint32_t(uint64_t(windowX)*uint32_t(drawableWidth)/uint32_t(width));
		pixelY=uint32_t(uint64_t(windowY)*uint32_t(drawableHeight)/uint32_t(height));
		captured=Empty(PSYX_NATIVE_PENDING); captured.requestSerial=serial;
		captured.sceneGeneration=scene.generation; captured.simulationTick=scene.snapshot.simulationTick;
		count=scene.snapshot.instanceCount;
		for (uint32_t i=0; i<count; ++i)
		{
			const PsyXNativeInstance& instance=scene.instances[i];
			records[i].identity=instance.identity; records[i].mesh=instance.mesh;
			records[i].material=scene.meshes[instance.mesh.slot].material;
			records[i].world=instance.layer==PSYX_NATIVE_WORLD;
		}
		recorded=true;
		return true;
	}
	void Submitted(uint64_t submission)
	{
		if (!recorded) return;
		captured.submittedSerial=submission; recorded=false; inFlight=true; pending=false;
	}
	void Complete(uint32_t pixel, const PsyXNativeScene& scene)
	{
		if (!inFlight) return;
		inFlight=false;
		if (result.reason!=PSYX_NATIVE_PENDING || result.requestSerial!=captured.requestSerial) return;
		result=captured;
		if (scene.generation!=captured.sceneGeneration) { result.reason=PSYX_NATIVE_STALE; return; }
		result.reason=PSYX_NATIVE_OK;
		if (!pixel) return;
		if (pixel>count || !records[pixel-1].world) { result.reason=PSYX_NATIVE_INVALID; return; }
		const Record& record=records[pixel-1];
		if (!scene.IsLive(record.mesh) || (record.material.generation && !scene.IsMaterialLive(record.material)))
		{ result.reason=PSYX_NATIVE_STALE; return; }
		result.hit=1; result.identity=record.identity; result.mesh=record.mesh;
		result.material=record.material; result.instanceIndex=pixel-1;
	}
	PsyXNativeResult Get(const PsyXNativeScene& scene, PsyXNativePickResult* output) const
	{
		if (!output) return PSYX_NATIVE_INVALID;
		*output=result;
		if (output->reason==PSYX_NATIVE_OK &&
			(output->sceneGeneration!=scene.generation || (output->hit &&
			(!scene.IsLive(output->mesh) || (output->material.generation && !scene.IsMaterialLive(output->material))))))
		{ output->hit=0; output->reason=PSYX_NATIVE_STALE; }
		return output->reason;
	}
private:
	struct Record { uint64_t identity; PsyXNativeMeshHandle mesh; PsyXNativeMaterialHandle material; bool world; };
	Record records[PSYX_NATIVE_MAX_INSTANCES];
	uint64_t serial;
	bool pending, recorded, inFlight;
	int windowX, windowY, windowWidth, windowHeight;
	uint32_t count;
	PsyXNativePickResult result, captured;
	static PsyXNativePickResult Empty(PsyXNativeResult reason)
	{
		PsyXNativePickResult value={}; value.size=sizeof(value); value.version=PSYX_NATIVE_VERSION;
		value.reason=reason; return value;
	}
};
#endif
