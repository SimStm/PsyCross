// Included by the sole Vulkan owner through PsyX_VkNative.inl. Uploads happen
// after the frame fence; this bounded path waits on an upload-specific fence.
// No borrowed legacy RGBA texture IDs or independently owned device/presenter.

// Internal self-test seam: submit real work, but conservatively report a wait
// failure and an unsignalled fence until the test explicitly releases the hold.
// This cannot be enabled by a game setting or the public resource API.
static int g_nativeTestHoldUpload = 0;

static int ReleaseNativeUpload(VkNativeMaterial& material)
{
	if (material.uploadSubmitted)
	{
		const VkResult status = g_nativeTestHoldUpload ? VK_NOT_READY : vkGetFenceStatus(g_vk.device, material.uploadFence);
		if (status != VK_SUCCESS && status != VK_ERROR_DEVICE_LOST)
		{
			if (status != VK_NOT_READY) VkOk(status, "native upload fence status; resources retained");
			return 0;
		}
	}
	if (material.uploadCommand) vkFreeCommandBuffers(g_vk.device, g_vk.commandPool, 1, &material.uploadCommand);
	if (material.staging) vkDestroyBuffer(g_vk.device, material.staging, NULL);
	if (material.stagingMemory) vkFreeMemory(g_vk.device, material.stagingMemory, NULL);
	if (material.uploadFence) vkDestroyFence(g_vk.device, material.uploadFence, NULL);
	material.uploadCommand = VK_NULL_HANDLE;
	material.staging = VK_NULL_HANDLE; material.stagingMemory = VK_NULL_HANDLE;
	material.uploadFence = VK_NULL_HANDLE; material.uploadSubmitted = 0;
	return 1;
}

static int DestroyNativeMaterial(VkNativeMaterial& material)
{
	if (!ReleaseNativeUpload(material)) return 0;
	for (unsigned int filter=0; filter<3; ++filter)
	{
		if (material.sets[filter] && g_nativeVk.materialPool)
			VkOk(vkFreeDescriptorSets(g_vk.device, g_nativeVk.materialPool, 1, &material.sets[filter]), "native material descriptor release");
		if (material.samplers[filter]) vkDestroySampler(g_vk.device, material.samplers[filter], NULL);
	}
	if (material.view) vkDestroyImageView(g_vk.device, material.view, NULL);
	if (material.image) vkDestroyImage(g_vk.device, material.image, NULL);
	if (material.memory) vkFreeMemory(g_vk.device, material.memory, NULL);
	memset(&material, 0, sizeof(material));
	return 1;
}

static int CreateNativeMaterialDescriptors()
{
	if (g_nativeVk.materialPool) return 1;
	VkDescriptorSetLayoutBinding binding = {};
	binding.binding = 0;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	binding.descriptorCount = 1;
	binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	VkDescriptorSetLayoutCreateInfo layout = {};
	layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout.bindingCount = 1; layout.pBindings = &binding;
	if (!VkOk(vkCreateDescriptorSetLayout(g_vk.device, &layout, NULL, &g_nativeVk.materialLayout), "native material layout")) return 0;
	VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (PSYX_NATIVE_MAX_MATERIALS + 1)*3 };
	VkDescriptorPoolCreateInfo pool = {};
	pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool.maxSets = (PSYX_NATIVE_MAX_MATERIALS + 1)*3;
	pool.poolSizeCount = 1; pool.pPoolSizes = &size;
	if (!VkOk(vkCreateDescriptorPool(g_vk.device, &pool, NULL, &g_nativeVk.materialPool), "native material pool"))
	{
		vkDestroyDescriptorSetLayout(g_vk.device, g_nativeVk.materialLayout, NULL);
		g_nativeVk.materialLayout = VK_NULL_HANDLE;
		return 0;
	}
	return 1;
}

static int CreateNativeSampling(VkNativeMaterial& gpu, uint32_t levelCount, bool linearSupported)
{
	// Each descriptor is written once, before use, and retained with the image.
	// Runtime snapshots choose a set; no descriptor or sampler is mutated in flight.
	for (unsigned int filter=0; filter<3; ++filter)
	{
		if (filter && !linearSupported) continue;
		VkSamplerCreateInfo sampler = {};
		sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		sampler.magFilter = sampler.minFilter = filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
		sampler.mipmapMode = filter == PSYX_NATIVE_TRILINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
		sampler.maxLod=filter==PSYX_NATIVE_TRILINEAR ? float(levelCount-1) : 0;
		sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		if (!VkOk(vkCreateSampler(g_vk.device, &sampler, NULL, &gpu.samplers[filter]), "native material sampler")) return 0;
		VkDescriptorSetAllocateInfo allocation = {};
		allocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocation.descriptorPool = g_nativeVk.materialPool;
		allocation.descriptorSetCount = 1; allocation.pSetLayouts = &g_nativeVk.materialLayout;
		if (!VkOk(vkAllocateDescriptorSets(g_vk.device, &allocation, &gpu.sets[filter]), "native material descriptor")) return 0;
		VkDescriptorImageInfo image = { gpu.samplers[filter], gpu.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		VkWriteDescriptorSet write = {};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = gpu.sets[filter]; write.dstBinding = 0;
		write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		write.descriptorCount = 1; write.pImageInfo = &image;
		vkUpdateDescriptorSets(g_vk.device, 1, &write, 0, NULL);
	}
	return 1;
}

static int UploadNativeMaterial(VkNativeMaterial& gpu, const PsyXNativeScene::Material& source)
{
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(g_vk.physicalDevice, &properties);
	VkFormatProperties format;
	vkGetPhysicalDeviceFormatProperties(g_vk.physicalDevice, VK_FORMAT_R8G8B8A8_SRGB, &format);
	VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
	if (source.filter != PSYX_NATIVE_NEAREST) required |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
	if (source.width > properties.limits.maxImageDimension2D || source.height > properties.limits.maxImageDimension2D ||
		(format.optimalTilingFeatures & required) != required || !CreateNativeMaterialDescriptors()) return 0;

	// A retry must not overwrite retained handles from an earlier failed wait.
	if (!DestroyNativeMaterial(gpu)) return 0;
	void* mapped = NULL;
	const uint32_t levelCount=uint32_t(source.levels.size());
	int ok = CreateImage2DLevels(source.width, source.height, int(levelCount), VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &gpu.image, &gpu.memory);
	if (ok) ok = CreateBuffer(source.rgba.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&gpu.staging, &gpu.stagingMemory, &mapped);
	if (ok) memcpy(mapped, source.rgba.data(), source.rgba.size());
	if (mapped) vkUnmapMemory(g_vk.device, gpu.stagingMemory);
	if (ok)
	{
		VkCommandBufferAllocateInfo allocation = {};
		allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocation.commandPool = g_vk.commandPool;
		allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocation.commandBufferCount = 1;
		ok = VkOk(vkAllocateCommandBuffers(g_vk.device, &allocation, &gpu.uploadCommand), "native texture upload command");
	}
	if (ok)
	{
		VkCommandBufferBeginInfo begin = {};
		begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		ok = VkOk(vkBeginCommandBuffer(gpu.uploadCommand, &begin), "native texture upload begin");
	}
	if (ok)
	{
		ImageBarrierLevels(gpu.uploadCommand, gpu.image, VK_IMAGE_ASPECT_COLOR_BIT, 0, int(levelCount), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
		// 4096x4096 permits at most thirteen levels; no allocation inside recording.
		VkBufferImageCopy copies[13] = {};
		for (uint32_t level=0; level<levelCount; ++level)
		{
			VkBufferImageCopy& copy=copies[level];
			copy.bufferOffset=source.levels[level].offset;
			copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copy.imageSubresource.mipLevel=level;
			copy.imageSubresource.layerCount = 1;
			copy.imageExtent.width = source.levels[level].width;
			copy.imageExtent.height = source.levels[level].height; copy.imageExtent.depth = 1;
		}
		vkCmdCopyBufferToImage(gpu.uploadCommand, gpu.staging, gpu.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, levelCount, copies);
		ImageBarrierLevels(gpu.uploadCommand, gpu.image, VK_IMAGE_ASPECT_COLOR_BIT, 0, int(levelCount), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
		ok = VkOk(vkEndCommandBuffer(gpu.uploadCommand), "native texture upload end");
	}
	if (ok)
	{
		VkFenceCreateInfo fence = {};
		fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		ok = VkOk(vkCreateFence(g_vk.device, &fence, NULL, &gpu.uploadFence), "native texture upload fence");
	}
	if (ok)
	{
		VkSubmitInfo submit = {};
		submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit.commandBufferCount = 1; submit.pCommandBuffers = &gpu.uploadCommand;
		ok = VkOk(vkQueueSubmit(g_vk.queue, 1, &submit, gpu.uploadFence), "native texture upload submit");
		gpu.uploadSubmitted = ok;
		if (ok)
		{
			const VkResult wait = g_nativeTestHoldUpload ? VK_TIMEOUT :
				vkWaitForFences(g_vk.device, 1, &gpu.uploadFence, VK_TRUE, UINT64_MAX);
			ok = VkOk(wait, "native texture upload completion");
		}
	}
	if (!ReleaseNativeUpload(gpu)) ok = 0;
	if (ok) ok = CreateImageView2DLevels(gpu.image, VK_FORMAT_R8G8B8A8_SRGB, int(levelCount), &gpu.view);
	if (ok) ok=CreateNativeSampling(gpu,levelCount,(format.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)!=0);
	if (!ok) DestroyNativeMaterial(gpu);
	return ok;
}

static int PrepareNativeWhiteMaterial()
{
	if (g_nativeVk.white.sets[0]) return 1;
	const uint8_t white[] = {255,255,255,255};
	PsyXNativeScene::Material source;
	source.width = source.height = 1;
	source.rgba.assign(white,white+4);
	PsyXNativeMip::Layout(1,1,false,source.levels);
	return UploadNativeMaterial(g_nativeVk.white, source);
}

static void NativeMaterialsCompleted()
{
	for (uint32_t slot = 0; slot < PSYX_NATIVE_MAX_MATERIALS; ++slot)
	{
		if (g_nativeScene.CanReclaimMaterial(slot, g_nativeVk.completed))
		{
			if (DestroyNativeMaterial(g_nativeVk.materials[slot])) g_nativeScene.ReclaimMaterial(slot);
		}
		PsyXNativeScene::Material& source = g_nativeScene.materials[slot];
		if (source.state == PsyXNativeScene::Failed) DestroyNativeMaterial(g_nativeVk.materials[slot]);
		if (source.state != PsyXNativeScene::Pending) continue;
		source.state = UploadNativeMaterial(g_nativeVk.materials[slot], source) ? PsyXNativeScene::Ready : PsyXNativeScene::Failed;
		if (source.state==PsyXNativeScene::Ready) ++g_nativeVk.materialUploads;
	}
}

static void DestroyNativeMaterialDescriptors()
{
	DestroyNativeMaterial(g_nativeVk.white);
	if (g_nativeVk.materialPool) vkDestroyDescriptorPool(g_vk.device, g_nativeVk.materialPool, NULL);
	if (g_nativeVk.materialLayout) vkDestroyDescriptorSetLayout(g_vk.device, g_nativeVk.materialLayout, NULL);
	g_nativeVk.materialPool = VK_NULL_HANDLE; g_nativeVk.materialLayout = VK_NULL_HANDLE;
}

PsyXNativeResult PsyX_Native_CreateMaterial(const PsyXNativeMaterialDesc* desc, PsyXNativeMaterialHandle* handle)
{
	if (!g_vk.initialised || !g_vk.gameMode)
	{
		if (handle) memset(handle, 0, sizeof(*handle));
		return g_nativeScene.Reject(PSYX_NATIVE_UNSUPPORTED);
	}
	return g_nativeScene.CreateMaterial(desc, handle);
}
PsyXNativeResult PsyX_Native_DestroyMaterial(PsyXNativeMaterialHandle handle)
{
	return g_nativeScene.DestroyMaterial(handle);
}
