/* Included only by the Vulkan owner, after its allocation/shader helpers.
 * This pass contributor shares g_vk's device/command buffer/fence/swapchain.
 * Keeping the native slice together avoids adding another presentation owner. */

static PsyXNativeScene g_nativeScene;
struct VkNativeMesh
{
	VkBuffer vertices;
	VkDeviceMemory vertexMemory;
	VkBuffer indices;
	VkDeviceMemory indexMemory;
};
struct VkNativeMaterial
{
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
	VkSampler sampler;
	VkDescriptorSet set;
	VkBuffer staging;
	VkDeviceMemory stagingMemory;
	VkCommandBuffer uploadCommand;
	VkFence uploadFence;
	int uploadSubmitted;
};

static struct
{
	VkNativeMesh meshes[PSYX_NATIVE_MAX_MESHES];
	VkNativeMaterial materials[PSYX_NATIVE_MAX_MATERIALS];
	VkNativeMaterial white;
	VkDescriptorSetLayout materialLayout;
	VkDescriptorPool materialPool;
	VkRenderPass pass;
	VkPipelineLayout layout;
	VkPipeline pipelines[4]; // layer*2 + explicit cull policy
	VkImage depth;
	VkDeviceMemory depthMemory;
	VkImageView depthView;
	VkFramebuffer targets[8];
	uint64_t submitted;
	uint64_t completed;
	PsyXNativeStats frame;
} g_nativeVk;

#include "PsyX_VkNativeMaterials.inl"

static void DestroyNativeMesh(uint32_t slot)
{
	VkNativeMesh& mesh = g_nativeVk.meshes[slot];
	if (mesh.vertices) vkDestroyBuffer(g_vk.device, mesh.vertices, NULL);
	if (mesh.indices) vkDestroyBuffer(g_vk.device, mesh.indices, NULL);
	if (mesh.vertexMemory) vkFreeMemory(g_vk.device, mesh.vertexMemory, NULL);
	if (mesh.indexMemory) vkFreeMemory(g_vk.device, mesh.indexMemory, NULL);
	memset(&mesh, 0, sizeof(mesh));
}

static void DestroyNativeTargets()
{
	for (unsigned int i = 0; i < 8; ++i)
	{
		if (g_nativeVk.targets[i]) vkDestroyFramebuffer(g_vk.device, g_nativeVk.targets[i], NULL);
		g_nativeVk.targets[i] = VK_NULL_HANDLE;
	}
	if (g_nativeVk.depthView) vkDestroyImageView(g_vk.device, g_nativeVk.depthView, NULL);
	if (g_nativeVk.depth) vkDestroyImage(g_vk.device, g_nativeVk.depth, NULL);
	if (g_nativeVk.depthMemory) vkFreeMemory(g_vk.device, g_nativeVk.depthMemory, NULL);
	g_nativeVk.depthView = VK_NULL_HANDLE;
	g_nativeVk.depth = VK_NULL_HANDLE;
	g_nativeVk.depthMemory = VK_NULL_HANDLE;
}

static int CreateNativeTargets(uint32_t width, uint32_t height)
{
	if (!g_nativeVk.pass) return 1; // lazy opt-in; legacy has no extra allocations
	if (!CreateImage2D(width, height, VK_FORMAT_D32_SFLOAT,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &g_nativeVk.depth, &g_nativeVk.depthMemory) ||
		!CreateImageView2D(g_nativeVk.depth, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, &g_nativeVk.depthView))
	{
		DestroyNativeTargets();
		return 0;
	}
	for (uint32_t i = 0; i < g_vk.swapchainImageCount; ++i)
	{
		VkImageView attachments[] = { g_vk.swapchainViews[i], g_nativeVk.depthView };
		VkFramebufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		info.renderPass = g_nativeVk.pass;
		info.attachmentCount = 2;
		info.pAttachments = attachments;
		info.width = width;
		info.height = height;
		info.layers = 1;
		if (!VkOk(vkCreateFramebuffer(g_vk.device, &info, NULL, &g_nativeVk.targets[i]), "native framebuffer"))
		{
			DestroyNativeTargets();
			return 0;
		}
	}
	return 1;
}

static int CreateNativePass()
{
	VkFormatProperties support;
	vkGetPhysicalDeviceFormatProperties(g_vk.physicalDevice, VK_FORMAT_D32_SFLOAT, &support);
	if (!(support.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) return 0;
	VkAttachmentDescription attachments[2] = {};
	attachments[0].format = g_vk.swapchainFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[1].format = VK_FORMAT_D32_SFLOAT;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	VkAttachmentReference color = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
	VkAttachmentReference depth = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color;
	subpass.pDepthStencilAttachment = &depth;
	VkSubpassDependency dependencies[2] = {};
	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
		VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	VkRenderPassCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	info.attachmentCount = 2;
	info.pAttachments = attachments;
	info.subpassCount = 1;
	info.pSubpasses = &subpass;
	info.dependencyCount = 2;
	info.pDependencies = dependencies;
	return VkOk(vkCreateRenderPass(g_vk.device, &info, NULL, &g_nativeVk.pass), "native render pass");
}

static int CreateNativePipeline()
{
	if (!PrepareNativeWhiteMaterial()) return 0;
	VkPushConstantRange push = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 88 };
	VkPipelineLayoutCreateInfo layout = {};
	layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges = &push;
	layout.setLayoutCount = 1;
	layout.pSetLayouts = &g_nativeVk.materialLayout;
	if (!VkOk(vkCreatePipelineLayout(g_vk.device, &layout, NULL, &g_nativeVk.layout), "native pipeline layout")) return 0;
	VkShaderModule vertex = CreateShaderModuleFromSpirv(native_vert_spv, native_vert_spv_size);
	VkShaderModule fragment = CreateShaderModuleFromSpirv(native_frag_spv, native_frag_spv_size);
	VkPipelineShaderStageCreateInfo stages[2] = {};
	stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vertex;
	stages[0].pName = "main";
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fragment;
	stages[1].pName = "main";
	VkVertexInputBindingDescription binding = { 0, sizeof(PsyXNativeVertex), VK_VERTEX_INPUT_RATE_VERTEX };
	VkVertexInputAttributeDescription attributes[] = {
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 }, { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12 },
		{ 2, 0, VK_FORMAT_R32G32_SFLOAT, 24 }, { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 32 }
	};
	VkPipelineVertexInputStateCreateInfo input = {};
	input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	input.vertexBindingDescriptionCount = 1;
	input.pVertexBindingDescriptions = &binding;
	input.vertexAttributeDescriptionCount = 4;
	input.pVertexAttributeDescriptions = attributes;
	VkPipelineInputAssemblyStateCreateInfo assembly = {};
	assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPipelineViewportStateCreateInfo viewport = {};
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = viewport.scissorCount = 1;
	VkPipelineRasterizationStateCreateInfo raster = {};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_BACK_BIT;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.lineWidth = 1;
	VkPipelineMultisampleStateCreateInfo samples = {};
	samples.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo depth = {};
	depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth.depthTestEnable = depth.depthWriteEnable = VK_TRUE;
	depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	VkPipelineColorBlendAttachmentState attachment = {};
	attachment.colorWriteMask = 15;
	VkPipelineColorBlendStateCreateInfo blend = {};
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 1;
	blend.pAttachments = &attachment;
	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamic = {};
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dynamicStates;
	VkGraphicsPipelineCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	info.stageCount = 2;
	info.pStages = stages;
	info.pVertexInputState = &input;
	info.pInputAssemblyState = &assembly;
	info.pViewportState = &viewport;
	info.pRasterizationState = &raster;
	info.pMultisampleState = &samples;
	info.pDepthStencilState = &depth;
	info.pColorBlendState = &blend;
	info.pDynamicState = &dynamic;
	info.layout = g_nativeVk.layout;
	info.renderPass = g_nativeVk.pass;
	int ok = vertex && fragment;
	for (unsigned int pipeline = 0; pipeline < 4 && ok; ++pipeline)
	{
		const unsigned int cull = pipeline%2;
		depth.depthTestEnable = depth.depthWriteEnable = pipeline/2 == PSYX_NATIVE_WORLD ? VK_TRUE : VK_FALSE;
		raster.cullMode = cull == PSYX_NATIVE_CULL_NONE ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
		ok = VkOk(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &info, NULL,
			&g_nativeVk.pipelines[pipeline]), "native world/backdrop pipeline");
	}
	if (vertex) vkDestroyShaderModule(g_vk.device, vertex, NULL);
	if (fragment) vkDestroyShaderModule(g_vk.device, fragment, NULL);
	return ok;
}

static void DestroyNativePipeline()
{
	for (unsigned int pipeline = 0; pipeline < 4; ++pipeline)
	{
		if (g_nativeVk.pipelines[pipeline]) vkDestroyPipeline(g_vk.device, g_nativeVk.pipelines[pipeline], NULL);
		g_nativeVk.pipelines[pipeline] = VK_NULL_HANDLE;
	}
	if (g_nativeVk.layout) vkDestroyPipelineLayout(g_vk.device, g_nativeVk.layout, NULL);
	if (g_nativeVk.pass) vkDestroyRenderPass(g_vk.device, g_nativeVk.pass, NULL);
	g_nativeVk.layout = VK_NULL_HANDLE;
	g_nativeVk.pass = VK_NULL_HANDLE;
}

static void NativeCompleted()
{
	// Called only after the owner's fence or device-idle succeeds. Persistent
	// CPU payloads never require reads from GPU-mapped memory.
	g_nativeVk.completed = g_nativeVk.submitted;
	NativeMaterialsCompleted();
	for (uint32_t slot = 0; slot < PSYX_NATIVE_MAX_MESHES; ++slot)
	{
		if (g_nativeScene.CanReclaim(slot, g_nativeVk.completed))
		{
			DestroyNativeMesh(slot);
			g_nativeScene.Reclaim(slot);
		}
		PsyXNativeScene::Mesh& source = g_nativeScene.meshes[slot];
		if (source.state != PsyXNativeScene::Pending) continue;
		VkNativeMesh& mesh = g_nativeVk.meshes[slot];
		void* vertices = NULL;
		void* indices = NULL;
		const int vertexOk = CreateBuffer(source.vertices.size() * sizeof(PsyXNativeVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&mesh.vertices, &mesh.vertexMemory, &vertices);
		const int indexOk = vertexOk && CreateBuffer(source.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&mesh.indices, &mesh.indexMemory, &indices);
		if (vertexOk && indexOk)
		{
			memcpy(vertices, source.vertices.data(), source.vertices.size() * sizeof(PsyXNativeVertex));
			memcpy(indices, source.indices.data(), source.indices.size() * sizeof(uint32_t));
		}
		if (vertices) vkUnmapMemory(g_vk.device, mesh.vertexMemory);
		if (indices) vkUnmapMemory(g_vk.device, mesh.indexMemory);
		if (!vertexOk || !indexOk) DestroyNativeMesh(slot);
		source.state = vertexOk && indexOk ? PsyXNativeScene::Ready : PsyXNativeScene::Failed;
	}
}

static PsyXNativeResult PrepareNativeFrame()
{
	g_nativeScene.GetStats(&g_nativeVk.frame);
	if (!g_nativeScene.requested) return PSYX_NATIVE_OK;
	if (!g_vk.gameMode) return PSYX_NATIVE_UNSUPPORTED;
	if (!g_nativeScene.snapshotPending) return PSYX_NATIVE_NO_SNAPSHOT;
	if (g_vk.psx.modernSceneBoundary < 0) return PSYX_NATIVE_NO_BOUNDARY;
	for (uint32_t i = 0; i < g_nativeScene.snapshot.instanceCount; ++i)
	{
		const PsyXNativeMeshHandle handle = g_nativeScene.instances[i].mesh;
		if (!g_nativeScene.IsLive(handle)) return PSYX_NATIVE_STALE;
		const PsyXNativeScene::State state = g_nativeScene.meshes[handle.slot].state;
		if (state == PsyXNativeScene::Failed) return PSYX_NATIVE_UPLOAD_FAILED;
		if (state != PsyXNativeScene::Ready) return PSYX_NATIVE_PENDING;
		const PsyXNativeMaterialHandle material = g_nativeScene.meshes[handle.slot].material;
		if (material.generation)
		{
			if (!g_nativeScene.IsMaterialLive(material)) return PSYX_NATIVE_STALE;
			const PsyXNativeScene::State materialState = g_nativeScene.materials[material.slot].state;
			if (materialState == PsyXNativeScene::Failed) return PSYX_NATIVE_UPLOAD_FAILED;
			if (materialState != PsyXNativeScene::Ready) return PSYX_NATIVE_PENDING;
		}
	}
	if (!g_nativeVk.pass && (!CreateNativePass() || !CreateNativePipeline()))
	{
		DestroyNativePipeline();
		return PSYX_NATIVE_UNSUPPORTED;
	}
	if (!g_nativeVk.depth && !CreateNativeTargets(g_vk.width, g_vk.height)) return PSYX_NATIVE_UPLOAD_FAILED;
	return PSYX_NATIVE_OK;
}

static void NativeMultiply(float out[16], const float a[16], const float b[16])
{
	for (unsigned int column = 0; column < 4; ++column)
		for (unsigned int row = 0; row < 4; ++row)
		{
			out[column * 4 + row] = 0;
			for (unsigned int k = 0; k < 4; ++k) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
		}
}

static void RecordNativeWorld(VkCommandBuffer cmd, uint32_t imageIndex)
{
	VkClearValue clears[2] = {};
	// Explicit unlit SDR backdrop; no legacy sky or depth participates.
	clears[0].color.float32[0] = 0.025f;
	clears[0].color.float32[1] = 0.035f;
	clears[0].color.float32[2] = 0.055f;
	clears[0].color.float32[3] = 1;
	clears[1].depthStencil.depth = 1;
	VkRenderPassBeginInfo begin = {};
	begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin.renderPass = g_nativeVk.pass;
	begin.framebuffer = g_nativeVk.targets[imageIndex];
	begin.renderArea.extent.width = g_vk.width;
	begin.renderArea.extent.height = g_vk.height;
	begin.clearValueCount = 2;
	begin.pClearValues = clears;
	vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
	VkViewport viewport = { 0, 0, float(g_vk.width), float(g_vk.height), 0, 1 };
	VkRect2D scissor = { { 0, 0 }, { uint32_t(g_vk.width), uint32_t(g_vk.height) } };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	VkPipeline activePipeline = VK_NULL_HANDLE;
	float viewProjection[2][16], backdropView[16];
	memcpy(backdropView, g_nativeScene.snapshot.view.view, sizeof(backdropView));
	backdropView[12] = backdropView[13] = backdropView[14] = 0;
	NativeMultiply(viewProjection[PSYX_NATIVE_WORLD], g_nativeScene.snapshot.view.projection, g_nativeScene.snapshot.view.view);
	NativeMultiply(viewProjection[PSYX_NATIVE_BACKDROP], g_nativeScene.snapshot.view.projection, backdropView);
	const PsyXNativeLayer order[] = { PSYX_NATIVE_BACKDROP, PSYX_NATIVE_WORLD };
	for (unsigned int pass = 0; pass < 2; ++pass)
	for (uint32_t i = 0; i < g_nativeScene.snapshot.instanceCount; ++i)
	{
		const PsyXNativeInstance& instance = g_nativeScene.instances[i];
		if (instance.layer != order[pass]) continue;
		PsyXNativeScene::Mesh& source = g_nativeScene.meshes[instance.mesh.slot];
		const VkNativeMesh& mesh = g_nativeVk.meshes[instance.mesh.slot];
		struct { float mvp[16]; float tint[4]; uint32_t encodeSRGB; float alphaCutoff; } push;
		NativeMultiply(push.mvp, viewProjection[instance.layer], instance.world);
		memcpy(push.tint, instance.tint, sizeof(push.tint));
		push.encodeSRGB = g_vk.srgbOutput ? 0 : 1;
		const bool textured = source.material.generation != 0;
		const unsigned int cull = textured ? g_nativeScene.materials[source.material.slot].cull : PSYX_NATIVE_CULL_BACK;
		const unsigned int pipeline = unsigned(instance.layer)*2+cull;
		if (activePipeline != g_nativeVk.pipelines[pipeline])
		{
			activePipeline = g_nativeVk.pipelines[pipeline];
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);
		}
		push.alphaCutoff = textured ? g_nativeScene.materials[source.material.slot].alphaCutoff : 0;
		const VkDescriptorSet descriptor = textured ? g_nativeVk.materials[source.material.slot].set : g_nativeVk.white.set;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_nativeVk.layout, 0, 1, &descriptor, 0, NULL);
		vkCmdPushConstants(cmd, g_nativeVk.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertices, &offset);
		vkCmdBindIndexBuffer(cmd, mesh.indices, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(cmd, uint32_t(source.indices.size()), 1, 0, 0, 0);
		++g_nativeVk.frame.nativeDraws;
		if (instance.layer == PSYX_NATIVE_BACKDROP) ++g_nativeVk.frame.nativeBackdropDraws;
	}
	vkCmdEndRenderPass(cmd);
}

static void NativeSubmitted()
{
	++g_nativeVk.submitted;
	if (g_nativeVk.frame.effective)
		for (uint32_t i = 0; i < g_nativeScene.snapshot.instanceCount; ++i)
		{
			PsyXNativeScene::Mesh& mesh = g_nativeScene.meshes[g_nativeScene.instances[i].mesh.slot];
			mesh.lastSubmission = g_nativeVk.submitted;
			if (mesh.material.generation) g_nativeScene.materials[mesh.material.slot].lastSubmission = g_nativeVk.submitted;
		}
	g_nativeVk.frame.submittedSerial = g_nativeVk.submitted;
	g_nativeVk.frame.completedSerial = g_nativeVk.completed;
	g_nativeScene.ConsumeSnapshot();
}

static void NativeShutdown()
{
	// The owner has confirmed device idle (or loss). This proof also covers
	// uploads submitted after the last ordinary frame fence was signalled.
	for (uint32_t slot = 0; slot < PSYX_NATIVE_MAX_MATERIALS; ++slot)
		g_nativeVk.materials[slot].uploadSubmitted = 0;
	g_nativeVk.white.uploadSubmitted = 0;
	g_nativeScene.Reset();
	g_nativeScene.SetRequested(0);
	NativeCompleted();
	DestroyNativeTargets();
	DestroyNativePipeline();
	DestroyNativeMaterialDescriptors();
	memset(&g_nativeVk, 0, sizeof(g_nativeVk));
}

PsyXNativeResult PsyX_Native_CreateMesh(const PsyXNativeMeshDesc* desc, PsyXNativeMeshHandle* handle)
{
	if (!g_vk.initialised || !g_vk.gameMode)
	{
		if (handle) memset(handle, 0, sizeof(*handle));
		return g_nativeScene.Reject(PSYX_NATIVE_UNSUPPORTED);
	}
	return g_nativeScene.Create(desc, handle);
}

PsyXNativeResult PsyX_Native_DestroyMesh(PsyXNativeMeshHandle handle) { return g_nativeScene.Destroy(handle); }
PsyXNativeResult PsyX_Native_Publish(const PsyXNativeSnapshot* snapshot)
{
	if (!g_vk.initialised || !g_vk.gameMode) return g_nativeScene.Reject(PSYX_NATIVE_UNSUPPORTED);
	return g_nativeScene.Publish(snapshot);
}
void PsyX_Native_SetRequested(int requested) { g_nativeScene.SetRequested(requested); }
uint64_t PsyX_Native_ResetScene() { return g_nativeScene.Reset(); }
uint64_t PsyX_Native_GetSceneGeneration() { return g_nativeScene.generation; }
void PsyX_Native_GetStats(PsyXNativeStats* stats)
{
	if (!stats) return;
	g_nativeScene.GetStats(stats);
	stats->effective = g_nativeVk.frame.effective;
	stats->reason = g_nativeVk.frame.reason;
	stats->nativeDraws = g_nativeVk.frame.nativeDraws;
	stats->nativeBackdropDraws = g_nativeVk.frame.nativeBackdropDraws;
	stats->legacyWorldDraws = g_nativeVk.frame.legacyWorldDraws;
	stats->legacyOverlayDraws = g_nativeVk.frame.legacyOverlayDraws;
	stats->suppressedWorldDraws = g_nativeVk.frame.suppressedWorldDraws;
	stats->submittedSerial = g_nativeVk.submitted;
	stats->completedSerial = g_nativeVk.completed;
	stats->simulationTick = g_nativeVk.frame.simulationTick;
	if (stats->requested && (!g_vk.initialised || !g_vk.gameMode)) stats->reason = PSYX_NATIVE_UNSUPPORTED;
}
