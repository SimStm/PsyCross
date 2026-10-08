/* Included only by the Vulkan owner, after its allocation/shader helpers.
 * This pass contributor shares g_vk's device/command buffer/fence/swapchain.
 * Keeping the native slice together avoids adding another presentation owner. */

static PsyXNativeScene g_nativeScene;
#include "PsyX_NativePick.h"
#include "PsyX_NativeTransform.h"
#include "PsyX_NativePlane.h"
static PsyXNativePick g_nativePick;
static int g_nativePickReadbackFault = 0; // actual-allocation rollback acceptance hook
struct NativeDrawPush
{
	float mvp[16], tint[4];
	uint32_t encodeSRGB;
	float alphaCutoff;
	uint32_t pickId, sampling;
	float depthPlane[4];
};
static_assert(sizeof(NativeDrawPush)==112, "Native push constants must match GLSL alignment");
static_assert(offsetof(NativeDrawPush,sampling)==92, "Native sampling must use the former GLSL padding word");
static_assert(offsetof(NativeDrawPush,depthPlane)==96, "Native plane must start at the GLSL vec4 offset");
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
	VkSampler samplers[3];
	VkDescriptorSet sets[3];
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
	VkImage pickImage;
	VkDeviceMemory pickImageMemory;
	VkImageView pickView;
	VkBuffer pickBuffer;
	VkDeviceMemory pickBufferMemory;
	uint32_t* pickMapped; // one pixel, read only after the owner fence
	VkFramebuffer targets[8];
	uint64_t submitted;
	uint64_t completed;
	uint64_t effectiveFrames, failedFrames, loadingFrames, auxiliaryFrames, materialUploads;
	PsyXNativeStats frame;
	double clipToWorld[16];
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
	if (g_nativeVk.pickView) vkDestroyImageView(g_vk.device, g_nativeVk.pickView, NULL);
	if (g_nativeVk.pickImage) vkDestroyImage(g_vk.device, g_nativeVk.pickImage, NULL);
	if (g_nativeVk.pickImageMemory) vkFreeMemory(g_vk.device, g_nativeVk.pickImageMemory, NULL);
	g_nativeVk.pickView = VK_NULL_HANDLE; g_nativeVk.pickImage = VK_NULL_HANDLE;
	g_nativeVk.pickImageMemory = VK_NULL_HANDLE;
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
	if (!CreateImage2D(width, height, VK_FORMAT_R32_UINT,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		&g_nativeVk.pickImage, &g_nativeVk.pickImageMemory) ||
		!CreateImageView2D(g_nativeVk.pickImage, VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, &g_nativeVk.pickView))
	{ DestroyNativeTargets(); return 0; }
	for (uint32_t i = 0; i < g_vk.swapchainImageCount; ++i)
	{
		VkImageView attachments[] = { g_vk.swapchainViews[i], g_nativeVk.pickView, g_nativeVk.depthView };
		VkFramebufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		info.renderPass = g_nativeVk.pass;
		info.attachmentCount = 3;
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
	vkGetPhysicalDeviceFormatProperties(g_vk.physicalDevice, VK_FORMAT_R32_UINT, &support);
	if (!(support.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) return 0;
	VkImageFormatProperties pickSupport;
	if (vkGetPhysicalDeviceImageFormatProperties(g_vk.physicalDevice, VK_FORMAT_R32_UINT, VK_IMAGE_TYPE_2D,
		VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		0, &pickSupport) != VK_SUCCESS) return 0;
	VkAttachmentDescription attachments[3] = {};
	attachments[0].format = g_vk.swapchainFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[1].format = VK_FORMAT_R32_UINT;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	attachments[2] = attachments[1];
	attachments[2].format = VK_FORMAT_D32_SFLOAT;
	attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	VkAttachmentReference colors[] = { { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL }, { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL } };
	VkAttachmentReference depth = { 2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 2;
	subpass.pColorAttachments = colors;
	subpass.pDepthStencilAttachment = &depth;
	VkSubpassDependency dependencies[2] = {};
	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
		VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
	VkRenderPassCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	info.attachmentCount = 3;
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
	VkPushConstantRange push = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(NativeDrawPush) };
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
	raster.depthBiasEnable = VK_TRUE;
	VkPipelineMultisampleStateCreateInfo samples = {};
	samples.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo depth = {};
	depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth.depthTestEnable = depth.depthWriteEnable = VK_TRUE;
	depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	VkPipelineColorBlendAttachmentState attachments[2] = {};
	// Identical disabled blend/write masks avoid requiring independentBlend.
	attachments[0].colorWriteMask = attachments[1].colorWriteMask = 15;
	VkPipelineColorBlendStateCreateInfo blend = {};
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 2;
	blend.pAttachments = attachments;
	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
	VkPipelineDynamicStateCreateInfo dynamic = {};
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 3;
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
	if (g_nativeVk.pickMapped) g_nativePick.Complete(*g_nativeVk.pickMapped, g_nativeScene);
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

static unsigned int NativeSamplingFilter(const PsyXNativeInstance& instance, PsyXNativeFilter materialFilter)
{
	return instance.sampling==PSYX_NATIVE_USE_MATERIAL_FILTER ? unsigned(materialFilter) : unsigned(instance.sampling)-1;
}

static PsyXNativeResult PrepareNativeFrame()
{
	g_nativeScene.GetStats(&g_nativeVk.frame);
	if (!g_nativeScene.requested) return PSYX_NATIVE_OK;
	if (!g_vk.gameMode) return PSYX_NATIVE_UNSUPPORTED;
	if (!g_nativeScene.worldExpected && !g_nativeScene.snapshotPending && g_vk.psx.modernSceneBoundary < 0)
		return PSYX_NATIVE_OK; // remaining frontend/loading consumers, not a world fallback
	if (g_nativeScene.worldExpected && g_nativeScene.producerReason != PSYX_NATIVE_OK)
		return g_nativeScene.producerReason;
	if (!g_nativeScene.snapshotPending) return PSYX_NATIVE_NO_SNAPSHOT;
	if (g_vk.psx.modernSceneBoundary < 0) return PSYX_NATIVE_NO_BOUNDARY;
	if (!PsyXNativePlane::Inverse(g_nativeVk.clipToWorld,g_nativeScene.snapshot.view.projection,
		g_nativeScene.snapshot.view.view)) return PSYX_NATIVE_INVALID;
	for (uint32_t i = 0; i < g_nativeScene.snapshot.instanceCount; ++i)
	{
		const PsyXNativeMeshHandle handle = g_nativeScene.instances[i].mesh;
		if (!g_nativeScene.IsLive(handle)) return PSYX_NATIVE_STALE;
		const PsyXNativeScene::Mesh& source=g_nativeScene.meshes[handle.slot];
		if (g_nativeScene.instances[i].layer==PSYX_NATIVE_WORLD)
		{
			for (size_t range=0; range<source.drawRanges.size(); ++range)
			{
				if (!source.drawRanges[range].horizontalPlane) continue;
				float plane[4];
				if (PsyXNativePlane::Horizontal(plane,g_nativeVk.clipToWorld,g_nativeScene.instances[i].world,
					source.drawRanges[range].planeY,g_vk.width,g_vk.height,source.depthLayer)==PsyXNativePlane::Invalid)
					return PSYX_NATIVE_INVALID;
			}
		}
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
			const unsigned int filter=NativeSamplingFilter(g_nativeScene.instances[i],g_nativeScene.materials[material.slot].filter);
			if (!g_nativeVk.materials[material.slot].sets[filter]) return PSYX_NATIVE_UNSUPPORTED;
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

static bool NativeWorldSelected()
{
	return g_nativeVk.frame.frameState == PSYX_NATIVE_FRAME_READY ||
		g_nativeVk.frame.frameState == PSYX_NATIVE_FRAME_LOADING ||
		g_nativeVk.frame.frameState == PSYX_NATIVE_FRAME_FAILED;
}

static void NativeFrameDiagnostic()
{
	static PsyXNativeFrameState lastState = PSYX_NATIVE_FRAME_OFF;
	static PsyXNativeResult lastReason = PSYX_NATIVE_OK;
	const PsyXNativeStats& frame = g_nativeVk.frame;
	if (frame.frameState == lastState && frame.reason == lastReason) return;
	lastState = frame.frameState; lastReason = frame.reason;
	if (!NativeWorldSelected()) return;
	const char* state = frame.frameState == PSYX_NATIVE_FRAME_READY ? "ready" :
		frame.frameState == PSYX_NATIVE_FRAME_LOADING ? "loading" : "FAILED";
	PsyX_Log_Info("Native world %s: %s; scene%llu tick%llu meshes ready%u/pending%u materials ready%u/pending%u; legacy world disabled\n",
		state, PsyX_Native_ResultName(frame.reason), (unsigned long long)frame.sceneGeneration,
		(unsigned long long)frame.simulationTick, frame.residentMeshes, frame.pendingMeshes,
		frame.residentMaterials, frame.pendingMaterials);
}

static void DrawNativeFrameDiagnostic()
{
	const PsyXNativeStats& frame = g_nativeVk.frame;
	if (frame.frameState != PSYX_NATIVE_FRAME_LOADING && frame.frameState != PSYX_NATIVE_FRAME_FAILED) return;
	const float displayWidth = ImGui::GetIO().DisplaySize.x;
	ImGui::SetNextWindowPos(ImVec2(displayWidth-12,12), ImGuiCond_Always, ImVec2(1,0));
	// Known dimensions avoid ImGui hiding the first failed frame for automatic
	// size measurement. Position uses logical UI coordinates, not drawable pixels.
	ImGui::SetNextWindowSize(ImVec2(displayWidth < 624 ? displayWidth-24 : 600,160), ImGuiCond_Always);
	ImGui::Begin("Native renderer status", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoInputs);
	ImGui::TextUnformatted(frame.frameState == PSYX_NATIVE_FRAME_LOADING ? "Loading native world" : "Native world rendering FAILED");
	ImGui::TextWrapped("%s", PsyX_Native_ResultName(frame.reason));
	ImGui::Text("Scene %llu | tick %llu | ready meshes %u | pending meshes %u",
		(unsigned long long)frame.sceneGeneration, (unsigned long long)frame.simulationTick,
		frame.residentMeshes, frame.pendingMeshes);
	ImGui::TextWrapped("Legacy world recovery is disabled. F11 shows resources and migration coverage; see the game log for source errors.");
	ImGui::End();
}

static int PrepareNativePickBuffer()
{
	if (g_nativeVk.pickBuffer) return 1;
	VkBuffer buffer=VK_NULL_HANDLE; VkDeviceMemory memory=VK_NULL_HANDLE; void* mapped=NULL;
	int ready=CreateBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, &memory, &mapped);
	if (g_nativePickReadbackFault) { g_nativePickReadbackFault=0; ready=0; }
	if (!ready)
	{
		// CreateBuffer can fail after creating/binding/mapping a partial payload.
		// Nothing has been submitted: destroy every successful step now, and do
		// not leave a non-null unbound buffer that a later click could reuse.
		if (mapped) vkUnmapMemory(g_vk.device,memory);
		if (buffer) vkDestroyBuffer(g_vk.device,buffer,NULL);
		if (memory) vkFreeMemory(g_vk.device,memory,NULL);
		return 0;
	}
	g_nativeVk.pickBuffer=buffer; g_nativeVk.pickBufferMemory=memory;
	g_nativeVk.pickMapped=static_cast<uint32_t*>(mapped);
	return 1;
}

static void RecordNativePick(VkCommandBuffer cmd)
{
	if (!g_nativePick.Pending()) return;
	if (!PrepareNativePickBuffer()) { g_nativePick.Fail(PSYX_NATIVE_UPLOAD_FAILED); return; }
	int width=0, height=0; SDL_GetWindowSize(g_vk.window, &width, &height);
	uint32_t x=0, y=0;
	if (!g_nativePick.Capture(g_nativeScene, width, height, g_vk.width, g_vk.height, x, y)) return;
	VkBufferImageCopy copy={};
	copy.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
	copy.imageSubresource.layerCount=1;
	copy.imageOffset.x=int32_t(x); copy.imageOffset.y=int32_t(y);
	copy.imageExtent.width=copy.imageExtent.height=copy.imageExtent.depth=1;
	vkCmdCopyImageToBuffer(cmd, g_nativeVk.pickImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		g_nativeVk.pickBuffer, 1, &copy);
	VkBufferMemoryBarrier barrier={}; barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
	barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer=g_nativeVk.pickBuffer; barrier.size=sizeof(uint32_t);
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
		0, NULL, 1, &barrier, 0, NULL);
}

static void RecordNativeWorld(VkCommandBuffer cmd, uint32_t imageIndex)
{
	VkClearValue clears[3] = {};
	// Explicit unlit SDR backdrop; no legacy sky or depth participates.
	clears[0].color.float32[0] = 0.025f;
	clears[0].color.float32[1] = 0.035f;
	clears[0].color.float32[2] = 0.055f;
	clears[0].color.float32[3] = 1;
	clears[2].depthStencil.depth = 1; // integer pick attachment stays zero (miss)
	VkRenderPassBeginInfo begin = {};
	begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin.renderPass = g_nativeVk.pass;
	begin.framebuffer = g_nativeVk.targets[imageIndex];
	begin.renderArea.extent.width = g_vk.width;
	begin.renderArea.extent.height = g_vk.height;
	begin.clearValueCount = 3;
	begin.pClearValues = clears;
	vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
	VkViewport viewport = { 0, 0, float(g_vk.width), float(g_vk.height), 0, 1 };
	VkRect2D scissor = { { 0, 0 }, { uint32_t(g_vk.width), uint32_t(g_vk.height) } };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	VkPipeline activePipeline = VK_NULL_HANDLE;
	float backdropView[16];
	memcpy(backdropView, g_nativeScene.snapshot.view.view, sizeof(backdropView));
	backdropView[12] = backdropView[13] = backdropView[14] = 0;
	const PsyXNativeLayer order[] = { PSYX_NATIVE_BACKDROP, PSYX_NATIVE_WORLD };
	for (unsigned int pass = 0; pass < 2; ++pass)
	for (uint32_t i = 0; i < g_nativeScene.snapshot.instanceCount; ++i)
	{
		const PsyXNativeInstance& instance = g_nativeScene.instances[i];
		if (instance.layer != order[pass]) continue;
		PsyXNativeScene::Mesh& source = g_nativeScene.meshes[instance.mesh.slot];
		const VkNativeMesh& mesh = g_nativeVk.meshes[instance.mesh.slot];
		NativeDrawPush push={};
		push.pickId = instance.layer == PSYX_NATIVE_WORLD ? i+1 : 0;
		PsyXNativeTransform::Compose(push.mvp,g_nativeScene.snapshot.view.projection,
			instance.layer==PSYX_NATIVE_BACKDROP ? backdropView : g_nativeScene.snapshot.view.view,instance.world);
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
		const unsigned int filter=textured ? NativeSamplingFilter(instance,g_nativeScene.materials[source.material.slot].filter) : PSYX_NATIVE_NEAREST;
		push.sampling=filter;
		const VkDescriptorSet descriptor = textured ? g_nativeVk.materials[source.material.slot].sets[filter] : g_nativeVk.white.sets[filter];
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_nativeVk.layout, 0, 1, &descriptor, 0, NULL);
		const VkDeviceSize offset = 0;
		// Only declared coplanar artwork gets a small D32 representable bias.
		// Ordinary geometry and backdrop reset it; alpha/depth/ID share this draw.
		// Zero clamp/slope needs no optional device feature or geometric offset.
		vkCmdSetDepthBias(cmd, instance.layer == PSYX_NATIVE_WORLD ? -4.0f*source.depthLayer : 0.0f, 0.0f, 0.0f);
		vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertices, &offset);
		vkCmdBindIndexBuffer(cmd, mesh.indices, 0, VK_INDEX_TYPE_UINT32);
		const bool world = instance.layer == PSYX_NATIVE_WORLD;
		const size_t rangeCount = world ? source.drawRanges.size() : 1;
		for (size_t rangeIndex=0; rangeIndex<rangeCount; ++rangeIndex)
		{
			const PsyXNativeScene::DrawRange& range = source.drawRanges[rangeIndex];
			memset(push.depthPlane,0,sizeof(push.depthPlane)); push.depthPlane[3]=-1;
			if (world && range.horizontalPlane)
				PsyXNativePlane::Horizontal(push.depthPlane,g_nativeVk.clipToWorld,instance.world,
					range.planeY,g_vk.width,g_vk.height,source.depthLayer);
			vkCmdPushConstants(cmd, g_nativeVk.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
			vkCmdDrawIndexed(cmd, world ? range.indexCount : uint32_t(source.indices.size()), 1,
				world ? range.firstIndex : 0, 0, 0);
			++g_nativeVk.frame.nativeDraws;
			if (!world) ++g_nativeVk.frame.nativeBackdropDraws;
		}
	}
	vkCmdEndRenderPass(cmd);
	RecordNativePick(cmd);
}

static void NativeSubmitted()
{
	++g_nativeVk.submitted;
	if (g_nativeVk.frame.effective) g_nativePick.Submitted(g_nativeVk.submitted);
	else g_nativePick.Fail(g_nativeVk.frame.reason == PSYX_NATIVE_OK ? PSYX_NATIVE_UNSUPPORTED : g_nativeVk.frame.reason);
	if (g_nativeVk.frame.effective) ++g_nativeVk.effectiveFrames;
	else if (g_nativeVk.frame.frameState == PSYX_NATIVE_FRAME_FAILED) ++g_nativeVk.failedFrames;
	else if (g_nativeVk.frame.frameState == PSYX_NATIVE_FRAME_LOADING) ++g_nativeVk.loadingFrames;
	else if (g_nativeVk.frame.frameState == PSYX_NATIVE_FRAME_AUXILIARY) ++g_nativeVk.auxiliaryFrames;
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
	g_nativePick.Cancel();
	if (g_nativeVk.pickMapped) vkUnmapMemory(g_vk.device, g_nativeVk.pickBufferMemory);
	if (g_nativeVk.pickBuffer) vkDestroyBuffer(g_vk.device, g_nativeVk.pickBuffer, NULL);
	if (g_nativeVk.pickBufferMemory) vkFreeMemory(g_vk.device, g_nativeVk.pickBufferMemory, NULL);
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
void PsyX_Native_SetRequested(int requested) { g_nativeScene.SetRequested(requested); if (!requested) g_nativePick.Cancel(); }
void PsyX_Native_SetFrameStatus(PsyXNativeResult reason) { g_nativeScene.SetFrameStatus(reason); }
uint64_t PsyX_Native_ResetScene() { g_nativePick.Cancel(); return g_nativeScene.Reset(); }
uint64_t PsyX_Native_GetSceneGeneration() { return g_nativeScene.generation; }
PsyXNativeResult PsyX_Native_RequestPick(int x, int y)
{
	if (!g_vk.initialised || !g_vk.gameMode || !g_nativeScene.requested) return PSYX_NATIVE_UNSUPPORTED;
	int width=0, height=0; SDL_GetWindowSize(g_vk.window, &width, &height);
	return g_nativePick.Request(x,y,width,height);
}
PsyXNativeResult PsyX_Native_GetPickResult(PsyXNativePickResult* result) { return g_nativePick.Get(g_nativeScene,result); }
void PsyX_Native_CancelPick() { g_nativePick.Cancel(); }
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
	stats->effectiveFrames = g_nativeVk.effectiveFrames;
	stats->failedFrames = g_nativeVk.failedFrames;
	stats->loadingFrames = g_nativeVk.loadingFrames;
	stats->auxiliaryFrames = g_nativeVk.auxiliaryFrames;
	stats->frameState = g_nativeVk.frame.frameState;
	stats->materialUploads = g_nativeVk.materialUploads;
	if (stats->requested && (!g_vk.initialised || !g_vk.gameMode)) stats->reason = PSYX_NATIVE_UNSUPPORTED;
}
