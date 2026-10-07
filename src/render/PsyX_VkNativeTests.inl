/* GPU acceptance checks for the native contributor. Included after the PSX
 * readback helpers; uses the same owner and public native resource boundary. */
static void NativeTestIdentity(float matrix[16])
{
	memset(matrix, 0, 16 * sizeof(float));
	matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1;
}

static void NativeTestRectangle(std::vector<PsyXNativeVertex>& vertices, std::vector<uint32_t>& indices,
	float left, float top, float right, float bottom, float distance, const float color[4], int reversed)
{
	const uint32_t first = uint32_t(vertices.size());
	const float xs[] = { left, right, right, left };
	const float ys[] = { top, top, bottom, bottom };
	for (int i = 0; i < 4; ++i)
	{
		PsyXNativeVertex vertex = {};
		vertex.position[0] = xs[i] * distance;
		vertex.position[1] = -ys[i] * distance;
		vertex.position[2] = -distance;
		vertex.normal[2] = 1;
		vertex.uv[0] = (i == 1 || i == 2) ? 1.0f : 0.0f;
		vertex.uv[1] = i >= 2 ? 1.0f : 0.0f;
		memcpy(vertex.color, color, sizeof(vertex.color));
		vertices.push_back(vertex);
	}
	const uint32_t order[] = { 0, 2, 1, 0, 3, 2 };
	for (int i = 0; i < 6; i += 3)
	{
		indices.push_back(first + order[i]);
		indices.push_back(first + order[i + (reversed ? 2 : 1)]);
		indices.push_back(first + order[i + (reversed ? 1 : 2)]);
	}
}

static int NativeTestFrame(PsyXNativeSnapshot& snapshot, int legacyDepth,
	std::vector<unsigned char>& pixels, char* report, int reportSize, int legacyBilinear = 0,
	PsyXNativeResult expectedReason = PSYX_NATIVE_OK)
{
	// Acquire can legitimately skip a frame while recreating an out-of-date
	// swapchain. Accept only a fresh successful submission, never stale readback.
	for (int attempt = 0; attempt < 4; ++attempt)
	{
		const uint64_t before = g_nativeVk.submitted;
		PsyX_Vk_GameBeginFrame();
		// RenderFrame processes resize after this producer runs. Match its next
		// extent so the legacy bottom-left scissor never becomes negative.
		int drawableWidth = 0, drawableHeight = 0;
		SDL_Vulkan_GetDrawableSize(g_vk.window, &drawableWidth, &drawableHeight);
		if (drawableWidth < 64 || drawableHeight < 64) return 0;
		if (PsyX_Native_Publish(&snapshot) != PSYX_NATIVE_OK)
		{
			ReportAppend(report, reportSize, "native publish FAIL\n");
			return 0;
		}
		float ortho[16];
		PsxBuildOrtho(0, float(drawableWidth), float(drawableHeight), 0, -1, 1, ortho);
		PsyX_Vk_GameSetProjection2D(ortho);
		VkPsxVertex quad[6];
		PsxFillQuad(float(drawableWidth), float(drawableHeight), 0, 0, 0, 0, quad);
		PsyX_Vk_GameUpdateVertexBuffer(quad, 6);
		PsyX_Vk_GameSetBlendMode(PSYX_VK_BLEND_NONE);
		PsyX_Vk_GameSetTexture(PSYX_VK_TEX_16BIT, 0);
		PsyX_Vk_GameEnableDepth(legacyDepth);
		PsyX_Vk_GameSetBilinear(legacyBilinear);
		PsyX_Vk_GameSetViewPort(0, 0, drawableWidth, drawableHeight);
		PsyX_Vk_GameSetScissor(0, 0, 0, drawableWidth, drawableHeight);
		PsyX_Vk_GameDrawTriangles(0, 2); // synthetic old sky/world: must be suppressed
		PsyX_Vk_GameModernSceneBoundary();
		PsyX_Vk_GameSetScissor(1, 0, drawableHeight - 64, 64, 64);
		PsyX_Vk_GameDrawTriangles(0, 2); // original UI, using its own legacy depth
		PsyX_Vk_GameSetScissor(0, 0, 0, drawableWidth, drawableHeight);
		if (!PsyX_Vk_RenderFrame()) return 0;
		if (g_nativeVk.submitted == before) continue;
		pixels.resize(size_t(g_vk.width) * g_vk.height * 4);
		int width = 0, height = 0;
		if (!PsyX_Vk_ReadbackRgba(pixels.data(), &width, &height) || width != g_vk.width || height != g_vk.height)
		{
			ReportAppend(report, reportSize, "native readback FAIL\n");
			return 0;
		}
		PsyXNativeStats stats;
		PsyX_Native_GetStats(&stats);
		uint32_t backdrops=0;
		for (uint32_t i=0; i<snapshot.instanceCount; ++i) if (snapshot.instances[i].layer==PSYX_NATIVE_BACKDROP) ++backdrops;
		const int ok = expectedReason == PSYX_NATIVE_OK ?
			(stats.effective && stats.nativeDraws == snapshot.instanceCount && stats.nativeBackdropDraws == backdrops && stats.legacyWorldDraws == 0 &&
			stats.suppressedWorldDraws == 1 && stats.legacyOverlayDraws == 1) :
			(!stats.effective && stats.reason == expectedReason && !stats.nativeDraws &&
			!stats.suppressedWorldDraws && stats.legacyWorldDraws + stats.legacyOverlayDraws == 2);
		ReportAppend(report, reportSize, ok ? "native/world/UI draw accounting ok\n" : "native draw accounting FAIL\n");
		if (!ok)
		{
			char line[256];
			snprintf(line, sizeof(line), "actual native stats: effective=%d reason=%d expected=%d native=%u world=%u suppressed=%u UI=%u\n",
				stats.effective, stats.reason, expectedReason, stats.nativeDraws, stats.legacyWorldDraws,
				stats.suppressedWorldDraws, stats.legacyOverlayDraws);
			ReportAppend(report, reportSize, line);
		}
		return ok;
	}
	ReportAppend(report, reportSize, "native frame never submitted FAIL\n");
	return 0;
}

static int NativeTestBackdrop(PsyXNativeSnapshot snapshot, char* report, int reportSize)
{
	const float red[]={1,0,0,1}, green[]={0,1,0,1}, blue[]={0,0,1,1}, ui[]={248.0f/255,0,0,1};
	snapshot.sceneGeneration=PsyX_Native_ResetScene();
	PsyX_Vk_GameBeginFrame();
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.projection[0]=1; snapshot.view.projection[5]=-1;
	PsyXNativeInstance instances[2]={}; // Deliberately world first in the producer array.
	for (unsigned int item=0; item<2; ++item)
	{
		NativeTestIdentity(instances[item].world);
		for (unsigned int channel=0; channel<4; ++channel) instances[item].tint[channel]=1;
		instances[item].layer=item ? PSYX_NATIVE_BACKDROP : PSYX_NATIVE_WORLD;
		std::vector<PsyXNativeVertex> vertices;
		std::vector<uint32_t> indices;
		const float extent=item ? .95f : .2f;
		if (item)
		{
			NativeTestRectangle(vertices,indices,-extent,-extent,0,extent,4,green,0);
			NativeTestRectangle(vertices,indices,0,-extent,extent,extent,4,blue,0);
		}
		else NativeTestRectangle(vertices,indices,-extent,-extent,extent,extent,8,red,0);
		PsyXNativeMeshDesc mesh={};
		mesh.size=sizeof(mesh); mesh.version=PSYX_NATIVE_VERSION;
		mesh.vertices=vertices.data(); mesh.vertexCount=uint32_t(vertices.size());
		mesh.indices=indices.data(); mesh.indexCount=uint32_t(indices.size());
		for (unsigned int axis=0; axis<3; ++axis) { mesh.boundsMin[axis]=-8; mesh.boundsMax[axis]=8; }
		if (PsyX_Native_CreateMesh(&mesh,&instances[item].mesh)!=PSYX_NATIVE_PENDING) return 1;
	}
	snapshot.instances=instances; snapshot.instanceCount=2;
	std::vector<unsigned char> pixels;
	int failures=0;
	for (unsigned int translated=0; translated<2; ++translated)
	{
		snapshot.view.view[12]=translated ? 4.0f : 0.0f;
		if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; continue; }
		const int worldX=translated ? g_vk.width*3/4 : g_vk.width/2;
		if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,worldX,g_vk.height/2,red,3,
			"farther world survives nearer backdrop without depth write",report,reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/4,g_vk.height/2,green,3,
			"original backdrop ignores camera world translation",report,reportSize)) ++failures;
		if (translated && !PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2-5,g_vk.height/2,green,3,
			"world moves while backdrop stays fixed",report,reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,32,32,ui,3,"original UI survives backdrop",report,reportSize)) ++failures;
	}
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.view[0]=snapshot.view.view[5]=0;
	snapshot.view.view[1]=1; snapshot.view.view[4]=-1; // Rotate the copied source geometry90 degrees.
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) ++failures;
	else
	{
		if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2,g_vk.height*3/4,green,3,
			"backdrop source geometry follows camera rotation",report,reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2,g_vk.height/4,blue,3,
			"rotated backdrop preserves original artwork correspondence",report,reportSize)) ++failures;
	}
	PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
	PsyXNativeStats stats; PsyX_Native_GetStats(&stats);
	if (stats.ownedBytes || stats.retiringMeshes || stats.residentMeshes) ++failures;
	ReportAppend(report,reportSize,failures ? "native backdrop order/depth/translation/retirement FAIL\n" :
		"native backdrop order/depth/translation/retirement ok\n");
	return failures;
}

static int NativeTestMips(PsyXNativeSnapshot snapshot, const float background[4], char* report, int reportSize)
{
	const float white[]={1,1,1,1}, black[]={0,0,0,1};
	const float middle[]={188.0f/255,188.0f/255,188.0f/255,1};
	std::vector<PsyXNativeVertex> vertices;
	std::vector<uint32_t> indices;
	// An eight-pixel rectangle minifies a 128x64 checker. Offset by half a
	// texel so level-zero sampling lands on a black texel, not a checker edge.
	NativeTestRectangle(vertices,indices,-8.0f/g_vk.width,-8.0f/g_vk.height,
		8.0f/g_vk.width,8.0f/g_vk.height,4,white,0);
	for (size_t i=0;i<vertices.size();++i) { vertices[i].uv[0]+=.5f/128; vertices[i].uv[1]+=.5f/64; }
	PsyXNativeMeshDesc mesh={};
	mesh.size=sizeof(mesh); mesh.version=PSYX_NATIVE_VERSION;
	mesh.vertices=vertices.data(); mesh.vertexCount=uint32_t(vertices.size());
	mesh.indices=indices.data(); mesh.indexCount=uint32_t(indices.size());
	for (unsigned int i=0;i<3;++i) { mesh.boundsMin[i]=-4; mesh.boundsMax[i]=4; }
	PsyXNativeInstance instance={};
	NativeTestIdentity(instance.world);
	for (unsigned int i=0;i<4;++i) instance.tint[i]=1;
	snapshot.instances=&instance; snapshot.instanceCount=1;
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.projection[0]=1; snapshot.view.projection[5]=-1;
	PsyXNativeMaterialDesc material={};
	material.size=sizeof(material); material.version=PSYX_NATIVE_VERSION;
	material.width=128; material.height=64; material.byteCount=128*64*4; material.alphaCutoff=.5f;
	std::vector<uint8_t> artwork(size_t(material.byteCount));
	std::vector<unsigned char> pixels;
	int failures=0;
	for (unsigned int mode=0;mode<4;++mode)
	{
		snapshot.sceneGeneration=PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
		for (unsigned int y=0;y<64;++y)
		for (unsigned int x=0;x<128;++x)
		{
			uint8_t* texel=artwork.data()+(y*128+x)*4;
			if (mode==3)
			{
				texel[0]=255; texel[1]=texel[2]=x<64 ? 0 : 255; texel[3]=x<64 ? 0 : 255;
			}
			else { texel[0]=texel[1]=texel[2]=((x+y)&1) ? 255 : 0; texel[3]=255; }
		}
		material.rgba=artwork.data();
		material.filter=mode<2 ? PsyXNativeFilter(mode) : PSYX_NATIVE_TRILINEAR;
		if (PsyX_Native_CreateMaterial(&material,&mesh.material)!=PSYX_NATIVE_PENDING ||
			PsyX_Native_CreateMesh(&mesh,&instance.mesh)!=PSYX_NATIVE_PENDING) { ++failures; break; }
		memset(artwork.data(),127,artwork.size()); // Upload must use the complete owned chain.
		if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; continue; }
		if (mode<3)
		{
			if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2,g_vk.height/2,
				mode==2 ? middle : black,4,mode==2 ? "minified trilinear checker is linear-light gray" :
				"base-only sampler preserves aliased black texel",report,reportSize)) ++failures;
		}
		else
		{
			if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2-3,g_vk.height/2,
				background,3,"minified cutout retains transparent half",report,reportSize)) ++failures;
			if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2+2,g_vk.height/2,
				white,3,"minified opaque half has no transparent red halo",report,reportSize)) ++failures;
		}
		PsyXNativeStats stats; PsyX_Native_GetStats(&stats);
		if (stats.ownedMaterialBytes!=(mode<2 ? 32768 : 43692) || stats.residentMaterials!=1) ++failures;
		if (mode==2)
		{
			const VkImage image=g_nativeVk.materials[mesh.material.slot].image;
			const uint64_t uploads=stats.materialUploads, fallback=stats.fallbackFrames;
			const PsyXNativeSampling choices[]={PSYX_NATIVE_SAMPLE_NEAREST,PSYX_NATIVE_SAMPLE_TRILINEAR,
				PSYX_NATIVE_SAMPLE_LINEAR,PSYX_NATIVE_USE_MATERIAL_FILTER};
			for (unsigned int change=0; change<8; ++change)
			{
				instance.sampling=choices[change%4];
				if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; continue; }
				const bool smooth=instance.sampling==PSYX_NATIVE_SAMPLE_TRILINEAR || instance.sampling==PSYX_NATIVE_USE_MATERIAL_FILTER;
				if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2,g_vk.height/2,
					smooth ? middle : black,4,"runtime sampler selection changes actual minified pixels",report,reportSize)) ++failures;
				PsyX_Native_GetStats(&stats);
				if (stats.materialUploads!=uploads || stats.fallbackFrames!=fallback || stats.ownedMaterialBytes!=43692 ||
					g_nativeVk.materials[mesh.material.slot].image!=image) ++failures;
			}
			PsyXNativeInstance pair[]={instance,instance};
			pair[0].sampling=PSYX_NATIVE_SAMPLE_NEAREST; pair[1].sampling=PSYX_NATIVE_SAMPLE_TRILINEAR;
			pair[0].world[12]=-320.0f/g_vk.width; pair[1].world[12]=320.0f/g_vk.width;
			snapshot.instances=pair; snapshot.instanceCount=2;
			if (!NativeTestFrame(snapshot,0,pixels,report,reportSize) ||
				!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2-40,g_vk.height/2,black,4,
					"shared image independently samples nearest instance",report,reportSize) ||
				!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2+40,g_vk.height/2,middle,4,
					"shared image independently samples trilinear instance",report,reportSize)) ++failures;
			snapshot.instances=&instance; snapshot.instanceCount=1; instance.sampling=PSYX_NATIVE_USE_MATERIAL_FILTER;
			ReportAppend(report,reportSize,failures ? "runtime sampler/image reuse FAIL\n" :
				"runtime sampler/image reuse: eight changes, independent instances, zero uploads/fallback ok\n");
		}
	}
	PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
	PsyXNativeStats stats; PsyX_Native_GetStats(&stats);
	if (stats.ownedMaterialBytes || stats.ownedBytes || stats.residentMaterials || stats.retiringMaterials) ++failures;
	ReportAppend(report,reportSize,failures ? "native mip minification/cutout/ownership FAIL\n" :
		"native mip minification/cutout/full-chain ownership and retirement ok\n");
	return failures;
}

static int NativeTestUploadFailure(PsyXNativeSnapshot snapshot, char* report, int reportSize)
{
	const float white[] = { 1, 1, 1, 1 };
	std::vector<PsyXNativeVertex> vertices;
	std::vector<uint32_t> indices;
	NativeTestRectangle(vertices, indices, -.8f, -.8f, .8f, .8f, 4, white, 0);
	snapshot.sceneGeneration = PsyX_Native_ResetScene();
	PsyX_Vk_GameBeginFrame();
	uint8_t artwork[] = { 255, 255, 255, 255, 255, 255, 255, 255 };
	PsyXNativeMaterialDesc material = {};
	material.size = sizeof(material); material.version = PSYX_NATIVE_VERSION;
	material.rgba = artwork; material.width = 2; material.height = 1; material.byteCount = 8;
	material.filter=PSYX_NATIVE_TRILINEAR;
	PsyXNativeMeshDesc mesh = {};
	mesh.size = sizeof(mesh); mesh.version = PSYX_NATIVE_VERSION;
	mesh.vertices = vertices.data(); mesh.vertexCount = uint32_t(vertices.size());
	mesh.indices = indices.data(); mesh.indexCount = uint32_t(indices.size());
	for (int i = 0; i < 3; ++i) { mesh.boundsMin[i] = -4; mesh.boundsMax[i] = 4; }
	PsyXNativeInstance instance = {};
	NativeTestIdentity(instance.world);
	for (int i = 0; i < 4; ++i) instance.tint[i] = 1;
	if (PsyX_Native_CreateMaterial(&material, &mesh.material) != PSYX_NATIVE_PENDING ||
		PsyX_Native_CreateMesh(&mesh, &instance.mesh) != PSYX_NATIVE_PENDING) return 1;
	snapshot.instances = &instance; snapshot.instanceCount = 1;
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.projection[0] = 1; snapshot.view.projection[5] = -1;
	g_nativeTestHoldUpload = 1;
	std::vector<unsigned char> pixels;
	int failures = !NativeTestFrame(snapshot, 0, pixels, report, reportSize, 0, PSYX_NATIVE_UPLOAD_FAILED);
	const float legacy[] = { 248.0f / 255, 0, 0, 1 };
	if (pixels.empty() || !PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 2,
		g_vk.height / 2, legacy, 3, "upload failure preserves original world pixels", report, reportSize)) ++failures;
	VkNativeMaterial& gpu = g_nativeVk.materials[mesh.material.slot];
	const VkImage retainedImage = gpu.image;
	const VkBuffer retainedStaging = gpu.staging;
	const VkCommandBuffer retainedCommand = gpu.uploadCommand;
	const VkFence retainedFence = gpu.uploadFence;
	if (!retainedImage || !retainedStaging || !retainedCommand || !retainedFence || !gpu.uploadSubmitted) ++failures;
	if (PsyX_Native_DestroyMaterial(mesh.material) != PSYX_NATIVE_OK) ++failures;
	PsyX_Vk_GameBeginFrame();
	PsyXNativeStats stats;
	PsyX_Native_GetStats(&stats);
	if (stats.retiringMaterials != 1 || stats.ownedMaterialBytes != 12 || gpu.image != retainedImage ||
		gpu.staging != retainedStaging || gpu.uploadCommand != retainedCommand || gpu.uploadFence != retainedFence) ++failures;
	// The fault seam never pretends the real upload completed. Wait for its
	// actual fence before restoring normal completion and releasing resources.
	if (!retainedFence || vkWaitForFences(g_vk.device, 1, &retainedFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) ++failures;
	g_nativeTestHoldUpload = 0;
	PsyX_Native_ResetScene();
	PsyX_Vk_GameBeginFrame();
	PsyX_Native_GetStats(&stats);
	if (stats.retiringMaterials || stats.ownedMaterialBytes || stats.ownedBytes || gpu.image || gpu.staging ||
		gpu.uploadCommand || gpu.uploadFence || gpu.uploadSubmitted) ++failures;
	ReportAppend(report, reportSize, failures ? "submitted upload failure/fallback/retention FAIL\n" :
		"submitted upload failure: legacy fallback, retained handles, fence recovery and zero owned bytes ok\n");
	return failures;
}

static int NativeTestMaterials(PsyXNativeSnapshot snapshot, const float background[4], char* report, int reportSize)
{
	const float white[] = { 1, 1, 1, 1 };
	const float black[] = { 0, 0, 0, 1 };
	// 50% linear light encodes to sRGB 188, not artwork byte 128.
	const float middle[] = { 188.0f / 255, 188.0f / 255, 188.0f / 255, 1 };
	std::vector<PsyXNativeVertex> vertices;
	std::vector<uint32_t> indices;
	NativeTestRectangle(vertices, indices, -.8f, -.8f, .8f, .8f, 4, white, 0);
	PsyXNativeMeshDesc mesh = {};
	mesh.size = sizeof(mesh); mesh.version = PSYX_NATIVE_VERSION;
	mesh.vertices = vertices.data(); mesh.vertexCount = uint32_t(vertices.size());
	mesh.indices = indices.data(); mesh.indexCount = uint32_t(indices.size());
	for (int i = 0; i < 3; ++i) { mesh.boundsMin[i] = -4; mesh.boundsMax[i] = 4; }
	PsyXNativeMaterialDesc material = {};
	material.size = sizeof(material); material.version = PSYX_NATIVE_VERSION;
	material.width = 2; material.height = 1; material.byteCount = 8;
	PsyXNativeInstance instance = {};
	NativeTestIdentity(instance.world);
	for (int i = 0; i < 4; ++i) instance.tint[i] = 1;
	instance.identity = 2;
	snapshot.instances = &instance; snapshot.instanceCount = 1;
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.cameraPosition[0] = snapshot.view.cameraPosition[1] = snapshot.view.cameraPosition[2] = 0;
	snapshot.view.projection[0] = 1; snapshot.view.projection[5] = -1;
	std::vector<unsigned char> pixels, previous;
	int failures = 0;
	// Every cycle allocates, uploads, submits and retires a texture/descriptor.
	for (int reload = 0; reload < 8; ++reload)
	{
		snapshot.sceneGeneration = PsyX_Native_ResetScene();
		PsyX_Vk_GameBeginFrame();
		uint8_t artwork[] = { 0, 0, 0, 255, 255, 255, 255, 255 };
		material.rgba = artwork;
		material.filter = (reload & 1) ? PSYX_NATIVE_LINEAR : PSYX_NATIVE_NEAREST;
		material.alphaCutoff = .5f;
		material.cull = reload >= 6 ? PSYX_NATIVE_CULL_NONE : PSYX_NATIVE_CULL_BACK;
		instance.world[0] = reload >= 6 ? -1.0f : 1.0f;
		if (reload >= 4) artwork[3] = 0;
		if (PsyX_Native_CreateMaterial(&material, &mesh.material) != PSYX_NATIVE_PENDING ||
			PsyX_Native_CreateMesh(&mesh, &instance.mesh) != PSYX_NATIVE_PENDING)
		{
			ReportAppend(report, reportSize, "native material resource creation FAIL\n"); ++failures; break;
		}
		// Destroy the producer bytes before GPU upload: the owner must have copied.
		memset(artwork, 127, sizeof(artwork));
		for (int legacyBilinear = 0; legacyBilinear < 2; ++legacyBilinear)
		{
			if (!NativeTestFrame(snapshot, 0, pixels, report, reportSize, legacyBilinear)) { ++failures; continue; }
			if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width * (reload >= 6 ? 3 : 1) / 4, g_vk.height / 2,
				reload >= 4 ? background : black, 3, reload >= 4 ? "native cutout reveals background" : "native owns original black texel",
				report, reportSize)) ++failures;
			if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width * (reload >= 6 ? 1 : 3) / 4, g_vk.height / 2,
				white, 3, reload >= 6 ? "back-facing two-sided cutout solid texel" : "native white artwork texel", report, reportSize)) ++failures;
			if ((reload & 1) && reload < 4 && !PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height,
				g_vk.width / 2, g_vk.height / 2, middle, 3, "native linear filtering in linear light", report, reportSize)) ++failures;
			if (!legacyBilinear) previous = pixels;
			else if (previous != pixels) { ReportAppend(report, reportSize, "legacy filter changes native artwork FAIL\n"); ++failures; }
		}
		if (reload == 4 || reload == 5)
		{
			instance.world[0] = -1;
			if (!NativeTestFrame(snapshot, 0, pixels, report, reportSize) ||
				!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 4, g_vk.height / 2,
					background, 3, "single-sided reversed cutout is culled", report, reportSize)) ++failures;
			instance.world[0] = 1;
		}
		PsyXNativeStats stats;
		PsyX_Native_GetStats(&stats);
		if (stats.residentMaterials != 1 || stats.pendingMaterials || stats.retiringMaterials || stats.ownedMaterialBytes != 8) ++failures;
		if (PsyX_Native_DestroyMaterial(mesh.material) != PSYX_NATIVE_OK || PsyX_Native_Publish(&snapshot) != PSYX_NATIVE_STALE) ++failures;
		PsyX_Native_GetStats(&stats);
		if (stats.retiringMaterials != 1) ++failures;
		PsyX_Vk_GameBeginFrame();
		PsyX_Native_GetStats(&stats);
		if (stats.residentMaterials || stats.retiringMaterials || stats.ownedMaterialBytes) ++failures;
	}
	PsyX_Native_ResetScene();
	PsyX_Vk_GameBeginFrame();
	PsyXNativeStats stats;
	PsyX_Native_GetStats(&stats);
	if (stats.residentMaterials || stats.pendingMaterials || stats.retiringMaterials || stats.ownedMaterialBytes || stats.ownedBytes) ++failures;
	ReportAppend(report, reportSize, failures ? "native material/filter/cutout/retirement FAIL\n" : "eight material/filter/cutout/retirement cycles ok\n");
	return failures;
}

int PsyX_Vk_NativeSelfTest(char* report, int reportSize)
{
	if (report && reportSize > 0) report[0] = 0;
	if (!g_vk.initialised || !g_vk.gameMode || !g_vk.psx.ready) return 0;
	const float red[] = { 1, 0, 0, 1 }, blue[] = { 0, 0, 1, 1 }, green[] = { 0, 1, 0, 1 };
	const float magenta[] = { 1, 0, 1, 1 }, yellow[] = { 1, 1, 0, 1 }, cyan[] = { 0, 1, 1, 1 };
	const float ui[] = { 248.0f / 255, 0, 0, 1 };
	// The clear value is linear only when the attachment encodes sRGB stores.
	float background[] = { .025f, .035f, .055f, 1 };
	if (g_vk.srgbOutput)
		for (int i = 0; i < 3; ++i) background[i] = 1.055f * powf(background[i], 1.0f / 2.4f) - .055f;
	std::vector<PsyXNativeVertex> vertices;
	std::vector<uint32_t> indices;
	NativeTestRectangle(vertices, indices, -.85f, -.6f, -.15f, .6f, 4, red, 0);
	NativeTestRectangle(vertices, indices, -.85f, -.6f, -.15f, .6f, 8, blue, 0); // later draw is farther
	NativeTestRectangle(vertices, indices, .15f, -.6f, .75f, -.05f, 4, green, 0);
	NativeTestRectangle(vertices, indices, .15f, .05f, .75f, .6f, 4, magenta, 1);
	NativeTestRectangle(vertices, indices, -.1f, -.1f, .1f, .1f, .5f, yellow, 0); // before near=1
	NativeTestRectangle(vertices, indices, -.9f, -.9f, .9f, .9f, 25, cyan, 0); // beyond far=20
	PsyXNativeMeshDesc desc = {};
	desc.size = sizeof(desc); desc.version = PSYX_NATIVE_VERSION;
	desc.vertices = vertices.data(); desc.vertexCount = uint32_t(vertices.size());
	desc.indices = indices.data(); desc.indexCount = uint32_t(indices.size());
	for (int i = 0; i < 3; ++i) { desc.boundsMin[i] = -30; desc.boundsMax[i] = 30; }
	PsyXNativeInstance instance = {};
	if (PsyX_Native_CreateMesh(&desc, &instance.mesh) != PSYX_NATIVE_PENDING) return 0;
	NativeTestIdentity(instance.world);
	for (int i = 0; i < 4; ++i) instance.tint[i] = 1;
	instance.identity = 1;
	PsyXNativeSnapshot snapshot = {};
	snapshot.size = sizeof(snapshot); snapshot.version = PSYX_NATIVE_VERSION;
	snapshot.sceneGeneration = PsyX_Native_GetSceneGeneration();
	snapshot.instances = &instance; snapshot.instanceCount = 1;
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.nearPlane = 1; snapshot.view.farPlane = 20;
	snapshot.view.projection[0] = 1; snapshot.view.projection[5] = -1;
	snapshot.view.projection[10] = -20.0f / 19;
	snapshot.view.projection[11] = -1;
	snapshot.view.projection[14] = -20.0f / 19;
	PsyX_Native_SetRequested(1);
	std::vector<unsigned short> vram(PSYX_VK_VRAM_WIDTH * PSYX_VK_VRAM_HEIGHT, 0);
	vram[0] = 0x001F;
	PsyX_Vk_GameSetVram(vram.data());
	std::vector<unsigned char> pixels, firstFrame;
	int failures = 0;
	for (int depth = 0; depth < 2; ++depth)
	{
		snapshot.simulationTick++;
		if (!NativeTestFrame(snapshot, depth, pixels, report, reportSize)) { ++failures; continue; }
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 4, g_vk.height / 2,
			red, 3, "near red occludes later far blue", report, reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width * 7 / 10, g_vk.height * 7 / 20,
			green, 3, "front winding", report, reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width * 7 / 10, g_vk.height * 13 / 20,
			background, 3, "back winding culled", report, reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 2, g_vk.height / 2,
			background, 3, "near and far clipping", report, reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, 32, 32,
			ui, 3, "legacy UI on independent depth", report, reportSize)) ++failures;
		if (!depth) firstFrame = pixels;
		else if (pixels != firstFrame) { ReportAppend(report, reportSize, "legacy depth affects native pixels FAIL\n"); ++failures; }
		else ReportAppend(report, reportSize, "native pixels independent of legacy depth state ok\n");
	}
	snapshot.view.projection[0] = .5f; snapshot.view.projection[5] = -.5f;
	if (!NativeTestFrame(snapshot, 0, pixels, report, reportSize)) ++failures;
	else
	{
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width * 3 / 8, g_vk.height / 2,
			red, 3, "wide FOV shrinks geometry", report, reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 4, g_vk.height / 2,
			background, 3, "wide FOV vacated pixel", report, reportSize)) ++failures;
	}
	snapshot.view.projection[0] = 1; snapshot.view.projection[5] = -1;
	snapshot.view.view[12] = 2; snapshot.view.cameraPosition[0] = -2;
	if (!NativeTestFrame(snapshot, 0, pixels, report, reportSize)) ++failures;
	else if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 2, g_vk.height / 2,
		red, 3, "camera translation", report, reportSize)) ++failures;
	const int originalWidth = g_vk.width, originalHeight = g_vk.height;
	SDL_SetWindowSize(g_vk.window, 800, 600);
	SDL_PumpEvents();
	if (!NativeTestFrame(snapshot, 0, pixels, report, reportSize) || g_vk.width != 800 || g_vk.height != 600) ++failures;
	else
	{
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, 400, 300,
			red, 3, "resized native depth/target", report, reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, 32, 32,
			ui, 3, "resized original UI scissor", report, reportSize)) ++failures;
	}
	const uint64_t beforeMinimize = g_nativeVk.submitted;
	SDL_MinimizeWindow(g_vk.window);
	Uint32 start = SDL_GetTicks();
	while (!(SDL_GetWindowFlags(g_vk.window) & SDL_WINDOW_MINIMIZED) && SDL_GetTicks() - start < 2000)
	{
		SDL_PumpEvents(); SDL_Delay(10);
	}
	if (!(SDL_GetWindowFlags(g_vk.window) & SDL_WINDOW_MINIMIZED) || !PsyX_Vk_RenderFrame() ||
		g_nativeVk.submitted != beforeMinimize)
	{
		ReportAppend(report, reportSize, "minimize must skip submission FAIL\n"); ++failures;
	}
	else ReportAppend(report, reportSize, "minimize skips submission ok\n");
	SDL_RestoreWindow(g_vk.window);
	start = SDL_GetTicks();
	while ((SDL_GetWindowFlags(g_vk.window) & SDL_WINDOW_MINIMIZED) && SDL_GetTicks() - start < 2000)
	{
		SDL_PumpEvents(); SDL_Delay(10);
	}
	SDL_SetWindowSize(g_vk.window, originalWidth, originalHeight);
	SDL_PumpEvents();
	if ((SDL_GetWindowFlags(g_vk.window) & SDL_WINDOW_MINIMIZED) ||
		!NativeTestFrame(snapshot, 0, pixels, report, reportSize)) ++failures;
	else
	{
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 2, g_vk.height / 2,
			red, 3, "restore native depth/target", report, reportSize)) ++failures;
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, 32, 32,
			ui, 3, "restored original UI scissor", report, reportSize)) ++failures;
	}
	// Destroy after a real submission: immediately stale, retained until owner fence.
	PsyXNativeStats stats;
	if (PsyX_Native_DestroyMesh(instance.mesh) != PSYX_NATIVE_OK ||
		PsyX_Native_Publish(&snapshot) != PSYX_NATIVE_STALE) ++failures;
	PsyX_Native_GetStats(&stats);
	if (stats.retiringMeshes != 1) ++failures;
	PsyX_Vk_GameBeginFrame();
	PsyX_Native_GetStats(&stats);
	if (stats.retiringMeshes || stats.residentMeshes || stats.ownedBytes) ++failures;
	else ReportAppend(report, reportSize, "submitted mesh retirement and stale handle ok\n");
	// Exercise repeated scene reload/upload/reclaim, never growing the resource pool.
	for (int reload = 0; reload < 8; ++reload)
	{
		snapshot.sceneGeneration = PsyX_Native_ResetScene();
		if (PsyX_Native_CreateMesh(&desc, &instance.mesh) != PSYX_NATIVE_PENDING ||
			!NativeTestFrame(snapshot, 0, pixels, report, reportSize)) { ++failures; break; }
		PsyX_Native_GetStats(&stats);
		if (stats.residentMeshes != 1 || stats.pendingMeshes || stats.retiringMeshes) ++failures;
	}
	PsyX_Native_ResetScene();
	PsyX_Vk_GameBeginFrame();
	PsyX_Native_GetStats(&stats);
	if (stats.residentMeshes || stats.pendingMeshes || stats.retiringMeshes || stats.ownedBytes) ++failures;
	else ReportAppend(report, reportSize, "eight reloads leave zero owned meshes/bytes ok\n");
	failures += NativeTestMaterials(snapshot, background, report, reportSize);
	failures += NativeTestMips(snapshot, background, report, reportSize);
	failures += NativeTestBackdrop(snapshot, report, reportSize);
	failures += NativeTestUploadFailure(snapshot, report, reportSize);
	PsyX_Native_SetRequested(0);
	// No native snapshot or requested mode: the original renderer remains usable.
	for (int requested = 0; requested < 2; ++requested)
	{
		PsyX_Native_SetRequested(requested);
		PsyX_Vk_GameBeginFrame();
		VkPsxVertex quad[6];
		PsxFillQuad(float(g_vk.width), float(g_vk.height), 0, 0, 0, 0, quad);
		PsyX_Vk_GameUpdateVertexBuffer(quad, 6);
		PsyX_Vk_GameDrawTriangles(0, 2);
		if (!PsyX_Vk_RenderFrame()) { ++failures; continue; }
		pixels.resize(size_t(g_vk.width) * g_vk.height * 4);
		if (!PsyX_Vk_ReadbackRgba(pixels.data(), NULL, NULL)) { ++failures; continue; }
		PsyX_Native_GetStats(&stats);
		if (stats.effective || stats.nativeDraws || stats.legacyWorldDraws != 1 ||
			stats.reason != (requested ? PSYX_NATIVE_NO_SNAPSHOT : PSYX_NATIVE_OK)) ++failures;
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 2, g_vk.height / 2,
			ui, 3, requested ? "missing snapshot explicit legacy fallback" : "native off selectable legacy",
			report, reportSize)) ++failures;
	}
	PsyX_Native_SetRequested(0);
	ReportAppend(report, reportSize, failures ? "native self-test FAIL\n" : "native self-test PASS\n");
	return failures == 0;
}
