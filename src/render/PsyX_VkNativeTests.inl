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
	PsyXNativeResult expectedReason = PSYX_NATIVE_OK, bool worldBoundary = true)
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
		if (worldBoundary)
		{
			PsyX_Vk_GameModernSceneBoundary();
			PsyX_Vk_GameSetScissor(1, 0, drawableHeight - 64, 64, 64);
			PsyX_Vk_GameDrawTriangles(0, 2); // original UI, using its own legacy depth
		}
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
			stats.frameState == PSYX_NATIVE_FRAME_FAILED && stats.suppressedWorldDraws == 1 &&
			!stats.legacyWorldDraws && stats.legacyOverlayDraws == (worldBoundary ? 1u : 0u));
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

static int NativeTestLargeOrigin(PsyXNativeSnapshot snapshot, char* report, int reportSize)
{
	snapshot.sceneGeneration=PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
	const float blue[]={0,0,1,1},red[]={1,0,0,1};
	PsyXNativeInstance instances[2]={};
	for (unsigned int i=0;i<2;++i)
	{
		std::vector<PsyXNativeVertex> vertices; std::vector<uint32_t> indices;
		NativeTestRectangle(vertices,indices,-.8f,-.8f,.8f,.8f,8,i ? red : blue,0);
		// A second local origin describes a genuinely nearer face in the same
		// view. Its separation is representable locally, not at the city origin.
		if (i) for(size_t vertex=0;vertex<vertices.size();++vertex) vertices[vertex].position[2]=-1031.996f;
		PsyXNativeMeshDesc mesh={}; mesh.size=sizeof(mesh); mesh.version=PSYX_NATIVE_VERSION;
		mesh.vertices=vertices.data(); mesh.vertexCount=uint32_t(vertices.size());
		mesh.indices=indices.data(); mesh.indexCount=uint32_t(indices.size());
		for (unsigned int axis=0;axis<3;++axis) { mesh.boundsMin[axis]=-1040; mesh.boundsMax[axis]=1040; }
		if (PsyX_Native_CreateMesh(&mesh,&instances[i].mesh)!=PSYX_NATIVE_PENDING) return 1;
		NativeTestIdentity(instances[i].world); for(unsigned int channel=0;channel<4;++channel) instances[i].tint[channel]=1;
		instances[i].identity=601+i;
	}
	snapshot.instances=instances; snapshot.instanceCount=2;
	std::vector<unsigned char> pixels; int failures=0;
	for (unsigned int origin=0;origin<2;++origin)
	for (unsigned int order=0;order<2;++order)
	{
		NativeTestIdentity(snapshot.view.view);
		snapshot.view.view[14]=origin ? 220000.0f : 0;
		for(unsigned int i=0;i<2;++i) instances[i].world[14]=(origin ? -220000.0f : 0)+(instances[i].identity==602 ? 1024 : 0);
		if (order) std::swap(instances[0],instances[1]);
		int width=0,height=0; SDL_GetWindowSize(g_vk.window,&width,&height);
		PsyXNativePickResult pick={}; PsyX_Native_RequestPick(width/2,height/2);
		if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; continue; }
		PsyX_Vk_GameBeginFrame();
		if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2,g_vk.height/2,red,3,
			"native nearer face at local/large city origin",report,reportSize) ||
			PsyX_Native_GetPickResult(&pick)!=PSYX_NATIVE_OK || !pick.hit || pick.identity!=602) ++failures;
	}
	PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame(); PsyXNativeStats stats; PsyX_Native_GetStats(&stats);
	if(stats.ownedBytes || stats.retiringMeshes) ++failures;
	ReportAppend(report,reportSize,failures ? "native large-origin occlusion/picking/retirement FAIL\n" :
		"native large-origin occlusion/picking/two orders/retirement PASS\n");
	return failures;
}

static int NativeTestHorizontal(PsyXNativeSnapshot snapshot, char* report, int reportSize)
{
	snapshot.sceneGeneration=PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.view[5]=snapshot.view.view[10]=.8660254f;
	snapshot.view.view[6]=.5f; snapshot.view.view[9]=-.5f;
	snapshot.view.view[13]=-6.9282032f; snapshot.view.view[14]=-4;
	snapshot.view.projection[0]=1; snapshot.view.projection[5]=-1;
	const float blue[]={0,0,1,1}, red[]={1,0,0,1}, green[]={0,1,0,1}, white[]={1,1,1,1};
	const uint8_t art[]={255,0,0,0,255,0,0,255};
	PsyXNativeMaterialDesc material={}; material.size=sizeof(material); material.version=PSYX_NATIVE_VERSION;
	material.rgba=art; material.width=2; material.height=1; material.byteCount=8;
	material.filter=PSYX_NATIVE_NEAREST; material.alphaCutoff=.5f; material.cull=PSYX_NATIVE_CULL_NONE;
	PsyXNativeMaterialHandle paint={};
	if(PsyX_Native_CreateMaterial(&material,&paint)!=PSYX_NATIVE_PENDING) return 1;
	PsyXNativeInstance instances[3]={};
	PsyXNativeMeshDesc paintMesh={}; std::vector<PsyXNativeVertex> paintVertices;
	std::vector<uint32_t> paintIndices;
	for(unsigned int i=0;i<3;++i)
	{
		std::vector<PsyXNativeVertex> vertices; std::vector<uint32_t> indices;
		NativeTestRectangle(vertices,indices,-.9f,-.9f,.9f,.9f,8,i==0 ? blue : i==1 ? red : green,0);
		for(size_t v=0;v<vertices.size();++v)
		{
			vertices[v].position[2]=-vertices[v].position[1]-16;
			vertices[v].position[1]=i==2 ? .1f : 0;
			vertices[v].normal[1]=1;vertices[v].normal[2]=0;
			if(i==1) vertices[v].position[0]-=1024;
		}
		if(i==1) { const uint32_t alternate[]={0,3,1,1,3,2}; indices.assign(alternate,alternate+6); }
		PsyXNativeMeshDesc mesh={}; mesh.size=sizeof(mesh); mesh.version=PSYX_NATIVE_VERSION;
		mesh.vertices=vertices.data(); mesh.vertexCount=uint32_t(vertices.size());
		mesh.indices=indices.data(); mesh.indexCount=uint32_t(indices.size());
		for(unsigned int axis=0;axis<3;++axis) { mesh.boundsMin[axis]=-1040;mesh.boundsMax[axis]=1040; }
		if(PsyX_Native_CreateMesh(&mesh,&instances[i].mesh)!=PSYX_NATIVE_PENDING) return 1;
		NativeTestIdentity(instances[i].world);if(i==1) instances[i].world[12]=1024;
		for(unsigned int channel=0;channel<4;++channel) instances[i].tint[channel]=1;
		instances[i].identity=701+i;
		if(i==1) { paintMesh=mesh;paintVertices.swap(vertices);paintIndices.swap(indices); }
	}
	snapshot.instances=instances; snapshot.instanceCount=2;
	std::vector<unsigned char> pixels;int failures=0;
	// Exact physical ties have a defined winner: the last submitted native face.
	// Different diagonals/local origins may not create alternating ownership bands.
	for(unsigned int order=0;order<2;++order)
	{
		if(order) std::swap(instances[0],instances[1]);
		int width=0,height=0;SDL_GetWindowSize(g_vk.window,&width,&height);
		PsyX_Native_RequestPick(width/2,height/2);PsyXNativePickResult pick={};
		if(!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures;continue; }
		const float* expected=instances[1].identity==702 ? red : blue;
		unsigned int checked=0,wrong=0;
		for(int y=g_vk.height*46/100;y<g_vk.height*54/100;++y)
		for(int x=g_vk.width*42/100;x<g_vk.width*58/100;++x)
		{
			++checked; const unsigned char* pixel=&pixels[(size_t(y)*g_vk.width+x)*4];
			for(unsigned int channel=0;channel<4;++channel)
				if(std::abs(int(pixel[channel])-int(expected[channel]*255))>3) { ++wrong;break; }
		}
		char line[128];snprintf(line,sizeof(line),"native horizontal tie order%u: pixels%u wrong%u\n",order,checked,wrong);
		ReportAppend(report,reportSize,line);if(wrong || !checked) ++failures;
		PsyX_Vk_GameBeginFrame();
		if(PsyX_Native_GetPickResult(&pick)!=PSYX_NATIVE_OK || !pick.hit || pick.identity!=instances[1].identity) ++failures;
	}
	// A truly nearer horizontal plane must win even against authored paint bias.
	const unsigned int paintSlot=instances[0].identity==702 ? 0 : 1;
	PsyX_Native_DestroyMesh(instances[paintSlot].mesh);
	for(size_t v=0;v<paintVertices.size();++v) memcpy(paintVertices[v].color,white,sizeof(white));
	paintMesh.vertices=paintVertices.data();paintMesh.indices=paintIndices.data();paintMesh.material=paint;paintMesh.depthLayer=1;
	if(PsyX_Native_CreateMesh(&paintMesh,&instances[paintSlot].mesh)!=PSYX_NATIVE_PENDING) return failures+1;
	for(unsigned int nearer=0;nearer<2;++nearer)
	for(unsigned int order=0;order<2;++order)
	{
		snapshot.instanceCount=nearer ? 3 : 2;
		if(order) std::swap(instances[0],instances[1]);
		for(unsigned int sample=0;sample<2;++sample)
		{
			int width=0,height=0;SDL_GetWindowSize(g_vk.window,&width,&height);
			PsyX_Native_RequestPick(width*(sample ? 55 : 45)/100,height/2);PsyXNativePickResult pick={};
			if(!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures;continue; }
			PsyX_Vk_GameBeginFrame();
			const float* expected=nearer ? green : sample ? red : blue;
			if(!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width*(sample ? 55 : 45)/100,g_vk.height/2,
				expected,3,"horizontal paint/hole/physical occluder",report,reportSize) ||
				PsyX_Native_GetPickResult(&pick)!=PSYX_NATIVE_OK || !pick.hit || pick.identity!=(nearer ? 703u : sample ? 702u : 701u)) ++failures;
		}
	}
	PsyX_Native_ResetScene();PsyX_Vk_GameBeginFrame();PsyXNativeStats stats;PsyX_Native_GetStats(&stats);
	if(stats.ownedBytes || stats.ownedMaterialBytes || stats.retiringMeshes || stats.retiringMaterials) ++failures;
	ReportAppend(report,reportSize,failures ? "native horizontal plane/ties/paint/alpha/occlusion/IDs/retirement FAIL\n" :
		"native horizontal plane/ties/paint/alpha/occlusion/IDs/two orders/retirement PASS\n");
	return failures;
}

static int NativeTestCoplanar(PsyXNativeSnapshot snapshot, char* report, int reportSize)
{
	snapshot.sceneGeneration=PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.projection[0]=1; snapshot.view.projection[5]=-1;
	const float white[]={1,1,1,1}, blue[]={0,0,1,1}, green[]={0,1,0,1}, red[]={1,0,0,1};
	const uint8_t art[]={255,0,0,0, 255,0,0,255};
	PsyXNativeMaterialDesc material={}; material.size=sizeof(material); material.version=PSYX_NATIVE_VERSION;
	material.rgba=art; material.width=2; material.height=1; material.byteCount=8;
	material.filter=PSYX_NATIVE_NEAREST; material.alphaCutoff=.5f; material.cull=PSYX_NATIVE_CULL_NONE;
	PsyXNativeMaterialHandle paint={};
	if (PsyX_Native_CreateMaterial(&material,&paint)!=PSYX_NATIVE_PENDING) return 1;
	PsyXNativeInstance instances[3]={}; PsyXNativeMeshDesc paintMesh={};
	std::vector<PsyXNativeVertex> paintVertices; std::vector<uint32_t> paintIndices;
	for (unsigned int i=0;i<3;++i)
	{
		std::vector<PsyXNativeVertex> vertices; std::vector<uint32_t> indices;
		// The paint fixture is a few D32 rounding units behind its base. A
		// declared coplanar layer must cover it, but never a genuinely nearer face.
		const float distance=i==0 ? 8.000002f : i==1 ? 8.0f : 7.9f;
		NativeTestRectangle(vertices,indices,i==2 ? .2f : -.9f,i==2 ? -.7f : -.9f,
			i==2 ? .7f : .9f,i==2 ? -.2f : .9f,distance,i==0 ? white : i==1 ? blue : green,0);
		PsyXNativeMeshDesc mesh={}; mesh.size=sizeof(mesh); mesh.version=PSYX_NATIVE_VERSION;
		mesh.vertices=vertices.data(); mesh.vertexCount=uint32_t(vertices.size());
		mesh.indices=indices.data(); mesh.indexCount=uint32_t(indices.size());
		for (unsigned int axis=0;axis<3;++axis) { mesh.boundsMin[axis]=-9; mesh.boundsMax[axis]=9; }
		if (!i) mesh.material=paint;
		if (PsyX_Native_CreateMesh(&mesh,&instances[i].mesh)!=PSYX_NATIVE_PENDING) return 1;
		NativeTestIdentity(instances[i].world); for(unsigned int j=0;j<4;++j) instances[i].tint[j]=1;
		instances[i].identity=501+i;
		if (!i) { paintMesh=mesh; paintVertices.swap(vertices); paintIndices.swap(indices); }
	}
	snapshot.instances=instances; snapshot.instanceCount=3;
	std::vector<unsigned char> pixels; int failures=0;
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize) ||
		!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width*13/20,g_vk.height/2,
			blue,3,"ordinary depth rejects behind-base paint control",report,reportSize)) ++failures;
	PsyX_Native_DestroyMesh(instances[0].mesh);
	paintMesh.vertices=paintVertices.data(); paintMesh.indices=paintIndices.data(); paintMesh.depthLayer=1;
	if (PsyX_Native_CreateMesh(&paintMesh,&instances[0].mesh)!=PSYX_NATIVE_PENDING) return failures+1;
	// Base and paint draw order reverses; the ordinary face resets bias explicitly.
	for (unsigned int order=0;order<2;++order)
	{
		if (order) std::swap(instances[0],instances[1]);
		for (unsigned int sample=0;sample<3;++sample)
		{
			int width=0,height=0; SDL_GetWindowSize(g_vk.window,&width,&height);
			const int x=width*(sample==0 ? 13 : sample==1 ? 7 : 14)/20;
			const int y=height*(sample==2 ? 6 : 10)/20;
			PsyXNativePickResult pick={}; PsyX_Native_RequestPick(x,y);
			if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; continue; }
			PsyX_Vk_GameBeginFrame();
			const float* expected=sample==0 ? red : sample==1 ? blue : green;
			const uint64_t identity=sample==0 ? 501 : sample==1 ? 502 : 503;
			if (!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width*(sample==0 ? 13 : sample==1 ? 7 : 14)/20,
				g_vk.height*(sample==2 ? 6 : 10)/20,expected,3,"native coplanar paint/hole/occluder",report,reportSize) ||
				PsyX_Native_GetPickResult(&pick)!=PSYX_NATIVE_OK || !pick.hit || pick.identity!=identity) ++failures;
		}
	}
	PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
	PsyXNativeStats stats; PsyX_Native_GetStats(&stats);
	if (stats.ownedBytes || stats.ownedMaterialBytes || stats.retiringMeshes || stats.retiringMaterials) ++failures;
	ReportAppend(report,reportSize,failures ? "native coplanar layer/alpha/occlusion/picking/retirement FAIL\n" :
		"native coplanar layer/alpha/occlusion/picking/two draw orders/retirement ok\n");
	return failures;
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
			const uint64_t uploads=stats.materialUploads, failed=stats.failedFrames;
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
				if (stats.materialUploads!=uploads || stats.failedFrames!=failed || stats.ownedMaterialBytes!=43692 ||
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
				"runtime sampler/image reuse: eight changes, independent instances, zero uploads/failures ok\n");
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
	const float failureColor[] = { 32.0f / 255, 0, 32.0f / 255, 1 };
	if (pixels.empty() || !PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 2,
		g_vk.height / 2, failureColor, 3, "upload failure explicit pixels, no original world", report, reportSize)) ++failures;
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
	// Recovery must actually render the new native resources, not old pixels.
	snapshot.sceneGeneration = PsyX_Native_GetSceneGeneration();
	if (PsyX_Native_CreateMaterial(&material, &mesh.material) != PSYX_NATIVE_PENDING ||
		PsyX_Native_CreateMesh(&mesh, &instance.mesh) != PSYX_NATIVE_PENDING ||
		!NativeTestFrame(snapshot,0,pixels,report,reportSize) ||
		!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2,g_vk.height/2,white,3,
			"new native image/mesh after actual upload fence recovery",report,reportSize)) ++failures;
	PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame(); PsyX_Native_GetStats(&stats);
	if (stats.ownedMaterialBytes || stats.ownedBytes || stats.retiringMaterials || stats.retiringMeshes) ++failures;
	ReportAppend(report, reportSize, failures ? "submitted upload failure/native recovery/retention FAIL\n" :
		"submitted upload failure: explicit error pixels, retained handles, native fence recovery and zero owned bytes ok\n");
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

// Characterize dark sRGB fractional filtering independently of city geometry
// and RenderDoc replay. Strict ideal-colour deviations remain diagnostics;
// uniformity, nearest texels, filter consistency and resource lifetime are gates.
static int NativeTestFractionalSampling(PsyXNativeSnapshot snapshot, char* report, int reportSize)
{
	struct Sample { float x, y; uint8_t rgba[16]; };
	const Sample samples[] = {
		{ .9559468f,.4726291f,{33,33,16,255,16,16,8,255,123,140,99,255,33,33,16,255} },
		{ .0920732f,.0887237f,{41,49,25,255,58,66,41,255,115,132,90,255,165,189,148,255} },
		{ .8511162f,.0247246f,{58,66,41,255,41,49,25,255,165,189,148,255,82,90,58,255} },
		{ .8737386f,.2212204f,{58,66,41,255,16,16,8,255,148,173,123,255,16,16,8,255} },
		{ .7409889f,.0629630f,{16,16,8,255,16,16,8,255,148,173,123,255,33,33,16,255} }
	};
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.cameraPosition[0]=snapshot.view.cameraPosition[1]=snapshot.view.cameraPosition[2]=0;
	snapshot.view.projection[0]=1; snapshot.view.projection[5]=-1;
	const float white[]={1,1,1,1};
	int failures=0, strictOutside=0;
	char line[256];
	for (unsigned int sample=0;sample<sizeof(samples)/sizeof(samples[0]);++sample)
	{
		const Sample& source=samples[sample];
		double ideal[3]={};
		for (unsigned int channel=0;channel<3;++channel)
		{
			double linear[4];
			for (unsigned int texel=0;texel<4;++texel)
			{
				const double value=source.rgba[texel*4+channel]/255.0;
				linear[texel]=value<=.04045 ? value/12.92 : pow((value+.055)/1.055,2.4);
			}
			const double value=(linear[0]*(1-source.x)+linear[1]*source.x)*(1-source.y)+
				(linear[2]*(1-source.x)+linear[3]*source.x)*source.y;
			ideal[channel]=255*(value<=.0031308 ? 12.92*value : 1.055*pow(value,1.0/2.4)-.055);
		}
		unsigned char bilinear[4]={};
		for (unsigned int filter=0;filter<3;++filter)
		{
			snapshot.sceneGeneration=PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
			PsyXNativeMaterialDesc material={}; material.size=sizeof(material); material.version=PSYX_NATIVE_VERSION;
			material.rgba=source.rgba; material.width=material.height=2; material.byteCount=16;
			material.filter=PsyXNativeFilter(filter); material.alphaCutoff=.5f; material.cull=PSYX_NATIVE_CULL_NONE;
			std::vector<PsyXNativeVertex> vertices; std::vector<uint32_t> indices;
			NativeTestRectangle(vertices,indices,-.8f,-.8f,.8f,.8f,4,white,0);
			for (size_t vertex=0;vertex<vertices.size();++vertex)
			{
				vertices[vertex].uv[0]=(.5f+source.x)/2;
				vertices[vertex].uv[1]=(.5f+source.y)/2;
			}
			PsyXNativeMeshDesc mesh={}; mesh.size=sizeof(mesh); mesh.version=PSYX_NATIVE_VERSION;
			mesh.vertices=vertices.data(); mesh.vertexCount=uint32_t(vertices.size());
			mesh.indices=indices.data(); mesh.indexCount=uint32_t(indices.size());
			for (int axis=0;axis<3;++axis) { mesh.boundsMin[axis]=-4; mesh.boundsMax[axis]=4; }
			PsyXNativeInstance instance={}; NativeTestIdentity(instance.world);
			for (int channel=0;channel<4;++channel) instance.tint[channel]=1;
			instance.identity=1000+sample;
			if (PsyX_Native_CreateMaterial(&material,&mesh.material)!=PSYX_NATIVE_PENDING ||
				PsyX_Native_CreateMesh(&mesh,&instance.mesh)!=PSYX_NATIVE_PENDING)
			{ ++failures; break; }
			snapshot.instances=&instance; snapshot.instanceCount=1;
			std::vector<unsigned char> pixels;
			if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; break; }
			const unsigned char* actual=&pixels[(size_t(g_vk.height/2)*g_vk.width+g_vk.width/2)*4];
			int nonuniform=0;
			for (int y=-16;y<16;++y) for (int x=-16;x<16;++x)
				if (memcmp(actual,&pixels[(size_t(g_vk.height/2+y)*g_vk.width+g_vk.width/2+x)*4],4)) ++nonuniform;
			if (nonuniform) { ++failures; ReportAppend(report,reportSize,"fractional sample nonuniform interior FAIL\n"); }
			if (!filter)
			{
				const unsigned int texel=(source.y>=.5f ? 2u : 0u)+(source.x>=.5f ? 1u : 0u);
				for (unsigned int channel=0;channel<3;++channel)
					if (abs(int(actual[channel])-int(source.rgba[texel*4+channel]))>1) ++failures;
			}
			else if (filter==1) memcpy(bilinear,actual,4);
			else if (memcmp(bilinear,actual,4)) ++failures;
			if (actual[3]!=255) ++failures;
			bool outside=false;
			for (unsigned int channel=0;channel<3;++channel)
				if (fabs(actual[channel]-ideal[channel])>2) outside=true;
			if (filter && outside) ++strictOutside;
			snprintf(line,sizeof(line),"fractional sample%u filter%u RGB=%u,%u,%u ideal=%.4f,%.4f,%.4f strict2=%s (accuracy diagnostic)\n",
				sample,filter,actual[0],actual[1],actual[2],ideal[0],ideal[1],ideal[2],outside ? "outside" : "inside");
			ReportAppend(report,reportSize,line);
			PsyXNativeStats stats; PsyX_Native_GetStats(&stats);
			const uint64_t meshBytes=vertices.size()*sizeof(PsyXNativeVertex)+indices.size()*sizeof(uint32_t);
			// Base-only filters own 16 bytes; trilinear additionally owns a 1x1 mip.
			const uint64_t materialBytes=filter==PSYX_NATIVE_TRILINEAR ? 20 : 16;
			const bool resources=stats.ownedBytes==meshBytes && stats.ownedMaterialBytes==materialBytes &&
				stats.residentMeshes==1 && stats.residentMaterials==1 && !stats.pendingMeshes &&
				!stats.retiringMeshes && !stats.pendingMaterials && !stats.retiringMaterials;
			snprintf(line,sizeof(line),"fractional owned mesh=%llu material=%llu expected=%llu/%llu %s\n",
				(unsigned long long)stats.ownedBytes,(unsigned long long)stats.ownedMaterialBytes,
				(unsigned long long)meshBytes,(unsigned long long)materialBytes,resources ? "ok" : "FAIL");
			ReportAppend(report,reportSize,line); if (!resources) ++failures;
		}
	}
	PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
	PsyXNativeStats stats; PsyX_Native_GetStats(&stats);
	if (stats.ownedBytes || stats.ownedMaterialBytes || stats.retiringMeshes || stats.retiringMaterials) ++failures;
	snprintf(line,sizeof(line),"fractional sampling readback/uniformity/nearest/filter-consistency/retirement %s; strict2 outside=%d, accuracy unaccepted\n",
		failures ? "FAIL" : "PASS",strictOutside);
	ReportAppend(report,reportSize,line);
	return failures;
}

static int NativeTestPicking(PsyXNativeSnapshot snapshot, char* report, int reportSize)
{
	snapshot.sceneGeneration=PsyX_Native_ResetScene();
	NativeTestIdentity(snapshot.view.view);
	snapshot.view.projection[0]=1; snapshot.view.projection[5]=-1;
	const float white[]={1,1,1,1};
	const uint8_t artwork[]={255,0,0,0, 255,255,255,255}; // transparent red must fall through
	PsyXNativeMaterialDesc material={};
	material.size=sizeof(material); material.version=PSYX_NATIVE_VERSION;
	material.rgba=artwork; material.width=2; material.height=1; material.byteCount=8;
	material.filter=PSYX_NATIVE_TRILINEAR; material.alphaCutoff=.5f; material.cull=PSYX_NATIVE_CULL_NONE;
	PsyXNativeMaterialHandle cutout={};
	if (PsyX_Native_CreateMaterial(&material,&cutout)!=PSYX_NATIVE_PENDING) return 1;
	PsyXNativeInstance instances[3]={};
	for (unsigned int item=0; item<3; ++item)
	{
		NativeTestIdentity(instances[item].world);
		for (unsigned int channel=0; channel<4; ++channel) instances[item].tint[channel]=1;
		instances[item].identity=UINT64_C(0x100000000)+101+item;
		instances[item].layer=item==2 ? PSYX_NATIVE_BACKDROP : PSYX_NATIVE_WORLD;
		const float extent=item==0 ? .5f : .999f;
		std::vector<PsyXNativeVertex> vertices; std::vector<uint32_t> indices;
		NativeTestRectangle(vertices,indices,-extent,-extent,extent,extent,item==0 ? 4.0f : 8.0f,white,0);
		PsyXNativeMeshDesc mesh={}; mesh.size=sizeof(mesh); mesh.version=PSYX_NATIVE_VERSION;
		mesh.vertices=vertices.data(); mesh.vertexCount=uint32_t(vertices.size());
		mesh.indices=indices.data(); mesh.indexCount=uint32_t(indices.size());
		for (unsigned int axis=0; axis<3; ++axis) { mesh.boundsMin[axis]=-8; mesh.boundsMax[axis]=8; }
		if (!item) mesh.material=cutout;
		if (PsyX_Native_CreateMesh(&mesh,&instances[item].mesh)!=PSYX_NATIVE_PENDING) return 1;
	}
	snapshot.instances=instances; snapshot.instanceCount=3;
	std::vector<unsigned char> pixels;
	int failures=0;
	// Fail after an actual local buffer allocation/map, before publication or
	// command submission. Native colour remains valid, then the next click retries.
	g_nativePickReadbackFault=1;
	PsyX_Native_RequestPick(g_vk.width/2,g_vk.height/2);
	PsyXNativePickResult allocationResult={};
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) ++failures;
	PsyX_Vk_GameBeginFrame();
	if (PsyX_Native_GetPickResult(&allocationResult)!=PSYX_NATIVE_UPLOAD_FAILED ||
		g_nativeVk.pickBuffer || g_nativeVk.pickBufferMemory || g_nativeVk.pickMapped) ++failures;
	else ReportAppend(report,reportSize,"native pick actual-allocation rollback preserves world and permits retry ok\n");
	// World array order reverses, while GPU depth and source identities stay fixed.
	for (unsigned int order=0; order<2; ++order)
	{
		if (order) std::swap(instances[0],instances[1]);
		for (unsigned int sample=0; sample<4; ++sample)
		{
			int width=0,height=0; SDL_GetWindowSize(g_vk.window,&width,&height);
			const int x=sample==0 ? width*13/20 : sample==1 ? width*7/20 : sample==2 ? 32 : width*9/10;
			const int y=sample==2 ? 32 : height/2;
			PsyXNativePickResult result={};
			if (PsyX_Native_RequestPick(x,y)!=PSYX_NATIVE_OK || PsyX_Native_GetPickResult(&result)!=PSYX_NATIVE_PENDING ||
				!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; continue; }
			PsyX_Vk_GameBeginFrame(); // ordinary owner fence, no special picking wait
			const PsyXNativeInstance& expected=instances[sample==0 ? order : 1-order];
			const bool ok=PsyX_Native_GetPickResult(&result)==PSYX_NATIVE_OK && result.hit &&
				result.identity==expected.identity && result.mesh.slot==expected.mesh.slot && result.mesh.generation==expected.mesh.generation &&
				result.sceneGeneration==snapshot.sceneGeneration && result.simulationTick==snapshot.simulationTick &&
				result.material.generation==(sample==0 ? cutout.generation : 0);
			char line[240]; snprintf(line,sizeof(line),"native pick order%u %s: identity%llu instance%u %s\n",order,
				sample==0 ? "near cutout solid" : sample==1 ? "cutout hole falls through" : sample==2 ? "UI leaves world ID" : "far world",
				(unsigned long long)result.identity,result.instanceIndex,ok ? "ok" : "FAIL");
			ReportAppend(report,reportSize,line); if (!ok) ++failures;
		}
	}
	// Two placements share the identical mesh/material but remain separate logical objects.
	PsyXNativeInstance pair[3]={instances[1],instances[1],instances[2]}; // instances[1] is the front after reversing
	pair[0].identity=9001; pair[1].identity=9002;
	pair[0].world[0]=pair[1].world[0]=.4f;
	pair[0].world[12]=-2; pair[1].world[12]=2;
	snapshot.instances=pair; snapshot.instanceCount=3;
	for (unsigned int item=0; item<2; ++item)
	{
		int width=0,height=0; SDL_GetWindowSize(g_vk.window,&width,&height);
		PsyXNativePickResult result={};
		PsyX_Native_RequestPick(item ? width*8/10 : width*3/10,height/2); // opaque half of each copy
		if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; continue; }
		PsyX_Vk_GameBeginFrame();
		const bool ok=PsyX_Native_GetPickResult(&result)==PSYX_NATIVE_OK && result.hit && result.identity==pair[item].identity &&
			result.mesh.slot==pair[item].mesh.slot && result.material.generation==cutout.generation;
		ReportAppend(report,reportSize,ok ? "shared native mesh retains independent logical pick ok\n" : "shared native mesh pick FAIL\n");
		if (!ok) ++failures;
	}
	// A pure backdrop writes zero; a hole without world behind also remains zero.
	snapshot.instances=&instances[1]; snapshot.instanceCount=2; // front followed by backdrop
	for (unsigned int sample=0; sample<2; ++sample)
	{
		int width=0,height=0; SDL_GetWindowSize(g_vk.window,&width,&height);
		PsyXNativePickResult result={};
		PsyX_Native_RequestPick(sample ? width/10 : width*7/20,height/2);
		if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) { ++failures; continue; }
		PsyX_Vk_GameBeginFrame();
		const bool ok=PsyX_Native_GetPickResult(&result)==PSYX_NATIVE_OK && !result.hit;
		ReportAppend(report,reportSize,ok ? "backdrop/cutout void remains a native pick miss ok\n" : "native backdrop/void pick FAIL\n");
		if (!ok) ++failures;
	}
	int width=0,height=0; SDL_GetWindowSize(g_vk.window,&width,&height);
	PsyXNativePickResult result={};
	PsyX_Native_RequestPick(width*13/20,height/2);
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) ++failures;
	PsyX_Native_RequestPick(width/10,height/2); // supersede before owner completion publishes old result
	PsyX_Vk_GameBeginFrame();
	if (PsyX_Native_GetPickResult(&result)!=PSYX_NATIVE_PENDING || !NativeTestFrame(snapshot,0,pixels,report,reportSize)) ++failures;
	PsyX_Vk_GameBeginFrame();
	if (PsyX_Native_GetPickResult(&result)!=PSYX_NATIVE_OK || result.hit) ++failures;
	else ReportAppend(report,reportSize,"new request supersedes submitted pick without old-hit flash ok\n");
	PsyX_Native_RequestPick(width*13/20,height/2);
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) ++failures;
	PsyX_Native_CancelPick(); PsyX_Vk_GameBeginFrame();
	if (PsyX_Native_GetPickResult(&result)!=PSYX_NATIVE_NO_SNAPSHOT) ++failures;
	else ReportAppend(report,reportSize,"cancel survives actual GPU completion ok\n");
	PsyX_Native_RequestPick(width*13/20,height/2);
	SDL_SetWindowSize(g_vk.window,800,600); SDL_PumpEvents();
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) ++failures;
	PsyX_Vk_GameBeginFrame();
	if (PsyX_Native_GetPickResult(&result)!=PSYX_NATIVE_STALE || result.hit) ++failures;
	else ReportAppend(report,reportSize,"resized native targets reject stale window request ok\n");
	SDL_SetWindowSize(g_vk.window,width,height); SDL_PumpEvents();
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) ++failures;
	PsyX_Native_RequestPick(width*13/20,height/2);
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize)) ++failures;
	PsyX_Native_DestroyMesh(instances[1].mesh); PsyX_Vk_GameBeginFrame();
	if (PsyX_Native_GetPickResult(&result)!=PSYX_NATIVE_STALE || result.hit) ++failures;
	else ReportAppend(report,reportSize,"submitted pick rejects retired mesh handle ok\n");
	PsyX_Native_ResetScene(); PsyX_Vk_GameBeginFrame();
	if (PsyX_Native_GetPickResult(&result)!=PSYX_NATIVE_NO_SNAPSHOT) ++failures;
	ReportAppend(report,reportSize,failures ? "native GPU picking FAIL\n" : "native GPU picking PASS\n");
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
	const float boundaryFailure[] = {32.0f/255,0,32.0f/255,1};
	if (!NativeTestFrame(snapshot,0,pixels,report,reportSize,0,PSYX_NATIVE_NO_BOUNDARY,false) ||
		!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,g_vk.width/2,g_vk.height/2,boundaryFailure,3,
			"valid native snapshot without boundary fails explicitly",report,reportSize)) ++failures;
	// The ordinary native frames below prove recovery after restoring the boundary.
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
	failures += NativeTestFractionalSampling(snapshot, report, reportSize);
	failures += NativeTestMips(snapshot, background, report, reportSize);
	failures += NativeTestBackdrop(snapshot, report, reportSize);
	failures += NativeTestPicking(snapshot, report, reportSize);
	failures += NativeTestLargeOrigin(snapshot, report, reportSize);
	failures += NativeTestHorizontal(snapshot, report, reportSize);
	failures += NativeTestCoplanar(snapshot, report, reportSize);
	failures += NativeTestUploadFailure(snapshot, report, reportSize);
	PsyX_Native_SetRequested(0);
	// Off/auxiliary consumers remain explicit. Every unavailable world state
	// suppresses the old world, distinguishes loading and has fresh GPU pixels.
	const PsyXNativeResult reasons[] = { PSYX_NATIVE_OK, PSYX_NATIVE_OK, PSYX_NATIVE_PENDING,
		PSYX_NATIVE_NO_SNAPSHOT, PSYX_NATIVE_NO_BOUNDARY, PSYX_NATIVE_UNSUPPORTED,
		PSYX_NATIVE_OUT_OF_BUDGET, PSYX_NATIVE_STALE, PSYX_NATIVE_INVALID };
	const float failureColor[] = {32.0f/255,0,32.0f/255,1};
	const float loadingColor[] = {0,0,32.0f/255,1};
	for (unsigned int mode = 0; mode < sizeof(reasons)/sizeof(reasons[0]); ++mode)
	{
		const int requested = mode != 0;
		PsyX_Native_SetRequested(requested);
		PsyX_Vk_GameBeginFrame();
		if (mode >= 2) PsyX_Native_SetFrameStatus(mode==3 ? PSYX_NATIVE_OK : reasons[mode]);
		const PsyXNativeResult pickRequest=PsyX_Native_RequestPick(1,1);
		if (pickRequest!=(requested ? PSYX_NATIVE_OK : PSYX_NATIVE_UNSUPPORTED)) ++failures;
		VkPsxVertex quad[6];
		PsxFillQuad(float(g_vk.width), float(g_vk.height), 0, 0, 0, 0, quad);
		PsyX_Vk_GameUpdateVertexBuffer(quad, 6);
		PsyX_Vk_GameDrawTriangles(0, 2);
		if (mode >= 2 && reasons[mode] != PSYX_NATIVE_NO_BOUNDARY)
		{
			PsyX_Vk_GameModernSceneBoundary();
			PsyX_Vk_GameSetScissor(1,0,g_vk.height-64,64,64);
			PsyX_Vk_GameDrawTriangles(0,2);
			PsyX_Vk_GameSetScissor(0,0,0,g_vk.width,g_vk.height);
		}
		if (!PsyX_Vk_RenderFrame()) { ++failures; continue; }
		pixels.resize(size_t(g_vk.width) * g_vk.height * 4);
		if (!PsyX_Vk_ReadbackRgba(pixels.data(), NULL, NULL)) { ++failures; continue; }
		PsyX_Native_GetStats(&stats);
		const PsyXNativeFrameState expectedState = mode == 0 ? PSYX_NATIVE_FRAME_OFF : mode == 1 ?
			PSYX_NATIVE_FRAME_AUXILIARY : mode == 2 ? PSYX_NATIVE_FRAME_LOADING : PSYX_NATIVE_FRAME_FAILED;
		if (stats.effective || stats.nativeDraws || stats.frameState != expectedState || stats.reason != reasons[mode] ||
			stats.legacyWorldDraws != (mode<2 ? 1u : 0u) || stats.suppressedWorldDraws != (mode<2 ? 0u : 1u) ||
			stats.legacyOverlayDraws != (mode<2 || reasons[mode]==PSYX_NATIVE_NO_BOUNDARY ? 0u : 1u)) ++failures;
		if (requested)
		{
			PsyXNativePickResult unavailable={};
			if (PsyX_Native_GetPickResult(&unavailable)!=(mode==1 ? PSYX_NATIVE_UNSUPPORTED : reasons[mode]) || unavailable.hit) ++failures;
		}
		if (!PsxCheckPixel(pixels.data(), g_vk.width, g_vk.height, g_vk.width / 2, g_vk.height / 2,
			mode<2 ? ui : mode==2 ? loadingColor : failureColor, 3, mode<2 ? "explicit off/auxiliary consumer" :
			mode==2 ? "expected loading, no old world pixels" : "native failure, no old world pixels",
			report, reportSize)) ++failures;
		if (mode>=2 && reasons[mode]!=PSYX_NATIVE_NO_BOUNDARY &&
			!PsxCheckPixel(pixels.data(),g_vk.width,g_vk.height,32,32,ui,3,"remaining UI consumer over native status",report,reportSize)) ++failures;
	}
	PsyX_Native_SetRequested(0);
	ReportAppend(report, reportSize, failures ? "native self-test FAIL\n" : "native self-test PASS\n");
	return failures == 0;
}
