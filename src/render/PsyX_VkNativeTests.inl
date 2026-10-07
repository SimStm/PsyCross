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
	std::vector<unsigned char>& pixels, char* report, int reportSize)
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
		PsyX_Vk_GameSetBilinear(0);
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
		const int ok = stats.effective && stats.nativeDraws == 1 && stats.legacyWorldDraws == 0 &&
			stats.suppressedWorldDraws == 1 && stats.legacyOverlayDraws == 1;
		ReportAppend(report, reportSize, ok ? "native/world/UI draw accounting ok\n" : "native draw accounting FAIL\n");
		return ok;
	}
	ReportAppend(report, reportSize, "native frame never submitted FAIL\n");
	return 0;
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
