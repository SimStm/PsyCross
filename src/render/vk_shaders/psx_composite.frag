#version 450
// Vulkan port of the modern shadow composite (renderer roadmap R5) and of the
// legacy lighting receptivity term (roadmap legacy-lighting-receptivity):
// rewrites the already-rendered legacy scene with the modern shadow map and,
// when enabled, a diffuse sun term from the modern light set applied. Legacy
// shading itself is preserved. The world position is reconstructed from the
// copied scene depth through the same projection and camera transform the
// modern meshes use.
//
// The scene colour is read from a copy instead of being multiplied through the
// blend stage: the blend stage clamps a source value above 1.0 on a
// fixed-point attachment, so the sunlight term (which brightens) cannot be
// expressed as a blend factor.

#define PSYX_VK_MAX_LIGHTS 8

layout(location = 0) out vec4 fragColor;

struct VkLight
{
	vec4 posRange;
	vec4 dirType;
	vec4 color;
};

layout(set = 0, binding = 0) uniform ModernUBO
{
	mat4 proj;
	mat4 projInverse;
	mat4 shadowMatrix;
	mat4 cameraViewInverse;
	mat4 cameraRotation;
	vec4 xyScale;
	vec4 shadowParams;	// x = shadows enabled, y = texel, z = strength, w = ao
	vec4 lightInfo;		// x = light count, y = srgb output, z = shadow debug, w = legacy light scale
	vec4 ambientExposure;
	vec4 cameraPos;
	vec4 viewport;
	VkLight lights[PSYX_VK_MAX_LIGHTS];
} u;

layout(set = 0, binding = 1) uniform sampler2D s_shadowMap;
layout(set = 0, binding = 2) uniform sampler2D s_sceneDepth;
layout(set = 0, binding = 3) uniform sampler2D s_sceneColor;

// Reconstructs the view-space position of a scene pixel from its depth. The
// PSX depth is a function of camera-space z, so the inverse of the projection
// the modern meshes use turns it back exactly.
vec3 ReconstructView(vec2 uv, float depth)
{
	vec3 ndc = vec3(uv * 2.0 - 1.0, depth * 2.0 - 1.0);
	vec4 s = u.projInverse * vec4(ndc, 1.0);
	s /= s.w;
	return vec3(s.x / u.xyScale.x, -s.y / u.xyScale.y, s.z / u.xyScale.z);
}

void main()
{
	vec2 uv = gl_FragCoord.xy / u.viewport.xy;
	float depth = texture(s_sceneDepth, uv).r;
	vec3 scene = texture(s_sceneColor, uv).rgb;
	int debugMode = int(u.lightInfo.z + 0.5);

	// Scene depth split over two channels, for reading the reconstruction's
	// input at more than 8-bit precision from a screenshot: red holds the
	// fraction, green the integral part, both of depth * 255.
	if (debugMode == 8)
	{
		float t = depth * 255.0;
		fragColor = vec4(fract(t), floor(t) / 255.0, 0.0, 1.0);
		return;
	}

	if (debugMode == 1)
	{
		fragColor = vec4(scene * depth, 1.0);
		return;
	}
	if (debugMode == 3)
	{
		fragColor = vec4(scene * texture(s_shadowMap, uv).r, 1.0);
		return;
	}

	vec3 debugColour = vec3(0.0, 0.0, 1.0);
	vec3 tint = vec3(1.0);
	vec3 N = vec3(0.0, 1.0, 0.0);
	float legacyNdl = 0.0;
	float legacyPointNdl = 0.0;

	// Reconstruct the view position for every pixel: the legacy lighting normal
	// needs the neighbouring positions too, and only world pixels consume the
	// result.
	vec3 viewPos = ReconstructView(uv, depth);
	vec3 world = (u.cameraViewInverse * vec4(viewPos, 1.0)).xyz;

	// Reconstructed distance from the camera, two-channel encoded like the
	// depth probe: value = (green + red / 255) / 255 * 65535 world units.
	if (debugMode == 9)
	{
		float t = clamp(length(world - u.cameraPos.xyz) / 65535.0, 0.0, 1.0) * 255.0;
		fragColor = vec4(fract(t), floor(t) / 255.0, 0.0, 1.0);
		return;
	}

	// The legacy 2D path (HUD, screen overlays) writes a constant ~0.5 depth;
	// such pixels are not world geometry and must not receive shadows or light.
	bool worldPixel = (depth < 0.999999) && (abs(depth - 0.5) > 1e-5);

	// The PSX depth buffer spends its top band on backdrop layers: the sky and
	// painted skyline sit at ~0.9997, the cloud/haze sheet at ~0.990. Those
	// pixels are flat images rather than surfaces, so the depth reconstruction
	// saturates and the reconstructed normal carries no information - shading
	// them with the sun washes the sky out.
	const float kSunMaxDepth = 0.999;
	bool sunPixel = worldPixel && depth < kSunMaxDepth;

	if (worldPixel)
	{
		vec4 sc = u.shadowMatrix * vec4(world, 1.0);
		// Vulkan shadow clip: xy in [-1,1], z already in [0,1].
		vec3 proj = vec3(sc.xy / sc.w * 0.5 + 0.5, sc.z / sc.w);
		bool inside = (proj.x >= 0.0 && proj.x <= 1.0 && proj.y >= 0.0 && proj.y <= 1.0 && proj.z <= 1.0);
		debugColour = inside ? vec3(0.0, 0.2 + 0.8 * proj.z, 0.0) : vec3(1.0, 0.0, 0.0);

		if (debugMode == 4)
		{
			float shadowDepth = inside ? texture(s_shadowMap, proj.xy).r : 1.0;
			fragColor = vec4(scene * vec3(proj.z, shadowDepth, 0.0), 1.0);
			return;
		}

		if (inside)
		{
			float ref = proj.z - 0.002;
			float lit = 0.0;
			for (int y = -1; y <= 1; y++)
				for (int x = -1; x <= 1; x++)
					lit += (ref > texture(s_shadowMap, proj.xy + vec2(float(x), float(y)) * u.shadowParams.y).r) ? 0.0 : 1.0;
			lit /= 9.0;
			tint = vec3(mix(1.0 - u.shadowParams.z, 1.0, lit));
		}

		// Legacy lighting receptivity: the legacy depth buffer has no surface
		// normal, so one is reconstructed from a five-tap cross of the
		// neighbour world positions. Screen-space derivatives at a single pixel
		// amplify the PGXP depth quantisation and the polygon-edge steps, which
		// showed up as flickering light on moving vehicles; the wider taps
		// average that out and an edge test keeps a neighbouring surface from
		// bending the normal. Each light term scales the already-lit legacy
		// colour; it does not replace the legacy shading model.
		float legacyScale = u.lightInfo.w;
		int lightCount = int(u.lightInfo.x + 0.5);
		if (sunPixel && legacyScale > 0.0 && lightCount > 0)
		{
			vec2 tapStep = 2.0 / u.viewport.xy;
			float dL = texture(s_sceneDepth, uv - vec2(tapStep.x, 0.0)).r;
			float dR = texture(s_sceneDepth, uv + vec2(tapStep.x, 0.0)).r;
			float dU = texture(s_sceneDepth, uv - vec2(0.0, tapStep.y)).r;
			float dD = texture(s_sceneDepth, uv + vec2(0.0, tapStep.y)).r;
			const float kEdgeTolerance = 0.0015;
			if (abs(dL - depth) < kEdgeTolerance && abs(dR - depth) < kEdgeTolerance &&
				abs(dU - depth) < kEdgeTolerance && abs(dD - depth) < kEdgeTolerance)
			{
				vec3 vL = ReconstructView(uv - vec2(tapStep.x, 0.0), dL);
				vec3 vR = ReconstructView(uv + vec2(tapStep.x, 0.0), dR);
				vec3 vU = ReconstructView(uv - vec2(0.0, tapStep.y), dU);
				vec3 vD = ReconstructView(uv + vec2(0.0, tapStep.y), dD);
				vec3 Nview = normalize(cross(vR - vL, vD - vU));
				vec3 V = normalize(-viewPos);
				if (dot(Nview, V) < 0.0)
					Nview = -Nview;
				N = mat3(u.cameraViewInverse) * Nview;

				// The directional sun (light 0). Its term fades out with
				// distance: beyond the shadow volume the reconstruction loses
				// precision and a hard cutoff left a visible edge. The volume
				// half-size is read from the shadow matrix itself, so the fade
				// follows the size the panel sets.
				if (u.lights[0].dirType.w < 0.5)
				{
					vec3 Lview = normalize(transpose(mat3(u.cameraViewInverse)) * u.lights[0].dirType.xyz);
					legacyNdl = max(dot(Nview, Lview), 0.0);
					float extent = 1.0 / max(u.shadowMatrix[0][0], 1e-6);
					float sunRange = 1.0 - smoothstep(extent * 2.0, extent * 4.0, length(world - u.cameraPos.xyz));
					tint *= vec3(1.0) + legacyScale * legacyNdl * u.lights[0].color.rgb * sunRange;
				}

				// Point lights (roadmap point-light-sources): the same
				// reconstructed normal, with distance attenuation from the
				// light's range. The game publishes at most a few, nearest
				// first, so the loop rejects the rest with one distance test.
				// Point lights cast no shadow: only the directional map exists.
				for (int i = 1; i < PSYX_VK_MAX_LIGHTS; i++)
				{
					if (i >= lightCount)
						break;
					if (u.lights[i].dirType.w < 0.5)
						continue;

					vec3 toLight = u.lights[i].posRange.xyz - world;
					float dist = length(toLight);
					float atten = clamp(1.0 - dist / max(u.lights[i].posRange.w, 1.0), 0.0, 1.0);
					if (atten <= 0.0)
						continue;

					vec3 Lview = normalize(transpose(mat3(u.cameraViewInverse)) * toLight);
					float pointNdl = max(dot(Nview, Lview), 0.0);
					legacyPointNdl = max(legacyPointNdl, pointNdl * atten * atten);
					tint *= vec3(1.0) + legacyScale * pointNdl * u.lights[i].color.rgb * atten * atten;
				}
			}
		}

		// Receptivity diagnostics. Each probe replaces the tint, so the frame
		// it is captured in shows the untouched scene scaled by the probe:
		// divide by a frame captured with the composite off to read the value.
		if (debugMode == 5)
		{
			fragColor = vec4(scene * vec3(sunPixel ? 0.25 + 0.75 * legacyNdl : 1.0), 1.0);
			return;
		}
		if (debugMode == 6)
		{
			fragColor = vec4(scene * (N * 0.5 + 0.5), 1.0);
			return;
		}
		if (debugMode == 7)
		{
			fragColor = vec4(scene * vec3(sunPixel ? 0.25 + 0.75 * clamp(legacyScale, 0.0, 1.0) : 1.0), 1.0);
			return;
		}
		if (debugMode == 10)
		{
			fragColor = vec4(scene * vec3(sunPixel ? 0.35 : 1.0), 1.0);
			return;
		}
		if (debugMode == 11)
		{
			fragColor = vec4(scene * vec3(sunPixel ? 0.25 + 0.75 * legacyPointNdl : 1.0), 1.0);
			return;
		}
	}

	if (debugMode == 2)
	{
		fragColor = vec4(scene * debugColour, 1.0);
		return;
	}

	fragColor = vec4(scene * tint, 1.0);
}
