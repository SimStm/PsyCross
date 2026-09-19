#version 450
// Vulkan port of the modern shadow composite (renderer roadmap R5): multiplies
// the pixels of the already-rendered legacy scene by the modern shadow map, so
// legacy geometry receives shadows cast by modern meshes. The world position is
// reconstructed from the copied scene depth through the same projection and
// camera transform the modern meshes use.

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
	vec4 lightInfo;		// x = light count, y = srgb output, z = shadow debug
	vec4 ambientExposure;
	vec4 cameraPos;
	vec4 viewport;
	VkLight lights[PSYX_VK_MAX_LIGHTS];
} u;

layout(set = 0, binding = 1) uniform sampler2D s_shadowMap;
layout(set = 0, binding = 2) uniform sampler2D s_sceneDepth;

void main()
{
	vec2 uv = gl_FragCoord.xy / u.viewport.xy;
	float depth = texture(s_sceneDepth, uv).r;
	int debugMode = int(u.lightInfo.z + 0.5);

	if (debugMode == 1)
	{
		fragColor = vec4(vec3(depth), 1.0);
		return;
	}
	if (debugMode == 3)
	{
		fragColor = vec4(vec3(texture(s_shadowMap, uv).r), 1.0);
		return;
	}

	vec3 debugColour = vec3(0.0, 0.0, 1.0);
	vec3 tint = vec3(1.0);

	// The legacy 2D path (HUD, screen overlays) writes a constant ~0.5 depth;
	// such pixels are not world geometry and must not receive shadows.
	bool worldPixel = (depth < 0.999999) && (abs(depth - 0.5) > 1e-5);
	if (worldPixel)
	{
		// The PSX vertex shader converted GL clip z to Vulkan's [0,1] depth, so
		// the stored value maps back to the GL NDC the inverse projection
		// expects: ndc.z = depth * 2 - 1.
		vec3 ndc = vec3(uv * 2.0 - 1.0, depth * 2.0 - 1.0);
		vec4 s = u.projInverse * vec4(ndc, 1.0);
		s /= s.w;
		vec3 viewPos = vec3(s.x / u.xyScale.x, -s.y / u.xyScale.y, s.z / u.xyScale.z);
		vec3 world = (u.cameraViewInverse * vec4(viewPos, 1.0)).xyz;

		vec4 sc = u.shadowMatrix * vec4(world, 1.0);
		// Vulkan shadow clip: xy in [-1,1], z already in [0,1].
		vec3 proj = vec3(sc.xy / sc.w * 0.5 + 0.5, sc.z / sc.w);
		bool inside = (proj.x >= 0.0 && proj.x <= 1.0 && proj.y >= 0.0 && proj.y <= 1.0 && proj.z <= 1.0);
		debugColour = inside ? vec3(0.0, 0.2 + 0.8 * proj.z, 0.0) : vec3(1.0, 0.0, 0.0);

		if (debugMode == 4)
		{
			float shadowDepth = inside ? texture(s_shadowMap, proj.xy).r : 1.0;
			fragColor = vec4(vec3(proj.z, shadowDepth, 0.0), 1.0);
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
	}

	if (debugMode == 2)
	{
		fragColor = vec4(debugColour, 1.0);
		return;
	}

	fragColor = vec4(tint, 1.0);
}
