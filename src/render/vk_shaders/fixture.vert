#version 450
// Vulkan port of the experimental modern mesh path (renderer roadmap R7).
// One push-constant block per instance: world transform + tint.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inUv;

layout(push_constant) uniform Push
{
	mat4 world;
	vec4 color;
	vec4 factors;	// x = metallic, y = roughness, z = emissive scale
} pc;

#define PSYX_VK_MAX_LIGHTS 8

struct VkLight
{
	vec4 posRange;	// xyz = world position, w = range
	vec4 dirType;	// xyz = world direction towards the light, w = type
	vec4 color;		// rgb = colour * intensity
};

layout(set = 0, binding = 0) uniform SceneUBO
{
	mat4 view;
	mat4 proj;
	mat4 shadowMatrix;
	vec4 cameraPos;
	vec4 ambientExposure;
	vec4 shadowParams;	// x = shadows enabled, y = texel, z = strength, w = ao
	vec4 lightInfo;		// x = light count, y = srgb output
	VkLight lights[PSYX_VK_MAX_LIGHTS];
} u;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec3 vViewPos;
layout(location = 2) out vec3 vNormal;
layout(location = 3) out vec2 vUv;
layout(location = 4) out vec3 vWorldPos;

void main()
{
	vec4 world = pc.world * vec4(inPosition, 1.0);
	vec4 view = u.view * world;

	gl_Position = u.proj * view;
	vWorldPos = world.xyz;
	vViewPos = view.xyz;
	vNormal = mat3(u.view) * mat3(pc.world) * inNormal;
	vUv = inUv;
	vColor = inColor * pc.color;
}
