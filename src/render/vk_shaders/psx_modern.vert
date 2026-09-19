#version 450
// Vulkan port of the in-game modern mesh vertex path (renderer roadmap R7b).
//
// The modern mesh must land in the same clip space as legacy geometry so the
// two mutually occlude in the shared depth buffer. It therefore reproduces the
// OpenGL path's GTE encoding exactly:
//
//   viewPos = cameraRotation * (world * position - cameraPosition)
//   clip    = Projection3D * vec4(viewPos.x * xyScaleX,
//                                 -viewPos.y * xyScale.y,
//                                 viewPos.z  * zScale, 1.0)
//
// and then converts GL clip space to Vulkan (y = -y, z = (z + w) * 0.5), the
// same conversion psx.vert performs for the emulated PSX vertices.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inUv;

layout(push_constant) uniform Push
{
	mat4 world;		// mesh local space -> world space
	vec4 color;		// per-instance tint
	vec4 factors;		// x = metallic, y = roughness, z = unused
	vec4 emissive;		// rgb = emissive factor (the map is always bound)
} pc;

#define PSYX_VK_MAX_LIGHTS 8

struct VkLight
{
	vec4 posRange;
	vec4 dirType;
	vec4 color;
};

layout(set = 0, binding = 0) uniform ModernUBO
{
	mat4 proj;		// GL-style Projection3D captured from GR_Perspective3D
	mat4 projInverse;	// same, inverted (shadow composite)
	mat4 shadowMatrix;	// world -> light clip (Vulkan depth convention)
	mat4 cameraViewInverse;	// view space -> world space (shadow composite)
	mat4 cameraRotation;	// rotation-only world -> camera (fixed point derived)
	vec4 xyScale;		// x, y = GTE screen scales, z = 1/128
	vec4 shadowParams;	// x = shadows enabled, y = texel, z = strength, w = ao
	vec4 lightInfo;		// x = light count, y = srgb output, z = shadow debug
	vec4 ambientExposure;	// rgb = ambient, w = exposure
	vec4 cameraPos;		// world-space camera position
	vec4 viewport;		// x = width, y = height
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
	vec3 viewPos = mat3(u.cameraRotation) * (world.xyz - u.cameraPos.xyz);

	vec4 src = vec4(viewPos.x * u.xyScale.x, -viewPos.y * u.xyScale.y,
		viewPos.z * u.xyScale.z, 1.0);
	gl_Position = u.proj * src;

	// GL clip space -> Vulkan: NDC y is flipped and depth is [0,1].
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;

	vColor = inColor * pc.color;
	vViewPos = viewPos;
	vNormal = normalize(mat3(u.cameraRotation) * mat3(pc.world) * inNormal);
	vUv = inUv;
	vWorldPos = world.xyz;
}
