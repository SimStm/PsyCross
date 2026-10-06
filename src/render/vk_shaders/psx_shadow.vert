#version 450
// Opaque legacy casters reuse the queued PGXP vertices (44-byte GrVertex).
// CPU selection rejects whole triangles without depth; this shader uses the
// same projection offsets and inverse reconstruction as the scene receiver.
layout(location = 0) in vec4 a_position;
layout(location = 1) in vec4 a_zw;
layout(location = 2) in uvec4 a_texcoord;
layout(location = 4) in ivec4 a_extra;
layout(location = 0) out vec4 v_texcoord;
layout(location = 2) out vec4 v_page_clut;

layout(set = 1, binding = 0) uniform ModernUBO
{
	mat4 proj;
	mat4 projInverse;
	mat4 shadowMatrix;
	mat4 cameraViewInverse;
	mat4 cameraRotation;
	vec4 xyScale;
} u;

void main()
{
	v_texcoord = vec4(a_texcoord);
	v_texcoord.xy += vec2(a_extra.xy) * 0.5;
	v_page_clut = vec4(fract(a_position.z / 16.0) * 1024.0,
		floor(a_position.z / 16.0) * 256.0, fract(a_position.w / 64.0),
		floor(a_position.w / 64.0) / 512.0) + 0.00025;
	vec4 encoded = vec4((a_position.xy + 0.5) * vec2(1.0, -1.0) * a_zw.y, a_zw.x, 1.0);
	vec4 clip = u.proj * encoded;
	clip.xy += vec2(a_zw.z, -a_zw.w) * clip.w;
	vec4 recovered = u.projInverse * clip;
	recovered /= recovered.w;
	vec3 view = vec3(recovered.x / u.xyScale.x, -recovered.y / u.xyScale.y,
		recovered.z / u.xyScale.z);
	gl_Position = u.shadowMatrix * u.cameraViewInverse * vec4(view, 1.0);
}
