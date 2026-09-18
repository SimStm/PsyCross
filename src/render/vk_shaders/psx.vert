#version 450
// Vulkan port of the emulated PSX GPU vertex path (renderer roadmap R7b).
// Faithful to PsyX_render.cpp's GTE_VERTEX_SHADER; the only additions are the
// GL-to-Vulkan clip-space conversion at the end (NDC Y down, depth [0,1]).
//
// Vertex layout is the packed GrVertex used by the game renderer with PGXP
// enabled (44 bytes): a_position at 0, a_zw at 16, a_texcoord at 32,
// a_color at 36, a_extra at 40.

layout(location = 0) in vec4 a_position;	// xy = screen pos, z = tpage, w = clut
layout(location = 1) in vec4 a_zw;		// x = z, y = scr_h, zw = projection offset
layout(location = 2) in uvec4 a_texcoord;	// uv, colour multiplier, dither
layout(location = 3) in vec4 a_color;		// normalised vertex colour
layout(location = 4) in ivec4 a_extra;		// texture coordinate nudge

layout(set = 0, binding = 0) uniform PsxUBO
{
	mat4 Projection;
	mat4 Projection3D;
} u;

layout(location = 0) out vec4 v_texcoord;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec4 v_page_clut;
layout(location = 3) out float v_z;

const vec2 c_UVFudge = vec2(0.00025, 0.00025);

void main()
{
	v_texcoord = vec4(a_texcoord);
	v_texcoord.xy += vec2(a_extra.xy) * 0.5;
	v_color = a_color;
	v_color.xyz *= float(a_texcoord.z);

	v_page_clut.x = fract(a_position.z / 16.0) * 1024.0;
	v_page_clut.y = floor(a_position.z / 16.0) * 256.0;
	v_page_clut.z = fract(a_position.w / 64.0);
	v_page_clut.w = floor(a_position.w / 64.0) / 512.0;
	v_page_clut.xy += c_UVFudge;
	v_page_clut.zw += c_UVFudge;

	mat4 ofsMat = mat4(
		vec4(1.0, 0.0, 0.0, 0.0),
		vec4(0.0, 1.0, 0.0, 0.0),
		vec4(0.0, 0.0, 1.0, 0.0),
		vec4(a_zw.z, -a_zw.w, 0.0, 1.0));
	vec2 geom_ofs = vec2(0.5, 0.5);

	vec4 pos = (a_zw.y > 100.0)
		? ofsMat * (u.Projection3D * vec4((a_position.xy + geom_ofs) * vec2(1.0, -1.0) * a_zw.y, a_zw.x, 1.0))
		: (u.Projection * vec4(a_position.xy, 0.5, 1.0));

	// Computed from the GL clip position, exactly like the OpenGL shader.
	v_z = (pos.z - 40.0) * 0.005;

	// GL clip space to Vulkan clip space: Y points down and depth is [0,1].
	pos.y = -pos.y;
	pos.z = (pos.z + pos.w) * 0.5;
	gl_Position = pos;
}
