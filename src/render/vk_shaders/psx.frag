#version 450
// Vulkan port of the emulated PSX GPU fragment path (renderer roadmap R7b).
// Faithful to PsyX_render.cpp's 4/8/16/32-bit PSX shaders, merged into a single
// shader selected by the texFormat push constant:
//   0 = 4-bit CLUT, 1 = 8-bit CLUT, 2 = 16-bit direct, 3 = 32-bit RGBA.
// The RG32F VRAM image is sampled directly and CLUT/texture-window lookups are
// done here, exactly as in the OpenGL renderer.

layout(location = 0) in vec4 v_texcoord;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec4 v_page_clut;
layout(location = 3) in float v_z;

layout(set = 0, binding = 1) uniform sampler2D s_vram;
layout(set = 0, binding = 2) uniform sampler2D s_rgLut;
layout(set = 0, binding = 3) uniform sampler2D s_texture;

layout(push_constant) uniform Push
{
	int texFormat;
	int bilinearFilter;
	vec2 texelSize;
	int overrideAlphaMode;
} pc;

layout(location = 0) out vec4 fragColor;

const vec2 c_VRAMTexel = vec2(1.0 / 1024.0, 1.0 / 512.0);
const vec2 c_LUTTexel = vec2(1.0 / 256.0, 1.0 / 256.0);

const mat4 c_dither = mat4(
	-4.0, +0.0, -3.0, +1.0,
	+2.0, -2.0, +3.0, -1.0,
	-3.0, +1.0, -4.0, +0.0,
	+3.0, -1.0, +2.0, -2.0) / 255.0;

vec2 VRAM(vec2 uv) { return texture(s_vram, uv).rg; }
float _idx2(vec2 array, int idx) { return array[idx]; }
vec4 lut(vec2 rg) { return texture(s_rgLut, rg - c_LUTTexel * 0.0001); }

vec4 dither(vec4 color)
{
	ivec2 dc = ivec2(fract(gl_FragCoord.xy / 4.0) * 4.0);
	color.xyz += vec3(c_dither[dc.x][dc.y] * v_texcoord.w);
	return color;
}

vec2 samplePSX(vec2 tc)
{
	if (pc.texFormat == 0)
	{
		// 4-bit CLUT: two nibbles per VRAM byte.
		vec2 uv = (tc * vec2(0.25, 1.0) + v_page_clut.xy) * c_VRAMTexel;
		vec2 comp = VRAM(uv);
		int index = int(fract(tc.x / 4.0 + 0.0001) * 4.0);
		float v = _idx2(comp, index / 2) * (255.0 / 16.0);
		float f = floor(v + 0.001);
		vec2 c = vec2((v - f) * 16.0, f);
		vec2 clut_pos = v_page_clut.zw;
		clut_pos.x += mix(c[0], c[1], mod(float(index), 2.0)) * c_VRAMTexel.x;
		return VRAM(clut_pos);
	}
	else if (pc.texFormat == 1)
	{
		// 8-bit CLUT.
		vec2 uv = (tc * vec2(0.5, 1.0) + v_page_clut.xy) * c_VRAMTexel;
		vec2 comp = VRAM(uv);
		vec2 clut_pos = v_page_clut.zw;
		int index = int(mod(tc.x, 2.0));
		clut_pos.x += _idx2(comp, index) * 255.0 * c_VRAMTexel.x;
		return VRAM(clut_pos);
	}

	// 16-bit direct colour.
	vec2 uv = (tc + v_page_clut.xy) * c_VRAMTexel;
	return VRAM(uv);
}

vec4 bilinearTextureSample(vec2 P)
{
	vec2 frac = fract(P);
	vec2 pixel = floor(P);
	vec2 C11 = samplePSX(pixel);
	vec2 C21 = samplePSX(pixel + vec2(1.0, 0.0));
	vec2 C12 = samplePSX(pixel + vec2(0.0, 1.0));
	vec2 C22 = samplePSX(pixel + vec2(1.0, 1.0));
	float ax1 = mix(float(C11.r + C11.g > 0.0), float(C21.r + C21.g > 0.0), frac.x);
	float ax2 = mix(float(C12.r + C12.g > 0.0), float(C22.r + C22.g > 0.0), frac.x);
	float axm = mix(ax1, ax2, frac.y);
	if (axm < 0.5)
		discard;

	vec4 x1 = mix(lut(C11), lut(C21), frac.x);
	vec4 x2 = mix(lut(C12), lut(C22), frac.x);
	vec4 t = mix(x1, x2, frac.y);
	t.w = 1.0 - t.w;
	return t;
}

vec4 nearestTextureSample(vec2 P)
{
	vec2 rg = samplePSX(P);
	float rgm = rg.x + rg.y;
	if (rgm == 0.0)
		discard;

	vec4 t = lut(rg);
	t.w = 1.0 - t.w;
	return t;
}

void main()
{
	vec4 color;

	if (pc.texFormat == 3)
	{
		// 32-bit RGBA (custom/override textures).
		vec2 tc = v_texcoord.xy * pc.texelSize + pc.texelSize * 0.5;
		color = texture(s_texture, tc);
		if (pc.overrideAlphaMode == 1 && color.a < 0.5)
			discard;
		else if (pc.overrideAlphaMode == 2 && color.a < (0.5 / 255.0))
			discard;
	}
	else
	{
		color = (pc.bilinearFilter > 0)
			? bilinearTextureSample(v_texcoord.xy)
			: nearestTextureSample(v_texcoord.xy);
	}

	fragColor = dither(color * v_color);
}
