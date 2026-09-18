#version 450
// Vulkan port of the reference Forward PBR shader (renderer roadmap R4/R7).

#define PSYX_VK_MAX_LIGHTS 8

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec3 vViewPos;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in vec2 vUv;
layout(location = 4) in vec3 vWorldPos;

layout(location = 0) out vec4 fragColor;

struct VkLight
{
	vec4 posRange;
	vec4 dirType;
	vec4 color;
};

layout(set = 0, binding = 0) uniform SceneUBO
{
	mat4 view;
	mat4 proj;
	mat4 shadowMatrix;
	vec4 cameraPos;
	vec4 ambientExposure;
	vec4 shadowParams;
	vec4 lightInfo;
	VkLight lights[PSYX_VK_MAX_LIGHTS];
} u;

layout(set = 0, binding = 1) uniform sampler2D s_shadowMap;
layout(set = 0, binding = 2) uniform sampler2D s_texture;
layout(set = 0, binding = 3) uniform sampler2D s_normalMap;
layout(set = 0, binding = 4) uniform sampler2D s_mrMap;
layout(set = 0, binding = 5) uniform sampler2D s_emissiveMap;

layout(push_constant) uniform Push
{
	mat4 world;
	vec4 color;
	vec4 factors;	// x = metallic, y = roughness, z = emissive scale
} pc;

const float kPi = 3.14159265359;

vec3 ToLinear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec3 ToSrgb(vec3 c) { return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)); }

float ShadowFactor(float NdotL)
{
	if (u.shadowParams.x < 0.5)
		return 1.0;

	vec4 sc = u.shadowMatrix * vec4(vWorldPos, 1.0);
	// Vulkan clip space: xy in [-1,1] but z already in [0,1].
	vec3 proj = vec3(sc.xy / sc.w * 0.5 + 0.5, sc.z / sc.w);
	if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0)
		return 1.0;

	float bias = max(0.0015 * (1.0 - NdotL), 0.0004);
	float depth = texture(s_shadowMap, proj.xy).r;
	return (proj.z - bias) > depth ? (1.0 - u.shadowParams.z) : 1.0;
}

vec3 ShadeLight(vec3 N, vec3 V, vec3 base, float metallic, float roughness, vec3 L, vec3 radiance)
{
	vec3 H = normalize(V + L);
	float NdotL = max(dot(N, L), 0.0);
	float NdotV = max(dot(N, V), 1e-4);
	float NdotH = max(dot(N, H), 0.0);
	float VdotH = max(dot(V, H), 0.0);
	float a = max(roughness * roughness, 1e-3);
	float a2 = a * a;
	float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
	float D = a2 / (kPi * denom * denom);
	float k = a * 0.5;
	float G = (NdotL / (NdotL * (1.0 - k) + k)) * (NdotV / (NdotV * (1.0 - k) + k));
	vec3 F0 = mix(vec3(0.04), base, metallic);
	vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
	vec3 spec = (D * G) * F / (4.0 * NdotV * NdotL + 1e-4);
	vec3 kd = (1.0 - F) * (1.0 - metallic);
	vec3 diffuse = kd * base / kPi;
	return (diffuse + spec) * radiance * NdotL;
}

void main()
{
	vec3 base = ToLinear(vColor.rgb);
	float alpha = vColor.a;
	vec4 texel = texture(s_texture, vUv);
	base *= ToLinear(texel.rgb);
	alpha *= texel.a;

	float metallic = pc.factors.x;
	float roughness = pc.factors.y;

	vec4 mr = texture(s_mrMap, vUv);
	roughness *= mr.g;
	metallic *= mr.b;

	roughness = clamp(roughness, 0.045, 1.0);
	metallic = clamp(metallic, 0.0, 1.0);

	vec3 N = normalize(vNormal);
	vec3 n = texture(s_normalMap, vUv).xyz * 2.0 - 1.0;

	vec3 dp1 = dFdx(vViewPos);
	vec3 dp2 = dFdy(vViewPos);
	vec2 duv1 = dFdx(vUv);
	vec2 duv2 = dFdy(vUv);
	vec3 dp2perp = cross(dp2, N);
	vec3 dp1perp = cross(N, dp1);
	vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
	vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
	float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
	mat3 TBN = mat3(T * invmax, B * invmax, N);
	N = normalize(TBN * n);

	vec3 V = normalize(-vViewPos);
	vec3 ambient = u.ambientExposure.rgb;
	if (u.shadowParams.w > 0.5)
		ambient *= clamp(N.y * 0.5 + 0.5, 0.25, 1.0);

	vec3 color = base * ambient;
	vec3 emission = pc.factors.z * texture(s_emissiveMap, vUv).rgb;
	color += ToLinear(emission);

	int lightCount = int(u.lightInfo.x);
	for (int i = 0; i < PSYX_VK_MAX_LIGHTS; i++)
	{
		if (i >= lightCount)
			break;

		vec3 L;
		vec3 radiance = u.lights[i].color.rgb;
		if (u.lights[i].dirType.w < 0.5)
		{
			L = normalize(mat3(u.view) * u.lights[i].dirType.xyz);
			radiance *= ShadowFactor(max(dot(N, L), 0.0));
		}
		else
		{
			vec3 lp = mat3(u.view) * u.lights[i].posRange.xyz;
			vec3 d = lp - vViewPos;
			float dist = length(d);
			L = d / max(dist, 1e-4);
			float atten = clamp(1.0 - dist / max(u.lights[i].posRange.w, 1.0), 0.0, 1.0);
			radiance *= atten * atten;
		}

		color += ShadeLight(N, V, base, metallic, roughness, L, radiance);
	}

	color *= u.ambientExposure.w;

	if (u.lightInfo.y < 0.5)
		color = ToSrgb(color);

	fragColor = vec4(color, alpha);
}
