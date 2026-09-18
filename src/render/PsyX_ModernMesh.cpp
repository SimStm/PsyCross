#include "PsyX/PsyX_public.h"
#include "PsyX/PsyX_render.h"
#include "PsyX/PsyX_globals.h"

#include "../platform.h"
#include "../gpu/PsyX_GPU.h"

#include "PsyX_ModernMesh.h"

#include "psx/gtereg.h"

#include <string.h>
#include <chrono>

/* The legacy 3D path feeds the projection vertices scaled by the GTE screen
   distance (scr_h = C2_H) after dividing the camera-space position by 128
   (512 * 1024 / 4096 = the GTE fixed-point normalisation used by PGXP).
   Reusing both keeps the modern mesh's clip position and depth identical to
   legacy geometry, which is what lets the two paths occlude each other. */
#define PSYX_MODERN_GTE_VIEW_SCALE 128.0f
#define PSYX_MODERN_MAX_MESHES 16

#if defined(USE_OPENGL)

static const char* kVertexShader =
	"#version 140\n"
	"in vec3 a_position;\n"
	"in vec4 a_color;\n"
	"in vec3 a_normal;\n"
	"in vec2 a_uv;\n"
	"uniform mat4 u_view;\n"
	"uniform mat4 u_projection;\n"
	"uniform vec2 u_xyScale;\n"
	"uniform float u_zScale;\n"
	"out vec4 v_color;\n"
	"out vec3 v_viewPos;\n"
	"out vec3 v_normal;\n"
	"out vec2 v_uv;\n"
	"out vec3 v_worldPos;\n"
	"void main()\n"
	"{\n"
	"	vec4 view = u_view * vec4(a_position, 1.0);\n"
	"	vec4 src = vec4(view.x * u_xyScale.x, -view.y * u_xyScale.y, view.z * u_zScale, 1.0);\n"
	"	gl_Position = u_projection * src;\n"
	"	v_color = a_color;\n"
	"	v_viewPos = view.xyz;\n"
	"	v_normal = mat3(u_view) * a_normal;\n"
	"	v_uv = a_uv;\n"
	"	v_worldPos = a_position;\n"
	"}\n";

static const char* kFragmentShader =
	"#version 140\n"
	"in vec4 v_color;\n"
	"in vec3 v_viewPos;\n"
	"in vec3 v_normal;\n"
	"in vec2 v_uv;\n"
	"in vec3 v_worldPos;\n"
	"out vec4 fragColor;\n"
	"uniform vec4 u_color;\n"
	"uniform vec4 u_baseColorFactor;\n"
	"uniform float u_metallicFactor;\n"
	"uniform float u_roughnessFactor;\n"
	"uniform int u_useTexture;\n"
	"uniform int u_useNormalMap;\n"
	"uniform int u_useMRMap;\n"
	"uniform int u_useEmissiveMap;\n"
	"uniform int u_useLighting;\n"
	"uniform sampler2D s_texture;\n"
	"uniform sampler2D s_normalMap;\n"
	"uniform sampler2D s_mrMap;\n"
	"uniform sampler2D s_emissiveMap;\n"
	"uniform sampler2D s_shadowMap;\n"
	"uniform mat4 u_view;\n"
	"uniform mat4 u_shadowMatrix;\n"
	"uniform int u_shadowEnabled;\n"
	"uniform int u_lightCount;\n"
	"uniform int u_lightType[8];\n"
	"uniform vec3 u_lightPos[8];\n"
	"uniform vec3 u_lightDir[8];\n"
	"uniform vec3 u_lightColor[8];\n"
	"uniform float u_lightRange[8];\n"
	"uniform vec3 u_emissiveFactor;\n"
	"uniform vec3 u_ambient;\n"
	"uniform float u_exposure;\n"
	"uniform int u_aoEnabled;\n"
	"const float kPi = 3.14159265359;\n"
	"vec3 ToLinear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }\n"
	"vec3 ToSrgb(vec3 c) { return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)); }\n"
	"float ShadowFactor(float NdotL)\n"
	"{\n"
	"	if (u_shadowEnabled == 0) return 1.0;\n"
	"	vec4 sc = u_shadowMatrix * vec4(v_worldPos, 1.0);\n"
	"	vec3 proj = sc.xyz / sc.w * 0.5 + 0.5;\n"
	"	if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0) return 1.0;\n"
	"	float bias = max(0.0015 * (1.0 - NdotL), 0.0004);\n"
	"	float depth = texture2D(s_shadowMap, proj.xy).r;\n"
	"	return (proj.z - bias) > depth ? 0.4 : 1.0;\n"
	"}\n"
	"vec3 ShadeLight(vec3 N, vec3 V, vec3 base, float metallic, float roughness, vec3 L, vec3 radiance)\n"
	"{\n"
	"	vec3 H = normalize(V + L);\n"
	"	float NdotL = max(dot(N, L), 0.0);\n"
	"	float NdotV = max(dot(N, V), 1e-4);\n"
	"	float NdotH = max(dot(N, H), 0.0);\n"
	"	float VdotH = max(dot(V, H), 0.0);\n"
	"	float a = max(roughness * roughness, 1e-3);\n"
	"	float a2 = a * a;\n"
	"	float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;\n"
	"	float D = a2 / (kPi * denom * denom);\n"
	"	float k = a * 0.5;\n"
	"	float G = (NdotL / (NdotL * (1.0 - k) + k)) * (NdotV / (NdotV * (1.0 - k) + k));\n"
	"	vec3 F0 = mix(vec3(0.04), base, metallic);\n"
	"	vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);\n"
	"	vec3 spec = (D * G) * F / (4.0 * NdotV * NdotL + 1e-4);\n"
	"	vec3 kd = (1.0 - F) * (1.0 - metallic);\n"
	"	vec3 diffuse = kd * base / kPi;\n"
	"	return (diffuse + spec) * radiance * NdotL;\n"
	"}\n"
	"void main()\n"
	"{\n"
	"	vec3 base = ToLinear(v_color.rgb * u_color.rgb * u_baseColorFactor.rgb);\n"
	"	float alpha = v_color.a * u_color.a * u_baseColorFactor.a;\n"
	"	if (u_useTexture != 0)\n"
	"	{\n"
	"		vec4 t = texture2D(s_texture, v_uv);\n"
	"		base *= ToLinear(t.rgb);\n"
	"		alpha *= t.a;\n"
	"	}\n"
	"	float metallic = u_metallicFactor;\n"
	"	float roughness = u_roughnessFactor;\n"
	"	if (u_useMRMap != 0)\n"
	"	{\n"
	"		vec4 mr = texture2D(s_mrMap, v_uv);\n"
	"		roughness *= mr.g;\n"
	"		metallic *= mr.b;\n"
	"	}\n"
	"	roughness = clamp(roughness, 0.045, 1.0);\n"
	"	metallic = clamp(metallic, 0.0, 1.0);\n"
	"	vec3 N = normalize(v_normal);\n"
	"	if (u_useNormalMap != 0)\n"
	"	{\n"
	"		vec3 n = texture2D(s_normalMap, v_uv).xyz * 2.0 - 1.0;\n"
	"		vec3 dp1 = dFdx(v_viewPos);\n"
	"		vec3 dp2 = dFdy(v_viewPos);\n"
	"		vec2 duv1 = dFdx(v_uv);\n"
	"		vec2 duv2 = dFdy(v_uv);\n"
	"		vec3 dp2perp = cross(dp2, N);\n"
	"		vec3 dp1perp = cross(N, dp1);\n"
	"		vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;\n"
	"		vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;\n"
	"		float invmax = inversesqrt(max(dot(T, T), dot(B, B)));\n"
	"		mat3 TBN = mat3(T * invmax, B * invmax, N);\n"
	"		N = normalize(TBN * n);\n"
	"	}\n"
	"	vec3 V = normalize(-v_viewPos);\n"
	"	vec3 ambient = u_ambient;\n"
	"	if (u_aoEnabled != 0)\n"
	"		ambient *= clamp(N.y * 0.5 + 0.5, 0.25, 1.0);\n"
	"	vec3 color = base * ambient;\n"
	"	vec3 emission = u_emissiveFactor;\n"
	"	if (u_useEmissiveMap != 0)\n"
	"		emission *= ToLinear(texture2D(s_emissiveMap, v_uv).rgb);\n"
	"	color += emission;\n"
	"	if (u_useLighting != 0)\n"
	"	{\n"
	"		for (int i = 0; i < 8; i++)\n"
	"		{\n"
	"			if (i >= u_lightCount) break;\n"
	"			vec3 L;\n"
	"			vec3 radiance = u_lightColor[i];\n"
	"			if (u_lightType[i] == 0)\n"
	"			{\n"
	"				L = normalize(mat3(u_view) * u_lightDir[i]);\n"
	"				radiance *= ShadowFactor(max(dot(N, L), 0.0));\n"
	"			}\n"
	"			else\n"
	"			{\n"
	"				vec3 lp = mat3(u_view) * u_lightPos[i];\n"
	"				vec3 d = lp - v_viewPos;\n"
	"				float dist = length(d);\n"
	"				L = d / max(dist, 1e-4);\n"
	"				float atten = clamp(1.0 - dist / max(u_lightRange[i], 1.0), 0.0, 1.0);\n"
	"				radiance *= atten * atten;\n"
	"			}\n"
	"			color += ShadeLight(N, V, base, metallic, roughness, L, radiance);\n"
	"		}\n"
	"	}\n"
	"	color *= u_exposure;\n"
	"	fragColor = vec4(ToSrgb(color), alpha);\n"
	"}\n";

static const char* kDepthVertexShader =
	"#version 140\n"
	"in vec3 a_position;\n"
	"uniform mat4 u_lightMatrix;\n"
	"uniform mat4 u_world;\n"
	"void main()\n"
	"{\n"
	"	gl_Position = u_lightMatrix * u_world * vec4(a_position, 1.0);\n"
	"}\n";

static const char* kDepthFragmentShader =
	"#version 140\n"
	"out vec4 fragColor;\n"
	"void main() { fragColor = vec4(1.0); }\n";

/* Draws over the already rendered legacy scene and multiplies the pixels that
   the modern shadow map says are behind a modern caster. The legacy scene has
   no world-space attributes left in the vertex stream, so the world position is
   reconstructed from the shared depth buffer through the same projection and
   camera transform the modern meshes use. */
static const char* kCompositeVertexShader =
	"#version 140\n"
	"void main()\n"
	"{\n"
	"	vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
	"	gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
	"}\n";

static const char* kCompositeFragmentShader =
	"#version 140\n"
	"uniform sampler2D s_sceneDepth;\n"
	"uniform sampler2D s_shadowMap;\n"
	"uniform mat4 u_projInverse;\n"
	"uniform mat4 u_cameraViewInverse;\n"
	"uniform vec2 u_xyScale;\n"
	"uniform float u_zScale;\n"
	"uniform mat4 u_shadowMatrix;\n"
	"uniform vec2 u_viewport;\n"
	"uniform float u_shadowTexel;\n"
	"uniform float u_strength;\n"
	"uniform int u_debugMode;\n"
	"out vec4 fragColor;\n"
	"void main()\n"
	"{\n"
	"	vec2 uv = gl_FragCoord.xy / u_viewport;\n"
	"	float depth = texture2D(s_sceneDepth, uv).r;\n"
	"	if (u_debugMode == 1)\n"
	"	{\n"
	"		fragColor = vec4(vec3(depth), 1.0);\n"
	"		return;\n"
	"	}\n"
	"	if (u_debugMode == 3)\n"
	"	{\n"
	"		fragColor = vec4(vec3(texture2D(s_shadowMap, uv).r), 1.0);\n"
	"		return;\n"
	"	}\n"
	"	vec3 debugColour = vec3(0.0, 0.0, 1.0);\n"
	"	vec3 tint = vec3(1.0);\n"
	// The legacy 2D path (HUD, screen overlays) writes a constant 0.5 depth;
	// such pixels are not world geometry and must not receive shadows.
	"	bool worldPixel = (depth < 0.999999) && (abs(depth - 0.5) > 1e-5);\n"
	"	if (worldPixel)\n"
	"	{\n"
	"		vec3 ndc = vec3(uv * 2.0 - 1.0, depth * 2.0 - 1.0);\n"
	"		vec4 s = u_projInverse * vec4(ndc, 1.0);\n"
	"		s /= s.w;\n"
	"		vec3 viewPos = vec3(s.x / u_xyScale.x, -s.y / u_xyScale.y, s.z / u_zScale);\n"
	"		vec3 world = (u_cameraViewInverse * vec4(viewPos, 1.0)).xyz;\n"
	"		vec4 sc = u_shadowMatrix * vec4(world, 1.0);\n"
	"		vec3 proj = sc.xyz / sc.w * 0.5 + 0.5;\n"
	"		bool inside = (proj.x >= 0.0 && proj.x <= 1.0 && proj.y >= 0.0 && proj.y <= 1.0 && proj.z <= 1.0);\n"
	"	debugColour = inside ? vec3(0.0, 0.2 + 0.8 * proj.z, 0.0) : vec3(1.0, 0.0, 0.0);\n"
	"		if (u_debugMode == 4)\n"
	"		{\n"
	"			float shadowDepth = inside ? texture2D(s_shadowMap, proj.xy).r : 1.0;\n"
	"			fragColor = vec4(vec3(proj.z, shadowDepth, 0.0), 1.0);\n"
	"			return;\n"
	"		}\n"
	"		if (inside)\n"
	"		{\n"
	"			float ref = proj.z - 0.002;\n"
	"			float lit = 0.0;\n"
	"			for (int y = -1; y <= 1; y++)\n"
	"				for (int x = -1; x <= 1; x++)\n"
	"					lit += (ref > texture2D(s_shadowMap, proj.xy + vec2(float(x), float(y)) * u_shadowTexel).r) ? 0.0 : 1.0;\n"
	"			lit /= 9.0;\n"
	"			tint = vec3(mix(1.0 - u_strength, 1.0, lit));\n"
	"		}\n"
	"	}\n"
	"	if (u_debugMode == 2)\n"
	"	{\n"
	"		fragColor = vec4(debugColour, 1.0);\n"
	"		return;\n"
	"	}\n"
	"	fragColor = vec4(tint, 1.0);\n"
	"}\n";

typedef struct
{
	int used;
	GLuint vao;
	GLuint vbo;
	GLuint ebo;
	GLuint cbo;
	GLuint nbo;
	GLuint ubo;
	int vertexCount;
	int indexCount;
	unsigned int texture;
	unsigned int normalTexture;
	unsigned int metallicRoughnessTexture;
	int useTexture;
	int useNormalMap;
	int useMRMap;
	unsigned int emissiveTexture;
	int useEmissiveMap;
	float emissiveFactor[3];
	float metallicFactor;
	float roughnessFactor;
	float baseColorFactor[4];
	float view[16];
	float world[16];
	float color[4];
	int visible;
} PsyXModernMesh;

static PsyXModernMesh g_meshes[PSYX_MODERN_MAX_MESHES];
static int g_meshCount = 0;
static int g_enabled = 0;
static GLuint g_program = 0;
static GLuint g_depthProgram = 0;
static GLint g_dpLightMatrix = -1;
static GLint g_dpWorld = -1;
static GLuint g_shadowFbo = 0;
static GLuint g_shadowTexture = 0;
static const int g_shadowSize = 2048;
static GLint g_uShadowMap = -1, g_uShadowMatrix = -1, g_uShadowEnabled = -1;
static float g_shadowMatrix[16] = { 0 };
static GLint g_uView = -1, g_uProjection = -1, g_uXyScale = -1, g_uZScale = -1, g_uColor = -1;
static GLint g_uBaseColorFactor = -1, g_uUseTexture = -1, g_uTexture = -1;
static GLint g_uMetallicFactor = -1, g_uRoughnessFactor = -1;
static GLint g_uUseNormalMap = -1, g_uUseMRMap = -1, g_uUseEmissiveMap = -1, g_uUseLighting = -1;
static GLint g_uNormalMap = -1, g_uMrMap = -1, g_uEmissiveMap = -1;
static GLint g_uAmbient = -1, g_uExposure = -1, g_uAoEnabled = -1;
static GLint g_uLightCount = -1, g_uLightType = -1, g_uLightPos = -1, g_uLightDir = -1, g_uLightColor = -1, g_uLightRange = -1;
static GLint g_uEmissiveFactor = -1;
static GLint g_aPosition = -1, g_aColor = -1, g_aNormal = -1, g_aUv = -1;

static PsyXModernLightSet g_lights;

static PsyXModernMeshStats g_stats;

float g_psyxModernProjection[16] = { 0 };
int   g_psyxModernProjectionValid = 0;
static float g_psyxModernProjectionInverse[16] = { 0 };
static int   g_psyxModernProjectionInverseValid = 0;

static GLuint g_compositeProgram = 0;
static GLuint g_fullscreenVao = 0;
static GLuint g_sceneDepthFbo = 0;
static GLuint g_sceneDepthTexture = 0;
static int    g_sceneDepthWidth = 0;
static int    g_sceneDepthHeight = 0;
static int    g_sceneDepthTargetOk = 0;
static GLint  g_cuSceneDepth = -1, g_cuShadowMap = -1, g_cuProjInverse = -1;
static GLint  g_cuCameraViewInverse = -1, g_cuXyScale = -1, g_cuZScale = -1;
static GLint  g_cuShadowMatrix = -1, g_cuViewport = -1, g_cuShadowTexel = -1, g_cuStrength = -1;
static GLint  g_cuDebugMode = -1;

static float g_modernCameraRotation[16] =
{
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};
static float g_modernCameraPosition[3] = { 0.0f, 0.0f, 0.0f };
static float g_modernCameraViewInverse[16] =
{
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};
static int   g_modernCameraValid = 0;
static int   g_shadowDebugMode = 0;

// Cofactor inverse of a column-major 4x4. Returns 0 when singular.
static int InvertMatrix4(const float m[16], float out[16])
{
	float inv[16];

	inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
		+ m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
		- m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
		+ m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
		- m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
		- m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
		+ m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
		- m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
		+ m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
		+ m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
	inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
		- m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
		+ m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
		- m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
	inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
		- m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
	inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
		+ m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
		- m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
		+ m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

	float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (det > -1e-12f && det < 1e-12f)
		return 0;

	det = 1.0f / det;
	for (int i = 0; i < 16; i++)
		out[i] = inv[i] * det;
	return 1;
}

void PsyX_ModernMesh_SetProjection(const float matrix[16])
{
	memcpy(g_psyxModernProjection, matrix, sizeof(g_psyxModernProjection));
	g_psyxModernProjectionValid = 1;
	g_psyxModernProjectionInverseValid = InvertMatrix4(g_psyxModernProjection, g_psyxModernProjectionInverse);
}

void PsyX_ModernMesh_SetCamera(const float viewRotation[16], const float cameraPosition[3])
{
	if (viewRotation)
		memcpy(g_modernCameraRotation, viewRotation, sizeof(g_modernCameraRotation));
	if (cameraPosition)
		memcpy(g_modernCameraPosition, cameraPosition, sizeof(g_modernCameraPosition));

	// Build and invert the full camera view so the composite can go from
	// view space back to world space exactly, without assuming the rotation is
	// orthonormal (the game's matrix can carry an aspect scale).
	float view[16];
	memcpy(view, g_modernCameraRotation, sizeof(view));
	const float cx = g_modernCameraPosition[0];
	const float cy = g_modernCameraPosition[1];
	const float cz = g_modernCameraPosition[2];
	view[12] = -(view[0] * cx + view[4] * cy + view[8] * cz);
	view[13] = -(view[1] * cx + view[5] * cy + view[9] * cz);
	view[14] = -(view[2] * cx + view[6] * cy + view[10] * cz);
	view[3] = view[7] = view[11] = 0.0f;
	view[15] = 1.0f;

	if (!InvertMatrix4(view, g_modernCameraViewInverse))
		memset(g_modernCameraViewInverse, 0, sizeof(g_modernCameraViewInverse));

	g_modernCameraValid = (viewRotation != NULL && cameraPosition != NULL);
}

void PsyX_ModernMesh_SetShadowDebug(int mode)
{
	g_shadowDebugMode = mode;
}

static GLuint CompileShader(GLenum type, const char* source)
{
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);

	GLint status = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (!status)
	{
		char log[512];
		glGetShaderInfoLog(shader, sizeof(log), NULL, log);
		eprinterr("PsyX modern mesh shader failed: %s\n", log);
		glDeleteShader(shader);
		return 0;
	}

	return shader;
}

static int CreateProgram()
{
	GLuint vs = CompileShader(GL_VERTEX_SHADER, kVertexShader);
	if (!vs)
		return 0;

	GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader);
	if (!fs)
	{
		glDeleteShader(vs);
		return 0;
	}

	g_program = glCreateProgram();
	glAttachShader(g_program, vs);
	glAttachShader(g_program, fs);
	glBindAttribLocation(g_program, 0, "a_position");
	glBindAttribLocation(g_program, 1, "a_color");
	glBindAttribLocation(g_program, 2, "a_normal");
	glBindAttribLocation(g_program, 3, "a_uv");
	glLinkProgram(g_program);
	glDeleteShader(vs);
	glDeleteShader(fs);

	GLint status = 0;
	glGetProgramiv(g_program, GL_LINK_STATUS, &status);
	if (!status)
	{
		char log[512];
		glGetProgramInfoLog(g_program, sizeof(log), NULL, log);
		eprinterr("PsyX modern mesh program failed: %s\n", log);
		glDeleteProgram(g_program);
		g_program = 0;
		return 0;
	}

	g_uView = glGetUniformLocation(g_program, "u_view");
	g_uProjection = glGetUniformLocation(g_program, "u_projection");
	g_uXyScale = glGetUniformLocation(g_program, "u_xyScale");
	g_uZScale = glGetUniformLocation(g_program, "u_zScale");
	g_uColor = glGetUniformLocation(g_program, "u_color");
	g_uBaseColorFactor = glGetUniformLocation(g_program, "u_baseColorFactor");
	g_uUseTexture = glGetUniformLocation(g_program, "u_useTexture");
	g_uTexture = glGetUniformLocation(g_program, "s_texture");
	g_uMetallicFactor = glGetUniformLocation(g_program, "u_metallicFactor");
	g_uRoughnessFactor = glGetUniformLocation(g_program, "u_roughnessFactor");
	g_uUseNormalMap = glGetUniformLocation(g_program, "u_useNormalMap");
	g_uUseMRMap = glGetUniformLocation(g_program, "u_useMRMap");
	g_uUseEmissiveMap = glGetUniformLocation(g_program, "u_useEmissiveMap");
	g_uUseLighting = glGetUniformLocation(g_program, "u_useLighting");
	g_uNormalMap = glGetUniformLocation(g_program, "s_normalMap");
	g_uMrMap = glGetUniformLocation(g_program, "s_mrMap");
	g_uEmissiveMap = glGetUniformLocation(g_program, "s_emissiveMap");
	g_uAmbient = glGetUniformLocation(g_program, "u_ambient");
	g_uExposure = glGetUniformLocation(g_program, "u_exposure");
	g_uAoEnabled = glGetUniformLocation(g_program, "u_aoEnabled");
	g_uLightCount = glGetUniformLocation(g_program, "u_lightCount");
	g_uLightType = glGetUniformLocation(g_program, "u_lightType");
	g_uLightPos = glGetUniformLocation(g_program, "u_lightPos");
	g_uLightDir = glGetUniformLocation(g_program, "u_lightDir");
	g_uLightColor = glGetUniformLocation(g_program, "u_lightColor");
	g_uLightRange = glGetUniformLocation(g_program, "u_lightRange");
	g_uEmissiveFactor = glGetUniformLocation(g_program, "u_emissiveFactor");
	g_uShadowMap = glGetUniformLocation(g_program, "s_shadowMap");
	g_uShadowMatrix = glGetUniformLocation(g_program, "u_shadowMatrix");
	g_uShadowEnabled = glGetUniformLocation(g_program, "u_shadowEnabled");
	g_aPosition = 0;
	g_aColor = 1;
	g_aNormal = 2;
	g_aUv = 3;

	return 1;
}

static void MakeLookAt(const float eye[3], const float center[3], const float up[3], float m[16])
{
	float f[3] = { center[0] - eye[0], center[1] - eye[1], center[2] - eye[2] };
	float fl = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
	if (fl < 1e-5f) fl = 1.0f;
	f[0] /= fl; f[1] /= fl; f[2] /= fl;

	float s[3] = { f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0] };
	float sl = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
	if (sl < 1e-5f) { s[0] = 1; s[1] = 0; s[2] = 0; sl = 1; }
	s[0] /= sl; s[1] /= sl; s[2] /= sl;

	const float u[3] = { s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0] };

	m[0] = s[0]; m[4] = s[1]; m[8] = s[2];  m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
	m[1] = u[0]; m[5] = u[1]; m[9] = u[2];  m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
	m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2]; m[14] = (f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]);
	m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
}

static void MakeOrtho(float l, float r, float b, float t, float n, float far, float m[16])
{
	m[0] = 2 / (r - l); m[4] = 0; m[8] = 0; m[12] = -(r + l) / (r - l);
	m[1] = 0; m[5] = 2 / (t - b); m[9] = 0; m[13] = -(t + b) / (t - b);
	m[2] = 0; m[6] = 0; m[10] = -2 / (far - n); m[14] = -(far + n) / (far - n);
	m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
}

static void MulMatrix(const float a[16], const float b[16], float out[16])
{
	for (int c = 0; c < 4; c++)
		for (int r = 0; r < 4; r++)
		{
			float sum = 0;
			for (int k = 0; k < 4; k++)
				sum += a[k * 4 + r] * b[c * 4 + k];
			out[c * 4 + r] = sum;
		}
}

static int CreateShadowResources()
{
	GLuint vs = CompileShader(GL_VERTEX_SHADER, kDepthVertexShader);
	GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kDepthFragmentShader);
	if (!vs || !fs)
		return 0;

	g_depthProgram = glCreateProgram();
	glAttachShader(g_depthProgram, vs);
	glAttachShader(g_depthProgram, fs);
	glBindAttribLocation(g_depthProgram, 0, "a_position");
	glLinkProgram(g_depthProgram);
	glDeleteShader(vs);
	glDeleteShader(fs);

	GLint status = 0;
	glGetProgramiv(g_depthProgram, GL_LINK_STATUS, &status);
	if (!status)
	{
		glDeleteProgram(g_depthProgram);
		g_depthProgram = 0;
		return 0;
	}
	g_dpLightMatrix = glGetUniformLocation(g_depthProgram, "u_lightMatrix");
	g_dpWorld = glGetUniformLocation(g_depthProgram, "u_world");

	glGenFramebuffers(1, &g_shadowFbo);
	glGenTextures(1, &g_shadowTexture);
	glBindTexture(GL_TEXTURE_2D, g_shadowTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, g_shadowSize, g_shadowSize, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

	glBindFramebuffer(GL_FRAMEBUFFER, g_shadowFbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_shadowTexture, 0);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);
	const GLenum fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	eprintinfo("PsyX modern shadow map: %dx%d depth-only, framebuffer %s\n",
		g_shadowSize, g_shadowSize,
		fbStatus == GL_FRAMEBUFFER_COMPLETE ? "complete" : "INCOMPLETE");

	return fbStatus == GL_FRAMEBUFFER_COMPLETE ? 1 : 0;
}

static int CreateCompositeProgram()
{
	GLuint vs = CompileShader(GL_VERTEX_SHADER, kCompositeVertexShader);
	if (!vs)
		return 0;

	GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kCompositeFragmentShader);
	if (!fs)
	{
		glDeleteShader(vs);
		return 0;
	}

	g_compositeProgram = glCreateProgram();
	glAttachShader(g_compositeProgram, vs);
	glAttachShader(g_compositeProgram, fs);
	glLinkProgram(g_compositeProgram);
	glDeleteShader(vs);
	glDeleteShader(fs);

	GLint status = 0;
	glGetProgramiv(g_compositeProgram, GL_LINK_STATUS, &status);
	if (!status)
	{
		char log[512];
		glGetProgramInfoLog(g_compositeProgram, sizeof(log), NULL, log);
		eprinterr("PsyX modern shadow composite failed: %s\n", log);
		glDeleteProgram(g_compositeProgram);
		g_compositeProgram = 0;
		return 0;
	}

	g_cuSceneDepth = glGetUniformLocation(g_compositeProgram, "s_sceneDepth");
	g_cuShadowMap = glGetUniformLocation(g_compositeProgram, "s_shadowMap");
	g_cuProjInverse = glGetUniformLocation(g_compositeProgram, "u_projInverse");
	g_cuCameraViewInverse = glGetUniformLocation(g_compositeProgram, "u_cameraViewInverse");
	g_cuXyScale = glGetUniformLocation(g_compositeProgram, "u_xyScale");
	g_cuZScale = glGetUniformLocation(g_compositeProgram, "u_zScale");
	g_cuShadowMatrix = glGetUniformLocation(g_compositeProgram, "u_shadowMatrix");
	g_cuViewport = glGetUniformLocation(g_compositeProgram, "u_viewport");
	g_cuShadowTexel = glGetUniformLocation(g_compositeProgram, "u_shadowTexel");
	g_cuStrength = glGetUniformLocation(g_compositeProgram, "u_strength");
	g_cuDebugMode = glGetUniformLocation(g_compositeProgram, "u_debugMode");

	glGenVertexArrays(1, &g_fullscreenVao);
	return 1;
}

/* Copies the legacy scene depth into a sampleable texture so the composite can
   reconstruct world positions for pixels that are already shaded. */
static int EnsureSceneDepthTarget(int width, int height)
{
	if (g_sceneDepthTexture != 0 && g_sceneDepthWidth == width && g_sceneDepthHeight == height)
		return g_sceneDepthTargetOk;

	if (!g_sceneDepthFbo)
		glGenFramebuffers(1, &g_sceneDepthFbo);
	if (!g_sceneDepthTexture)
		glGenTextures(1, &g_sceneDepthTexture);

	GLint prevFbo = 0;
	GLint prevTexture = 0;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexture);

	glBindTexture(GL_TEXTURE_2D, g_sceneDepthTexture);
	// The window framebuffer is depth-stencil, so the copy target must match or
	// the depth blit is rejected by the driver.
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, width, height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

	glBindFramebuffer(GL_FRAMEBUFFER, g_sceneDepthFbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, g_sceneDepthTexture, 0);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);
	g_sceneDepthTargetOk = (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prevFbo);
	glBindTexture(GL_TEXTURE_2D, (GLuint)prevTexture);

	g_sceneDepthWidth = width;
	g_sceneDepthHeight = height;
	return g_sceneDepthTargetOk;
}

int PsyX_ModernMesh_CreateEx(const PsyXModernMeshDesc* desc)
{
	if (!desc || !desc->positions || desc->vertexCount <= 0 || g_meshCount >= PSYX_MODERN_MAX_MESHES)
		return -1;

	if (!g_program && !CreateProgram())
		return -1;

	if (!g_depthProgram)
		CreateShadowResources();

	int slot = -1;
	for (int i = 0; i < PSYX_MODERN_MAX_MESHES; i++)
	{
		if (!g_meshes[i].used)
		{
			slot = i;
			break;
		}
	}
	if (slot < 0)
		return -1;

	const int vertexCount = desc->vertexCount;
	const int indexCount = desc->indexCount;

	PsyXModernMesh* mesh = &g_meshes[slot];
	memset(mesh, 0, sizeof(*mesh));
	mesh->used = 1;
	mesh->vertexCount = vertexCount;
	mesh->indexCount = indexCount;
	mesh->color[0] = mesh->color[1] = mesh->color[2] = mesh->color[3] = 1.0f;
	mesh->world[0] = mesh->world[5] = mesh->world[10] = mesh->world[15] = 1.0f;
	mesh->texture = desc->baseColorTexture;
	mesh->useTexture = desc->baseColorTexture != 0;
	mesh->normalTexture = desc->normalTexture;
	mesh->useNormalMap = desc->normalTexture != 0;
	mesh->metallicRoughnessTexture = desc->metallicRoughnessTexture;
	mesh->useMRMap = desc->metallicRoughnessTexture != 0;
	mesh->emissiveTexture = desc->emissiveTexture;
	mesh->useEmissiveMap = desc->emissiveTexture != 0;
	mesh->emissiveFactor[0] = desc->emissiveFactor ? desc->emissiveFactor[0] : 0.0f;
	mesh->emissiveFactor[1] = desc->emissiveFactor ? desc->emissiveFactor[1] : 0.0f;
	mesh->emissiveFactor[2] = desc->emissiveFactor ? desc->emissiveFactor[2] : 0.0f;
	mesh->metallicFactor = desc->metallicFactor;
	mesh->roughnessFactor = desc->roughnessFactor;
	for (int i = 0; i < 4; i++)
		mesh->baseColorFactor[i] = desc->baseColorFactor ? desc->baseColorFactor[i] : 1.0f;

	GLint prevVao = 0;
	GLint prevArrayBuffer = 0;
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrayBuffer);

	glGenVertexArrays(1, &mesh->vao);
	glGenBuffers(1, &mesh->vbo);
	glGenBuffers(1, &mesh->cbo);
	if (indexCount > 0)
		glGenBuffers(1, &mesh->ebo);
	if (desc->normals)
		glGenBuffers(1, &mesh->nbo);
	if (desc->uvs)
		glGenBuffers(1, &mesh->ubo);

	glBindVertexArray(mesh->vao);

	glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo);
	glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vertexCount * 3 * sizeof(float), desc->positions, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	unsigned char* white = new unsigned char[(size_t)vertexCount * 4];
	if (desc->colors)
		memcpy(white, desc->colors, (size_t)vertexCount * 4);
	else
		memset(white, 255, (size_t)vertexCount * 4);

	glBindBuffer(GL_ARRAY_BUFFER, mesh->cbo);
	glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vertexCount * 4, white, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, 4, (void*)0);
	delete[] white;

	if (mesh->nbo)
	{
		glBindBuffer(GL_ARRAY_BUFFER, mesh->nbo);
		glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vertexCount * 3 * sizeof(float), desc->normals, GL_STATIC_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	}

	if (mesh->ubo)
	{
		glBindBuffer(GL_ARRAY_BUFFER, mesh->ubo);
		glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vertexCount * 2 * sizeof(float), desc->uvs, GL_STATIC_DRAW);
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	}

	if (indexCount > 0)
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)indexCount * sizeof(unsigned short), desc->indices, GL_STATIC_DRAW);
	}

	// Restore the caller's bindings instead of leaving VAO 0 bound: the
	// emulator keeps its own vertex buffers in a VAO, and disturbing the
	// binding here corrupted later legacy draws.
	glBindVertexArray((GLuint)prevVao);
	glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevArrayBuffer);

	g_meshCount++;
	return slot;
}

int PsyX_ModernMesh_Create(const float* positions, int vertexCount,
	const unsigned short* indices, int indexCount, const unsigned char* colors)
{
	PsyXModernMeshDesc desc = {};
	desc.positions = positions;
	desc.vertexCount = vertexCount;
	desc.indices = indices;
	desc.indexCount = indexCount;
	desc.colors = colors;
	const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	desc.baseColorFactor = white;
	desc.metallicFactor = 0.0f;
	desc.roughnessFactor = 1.0f;
	return PsyX_ModernMesh_CreateEx(&desc);
}

void PsyX_ModernMesh_SetLights(const PsyXModernLightSet* lights)
{
	if (lights)
		g_lights = *lights;
}

void PsyX_ModernMesh_Destroy(int mesh)
{
	if (mesh < 0 || mesh >= PSYX_MODERN_MAX_MESHES || !g_meshes[mesh].used)
		return;

	PsyXModernMesh* m = &g_meshes[mesh];
	if (m->vao) glDeleteVertexArrays(1, &m->vao);
	if (m->vbo) glDeleteBuffers(1, &m->vbo);
	if (m->ebo) glDeleteBuffers(1, &m->ebo);
	if (m->cbo) glDeleteBuffers(1, &m->cbo);
	if (m->nbo) glDeleteBuffers(1, &m->nbo);
	if (m->ubo) glDeleteBuffers(1, &m->ubo);

	memset(m, 0, sizeof(*m));
	if (g_meshCount > 0)
		g_meshCount--;
}

void PsyX_ModernMesh_SetInstance(int mesh, const float viewMatrix[16], const float color[4], int visible)
{
	if (mesh < 0 || mesh >= PSYX_MODERN_MAX_MESHES || !g_meshes[mesh].used)
		return;

	PsyXModernMesh* m = &g_meshes[mesh];
	if (viewMatrix)
		memcpy(m->view, viewMatrix, sizeof(m->view));
	if (color)
		memcpy(m->color, color, sizeof(m->color));
	m->visible = visible != 0;
}

void PsyX_ModernMesh_SetInstanceWorld(int mesh, const float worldMatrix[16])
{
	if (mesh < 0 || mesh >= PSYX_MODERN_MAX_MESHES || !g_meshes[mesh].used)
		return;

	PsyXModernMesh* m = &g_meshes[mesh];
	if (worldMatrix)
		memcpy(m->world, worldMatrix, sizeof(m->world));
}

void PsyX_ModernMesh_SetEnabled(int enabled)
{
	g_enabled = enabled != 0;
}

int PsyX_ModernMesh_GetEnabled(void)
{
	return g_enabled;
}

void PsyX_ModernMesh_GetStats(PsyXModernMeshStats* stats)
{
	if (stats)
		*stats = g_stats;
}

void PsyX_ModernMesh_RenderFrame(void)
{
	memset(&g_stats, 0, sizeof(g_stats));

	if (!g_enabled || !g_psyxModernProjectionValid || !g_program || g_meshCount == 0)
		return;

	GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
	GLboolean prevCullFace = glIsEnabled(GL_CULL_FACE);
	GLboolean prevBlend = glIsEnabled(GL_BLEND);
	GLint prevDepthFunc = GL_LESS;
	GLboolean prevDepthMask = GL_TRUE;
	GLint prevProgram = 0;
	GLint prevVao = 0;
	GLint prevViewport[4] = { 0, 0, 0, 0 };
	GLint prevBlendSrcRgb = GL_SRC_ALPHA, prevBlendDstRgb = GL_ONE_MINUS_SRC_ALPHA;
	GLint prevBlendSrcAlpha = GL_ONE, prevBlendDstAlpha = GL_ONE_MINUS_SRC_ALPHA;
	GLint prevBlendEqRgb = GL_FUNC_ADD, prevBlendEqAlpha = GL_FUNC_ADD;
	GLboolean prevScissor = glIsEnabled(GL_SCISSOR_TEST);
	GLint prevScissorBox[4] = { 0, 0, 0, 0 };

	glGetIntegerv(GL_SCISSOR_BOX, prevScissorBox);

	// The legacy path toggles scissor per draw; a leftover scissor must not clip
	// the modern scene or the shadow projection.
	glDisable(GL_SCISSOR_TEST);

	glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);
	glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
	glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
	glGetIntegerv(GL_VIEWPORT, prevViewport);
	glGetIntegerv(GL_BLEND_SRC_RGB, &prevBlendSrcRgb);
	glGetIntegerv(GL_BLEND_DST_RGB, &prevBlendDstRgb);
	glGetIntegerv(GL_BLEND_SRC_ALPHA, &prevBlendSrcAlpha);
	glGetIntegerv(GL_BLEND_DST_ALPHA, &prevBlendDstAlpha);
	glGetIntegerv(GL_BLEND_EQUATION_RGB, &prevBlendEqRgb);
	glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &prevBlendEqAlpha);

	const auto start = std::chrono::steady_clock::now();

	glViewport(0, 0, g_windowWidth, g_windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_TRUE);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);

	glUseProgram(g_program);

	// Mirror the legacy vertex shader's 3D encoding: it receives normalized
	// device coordinates (camera-space xy divided by the display size) and the
	// GTE screen distance, and divides depth by the same fixed-point factor.
	float displayWidth = 320.0f;
	float displayHeight = 240.0f;
	if (activeDispEnv.disp.w > 0 && activeDispEnv.disp.h > 0)
	{
		displayWidth = (float)activeDispEnv.disp.w;
		displayHeight = (float)activeDispEnv.disp.h;
	}

	const float xyScaleX = (float)C2_H / (PSYX_MODERN_GTE_VIEW_SCALE * displayWidth);
	const float xyScaleY = (float)C2_H / (PSYX_MODERN_GTE_VIEW_SCALE * displayHeight);

	glUniformMatrix4fv(g_uProjection, 1, GL_FALSE, g_psyxModernProjection);
	glUniform2f(g_uXyScale, xyScaleX, xyScaleY);
	glUniform1f(g_uZScale, 1.0f / PSYX_MODERN_GTE_VIEW_SCALE);

	// Light set: directional and point lights, small ambient, exposure.
	glUniform3f(g_uAmbient, g_lights.ambient[0], g_lights.ambient[1], g_lights.ambient[2]);
	glUniform1f(g_uExposure, g_lights.exposure > 0.0f ? g_lights.exposure : 1.0f);
	glUniform1i(g_uAoEnabled, g_lights.aoEnabled);
	glUniform1i(g_uUseLighting, 1);

	int lightCount = g_lights.count;
	if (lightCount < 0) lightCount = 0;
	if (lightCount > PSYX_MODERN_MAX_LIGHTS) lightCount = PSYX_MODERN_MAX_LIGHTS;
	glUniform1i(g_uLightCount, lightCount);
	if (lightCount > 0)
	{
		int types[PSYX_MODERN_MAX_LIGHTS];
		float positions[PSYX_MODERN_MAX_LIGHTS * 3];
		float directions[PSYX_MODERN_MAX_LIGHTS * 3];
		float colors[PSYX_MODERN_MAX_LIGHTS * 3];
		float ranges[PSYX_MODERN_MAX_LIGHTS];
		for (int i = 0; i < lightCount; i++)
		{
			const PsyXModernLight& l = g_lights.lights[i];
			types[i] = l.type;
			positions[i * 3 + 0] = l.position[0];
			positions[i * 3 + 1] = l.position[1];
			positions[i * 3 + 2] = l.position[2];
			directions[i * 3 + 0] = l.direction[0];
			directions[i * 3 + 1] = l.direction[1];
			directions[i * 3 + 2] = l.direction[2];
			colors[i * 3 + 0] = l.color[0] * l.intensity;
			colors[i * 3 + 1] = l.color[1] * l.intensity;
			colors[i * 3 + 2] = l.color[2] * l.intensity;
			ranges[i] = l.range > 0.0f ? l.range : 1000.0f;
		}
		glUniform1iv(g_uLightType, lightCount, types);
		glUniform3fv(g_uLightPos, lightCount, positions);
		glUniform3fv(g_uLightDir, lightCount, directions);
		glUniform3fv(g_uLightColor, lightCount, colors);
		glUniform1fv(g_uLightRange, lightCount, ranges);
	}

	// Directional shadow map. Coverage is explicit: modern meshes cast and
	// receive; legacy geometry does not participate in this pass.
	int shadowActive = 0;
	if (g_lights.shadowsEnabled && g_depthProgram && g_shadowFbo)
	{
		const PsyXModernLight* sun = NULL;
		for (int i = 0; i < lightCount; i++)
			if (g_lights.lights[i].type == 0) { sun = &g_lights.lights[i]; break; }

		if (sun)
		{
			const float extent = g_lights.shadowExtent > 0.0f ? g_lights.shadowExtent : 4000.0f;
			float dl = sqrtf(sun->direction[0] * sun->direction[0] + sun->direction[1] * sun->direction[1] + sun->direction[2] * sun->direction[2]);
			dl = dl > 1e-5f ? dl : 1.0f;
			const float dir[3] = { sun->direction[0] / dl, sun->direction[1] / dl, sun->direction[2] / dl };
			const float c[3] = { g_lights.shadowCenter[0], g_lights.shadowCenter[1], g_lights.shadowCenter[2] };
			const float eye[3] = { c[0] + dir[0] * extent * 2.0f, c[1] + dir[1] * extent * 2.0f, c[2] + dir[2] * extent * 2.0f };
			const int vertical = (dir[1] > 0.95f || dir[1] < -0.95f);
			const float up[3] = { 0.0f, vertical ? 0.0f : 1.0f, vertical ? 1.0f : 0.0f };
			float lightView[16], lightProj[16];
			MakeLookAt(eye, c, up, lightView);
			MakeOrtho(-extent, extent, -extent, extent, 0.05f, extent * 4.0f, lightProj);
			MulMatrix(lightProj, lightView, g_shadowMatrix);

			glBindFramebuffer(GL_FRAMEBUFFER, g_shadowFbo);
			glViewport(0, 0, g_shadowSize, g_shadowSize);
			glClear(GL_DEPTH_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LESS);
			glDepthMask(GL_TRUE);
			glDisable(GL_CULL_FACE);
			glDisable(GL_BLEND);
			glUseProgram(g_depthProgram);
			glUniformMatrix4fv(g_dpLightMatrix, 1, GL_FALSE, g_shadowMatrix);
			for (int i = 0; i < PSYX_MODERN_MAX_MESHES; i++)
			{
				PsyXModernMesh* m = &g_meshes[i];
				if (!m->used || !m->visible || m->vao == 0)
					continue;
				glUniformMatrix4fv(g_dpWorld, 1, GL_FALSE, m->world);
				glBindVertexArray(m->vao);
				if (m->indexCount > 0)
					glDrawElements(GL_TRIANGLES, m->indexCount, GL_UNSIGNED_SHORT, (void*)0);
				else
					glDrawArrays(GL_TRIANGLES, 0, m->vertexCount);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glViewport(0, 0, g_windowWidth, g_windowHeight);
			shadowActive = 1;
		}
	}

	// Legacy receivers: project the shadow map onto the pixels of the already
	// rendered legacy scene. The modern meshes draw after this pass and shade
	// their own shadows, so they are not darkened twice.
	if (shadowActive && g_modernCameraValid && g_psyxModernProjectionInverseValid)
	{
		if (!g_compositeProgram)
			CreateCompositeProgram();

		if (g_compositeProgram && EnsureSceneDepthTarget(g_windowWidth, g_windowHeight))
		{
			while (glGetError() != GL_NO_ERROR) {}

			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_sceneDepthFbo);
			glBlitFramebuffer(0, 0, g_windowWidth, g_windowHeight,
				0, 0, g_windowWidth, g_windowHeight,
				GL_DEPTH_BUFFER_BIT, GL_NEAREST);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			static int s_blitErrorLogged = 0;
			if (!s_blitErrorLogged)
			{
				const GLenum blitError = glGetError();
				if (blitError != GL_NO_ERROR)
				{
					s_blitErrorLogged = 1;
					eprinterr("PsyX modern shadow: scene depth blit failed (0x%04X)\n", (unsigned)blitError);
				}
			}

			const int s_debugMode = g_shadowDebugMode;

			glUseProgram(g_compositeProgram);
			glDisable(GL_DEPTH_TEST);
			glDepthMask(GL_FALSE);
			glDisable(GL_CULL_FACE);
			if (s_debugMode != 0)
			{
				glDisable(GL_BLEND);
			}
			else
			{
				glEnable(GL_BLEND);
				glBlendEquation(GL_FUNC_ADD);
				glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_ZERO, GL_ONE);
			}
			glUniform1i(g_cuDebugMode, s_debugMode);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, g_sceneDepthTexture);
			glUniform1i(g_cuSceneDepth, 0);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, g_shadowTexture);
			glUniform1i(g_cuShadowMap, 1);

			glUniformMatrix4fv(g_cuProjInverse, 1, GL_FALSE, g_psyxModernProjectionInverse);
			glUniformMatrix4fv(g_cuCameraViewInverse, 1, GL_FALSE, g_modernCameraViewInverse);
			glUniform2f(g_cuXyScale, xyScaleX, xyScaleY);
			glUniform1f(g_cuZScale, 1.0f / PSYX_MODERN_GTE_VIEW_SCALE);
			glUniformMatrix4fv(g_cuShadowMatrix, 1, GL_FALSE, g_shadowMatrix);
			glUniform2f(g_cuViewport, (float)g_windowWidth, (float)g_windowHeight);
			glUniform1f(g_cuShadowTexel, 1.0f / (float)g_shadowSize);
			glUniform1f(g_cuStrength, 0.45f);

			glBindVertexArray(g_fullscreenVao);

			glDrawArrays(GL_TRIANGLES, 0, 3);

			glBlendEquationSeparate(prevBlendEqRgb, prevBlendEqAlpha);
			glBlendFuncSeparate(prevBlendSrcRgb, prevBlendDstRgb, prevBlendSrcAlpha, prevBlendDstAlpha);
			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);
			glDepthMask(GL_TRUE);
			glDisable(GL_CULL_FACE);

			g_stats.legacyShadowPass = 1;
		}
	}

	glUseProgram(g_program);
	glUniform1i(g_uShadowEnabled, shadowActive);
	if (shadowActive)
	{
		glUniformMatrix4fv(g_uShadowMatrix, 1, GL_FALSE, g_shadowMatrix);
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, g_shadowTexture);
		glUniform1i(g_uShadowMap, 4);
	}

	g_stats.depthShared = 1;
	g_stats.meshCount = g_meshCount;

	for (int i = 0; i < PSYX_MODERN_MAX_MESHES; i++)
	{
		PsyXModernMesh* m = &g_meshes[i];
		if (!m->used || !m->visible || m->vao == 0)
			continue;

		glUniformMatrix4fv(g_uView, 1, GL_FALSE, m->view);
		glUniform4fv(g_uColor, 1, m->color);
		glUniform4fv(g_uBaseColorFactor, 1, m->baseColorFactor);
		glUniform1f(g_uMetallicFactor, m->metallicFactor);
		glUniform1f(g_uRoughnessFactor, m->roughnessFactor);

		glUniform1i(g_uUseTexture, m->useTexture);
		if (m->useTexture)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, m->texture);
			glUniform1i(g_uTexture, 0);
		}

		glUniform1i(g_uUseNormalMap, m->useNormalMap);
		if (m->useNormalMap)
		{
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, m->normalTexture);
			glUniform1i(g_uNormalMap, 1);
		}

		glUniform1i(g_uUseMRMap, m->useMRMap);
		if (m->useMRMap)
		{
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, m->metallicRoughnessTexture);
			glUniform1i(g_uMrMap, 2);
		}

		glUniform3f(g_uEmissiveFactor, m->emissiveFactor[0], m->emissiveFactor[1], m->emissiveFactor[2]);
		glUniform1i(g_uUseEmissiveMap, m->useEmissiveMap);
		if (m->useEmissiveMap)
		{
			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_2D, m->emissiveTexture);
			glUniform1i(g_uEmissiveMap, 3);
		}

		glBindVertexArray(m->vao);

		if (m->indexCount > 0)
			glDrawElements(GL_TRIANGLES, m->indexCount, GL_UNSIGNED_SHORT, (void*)0);
		else
			glDrawArrays(GL_TRIANGLES, 0, m->vertexCount);

		g_stats.visibleInstances++;
		g_stats.vertexCount += m->vertexCount;
		g_stats.drawCalls++;
	}

	const auto end = std::chrono::steady_clock::now();
	g_stats.lastFrameMicros = (int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	glBindVertexArray((GLuint)prevVao);
	glUseProgram((GLuint)prevProgram);
	glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
	glDepthFunc(prevDepthFunc);
	glDepthMask(prevDepthMask);

	if (prevDepthTest) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
	if (prevCullFace) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
	if (prevBlend) glEnable(GL_BLEND); else glDisable(GL_BLEND);

	glScissor(prevScissorBox[0], prevScissorBox[1], prevScissorBox[2], prevScissorBox[3]);
	if (prevScissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
}

void PsyX_ModernMesh_Shutdown(void)
{
	for (int i = 0; i < PSYX_MODERN_MAX_MESHES; i++)
		PsyX_ModernMesh_Destroy(i);

	if (g_program)
	{
		glDeleteProgram(g_program);
		g_program = 0;
	}
	if (g_depthProgram)
	{
		glDeleteProgram(g_depthProgram);
		g_depthProgram = 0;
	}
	if (g_shadowFbo)
	{
		glDeleteFramebuffers(1, &g_shadowFbo);
		g_shadowFbo = 0;
	}
	if (g_shadowTexture)
	{
		glDeleteTextures(1, &g_shadowTexture);
		g_shadowTexture = 0;
	}
	if (g_compositeProgram)
	{
		glDeleteProgram(g_compositeProgram);
		g_compositeProgram = 0;
	}
	if (g_fullscreenVao)
	{
		glDeleteVertexArrays(1, &g_fullscreenVao);
		g_fullscreenVao = 0;
	}
	if (g_sceneDepthFbo)
	{
		glDeleteFramebuffers(1, &g_sceneDepthFbo);
		g_sceneDepthFbo = 0;
	}
	if (g_sceneDepthTexture)
	{
		glDeleteTextures(1, &g_sceneDepthTexture);
		g_sceneDepthTexture = 0;
	}
	g_sceneDepthWidth = 0;
	g_sceneDepthHeight = 0;
	g_sceneDepthTargetOk = 0;
	g_modernCameraValid = 0;
	g_psyxModernProjectionInverseValid = 0;

	g_enabled = 0;
	g_psyxModernProjectionValid = 0;
}

#else // !USE_OPENGL

int PsyX_ModernMesh_Create(const float*, int, const unsigned short*, int, const unsigned char*) { return -1; }
int PsyX_ModernMesh_CreateEx(const PsyXModernMeshDesc*) { return -1; }
void PsyX_ModernMesh_SetLights(const PsyXModernLightSet*) {}
void PsyX_ModernMesh_Destroy(int) {}
void PsyX_ModernMesh_SetInstance(int, const float[16], const float[4], int) {}
void PsyX_ModernMesh_SetInstanceWorld(int, const float[16]) {}
void PsyX_ModernMesh_SetEnabled(int) {}
int PsyX_ModernMesh_GetEnabled(void) { return 0; }
void PsyX_ModernMesh_GetStats(PsyXModernMeshStats* stats) { if (stats) memset(stats, 0, sizeof(*stats)); }

void PsyX_ModernMesh_SetProjection(const float matrix[16]) { (void)matrix; }
void PsyX_ModernMesh_SetCamera(const float viewRotation[16], const float cameraPosition[3]) { (void)viewRotation; (void)cameraPosition; }
void PsyX_ModernMesh_SetShadowDebug(int mode) { (void)mode; }
void PsyX_ModernMesh_RenderFrame(void) {}
void PsyX_ModernMesh_Shutdown(void) {}

#endif // USE_OPENGL
