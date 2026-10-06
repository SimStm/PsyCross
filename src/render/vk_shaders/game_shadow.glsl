// Shared game receivers. Two nested player-centred volumes retain near detail
// while extending resident-only reach. They are not camera-frustum cascades.
vec3 GameShadowProjection(mat4 matrix, vec3 world)
{
	vec4 clip = matrix * vec4(world, 1.0);
	return vec3(clip.xy / clip.w * 0.5 + 0.5, clip.z / clip.w);
}

bool GameShadowInside(vec3 p)
{
	return all(greaterThanEqual(p, vec3(0.0))) && all(lessThanEqual(p, vec3(1.0)));
}

vec2 GameShadowUv(vec2 p, int volume)
{
	return vec2(p.x * 0.5 + float(volume) * 0.5, p.y);
}

float GameShadowEdge(vec3 p)
{
	vec3 edge = abs(p * 2.0 - 1.0);
	return max(max(edge.x, edge.y), edge.z);
}

float GameShadowPcf(vec3 p, int volume, float bias)
{
	if (!GameShadowInside(p)) return 1.0;
	vec2 texel = 1.0 / vec2(textureSize(s_shadowMap, 0));
	vec2 low = vec2(float(volume) * 0.5, 0.0) + texel * 0.5;
	vec2 high = vec2(float(volume + 1) * 0.5, 1.0) - texel * 0.5;
	float lit = 0.0;
	for (int y = -1; y <= 1; y++)
		for (int x = -1; x <= 1; x++)
		{
			// Keep every tap inside its own tile, including the atlas seam.
			vec2 uv = clamp(GameShadowUv(p.xy, volume) + vec2(x, y) * texel, low, high);
			lit += p.z - bias > texture(s_shadowMap, uv).r ? 0.0 : 1.0;
		}
	return lit / 9.0;
}

float GameShadowLit(vec3 world, float nearBias)
{
	if (u.shadowParams.x < 0.5) return 1.0;
	vec3 nearProjection = GameShadowProjection(u.shadowMatrix, world);
	float nearLit = GameShadowPcf(nearProjection, 0, nearBias);
	if (u.shadowCascade.x <= 0.0) return nearLit;
	vec3 farProjection = GameShadowProjection(u.shadowFarMatrix, world);
	// Keep bias in the same world units rather than scaling it with reach.
	float farLit = GameShadowPcf(farProjection, 1, nearBias * u.shadowVolume.w / u.shadowCascade.x);
	farLit = mix(farLit, 1.0, smoothstep(0.9, 1.0, GameShadowEdge(farProjection)));
	return mix(nearLit, farLit, smoothstep(0.75, 0.95, GameShadowEdge(nearProjection)));
}
