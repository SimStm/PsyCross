#version 450
layout(location=0) in vec4 vertexColor;
layout(location=1) in vec3 sourceNormal;
layout(location=2) in vec2 sourceUV;
layout(location=0) out vec4 outColor;
layout(location=1) out uint outPick;
layout(push_constant) uniform NativeDraw { mat4 mvp; vec4 tint; uint encodeSRGB; float alphaCutoff; uint pickId; uint sampling; vec4 depthPlane; } pc;
layout(set=0,binding=0) uniform sampler2D artwork;
// The swapchain owner selects explicit linear -> sRGB encoding for UNORM;
// an sRGB attachment performs the conversion itself.
vec3 encodeSRGB(vec3 linearRGB)
{
    return mix(12.92 * linearRGB, 1.055 * pow(linearRGB, vec3(1.0 / 2.4)) - 0.055,
               greaterThan(linearRGB, vec3(0.0031308)));
}
// Interpolate decoded texels in FP32. Fixed-format texture-unit interpolation
// can lose multiple final sRGB bytes in dark artwork even at the correct UV/LOD.
vec4 linearLevel(vec2 uv, int level)
{
    ivec2 dimensions = textureSize(artwork, level);
    // Original named UVs may exceed [0,1]. Clamp before integer conversion so
    // fetches preserve the material's clamp-edge policy without overflow/OOB.
    vec2 coordinate = clamp(uv, vec2(0.0), vec2(1.0)) * vec2(dimensions) - 0.5;
    ivec2 location = ivec2(floor(coordinate));
    ivec2 limit = dimensions - 1;
    vec2 weight = fract(coordinate);
    vec4 top = mix(texelFetch(artwork, clamp(location, ivec2(0), limit), level),
                   texelFetch(artwork, clamp(location + ivec2(1,0), ivec2(0), limit), level), weight.x);
    vec4 bottom = mix(texelFetch(artwork, clamp(location + ivec2(0,1), ivec2(0), limit), level),
                      texelFetch(artwork, clamp(location + ivec2(1,1), ivec2(0), limit), level), weight.x);
    return mix(top, bottom, weight.y);
}
vec4 sampleArtwork(vec2 uv, float lod)
{
    if (pc.sampling == 0u) return texture(artwork, uv); // native nearest unchanged
    if (pc.sampling == 1u) return linearLevel(uv, 0); // base-only bilinear
    lod = clamp(lod, 0.0, float(textureQueryLevels(artwork) - 1));
    int lower = int(floor(lod));
    float fraction = fract(lod);
    vec4 base = linearLevel(uv, lower);
    if (fraction == 0.0) return base;
    return mix(base, linearLevel(uv, lower + 1), fraction);
}
void main()
{
    // Query implicit LOD before any per-fragment discard. The sampling choice
    // is draw-uniform; .x retains the selected sampler/image LOD clamps.
    float lod = pc.sampling == 2u ? textureQueryLod(artwork, sourceUV).x : 0.0;
    // Retain and validate source attributes in the analytic pass. They never
    // reconstruct normals/positions from legacy depth or GTE-encoded vertices.
    if (any(isnan(sourceNormal)) || any(isnan(sourceUV)) || any(isinf(sourceUV))) discard;
    vec4 texel = sampleArtwork(sourceUV, lod); // sRGB image decoding precedes filtering.
    if (texel.a < pc.alphaCutoff) discard;
    gl_FragDepth = gl_FragCoord.z;
    if (pc.depthPlane.w >= 0.0)
    {
        float depth = clamp(dot(pc.depthPlane.xyz, vec3(gl_FragCoord.xy, 1.0)), 0.0, 1.0);
        // Declared paint layers keep their small D32 bias after depth replacement.
        // Ordinary equal-height terrain uses the same exact value and native order.
        uint bits = floatBitsToUint(depth), units = uint(pc.depthPlane.w);
        gl_FragDepth = depth > 0.0 ? uintBitsToFloat(bits > units ? bits - units : 0u) : 0.0;
    }
    vec3 linearRGB = clamp(vertexColor.rgb * texel.rgb, 0.0, 1.0);
    outColor = vec4(pc.encodeSRGB != 0 ? encodeSRGB(linearRGB) : linearRGB, 1.0);
    outPick = pc.pickId;
}
