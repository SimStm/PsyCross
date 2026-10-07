#version 450
layout(location=0) in vec4 vertexColor;
layout(location=1) in vec3 sourceNormal;
layout(location=2) in vec2 sourceUV;
layout(location=0) out vec4 outColor;
layout(location=1) out uint outPick;
layout(push_constant) uniform NativeDraw { mat4 mvp; vec4 tint; uint encodeSRGB; float alphaCutoff; uint pickId; vec4 depthPlane; } pc;
layout(set=0,binding=0) uniform sampler2D artwork;
// The swapchain owner selects explicit linear -> sRGB encoding for UNORM;
// an sRGB attachment performs the conversion itself.
vec3 encodeSRGB(vec3 linearRGB)
{
    return mix(12.92 * linearRGB, 1.055 * pow(linearRGB, vec3(1.0 / 2.4)) - 0.055,
               greaterThan(linearRGB, vec3(0.0031308)));
}
void main()
{
    // Retain and validate source attributes in the analytic pass. They never
    // reconstruct normals/positions from legacy depth or GTE-encoded vertices.
    if (any(isnan(sourceNormal)) || any(isnan(sourceUV))) discard;
    vec4 texel = texture(artwork, sourceUV); // sRGB image decoding precedes filtering.
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
