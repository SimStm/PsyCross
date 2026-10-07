#version 450
layout(location=0) in vec4 vertexColor;
layout(location=1) in vec3 sourceNormal;
layout(location=2) in vec2 sourceUV;
layout(location=0) out vec4 outColor;
layout(push_constant) uniform NativeDraw { mat4 mvp; vec4 tint; uint encodeSRGB; } pc;
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
    vec3 linearRGB = clamp(vertexColor.rgb, 0.0, 1.0);
    outColor = vec4(pc.encodeSRGB != 0 ? encodeSRGB(linearRGB) : linearRGB, 1.0);
}
