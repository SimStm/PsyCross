#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;
layout(location=3) in vec4 color;
layout(push_constant) uniform NativeDraw { mat4 mvp; vec4 tint; uint encodeSRGB; } pc;
layout(location=0) out vec4 vertexColor;
layout(location=1) out vec3 sourceNormal;
layout(location=2) out vec2 sourceUV;
void main()
{
    gl_Position = pc.mvp * vec4(position, 1.0);
    vertexColor = color * pc.tint;
    sourceNormal = normal;
    sourceUV = uv;
}
