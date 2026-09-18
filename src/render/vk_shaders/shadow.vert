#version 450
// Depth-only pass for the modern shadow map.

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform Push
{
	mat4 lightWorld;	// lightMatrix * instanceWorld
} pc;

void main()
{
	gl_Position = pc.lightWorld * vec4(inPosition, 1.0);
}
