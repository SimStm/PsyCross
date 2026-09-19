#version 450
// Fullscreen triangle used by the modern shadow composite. No vertex buffer is
// bound; the position is generated from the vertex index (same construction as
// the OpenGL composite draw).

void main()
{
	vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
	gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
