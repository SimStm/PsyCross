#ifndef PSYX_MODERN_MESH_INTERNAL_H
#define PSYX_MODERN_MESH_INTERNAL_H

/* Internal glue between the fixed-function renderer and the experimental
   modern mesh path. Not part of the public PsyCross API. */

/* Column-major 3D projection captured from GR_Perspective3D so the modern
   path uses exactly the same clip-space mapping as legacy geometry. */
extern float g_psyxModernProjection[16];
extern int   g_psyxModernProjectionValid;

/* Directional shadow volume alignment shared by the OpenGL and Vulkan modern
   paths: snap a world-space shadow-volume centre to the light-space texel grid
   of a `shadowSize` map covering `extent` units either way. `outUp` returns the
   up vector of the light basis the snap used, so the caller's look-at matrix
   and the snap agree. */
void PsyX_ModernShadowSnapCentre(const float lightDirection[3], const float centre[3],
	float extent, int shadowSize, float outCentre[3], float outUp[3]);

void PsyX_ModernMesh_SetProjection(const float matrix[16]);
void PsyX_ModernMesh_RenderFrame(void);
void PsyX_ModernMesh_Shutdown(void);

#endif // PSYX_MODERN_MESH_INTERNAL_H
