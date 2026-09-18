#ifndef PSYX_MODERN_MESH_INTERNAL_H
#define PSYX_MODERN_MESH_INTERNAL_H

/* Internal glue between the fixed-function renderer and the experimental
   modern mesh path. Not part of the public PsyCross API. */

/* Column-major 3D projection captured from GR_Perspective3D so the modern
   path uses exactly the same clip-space mapping as legacy geometry. */
extern float g_psyxModernProjection[16];
extern int   g_psyxModernProjectionValid;

void PsyX_ModernMesh_SetProjection(const float matrix[16]);
void PsyX_ModernMesh_RenderFrame(void);
void PsyX_ModernMesh_Shutdown(void);

#endif // PSYX_MODERN_MESH_INTERNAL_H
