#ifndef PSYX_NATIVE_PLANE_H
#define PSYX_NATIVE_PLANE_H

#include <cmath>

namespace PsyXNativePlane
{
// A frame-wide clip-to-world inverse makes equal world heights produce equal
// depth at every fragment, independent of mesh origins and triangulation.
inline bool Inverse(double out[16], const float projection[16], const float view[16])
{
	double rows[4][8] = {};
	for (unsigned int row=0; row<4; ++row)
	{
		for (unsigned int column=0; column<4; ++column)
			for (unsigned int k=0; k<4; ++k)
				rows[row][column]+=double(projection[k*4+row])*view[column*4+k];
		rows[row][row+4]=1;
	}
	for (unsigned int column=0; column<4; ++column)
	{
		unsigned int pivot=column;
		for (unsigned int row=column+1; row<4; ++row)
			if (std::fabs(rows[row][column])>std::fabs(rows[pivot][column])) pivot=row;
		if (!std::isfinite(rows[pivot][column]) || rows[pivot][column]==0) return false;
		for (unsigned int k=0; k<8; ++k)
		{ const double value=rows[column][k]; rows[column][k]=rows[pivot][k]; rows[pivot][k]=value; }
		const double divisor=rows[column][column];
		for (unsigned int k=0; k<8; ++k) rows[column][k]/=divisor;
		for (unsigned int row=0; row<4; ++row) if (row!=column)
		{
			const double factor=rows[row][column];
			for (unsigned int k=0; k<8; ++k) rows[row][k]-=factor*rows[column][k];
		}
	}
	for (unsigned int column=0; column<4; ++column)
		for (unsigned int row=0; row<4; ++row)
		{
			out[column*4+row]=rows[row][column+4];
			if (!std::isfinite(out[column*4+row])) return false;
		}
	return true;
}

enum Result { RasterDepth, PlanarDepth, Invalid };
inline Result Horizontal(float out[4], const double inverse[16], const float world[16],
	float localY, unsigned int width, unsigned int height, unsigned int depthLayer)
{
	out[0]=out[1]=out[2]=0; out[3]=-1;
	if (world[1]!=0 || world[9]!=0) return RasterDepth;
	if (!width || !height) return Invalid;
	const double worldY=double(world[5])*localY+world[13];
	double plane[4];
	for (unsigned int column=0; column<4; ++column)
		plane[column]=inverse[column*4+1]-worldY*inverse[column*4+3];
	// A plane through the eye projects to a line; it has no surface depth.
	if (plane[2]==0) return RasterDepth;
	out[0]=float(-2*plane[0]/plane[2]/width);
	out[1]=float(-2*plane[1]/plane[2]/height);
	out[2]=float((plane[0]+plane[1]-plane[3])/plane[2]);
	for (unsigned int i=0; i<3; ++i) if (!std::isfinite(out[i])) return Invalid;
	out[3]=float(4*depthLayer);
	return PlanarDepth;
}
}
#endif
