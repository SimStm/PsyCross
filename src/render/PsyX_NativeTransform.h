#ifndef PSYX_NATIVE_TRANSFORM_H
#define PSYX_NATIVE_TRANSFORM_H

namespace PsyXNativeTransform
{
// Cancel large city/camera translations before projection. Keep both CPU
// products in double until the final shader matrix; no GPU float64 is needed.
inline void Compose(float out[16], const float projection[16], const float view[16], const float world[16])
{
	double viewModel[16];
	for (unsigned int column=0; column<4; ++column)
		for (unsigned int row=0; row<4; ++row)
		{
			double value=0;
			for (unsigned int k=0; k<4; ++k) value+=double(view[k*4+row])*world[column*4+k];
			viewModel[column*4+row]=value;
		}
	for (unsigned int column=0; column<4; ++column)
		for (unsigned int row=0; row<4; ++row)
		{
			double value=0;
			for (unsigned int k=0; k<4; ++k) value+=double(projection[k*4+row])*viewModel[column*4+k];
			out[column*4+row]=float(value);
		}
}
}
#endif
