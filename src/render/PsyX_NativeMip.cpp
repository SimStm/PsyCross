#include "PsyX_NativeMip.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{
struct LinearTable
{
	double values[256];
	LinearTable()
	{
		for (unsigned int i=0; i<256; ++i)
		{
			const double encoded=i/255.0;
			values[i]=encoded<=0.04045 ? encoded/12.92 : std::pow((encoded+0.055)/1.055,2.4);
		}
	}
};

uint8_t Byte(double value)
{
	return uint8_t(std::max(0.0,std::min(255.0,std::floor(value+0.5))));
}

uint8_t Encode(double linear)
{
	return Byte(255.0*(linear<=0.0031308 ? 12.92*linear : 1.055*std::pow(linear,1.0/2.4)-0.055));
}

void PreserveCoverage(uint8_t* pixels, size_t count, unsigned int threshold, double target)
{
	if (!threshold) return;
	size_t histogram[256]={}, covered=0;
	for (size_t i=0; i<count; ++i) ++histogram[pixels[i*4+3]];
	for (unsigned int a=threshold; a<256; ++a) covered+=histogram[a];
	double scale=1.0, error=std::fabs(double(covered)/count-target);
	// A uniform scale cannot split a bucket of equal alphas. Keep the closest
	// representable coverage, preferring the least change on an equal error.
	covered=0;
	for (unsigned int a=255; a>0; --a)
	{
		if (!histogram[a]) continue;
		covered+=histogram[a];
		const double candidate=(threshold-0.5+1e-7)/a;
		const double candidateError=std::fabs(double(covered)/count-target);
		if (candidateError<error || (candidateError==error && std::fabs(candidate-1)<std::fabs(scale-1)))
		{
			scale=candidate; error=candidateError;
		}
	}
	// Zero coverage is also attainable without turning transparent texels opaque.
	if (target<error) scale=0;
	if (scale==1.0) return;
	for (size_t i=0; i<count; ++i) pixels[i*4+3]=Byte(pixels[i*4+3]*scale);
}

void Downsample(const uint8_t* source, const PsyXNativeMip::Level& from,
	uint8_t* destination, const PsyXNativeMip::Level& to)
{
	static const LinearTable linear;
	for (uint32_t y=0; y<to.height; ++y)
	for (uint32_t x=0; x<to.width; ++x)
	{
		const double left=double(x)*from.width/to.width, right=double(x+1)*from.width/to.width;
		const double top=double(y)*from.height/to.height, bottom=double(y+1)*from.height/to.height;
		double alpha=0, rgb[3]={};
		for (uint32_t sy=uint32_t(top); sy<uint32_t(std::ceil(bottom)); ++sy)
		for (uint32_t sx=uint32_t(left); sx<uint32_t(std::ceil(right)); ++sx)
		{
			const double weight=(std::min(right,double(sx+1))-std::max(left,double(sx)))*
				(std::min(bottom,double(sy+1))-std::max(top,double(sy)));
			const uint8_t* pixel=source+(size_t(sy)*from.width+sx)*4;
			const double contribution=weight*pixel[3];
			alpha+=contribution;
			for (unsigned int channel=0; channel<3; ++channel) rgb[channel]+=linear.values[pixel[channel]]*contribution;
		}
		uint8_t* pixel=destination+(size_t(y)*to.width+x)*4;
		for (unsigned int channel=0; channel<3; ++channel) pixel[channel]=alpha>0 ? Encode(rgb[channel]/alpha) : 0;
		pixel[3]=Byte(alpha/((right-left)*(bottom-top)));
	}
}
}

uint64_t PsyXNativeMip::Layout(uint32_t width, uint32_t height, bool mipmapped, std::vector<Level>& levels)
{
	levels.clear();
	if (!width || !height || width>4096 || height>4096) return 0;
	uint64_t bytes=0;
	for (;;)
	{
		Level level={width,height,bytes};
		levels.push_back(level);
		bytes+=uint64_t(width)*height*4;
		if (!mipmapped || (width==1 && height==1)) return bytes;
		width=std::max(1u,width/2); height=std::max(1u,height/2);
	}
}

void PsyXNativeMip::Build(const uint8_t* base, const std::vector<Level>& levels, uint64_t bytes,
	float alphaCutoff, std::vector<uint8_t>& pixels)
{
	pixels.resize(size_t(bytes));
	const size_t baseCount=size_t(levels[0].width)*levels[0].height;
	memcpy(pixels.data(),base,baseCount*4);
	if (levels.size()==1) return;
	// Match the shader's normalized float comparison, including cutoffs such as
	// 0.8f whose double representation would incorrectly ceil to the next byte.
	unsigned int threshold=0;
	while (threshold<255 && float(threshold)/255.0f<alphaCutoff) ++threshold;
	size_t covered=0;
	for (size_t i=0; i<baseCount; ++i) if (base[i*4+3]>=threshold) ++covered;
	const double coverage=double(covered)/baseCount;
	for (size_t i=1; i<levels.size(); ++i)
	{
		uint8_t* destination=pixels.data()+size_t(levels[i].offset);
		Downsample(pixels.data()+size_t(levels[i-1].offset),levels[i-1],destination,levels[i]);
		PreserveCoverage(destination,size_t(levels[i].width)*levels[i].height,threshold,coverage);
	}
}
