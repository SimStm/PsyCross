#ifndef PSYX_NATIVE_MIP_H
#define PSYX_NATIVE_MIP_H

#include <stdint.h>
#include <vector>

namespace PsyXNativeMip
{
struct Level
{
	uint32_t width, height;
	uint64_t offset;
};

// Dimensions are validated before any caller pixel memory is traversed.
uint64_t Layout(uint32_t width, uint32_t height, bool mipmapped, std::vector<Level>& levels);
// Level zero is byte-exact. Lower levels use linear-light, alpha-weighted area
// filtering and the closest attainable alpha-test coverage; zero alpha stays zero.
void Build(const uint8_t* base, const std::vector<Level>& levels, uint64_t bytes,
	float alphaCutoff, std::vector<uint8_t>& pixels);
}
#endif
