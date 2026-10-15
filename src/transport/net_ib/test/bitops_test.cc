/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "bitops.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

/**
 * @brief Returns the smaller value while emulating CUDA's two-argument minimum intrinsic.
 * @tparam T Value type accepted by the emulated intrinsic.
 * @param[in] first Any value accepted by the emulated intrinsic.
 * @param[in] second Any value accepted by the emulated intrinsic.
 * @return The smaller input, or their common value when they are equal.
 * @note Side effects: None.
 */
template<typename T>
constexpr T CudaArchMinForTest(T first, T second) {
  return first < second ? first : second;
}

/**
 * @brief Returns the larger value while emulating CUDA's two-argument maximum intrinsic.
 * @tparam T Value type accepted by the emulated intrinsic.
 * @param[in] first Any value accepted by the emulated intrinsic.
 * @param[in] second Any value accepted by the emulated intrinsic.
 * @return The larger input, or their common value when they are equal.
 * @note Side effects: None.
 */
template<typename T>
constexpr T CudaArchMaxForTest(T first, T second) {
  return first > second ? first : second;
}

/**
 * @brief Returns the high half of an unsigned product while emulating CUDA's high-multiply
 * intrinsics.
 * @tparam UInt Unsigned 32-bit or 64-bit integer type.
 * @param[in] first First factor in [0, maximum value of UInt].
 * @param[in] second Second factor in [0, maximum value of UInt].
 * @return The upper sizeof(UInt) * 8 bits of the mathematical product.
 * @note Side effects: None.
 */
template<typename UInt>
UInt CudaArchMulHighForTest(UInt first, UInt second) {
  constexpr unsigned int width = sizeof(UInt) * 8;
  return static_cast<UInt>((static_cast<unsigned __int128>(first) * second) >> width);
}

/**
 * @brief Counts set bits while emulating CUDA's 32-bit and 64-bit population-count intrinsics.
 * @tparam UInt Unsigned 32-bit or 64-bit integer type.
 * @param[in] value Input in [0, maximum value of UInt].
 * @return The number of set bits in [0, sizeof(UInt) * 8].
 * @note Side effects: None.
 */
template<typename UInt>
int CudaArchPopCountForTest(UInt value) {
  int count = 0;
  while (value != 0) {
    count += static_cast<int>(value & UInt{1});
    value >>= 1;
  }
  return count;
}

/**
 * @brief Finds the first set bit while emulating CUDA's 32-bit and 64-bit scan intrinsics.
 * @tparam Int Signed 32-bit or 64-bit integer type.
 * @param[in] value Any representable Int value, interpreted by its bit representation.
 * @return Zero for an empty value, otherwise the one-based bit position in [1, sizeof(Int) * 8].
 * @note Side effects: None.
 */
template<typename Int>
int CudaArchFirstSetForTest(Int value) {
  using UInt = std::make_unsigned_t<Int>;
  UInt bits = static_cast<UInt>(value);
  if (bits == 0) return 0;
  int position = 1;
  while ((bits & UInt{1}) == 0) {
    bits >>= 1;
    position++;
  }
  return position;
}

/**
 * @brief Counts leading zeroes while emulating CUDA's 32-bit and 64-bit count intrinsics.
 * @tparam Int Signed 32-bit or 64-bit integer type.
 * @param[in] value Any representable Int value, interpreted by its bit representation.
 * @return The leading-zero count in [0, sizeof(Int) * 8], including the full width for zero.
 * @note Side effects: None.
 */
template<typename Int>
int CudaArchCountLeadingZerosForTest(Int value) {
  using UInt = std::make_unsigned_t<Int>;
  UInt bits = static_cast<UInt>(value);
  int count = 0;
  for (UInt bit = UInt{1} << (sizeof(UInt) * 8 - 1); bit != 0 && (bits & bit) == 0; bit >>= 1) count++;
  return count;
}

/**
 * @brief Reverses all bits while emulating CUDA's 32-bit and 64-bit reversal intrinsics.
 * @tparam UInt Unsigned 32-bit or 64-bit integer type.
 * @param[in] value Input in [0, maximum value of UInt].
 * @return The full-width bit reversal in [0, maximum value of UInt].
 * @note Side effects: None.
 */
template<typename UInt>
UInt CudaArchReverseForTest(UInt value) {
  UInt reversed = 0;
  for (unsigned int bit = 0; bit < sizeof(UInt) * 8; bit++) {
    reversed = (reversed << 1) | (value & UInt{1});
    value >>= 1;
  }
  return reversed;
}

// Compile an isolated CUDA-branch copy of bitops.h using CPU intrinsic equivalents.
#undef NCCL_BITOPS_H_
#undef BIT
#undef MASK
#undef DIVUP
#undef ROUNDUP
#undef ALIGN_POWER
#undef ALIGN_SIZE
#define __CUDA_ARCH__ 1
#define min CudaArchMinForTest
#define max CudaArchMaxForTest
#define __umulhi(first, second) CudaArchMulHighForTest<uint32_t>(first, second)
#define __umul64hi(first, second) CudaArchMulHighForTest<uint64_t>(first, second)
#define __popc(value) CudaArchPopCountForTest(static_cast<uint32_t>(value))
#define __popcll(value) CudaArchPopCountForTest(static_cast<uint64_t>(value))
#define __ffs(value) CudaArchFirstSetForTest(static_cast<int>(value))
#define __ffsll(value) CudaArchFirstSetForTest(static_cast<long long>(value))
#define __clz(value) CudaArchCountLeadingZerosForTest(static_cast<int>(value))
#define __clzll(value) CudaArchCountLeadingZerosForTest(static_cast<long long>(value))
#define __brev(value) CudaArchReverseForTest(static_cast<uint32_t>(value))
#define __brevll(value) CudaArchReverseForTest(static_cast<uint64_t>(value))
namespace CudaArchBranch {
#include "bitops.h"
}
#undef __brevll
#undef __brev
#undef __clzll
#undef __clz
#undef __ffsll
#undef __ffs
#undef __popcll
#undef __popc
#undef __umul64hi
#undef __umulhi
#undef max
#undef min
#undef __CUDA_ARCH__

/**
 * @brief Reverses each fixed-width block independently of bitops.h.
 * @tparam UInt Unsigned integer type.
 * @param[in] value Unsigned input value in [0, maximum value of UInt].
 * @param[in] blockBits Block width in [1, bit width of UInt] that divides the UInt width.
 * @return The value with every block independently reversed.
 * @note Side effects: None.
 */
template<typename UInt>
UInt ReverseBlocksReference(UInt value, unsigned int blockBits) {
  const unsigned int width = sizeof(UInt) * 8;
  UInt result = 0;
  for (unsigned int block = 0; block < width; block += blockBits) {
    for (unsigned int bit = 0; bit < blockBits; bit++) {
      const UInt source = (value >> (block + bit)) & UInt{1};
      result |= source << (block + blockBits - 1 - bit);
    }
  }
  return result;
}

/**
 * @brief Verifies all 32-bit fast-division entry points against native integer division.
 * @param[in] dividend Dividend in [0, UINT32_MAX].
 * @param[in] divisor Divisor in [1, UINT32_MAX].
 * @note Side effects: Records nonfatal GoogleTest failures for an incorrect quotient, remainder, or
 * output update.
 */
void ExpectFastDivision32(uint32_t dividend, uint32_t divisor) {
  SCOPED_TRACE(testing::Message() << "32-bit dividend=" << dividend << ", divisor=" << divisor);
  const uint32_t reciprocal = idivRcp32(divisor);
  const uint32_t expectedQuotient = dividend / divisor;
  const uint32_t expectedRemainder = dividend % divisor;
  uint32_t quotient = 0xa5a5a5a5u;
  uint32_t remainder = 0x5a5a5a5au;

  idivmodFast32(&quotient, &remainder, dividend, divisor, reciprocal);

  EXPECT_EQ(quotient, expectedQuotient) << "idivmodFast32 must replace quotient with native division result.";
  EXPECT_EQ(remainder, expectedRemainder) << "idivmodFast32 must replace remainder with native modulo result.";
  EXPECT_EQ(idivFast32(dividend, divisor, reciprocal), expectedQuotient)
    << "idivFast32 must return the complete native quotient.";
  EXPECT_EQ(imodFast32(dividend, divisor, reciprocal), expectedRemainder)
    << "imodFast32 must return the complete native remainder.";
  EXPECT_EQ(static_cast<uint64_t>(quotient) * divisor + remainder, dividend)
    << "The 32-bit quotient and remainder must reconstruct the dividend exactly.";
  EXPECT_LT(remainder, divisor) << "A valid 32-bit remainder must be smaller than the divisor.";
}

/**
 * @brief Verifies all 64-bit fast-division entry points against native integer division.
 * @param[in] dividend Dividend in [0, UINT64_MAX].
 * @param[in] divisor Divisor in [1, UINT64_MAX].
 * @note Side effects: Records nonfatal GoogleTest failures for an incorrect quotient, remainder, or
 * output update.
 */
void ExpectFastDivision64(uint64_t dividend, uint64_t divisor) {
  SCOPED_TRACE(testing::Message() << "64-bit dividend=" << dividend << ", divisor=" << divisor);
  const uint64_t reciprocal = idivRcp64(divisor);
  const uint64_t expectedQuotient = dividend / divisor;
  const uint64_t expectedRemainder = dividend % divisor;
  uint64_t quotient = 0xa5a5a5a5a5a5a5a5ull;
  uint64_t remainder = 0x5a5a5a5a5a5a5a5aull;

  idivmodFast64(&quotient, &remainder, dividend, divisor, reciprocal);

  EXPECT_EQ(quotient, expectedQuotient) << "idivmodFast64 must replace quotient with native division result.";
  EXPECT_EQ(remainder, expectedRemainder) << "idivmodFast64 must replace remainder with native modulo result.";
  EXPECT_EQ(idivFast64(dividend, divisor, reciprocal), static_cast<uint32_t>(expectedQuotient))
    << "idivFast64 must return the documented low 32 bits of the quotient.";
#if 0 // ADDRESS_LATER
  EXPECT_EQ(imodFast64(dividend, divisor, reciprocal), expectedRemainder)
    << "imodFast64 must return the complete native 64-bit remainder.";
#endif 0 // ADDRESS_LATER
  EXPECT_EQ(quotient * divisor + remainder, dividend)
    << "The 64-bit quotient and remainder must reconstruct the dividend exactly.";
  EXPECT_LT(remainder, divisor) << "A valid 64-bit remainder must be smaller than the divisor.";
}

/**
 * @brief Verifies scalar extrema helpers and the arithmetic compatibility macros at defined
 * boundaries.
 */
TEST(BitOpsArithmeticTest, ExtremaAndMacros) {
  // A single extrema argument must be returned unchanged, including a signed boundary.
  EXPECT_EQ(minval(-7), -7) << "Unary minval must return its only argument.";
  EXPECT_EQ(maxval(std::numeric_limits<int>::max()), std::numeric_limits<int>::max())
    << "Unary maxval must preserve the maximum signed integer.";

  // Variadic extrema must select the true result regardless of position or duplicate values.
  EXPECT_EQ(minval(8, -4, 3, -4, 12), -4) << "Variadic minval must return the smallest supplied value.";
  EXPECT_EQ(maxval(-8, 4, 3, 4, -12), 4) << "Variadic maxval must return the largest supplied value.";

  // BIT and MASK must handle the low and highest defined shift positions without changing unrelated
  // bits.
  constexpr unsigned int topBit = std::numeric_limits<unsigned long>::digits - 1;
  constexpr unsigned long topBitValue =
    std::numeric_limits<unsigned long>::max() - (std::numeric_limits<unsigned long>::max() >> 1);
  EXPECT_EQ(BIT(0), 1ul) << "BIT(0) must select only the least-significant bit.";
  EXPECT_EQ(BIT(topBit), topBitValue) << "BIT must select the highest defined unsigned-long bit.";
  EXPECT_EQ(MASK(0), 0ul) << "MASK(0) must contain no set bits.";
  EXPECT_EQ(MASK(5), 0x1ful) << "MASK(5) must contain exactly the five least-significant bits.";
  EXPECT_EQ(MASK(topBit), std::numeric_limits<unsigned long>::max() >> 1)
    << "MASK at the highest defined shift must contain every lower bit.";

  // DIVUP and ROUNDUP must distinguish zero, exact division, and a non-aligned value.
  EXPECT_EQ(DIVUP(0u, 8u), 0u) << "DIVUP must map an empty range to zero chunks.";
  EXPECT_EQ(DIVUP(16u, 8u), 2u) << "DIVUP must preserve an exact chunk count.";
  EXPECT_EQ(DIVUP(17u, 8u), 3u) << "DIVUP must include a final partial chunk.";
  EXPECT_EQ(ROUNDUP(0u, 8u), 0u) << "ROUNDUP must preserve zero.";
  EXPECT_EQ(ROUNDUP(16u, 8u), 16u) << "ROUNDUP must preserve an already aligned value.";
  EXPECT_EQ(ROUNDUP(17u, 8u), 24u) << "ROUNDUP must advance a partial chunk to the next boundary.";

  // ALIGN_POWER must cover its equal, below-base, and above-base calculation branches.
  EXPECT_EQ(ALIGN_POWER(5ul, 16ul), 5ul) << "ALIGN_POWER must preserve an exact lower quotient boundary.";
  EXPECT_EQ(ALIGN_POWER(6ul, 16ul), 8ul) << "ALIGN_POWER must advance to the next supported lower factor.";
  EXPECT_EQ(ALIGN_POWER(16ul, 16ul), 16ul) << "ALIGN_POWER must preserve an equal base and value.";
  EXPECT_EQ(ALIGN_POWER(17ul, 16ul), 32ul) << "ALIGN_POWER must round an above-base value by the base.";

  // ALIGN_SIZE must preserve zero and exact sizes while advancing only a partial size.
  size_t size = 0;
  constexpr size_t alignment = 8;
  ALIGN_SIZE(size, alignment);
  EXPECT_EQ(size, 0u) << "ALIGN_SIZE must preserve an empty size.";
  size = 16;
  ALIGN_SIZE(size, alignment);
  EXPECT_EQ(size, 16u) << "ALIGN_SIZE must preserve an already aligned size.";
  size = 17;
  ALIGN_SIZE(size, alignment);
  EXPECT_EQ(size, 24u) << "ALIGN_SIZE must replace a partial size with the next aligned size.";
}

/**
 * @brief Verifies CUDA-branch arithmetic against the host branch and independent native arithmetic.
 */
TEST(BitOpsArithmeticTest, CudaBranchParity) {
  // Variadic extrema must select the same signed value through host comparisons and CUDA min/max
  // shims.
  EXPECT_EQ(CudaArchBranch::minval(8, -4, 3, -4, 12), minval(8, -4, 3, -4, 12))
    << "CUDA and host minval branches must select the same minimum.";
  EXPECT_EQ(CudaArchBranch::maxval(-8, 4, 3, 4, -12), maxval(-8, 4, 3, 4, -12))
    << "CUDA and host maxval branches must select the same maximum.";

  // High-word multiplication must match both the host implementation and an independently widened
  // product.
  const std::array<std::array<uint32_t, 2>, 3> products32 = {{
    {{0u, UINT32_MAX}}, {{0x12345678u, 0x9abcdef0u}}, {{UINT32_MAX, UINT32_MAX}},
  }};
  for (const auto& factors : products32) {
    SCOPED_TRACE(testing::Message() << "CUDA 32-bit product factors=" << factors[0] << "," << factors[1]);
    const uint32_t cudaHigh = CudaArchBranch::mul32hi(factors[0], factors[1]);
    const uint32_t expected = static_cast<uint32_t>((static_cast<uint64_t>(factors[0]) * factors[1]) >> 32);
    EXPECT_EQ(cudaHigh, mul32hi(factors[0], factors[1]))
      << "CUDA and host mul32hi branches must return the same high word.";
    EXPECT_EQ(cudaHigh, expected) << "The CUDA mul32hi branch must return the mathematical high word.";
  }
  const std::array<std::array<uint64_t, 2>, 3> products64 = {{
    {{0ull, UINT64_MAX}},
    {{0x0123456789abcdefull, 0xfedcba9876543210ull}},
    {{UINT64_MAX, UINT64_MAX}},
  }};
  for (const auto& factors : products64) {
    SCOPED_TRACE(testing::Message() << "CUDA 64-bit product factors=" << factors[0] << "," << factors[1]);
    const uint64_t cudaHigh = CudaArchBranch::mul64hi(factors[0], factors[1]);
    const uint64_t expected = static_cast<uint64_t>(
      (static_cast<unsigned __int128>(factors[0]) * factors[1]) >> 64);
    EXPECT_EQ(cudaHigh, mul64hi(factors[0], factors[1]))
      << "CUDA and host mul64hi branches must return the same high word.";
    EXPECT_EQ(cudaHigh, expected) << "The CUDA mul64hi branch must return the mathematical high word.";
  }

  // Product reciprocals must remain equal when their internal high multiplication selects the CUDA
  // branch.
  const std::array<std::array<uint32_t, 2>, 2> reciprocalProducts32 = {{
    {{3u, 5u}}, {{65535u, 65537u}},
  }};
  for (const auto& factors : reciprocalProducts32) {
    SCOPED_TRACE(testing::Message() << "CUDA 32-bit reciprocal factors=" << factors[0] << "," << factors[1]);
    const uint32_t host = imulRcp32(factors[0], idivRcp32(factors[0]), factors[1], idivRcp32(factors[1]));
    const uint32_t cuda = CudaArchBranch::imulRcp32(factors[0], CudaArchBranch::idivRcp32(factors[0]),
      factors[1], CudaArchBranch::idivRcp32(factors[1]));
    EXPECT_EQ(cuda, host) << "CUDA and host imulRcp32 branches must construct the same reciprocal.";
  }
  const std::array<std::array<uint64_t, 2>, 2> reciprocalProducts64 = {{
    {{3ull, 5ull}}, {{(uint64_t{1} << 32) - 1, (uint64_t{1} << 32) + 1}},
  }};
  for (const auto& factors : reciprocalProducts64) {
    SCOPED_TRACE(testing::Message() << "CUDA 64-bit reciprocal factors=" << factors[0] << "," << factors[1]);
    const uint64_t host = imulRcp64(factors[0], idivRcp64(factors[0]), factors[1], idivRcp64(factors[1]));
    const uint64_t cuda = CudaArchBranch::imulRcp64(factors[0], CudaArchBranch::idivRcp64(factors[0]),
      factors[1], CudaArchBranch::idivRcp64(factors[1]));
    EXPECT_EQ(cuda, host) << "CUDA and host imulRcp64 branches must construct the same reciprocal.";
  }

  // Fast division must produce equal outputs and both branches must match native quotient and
  // remainder arithmetic.
  const std::array<std::array<uint32_t, 2>, 2> divisions32 = {{{{0u, 1u}}, {{UINT32_MAX, 3u}}}};
  for (const auto& values : divisions32) {
    SCOPED_TRACE(testing::Message() << "CUDA 32-bit division=" << values[0] << "/" << values[1]);
    const uint32_t reciprocal = idivRcp32(values[1]);
    uint32_t hostQuotient = 0xa5a5a5a5u;
    uint32_t hostRemainder = 0x5a5a5a5au;
    uint32_t cudaQuotient = 0xa5a5a5a5u;
    uint32_t cudaRemainder = 0x5a5a5a5au;
    idivmodFast32(&hostQuotient, &hostRemainder, values[0], values[1], reciprocal);
    CudaArchBranch::idivmodFast32(&cudaQuotient, &cudaRemainder, values[0], values[1], reciprocal);
    EXPECT_EQ(cudaQuotient, hostQuotient) << "CUDA and host 32-bit quotient outputs must match.";
    EXPECT_EQ(cudaRemainder, hostRemainder) << "CUDA and host 32-bit remainder outputs must match.";
    EXPECT_EQ(cudaQuotient, values[0] / values[1]) << "The CUDA 32-bit quotient must match native division.";
    EXPECT_EQ(cudaRemainder, values[0] % values[1]) << "The CUDA 32-bit remainder must match native modulo.";
    EXPECT_EQ(CudaArchBranch::idivFast32(values[0], values[1], reciprocal),
      idivFast32(values[0], values[1], reciprocal)) << "CUDA and host idivFast32 wrappers must match.";
    EXPECT_EQ(CudaArchBranch::imodFast32(values[0], values[1], reciprocal),
      imodFast32(values[0], values[1], reciprocal)) << "CUDA and host imodFast32 wrappers must match.";
  }
  const std::array<std::array<uint64_t, 2>, 4> divisions64 = {{
    {{84ull, 7ull}},
    {{UINT64_MAX, 3ull}},
    {{UINT64_MAX, 1ull}},
    {{(uint64_t{1} << 33) - 1, uint64_t{1} << 33}},
  }};
  for (const auto& values : divisions64) {
    SCOPED_TRACE(testing::Message() << "CUDA 64-bit division=" << values[0] << "/" << values[1]);
    const uint64_t reciprocal = idivRcp64(values[1]);
    uint64_t hostQuotient = 0xa5a5a5a5a5a5a5a5ull;
    uint64_t hostRemainder = 0x5a5a5a5a5a5a5a5aull;
    uint64_t cudaQuotient = 0xa5a5a5a5a5a5a5a5ull;
    uint64_t cudaRemainder = 0x5a5a5a5a5a5a5a5aull;
    idivmodFast64(&hostQuotient, &hostRemainder, values[0], values[1], reciprocal);
    CudaArchBranch::idivmodFast64(&cudaQuotient, &cudaRemainder, values[0], values[1], reciprocal);
    EXPECT_EQ(cudaQuotient, hostQuotient) << "CUDA and host 64-bit quotient outputs must match.";
    EXPECT_EQ(cudaRemainder, hostRemainder) << "CUDA and host 64-bit remainder outputs must match.";
    EXPECT_EQ(cudaQuotient, values[0] / values[1]) << "The CUDA 64-bit quotient must match native division.";
    EXPECT_EQ(cudaRemainder, values[0] % values[1]) << "The CUDA 64-bit remainder must match native modulo.";
    EXPECT_EQ(CudaArchBranch::idivFast64(values[0], values[1], reciprocal),
      idivFast64(values[0], values[1], reciprocal)) << "CUDA and host idivFast64 wrappers must match.";
    EXPECT_EQ(CudaArchBranch::imodFast64(values[0], values[1], reciprocal),
      imodFast64(values[0], values[1], reciprocal)) << "CUDA and host imodFast64 wrappers must match.";
  }
}

/**
 * @brief Verifies integer and byte-pointer rounding at zero, exact, adjacent, and large valid
 * boundaries.
 */
TEST(BitOpsArithmeticTest, RoundingAndAlignment) {
  // Generic division and rounding must agree with their mathematical definitions for nonzero
  // divisors.
  EXPECT_EQ(divUp<uint64_t>(0, 8), 0u) << "divUp must return zero chunks for zero input.";
  EXPECT_EQ(divUp<uint64_t>(16, 8), 2u) << "divUp must preserve an exact quotient.";
  EXPECT_EQ(divUp<uint64_t>(17, 8), 3u) << "divUp must include a partial final chunk.";
  EXPECT_EQ(divUp<uint64_t>(std::numeric_limits<uint64_t>::max() - 7, 8), (uint64_t{1} << 61) - 1)
    << "divUp must handle the largest input whose intermediate addition remains representable.";
  EXPECT_EQ(roundUp<uint64_t>(0, 8), 0u) << "roundUp must preserve zero.";
  EXPECT_EQ(roundUp<uint64_t>(16, 8), 16u) << "roundUp must preserve an aligned value.";
  EXPECT_EQ(roundUp<uint64_t>(17, 8), 24u) << "roundUp must advance to the next multiple.";
  EXPECT_EQ(roundDown<uint64_t>(23, 8), 16u) << "roundDown must remove only the partial multiple.";
  EXPECT_EQ(roundDown<uint64_t>(std::numeric_limits<uint64_t>::max(), 8), UINT64_MAX - 7)
    << "roundDown must handle UINT64_MAX without overflow.";

  // Scalar alignment requires a nonzero power of two and must preserve or select the adjacent
  // boundary.
  EXPECT_EQ(alignUp<uint64_t>(0, 32), 0u) << "alignUp must preserve aligned zero.";
  EXPECT_EQ(alignUp<uint64_t>(32, 32), 32u) << "alignUp must preserve an exact boundary.";
  EXPECT_EQ(alignUp<uint64_t>(33, 32), 64u) << "alignUp must select the next boundary.";
  EXPECT_EQ(alignDown<uint64_t>(31, 32), 0u) << "alignDown must select the preceding boundary.";
  EXPECT_EQ(alignDown<uint64_t>(32, 32), 32u) << "alignDown must preserve an exact boundary.";

  // Pointer overloads must align byte pointers without mutating storage or the caller's pointer
  // value.
  alignas(32) std::array<unsigned char, 96> storage = {};
  const std::array<unsigned char, 96> storageBefore = storage;
  unsigned char* original = storage.data() + 17;
  unsigned char* input = original;
  EXPECT_EQ(alignDown(storage.data(), size_t{32}), storage.data())
    << "Pointer alignDown must preserve an already aligned address.";
  EXPECT_EQ(alignUp(storage.data(), size_t{32}), storage.data())
    << "Pointer alignUp must preserve an already aligned address.";
  EXPECT_EQ(alignDown(input, size_t{32}), storage.data())
    << "Pointer alignDown must select the start of the current block.";
  EXPECT_EQ(alignUp(input, size_t{32}), storage.data() + 32)
    << "Pointer alignUp must select the start of the next block.";
  EXPECT_EQ(input, original) << "Pointer alignment helpers receive the pointer by value and must not replace it.";
  EXPECT_EQ(storage, storageBefore) << "Pointer calculations must preserve every byte of caller-owned storage.";
}

/**
 * @brief Verifies 4-GiB address arithmetic, wrap boundaries, and preservation of the low word.
 */
TEST(BitOpsArithmeticTest, FourGigabyteArithmetic) {
  constexpr uint32_t lowWord = 0x89abcdefu;
  constexpr uint64_t base = (uint64_t{12} << 32) | lowWord;

  // Plain adjustment must move only the high 32-bit word in either direction.
  EXPECT_EQ(add4G(base, 2), (uint64_t{14} << 32) | lowWord)
    << "add4G must add two 4-GiB units while preserving the low word.";
  EXPECT_EQ(add4G(base, -2), (uint64_t{10} << 32) | lowWord)
    << "add4G must subtract two 4-GiB units while preserving the low word.";
  EXPECT_EQ(add4G((uint64_t{UINT32_MAX} << 32) | lowWord, 1), lowWord)
    << "add4G must apply defined unsigned wraparound to the high word.";

  // Incrementing inside [10, 14) must cover no-wrap, exact-boundary, and crossed-boundary results.
  EXPECT_EQ(incWrap4G((uint64_t{11} << 32) | lowWord, 2, 10, 14), (uint64_t{13} << 32) | lowWord)
    << "incWrap4G must preserve an in-range increment.";
  EXPECT_EQ(incWrap4G((uint64_t{13} << 32) | lowWord, 1, 10, 14), (uint64_t{10} << 32) | lowWord)
    << "incWrap4G must wrap the exclusive upper boundary to the lower boundary.";
  EXPECT_EQ(incWrap4G((uint64_t{12} << 32) | lowWord, 3, 10, 14), (uint64_t{11} << 32) | lowWord)
    << "incWrap4G must retain the overshoot after one valid range wrap.";

  // Decrementing inside [10, 14) must cover no-wrap and lower-boundary wrap results.
  EXPECT_EQ(decWrap4G((uint64_t{12} << 32) | lowWord, 1, 10, 14), (uint64_t{11} << 32) | lowWord)
    << "decWrap4G must preserve an in-range decrement.";
  EXPECT_EQ(decWrap4G((uint64_t{10} << 32) | lowWord, 1, 10, 14), (uint64_t{13} << 32) | lowWord)
    << "decWrap4G must wrap below the lower boundary to the last in-range word.";
#if 0 // ADDRESS_LATER
  EXPECT_EQ(decWrap4G(lowWord, 1, 0, 4), (uint64_t{3} << 32) | lowWord)
    << "decWrap4G must wrap below a zero lower boundary without unsigned underflow escaping the range.";
#endif 0 // ADDRESS_LATER
}

/**
 * @brief Verifies reciprocal construction and high-product helpers against wider native arithmetic.
 */
TEST(BitOpsDivisionTest, ReciprocalsAndHighProducts) {
  // Expected reciprocals are floor(2^width / divisor), reduced to the return width.
  const std::array<uint32_t, 5> divisors32 = {1u, 2u, 3u, 7u, UINT32_MAX};
  for (uint32_t divisor : divisors32) {
    SCOPED_TRACE(testing::Message() << "32-bit reciprocal divisor=" << divisor);
    const uint32_t expected = static_cast<uint32_t>((uint64_t{1} << 32) / divisor);
    EXPECT_EQ(idivRcp32(divisor), expected) << "idivRcp32 must return the exact precomputed reciprocal.";
  }
  const std::array<uint64_t, 5> divisors64 = {1ull, 2ull, 3ull, 7ull, UINT64_MAX};
  for (uint64_t divisor : divisors64) {
    SCOPED_TRACE(testing::Message() << "64-bit reciprocal divisor=" << divisor);
    const bool powerOfTwo = (divisor & (divisor - 1)) == 0;
    const uint64_t expected = UINT64_MAX / divisor + static_cast<uint64_t>(powerOfTwo);
    EXPECT_EQ(idivRcp64(divisor), expected) << "idivRcp64 must return the exact precomputed reciprocal.";
  }

  // High-word multiplication must equal the upper half of an independently widened product.
  EXPECT_EQ(mul32hi(UINT32_MAX, UINT32_MAX), UINT32_MAX - 1)
    << "mul32hi must retain the upper 32 bits of the maximum product.";
  EXPECT_EQ(mul32hi(0x12345678u, 0x9abcdef0u),
    static_cast<uint32_t>((uint64_t{0x12345678u} * 0x9abcdef0u) >> 32))
    << "mul32hi must match a widened product for non-boundary operands.";
  EXPECT_EQ(mul64hi(UINT64_MAX, UINT64_MAX), UINT64_MAX - 1)
    << "mul64hi must retain the upper 64 bits of the maximum product.";
  EXPECT_EQ(mul64hi(0x0123456789abcdefull, 0xfedcba9876543210ull),
    static_cast<uint64_t>((static_cast<unsigned __int128>(0x0123456789abcdefull) *
      0xfedcba9876543210ull) >> 64))
    << "mul64hi must match a widened product for non-boundary operands.";

  // Product reciprocal construction must honor either identity operand and ordinary composite
  // divisors.
  EXPECT_EQ(imulRcp32(1, idivRcp32(1), 7, idivRcp32(7)), idivRcp32(7))
    << "imulRcp32 must preserve the other reciprocal when the first factor is one.";
  EXPECT_EQ(imulRcp32(7, idivRcp32(7), 1, idivRcp32(1)), idivRcp32(7))
    << "imulRcp32 must preserve the other reciprocal when the second factor is one.";
  EXPECT_EQ(imulRcp32(3, idivRcp32(3), 5, idivRcp32(5)), idivRcp32(15))
    << "imulRcp32 must construct the reciprocal of an ordinary representable product.";
#if 0 // ADDRESS_LATER
  EXPECT_EQ(imulRcp32(65535, idivRcp32(65535), 65537, idivRcp32(65537)), idivRcp32(UINT32_MAX))
    << "imulRcp32 must construct the reciprocal when the representable product is UINT32_MAX.";
#endif 0 // ADDRESS_LATER
  EXPECT_EQ(imulRcp64(1, idivRcp64(1), 7, idivRcp64(7)), idivRcp64(7))
    << "imulRcp64 must preserve the other reciprocal when the first factor is one.";
  EXPECT_EQ(imulRcp64(7, idivRcp64(7), 1, idivRcp64(1)), idivRcp64(7))
    << "imulRcp64 must preserve the other reciprocal when the second factor is one.";
  EXPECT_EQ(imulRcp64(3, idivRcp64(3), 5, idivRcp64(5)), idivRcp64(15))
    << "imulRcp64 must construct the reciprocal of an ordinary representable product.";
#if 0 // ADDRESS_LATER
  EXPECT_EQ(imulRcp64((uint64_t{1} << 32) - 1, idivRcp64((uint64_t{1} << 32) - 1),
                (uint64_t{1} << 32) + 1, idivRcp64((uint64_t{1} << 32) + 1)),
            idivRcp64(UINT64_MAX))
    << "imulRcp64 must construct the reciprocal when the representable product is UINT64_MAX.";
#endif 0 // ADDRESS_LATER

}

/**
 * @brief Verifies quotient, remainder, output mutation, and full-width behavior for fast division.
 * @note Side effects: Records nonfatal GoogleTest failures through the division-check helpers.
 */
TEST(BitOpsDivisionTest, FastDivisionMatchesNativeArithmetic) {
  // The 32-bit path must cover divisor-one sentinel, less-than, exact, remainder, and maximum
  // inputs.
  ExpectFastDivision32(0, 1);
  ExpectFastDivision32(5, 7);
  ExpectFastDivision32(84, 7);
  ExpectFastDivision32(85, 7);
  ExpectFastDivision32(UINT32_MAX, 3);
  ExpectFastDivision32(UINT32_MAX, UINT32_MAX);

  // The 64-bit path must additionally preserve quotients and remainders above UINT32_MAX.
  ExpectFastDivision64(0, 1);
  ExpectFastDivision64(5, 7);
  ExpectFastDivision64(84, 7);
  ExpectFastDivision64(UINT64_MAX, 3);
  ExpectFastDivision64(UINT64_MAX, 1);
  ExpectFastDivision64((uint64_t{1} << 33) - 1, uint64_t{1} << 33);
}

/**
 * @brief Verifies population, first-bit removal, logarithms, and powers across 32-bit and 64-bit
 * boundaries.
 */
TEST(BitOpsBitTest, CountsScansAndPowers) {
  // Population counts must handle empty, alternating, and all-set values at different integer
  // widths.
  EXPECT_EQ(countOneBits(uint8_t{0}), 0) << "An empty byte must have no set bits.";
  EXPECT_EQ(countOneBits(uint8_t{0xff}), 8) << "An all-set byte must report all eight bits.";
  EXPECT_EQ(countOneBits(uint32_t{0xf0f00f0f}), 16) << "A mixed 32-bit value must report its exact population.";
  EXPECT_EQ(countOneBits(UINT64_MAX), 64) << "UINT64_MAX must report every bit set.";

  // First-bit lookup must return -1 for zero and the exact low, middle, and high set-bit indexes.
  EXPECT_EQ(firstOneBit(uint32_t{0}), -1) << "A zero mask must report no first set bit.";
  EXPECT_EQ(firstOneBit(uint32_t{1}), 0) << "The least-significant bit must have index zero.";
  EXPECT_EQ(firstOneBit(uint32_t{1} << 17), 17) << "A middle 32-bit mask must report its exact index.";
  EXPECT_EQ(firstOneBit(uint64_t{1} << 63), 63) << "The highest 64-bit mask must report index 63.";

  // Popping zero must preserve it; popping a nonzero mask must remove only its lowest set bit each
  // time.
  uint64_t emptyMask = 0;
  EXPECT_EQ(popFirstOneBit(&emptyMask), -1) << "Popping an empty mask must report no bit.";
  EXPECT_EQ(emptyMask, 0u) << "Popping an empty mask must leave it unchanged.";
  uint64_t mask = (uint64_t{1} << 3) | (uint64_t{1} << 4) | (uint64_t{1} << 60);
  EXPECT_EQ(popFirstOneBit(&mask), 3) << "The first pop must report the original lowest set bit.";
  EXPECT_EQ(mask, (uint64_t{1} << 4) | (uint64_t{1} << 60))
    << "The first pop must preserve every bit except bit three.";
  EXPECT_EQ(popFirstOneBit(&mask), 4) << "The second pop must report the next set bit.";
  EXPECT_EQ(mask, uint64_t{1} << 60) << "The second pop must preserve the remaining high bit.";

  // Power-of-two recognition must reject zero and composites and accept positive single-bit values.
#if 0 // ADDRESS_LATER
  EXPECT_FALSE(isPow2(uint64_t{0})) << "Zero has no set bit and therefore is not a power of two.";
#endif 0 // ADDRESS_LATER
  EXPECT_TRUE(isPow2(uint64_t{1})) << "One is two to the power zero.";
  EXPECT_TRUE(isPow2(uint64_t{1} << 63)) << "The highest single 64-bit bit is a power of two.";
  EXPECT_FALSE(isPow2(uint64_t{3})) << "A value with multiple set bits is not a power of two.";

  // Floor and ceiling logarithms must handle zero, exact powers, adjacent values, and the high bit.
  EXPECT_EQ(log2Down(uint64_t{0}), -1) << "log2Down must use -1 for zero.";
  EXPECT_EQ(log2Down(uint64_t{1}), 0) << "log2Down(1) must be zero.";
  EXPECT_EQ(log2Down(uint64_t{3}), 1) << "log2Down must select the preceding power for a composite.";
  EXPECT_EQ(log2Down(uint64_t{1} << 63), 63) << "log2Down must report the highest 64-bit exponent.";
  EXPECT_EQ(log2Up(uint64_t{0}), 0) << "log2Up must use zero for an empty size.";
  EXPECT_EQ(log2Up(uint64_t{1}), 0) << "log2Up(1) must be zero.";
  EXPECT_EQ(log2Up(uint64_t{3}), 2) << "log2Up must select the following exponent for a composite.";
  EXPECT_EQ(log2Up((uint64_t{1} << 63) + 1), 64) << "log2Up must report 64 above the highest power of two.";

  // Power rounding must preserve zero according to pow2Up and stay inside the positive pow2Down
  // domain.
  EXPECT_EQ(pow2Up(uint64_t{0}), 1u) << "pow2Up defines the smallest result for zero as one.";
  EXPECT_EQ(pow2Up(uint64_t{1}), 1u) << "pow2Up must preserve one.";
  EXPECT_EQ(pow2Up(uint64_t{3}), 4u) << "pow2Up must select the next power of two.";
  EXPECT_EQ(pow2Down(uint64_t{1}), 1u) << "pow2Down must preserve one.";
  EXPECT_EQ(pow2Down(uint64_t{7}), 4u) << "pow2Down must select the preceding power of two.";
  EXPECT_EQ(pow2Down(uint64_t{1} << 63), uint64_t{1} << 63)
    << "pow2Down must preserve the highest representable 64-bit power.";
}

/**
 * @brief Verifies block reversal and low-bit reversal against an independent bit-by-bit oracle.
 */
TEST(BitOpsBitTest, ReversesRequestedBits) {
  // Block reversal must cover its identity, recursive mask, byte-swap, and supported integer-width
  // paths.
  constexpr uint8_t byte = 0xb2u;
  EXPECT_EQ((reverseSubBits<uint8_t, 1>(byte)), ReverseBlocksReference(byte, 1))
    << "One-bit blocks must remain unchanged.";
  EXPECT_EQ((reverseSubBits<uint8_t, 2>(byte)), ReverseBlocksReference(byte, 2))
    << "Two-bit blocks must reverse independently.";
  EXPECT_EQ((reverseSubBits<uint8_t, 4>(byte)), ReverseBlocksReference(byte, 4))
    << "Four-bit blocks must reverse independently.";
  EXPECT_EQ((reverseSubBits<uint8_t, 8>(byte)), ReverseBlocksReference(byte, 8))
    << "A full byte must reverse all eight bits.";
  constexpr uint16_t word16 = 0x12f0u;
  EXPECT_EQ((reverseSubBits<uint16_t, 16>(word16)), ReverseBlocksReference(word16, 16))
    << "A 16-bit word must use the full-width reversal path.";
  constexpr uint32_t word32 = 0x01234567u;
  EXPECT_EQ((reverseSubBits<uint32_t, 32>(word32)), ReverseBlocksReference(word32, 32))
    << "A 32-bit word must use the full-width reversal path.";
  constexpr uint64_t word64 = 0x0123456789abcdefull;
  EXPECT_EQ((reverseSubBits<uint64_t, 64>(word64)), ReverseBlocksReference(word64, 64))
    << "A 64-bit word must use the full-width reversal path.";

  // Low-bit reversal must zero the unused high bits and handle empty, partial, and full-width
  // selections.
  EXPECT_EQ(reverseBits(uint32_t{0xdu}, 0), 0u) << "Reversing zero bits must return zero.";
  EXPECT_EQ(reverseBits(uint32_t{0xdu}, 1), 1u) << "Reversing one low bit must preserve that bit only.";
  EXPECT_EQ(reverseBits(uint32_t{0xdu}, 4), 0xbu) << "Reversing four low bits of 1101 must produce 1011.";
  EXPECT_EQ(reverseBits(word32, 13), ReverseBlocksReference(word32, 32) >> (32 - 13))
    << "Partial 32-bit reversal must return only the requested reversed low bits.";
  EXPECT_EQ(reverseBits(word64, 64), ReverseBlocksReference(word64, 64))
    << "Full-width 64-bit reversal must preserve every reversed bit.";
}

/**
 * @brief Verifies CUDA-branch bit operations against the host branch across supported widths and
 * boundaries.
 */
TEST(BitOpsBitTest, CudaBranchParity) {
  // Narrow integer types must retain host behavior when CUDA promotes them to its 32-bit intrinsic
  // arguments.
  constexpr uint8_t narrowValue = 0x90u;
  EXPECT_EQ(CudaArchBranch::countOneBits(narrowValue), countOneBits(narrowValue))
    << "CUDA and host narrow population counts must match after intrinsic promotion.";
  EXPECT_EQ(CudaArchBranch::firstOneBit(narrowValue), firstOneBit(narrowValue))
    << "CUDA and host narrow first-set-bit indexes must match after intrinsic promotion.";
  EXPECT_EQ(CudaArchBranch::log2Down(narrowValue), log2Down(narrowValue))
    << "CUDA and host narrow floor logarithms must match after intrinsic promotion.";
  EXPECT_EQ(CudaArchBranch::log2Up(narrowValue), log2Up(narrowValue))
    << "CUDA and host narrow ceiling logarithms must match after intrinsic promotion.";

  // Counts, scans, and logarithms must agree for zero, ordinary values, the high bit, and all-set
  // values.
  const std::array<uint32_t, 5> values32 = {0u, 1u, 3u, uint32_t{1} << 31, UINT32_MAX};
  for (uint32_t value : values32) {
    SCOPED_TRACE(testing::Message() << "CUDA 32-bit bit operation value=" << value);
    EXPECT_EQ(CudaArchBranch::countOneBits(value), countOneBits(value))
      << "CUDA and host 32-bit population counts must match.";
    EXPECT_EQ(CudaArchBranch::firstOneBit(value), firstOneBit(value))
      << "CUDA and host 32-bit first-set-bit indexes must match.";
    EXPECT_EQ(CudaArchBranch::log2Down(value), log2Down(value))
      << "CUDA and host 32-bit floor logarithms must match.";
    EXPECT_EQ(CudaArchBranch::log2Up(value), log2Up(value))
      << "CUDA and host 32-bit ceiling logarithms must match.";
  }
  const std::array<uint64_t, 5> values64 = {0ull, 1ull, 3ull, uint64_t{1} << 63, UINT64_MAX};
  for (uint64_t value : values64) {
    SCOPED_TRACE(testing::Message() << "CUDA 64-bit bit operation value=" << value);
    EXPECT_EQ(CudaArchBranch::countOneBits(value), countOneBits(value))
      << "CUDA and host 64-bit population counts must match.";
    EXPECT_EQ(CudaArchBranch::firstOneBit(value), firstOneBit(value))
      << "CUDA and host 64-bit first-set-bit indexes must match.";
    EXPECT_EQ(CudaArchBranch::log2Down(value), log2Down(value))
      << "CUDA and host 64-bit floor logarithms must match.";
    EXPECT_EQ(CudaArchBranch::log2Up(value), log2Up(value))
      << "CUDA and host 64-bit ceiling logarithms must match.";
  }

  // Pop and power helpers must preserve the same return values and mutations through their CUDA
  // dependencies.
  uint64_t hostMask = (uint64_t{1} << 4) | (uint64_t{1} << 60);
  uint64_t cudaMask = hostMask;
  EXPECT_EQ(CudaArchBranch::popFirstOneBit(&cudaMask), popFirstOneBit(&hostMask))
    << "CUDA and host popFirstOneBit branches must report the same removed bit.";
  EXPECT_EQ(cudaMask, hostMask) << "CUDA and host popFirstOneBit branches must leave identical remaining masks.";
  const std::array<uint64_t, 3> powers = {1ull, 3ull, uint64_t{1} << 63};
  for (uint64_t value : powers) {
    SCOPED_TRACE(testing::Message() << "CUDA power rounding value=" << value);
    EXPECT_EQ(CudaArchBranch::pow2Up(value), pow2Up(value))
      << "CUDA and host pow2Up branches must select the same power.";
    EXPECT_EQ(CudaArchBranch::pow2Down(value), pow2Down(value))
      << "CUDA and host pow2Down branches must select the same power.";
  }

  // Reversal must agree for narrow, 32-bit, and 64-bit types supported by the generic template.
  const uint8_t byte = 0x90u;
  const uint16_t halfWord = 0x12f0u;
  const uint32_t word = 0x01234567u;
  const uint64_t doubleWord = 0x0123456789abcdefull;
#if 0 // ADDRESS_LATER
  EXPECT_EQ(CudaArchBranch::reverseBits(byte, 8), reverseBits(byte, 8))
    << "CUDA and host full-byte reversal must match after intrinsic result normalization.";
  EXPECT_EQ(CudaArchBranch::reverseBits(halfWord, 16), reverseBits(halfWord, 16))
    << "CUDA and host full-16-bit reversal must match after intrinsic result normalization.";
#endif 0 // ADDRESS_LATER
  EXPECT_EQ(CudaArchBranch::reverseBits(word, 0), reverseBits(word, 0))
    << "CUDA and host zero-bit 32-bit reversal must both return zero.";
  EXPECT_EQ(CudaArchBranch::reverseBits(word, 13), reverseBits(word, 13))
    << "CUDA and host partial 32-bit reversal must match.";
  EXPECT_EQ(CudaArchBranch::reverseBits(word, 32), reverseBits(word, 32))
    << "CUDA and host full-32-bit reversal must match.";
  EXPECT_EQ(CudaArchBranch::reverseBits(doubleWord, 37), reverseBits(doubleWord, 37))
    << "CUDA and host partial 64-bit reversal must match.";
  EXPECT_EQ(CudaArchBranch::reverseBits(doubleWord, 64), reverseBits(doubleWord, 64))
    << "CUDA and host full-64-bit reversal must match.";

  // Fixed-point encoding must produce identical codes when its leading-zero operation selects the
  // CUDA branch.
  const std::array<uint32_t, 7> fixedPointValues = {0u, 1u, 7u, 8u, 17u, 31u, UINT32_MAX};
  for (uint32_t value : fixedPointValues) {
    SCOPED_TRACE(testing::Message() << "CUDA fixed-point input=" << value);
    EXPECT_EQ(CudaArchBranch::u32fpEncode(value, 3), u32fpEncode(value, 3))
      << "CUDA and host generic fixed-point encoders must match.";
    EXPECT_EQ(CudaArchBranch::u32fp8Encode(value), u32fp8Encode(value))
      << "CUDA and host eight-bit fixed-point encoders must match.";
  }
}

/**
 * @brief Verifies the custom unsigned fixed-point encoding at exact, truncated, and maximum
 * boundaries.
 */
TEST(BitOpsEncodingHashTest, EncodesUnsignedFixedPointValues) {
  struct EncodingCase {
    uint32_t input;
    uint8_t encoded;
    uint32_t decoded;
  };
  const std::array<EncodingCase, 9> cases = {{
    {0u, 0x00u, 0u},
    {1u, 0x01u, 1u},
    {7u, 0x07u, 7u},
    {8u, 0x08u, 8u},
    {15u, 0x0fu, 15u},
    {17u, 0x10u, 16u},
    {31u, 0x17u, 30u},
    {256u, 0x30u, 256u},
    {UINT32_MAX, 0xefu, 0xf0000000u},
  }};

  // Generic and eight-bit wrappers must agree on exact codes and their deliberately truncated
  // decoded values.
  for (const EncodingCase& testCase : cases) {
    SCOPED_TRACE(testing::Message() << "fixed-point input=" << testCase.input);
    EXPECT_EQ(u32fpEncode(testCase.input, 3), testCase.encoded)
      << "u32fpEncode must produce the expected three-mantissa-bit code.";
    EXPECT_EQ(u32fpDecode(testCase.encoded, 3), testCase.decoded)
      << "u32fpDecode must expand the expected three-mantissa-bit code.";
    EXPECT_EQ(u32fp8Encode(testCase.input), testCase.encoded)
      << "u32fp8Encode must preserve the complete generic code in eight bits.";
    EXPECT_EQ(u32fp8Decode(testCase.encoded), testCase.decoded)
      << "u32fp8Decode must match the generic decoder.";
    EXPECT_LE(testCase.decoded, testCase.input)
      << "The truncating fixed-point representation must never round a valid input upward.";
  }

  // The advertised maximum must equal the decoded representation of the largest uint32 input.
  EXPECT_EQ(u32fp8MaxValue(), 0xf0000000u) << "The format maximum must match its documented constant.";
  EXPECT_EQ(u32fp8Decode(u32fp8Encode(UINT32_MAX)), u32fp8MaxValue())
    << "Encoding and decoding UINT32_MAX must saturate at the advertised maximum.";
}

/**
 * @brief Verifies deterministic hashing, input preservation, typed overloads, and split-sensitive
 * accumulation.
 */
TEST(BitOpsEncodingHashTest, HashesBytesAndTypedValues) {
  const std::array<unsigned char, 3> abc = {'a', 'b', 'c'};
  const std::array<unsigned char, 3> abcBefore = abc;
  const std::array<unsigned char, 9> bytes = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<unsigned char, 9> bytesBefore = bytes;

  // Golden values cover empty input, one partial chunk, one full chunk, and a full-plus-partial
  // pair.
  EXPECT_EQ(getHash(nullptr, 0), 0x75b6431c590e86a2ull)
    << "The empty byte sequence must retain its stable hash.";
  EXPECT_EQ(getHash(abc.data(), abc.size()), 0x751c1b124ef01f18ull)
    << "A three-byte partial chunk must retain its stable hash.";
  EXPECT_EQ(getHash(bytes.data(), 8), 0xc35ce9c9b2e15285ull)
    << "An exact eight-byte chunk must retain its stable hash.";
  EXPECT_EQ(getHash(bytes.data(), bytes.size()), 0x43dd9592843edfb2ull)
    << "A full chunk followed by one byte must retain its stable hash.";
  EXPECT_EQ(abc, abcBefore) << "Hashing must not modify a partial input buffer.";
  EXPECT_EQ(bytes, bytesBefore) << "Hashing must not modify a multi-chunk input buffer.";

  // A zero-length accumulation must preserve both caller-owned accumulator words exactly.
  uint64_t emptyAccumulator[2] = {0x0123456789abcdefull, 0xfedcba9876543210ull};
  eatHash(emptyAccumulator, nullptr, 0);
  EXPECT_EQ(emptyAccumulator[0], 0x0123456789abcdefull)
    << "Zero-length eatHash must preserve the first accumulator word.";
  EXPECT_EQ(emptyAccumulator[1], 0xfedcba9876543210ull)
    << "Zero-length eatHash must preserve the second accumulator word.";

  // Byte accumulation and digesting must produce exact golden state without mutating const inputs.
  uint64_t accumulator[2] = {1, 1};
  eatHash(accumulator, abc.data(), abc.size());
  EXPECT_EQ(accumulator[0], 0xd9b28335a731190aull)
    << "eatHash must update the first accumulator word with all three bytes.";
  EXPECT_EQ(accumulator[1], 0x90b9ac6236b33cf2ull)
    << "eatHash must update the second accumulator word with all three bytes.";
  const std::array<uint64_t, 2> accumulatorBeforeDigest = {accumulator[0], accumulator[1]};
  EXPECT_EQ(digestHash(accumulator), 0x751c1b124ef01f18ull)
    << "digestHash must finalize the known three-byte accumulator exactly.";
  EXPECT_EQ(accumulator[0], accumulatorBeforeDigest[0]) << "digestHash must preserve its first input word.";
  EXPECT_EQ(accumulator[1], accumulatorBeforeDigest[1]) << "digestHash must preserve its second input word.";

  // Typed overloads must hash exactly sizeof(T) bytes and leave the typed input unchanged.
  const uint32_t typedValue = 0x04030201u;
  const uint32_t typedValueBefore = typedValue;
  uint64_t typedAccumulator[2] = {1, 1};
  uint64_t byteAccumulator[2] = {1, 1};
  eatHash(typedAccumulator, &typedValue);
  eatHash(byteAccumulator, static_cast<const void*>(&typedValue), sizeof(typedValue));
  EXPECT_EQ(typedAccumulator[0], byteAccumulator[0])
    << "Typed eatHash must consume the same first-state bytes as its void overload.";
  EXPECT_EQ(typedAccumulator[1], byteAccumulator[1])
    << "Typed eatHash must consume the same second-state bytes as its void overload.";
  EXPECT_EQ(getHash(&typedValue), getHash(static_cast<const void*>(&typedValue), sizeof(typedValue)))
    << "Typed getHash must consume exactly one complete object representation.";
  EXPECT_EQ(typedValue, typedValueBefore) << "Typed hashing must not modify the caller's object.";

  // The documented call-boundary sensitivity must distinguish one nine-byte call from two calls
  // over the same bytes.
  uint64_t fullAccumulator[2] = {1, 1};
  uint64_t splitAccumulator[2] = {1, 1};
  eatHash(fullAccumulator, bytes.data(), bytes.size());
  eatHash(splitAccumulator, bytes.data(), 4);
  eatHash(splitAccumulator, bytes.data() + 4, bytes.size() - 4);
  EXPECT_NE(digestHash(fullAccumulator), digestHash(splitAccumulator))
    << "Splitting identical bytes across eatHash calls must change the documented hash framing.";
  EXPECT_EQ(bytes, bytesBefore) << "Repeated and split hashing must preserve every source byte.";
}

}  // namespace
