/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "bitops.h"
#include "debug.h"
#include "nccl.h"
#include "os.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

/** @brief Scripts and records standalone host-allocation dependencies. */
struct HostAllocationState {
  bool failAllocation = false;
  bool failAlignedAllocation = false;
  size_t pageSize = 4096;
  int allocationCalls = 0;
  int releaseCalls = 0;
  int pageSizeCalls = 0;
  int alignedAllocationCalls = 0;
  size_t allocatedBytes = 0;
  size_t requestedAlignment = 0;
  size_t requestedAlignedBytes = 0;
};

HostAllocationState hostAllocationState;

/**
 * @brief Resets deterministic host-allocation dependencies.
 * @note Side effects: Replaces hostAllocationState.
 */
void ResetHostAllocationForTest() {
  hostAllocationState = HostAllocationState{};
}

/**
 * @brief Allocates ordinary heap storage or supplies a deterministic failure.
 * @param[in] bytes Requested allocation size.
 * @return Caller-owned heap storage releasable with ReleaseHostForTest on success; otherwise
 * NULL.
 * @note Side effects: Records the call and requested size and may allocate storage.
 */
void* AllocateHostForTest(size_t bytes) {
  ++hostAllocationState.allocationCalls;
  hostAllocationState.allocatedBytes = bytes;
  return hostAllocationState.failAllocation ? nullptr : std::malloc(bytes);
}

/**
 * @brief Releases one allocation made by the host-allocation seam.
 * @param[in] pointer NULL or a pointer returned by AllocateHostForTest.
 * @note Side effects: Records the call and releases pointer.
 */
void ReleaseHostForTest(void* pointer) {
  ++hostAllocationState.releaseCalls;
  std::free(pointer);
}

/**
 * @brief Returns the configured operating-system page size.
 * @return hostAllocationState.pageSize.
 * @note Side effects: Increments pageSizeCalls.
 */
size_t PageSizeForHostTest() {
  ++hostAllocationState.pageSizeCalls;
  return hostAllocationState.pageSize;
}

/**
 * @brief Allocates aligned storage or supplies a deterministic allocation error.
 * @param[out] pointer Valid non-NULL storage receiving caller-owned aligned memory on success
 * and preserved on error.
 * @param[in] alignment Power of two and multiple of sizeof(void*), as required by posix_memalign.
 * @param[in] bytes Requested allocation size.
 * @return Zero on success, EINVAL when failure is injected, or another posix_memalign error.
 * @note Side effects: Records alignment and bytes and may allocate storage through pointer.
 */
int AllocateAlignedForHostTest(void** pointer, size_t alignment, size_t bytes) {
  ++hostAllocationState.alignedAllocationCalls;
  hostAllocationState.requestedAlignment = alignment;
  hostAllocationState.requestedAlignedBytes = bytes;
  if (hostAllocationState.failAlignedAllocation) return EINVAL;
  return ::posix_memalign(pointer, alignment, bytes);
}

/** @brief Provides a nontrivial 32-byte-aligned allocation type. */
struct AlignedValueForTest {
  alignas(32) unsigned char bytes[3];
};

}  // namespace

/**
 * @brief Instantiates allocation helpers with heap, page-size, and alignment calls redirected
 * to test seams.
 */
#define malloc AllocateHostForTest
#define free ReleaseHostForTest
#define ncclOsGetPageSize PageSizeForHostTest
#define posix_memalign AllocateAlignedForHostTest
#include "standalone_host_alloc.h"
#undef posix_memalign
#undef ncclOsGetPageSize
#undef free
#undef malloc

/**
 * @brief Verifies that typed sizes use complete objects while void arithmetic remains byte-based.
 */
TEST(StandaloneHostAllocationTest, ReportsElementSizes) {
  SCOPED_TRACE("Host allocation sizing must match sizeof for objects and one byte for void storage.");

  EXPECT_EQ(occlSizeOfT<uint32_t>(), sizeof(uint32_t))
      << "A scalar allocation element must use its complete object size";
  EXPECT_EQ(occlSizeOfT<AlignedValueForTest>(), sizeof(AlignedValueForTest))
      << "An over-aligned object must include its padding in the allocation element size";
  EXPECT_EQ(occlSizeOfT<void>(), 1u)
      << "Void storage must advance in single-byte allocation units";
}

/**
 * @brief Verifies direct host calloc handles zero, success, and allocation failure with exact
 * output ownership.
 */
TEST(StandaloneHostAllocationTest, AllocatesZeroedTypedStorage) {
  SCOPED_TRACE("Typed host calloc must zero successful storage and preserve outputs when allocation fails.");
  ResetHostAllocationForTest();
  int* values = reinterpret_cast<int*>(uintptr_t{1});
  const std::string location = "allocation-site";

  EXPECT_EQ(occlHostCallocDebug(&values, 0, location.c_str(), 17), ncclSuccess)
      << "A zero-element allocation must succeed";
  EXPECT_EQ(values, nullptr) << "A zero-element allocation must clear its output pointer";
  EXPECT_EQ(hostAllocationState.allocationCalls, 0)
      << "A zero-element allocation must not call malloc";

  ASSERT_EQ(occlHostCallocDebug(&values, 4, location.c_str(), 17), ncclSuccess)
      << "A normal typed allocation must succeed";
  ASSERT_NE(values, nullptr) << "A successful allocation must publish storage";
  EXPECT_EQ(hostAllocationState.allocatedBytes, 4 * sizeof(int))
      << "Typed allocation must request the complete array size";
  const std::array<int, 4> actualValues = {values[0], values[1], values[2], values[3]};
  const std::array<int, 4> expectedValues = {0, 0, 0, 0};
  EXPECT_EQ(actualValues, expectedValues)
      << "Every allocated element must be zero initialized";
  EXPECT_EQ(location, "allocation-site") << "Allocation diagnostics must not modify their source string";
  std::free(values);

  ResetHostAllocationForTest();
  hostAllocationState.failAllocation = true;
  values = reinterpret_cast<int*>(uintptr_t{1});
  EXPECT_EQ(occlHostCallocDebug(&values, 4, location.c_str(), 17), ncclSystemError)
      << "Malloc failure must return ncclSystemError";
  EXPECT_EQ(values, reinterpret_cast<int*>(uintptr_t{1}))
      << "Malloc failure must preserve the caller's output pointer";
}

/**
 * @brief Verifies direct host realloc covers validation, no-op, successful growth, and allocation
 * failure.
 */
TEST(StandaloneHostAllocationTest, ReallocatesTypedStorage) {
  SCOPED_TRACE("Typed realloc must preserve old storage on errors and zero only the newly grown tail.");
  ResetHostAllocationForTest();
  int* values = static_cast<int*>(std::malloc(2 * sizeof(int)));
  ASSERT_NE(values, nullptr) << "The test fixture must allocate original storage";
  values[0] = 7;
  values[1] = 9;
  int* original = values;

  EXPECT_EQ(occlHostRealloc(&values, 2, 1), ncclInternalError)
      << "Shrinking through the grow-only helper must return ncclInternalError";
  EXPECT_EQ(values, original) << "Rejected shrinkage must preserve the original pointer";
  EXPECT_EQ(values[0], 7) << "Rejected shrinkage must preserve original data";
  EXPECT_EQ(occlHostRealloc(&values, 2, 2), ncclSuccess)
      << "An unchanged element count must be a successful no-op";
  EXPECT_EQ(values, original) << "A no-op realloc must preserve pointer identity";
  EXPECT_EQ(hostAllocationState.allocationCalls, 0) << "A no-op realloc must not allocate";

  ASSERT_EQ(occlHostRealloc(&values, 2, 4), ncclSuccess)
      << "Growing valid storage must succeed";
  ASSERT_NE(values, nullptr) << "Successful growth must publish replacement storage";
  EXPECT_EQ(values[0], 7) << "Growth must preserve the first existing element";
  EXPECT_EQ(values[1], 9) << "Growth must preserve the final existing element";
  EXPECT_EQ(values[2], 0) << "Growth must zero the first new element";
  EXPECT_EQ(values[3], 0) << "Growth must zero the final new element";
  EXPECT_EQ(hostAllocationState.releaseCalls, 1) << "Successful growth must release old storage once";
  std::free(values);

  ResetHostAllocationForTest();
  values = static_cast<int*>(std::malloc(2 * sizeof(int)));
  ASSERT_NE(values, nullptr) << "The failure fixture must allocate original storage";
  values[0] = 11;
  values[1] = 13;
  original = values;
  hostAllocationState.failAllocation = true;
  EXPECT_EQ(occlHostRealloc(&values, 2, 4), ncclSystemError)
      << "Replacement allocation failure must return ncclSystemError";
  EXPECT_EQ(values, original) << "Failed growth must preserve the original pointer";
  EXPECT_EQ(values[0], 11) << "Failed growth must preserve original contents";
  EXPECT_EQ(values[1], 13) << "Failed growth must preserve the complete original range";
  EXPECT_EQ(hostAllocationState.releaseCalls, 0) << "Failed growth must not release live original storage";
  std::free(values);

  values = nullptr;
  EXPECT_EQ(occlHostRealloc(&values, 1, 2), ncclInternalError)
      << "A NULL pointer claiming existing elements must return ncclInternalError";
  EXPECT_EQ(values, nullptr) << "Invalid NULL input must remain NULL";

  ResetHostAllocationForTest();
  ASSERT_EQ(occlHostRealloc(&values, 0, 2), ncclSuccess)
      << "Growing an empty allocation must succeed";
  ASSERT_NE(values, nullptr) << "Growth from an empty allocation must publish storage";
  EXPECT_EQ(hostAllocationState.allocatedBytes, 2 * sizeof(int))
      << "Growth from empty storage must allocate the requested typed extent";
  EXPECT_EQ(values[0], 0) << "Growth from empty storage must zero the first element";
  EXPECT_EQ(values[1], 0) << "Growth from empty storage must zero the final element";
  EXPECT_EQ(hostAllocationState.releaseCalls, 0)
      << "Growth from a NULL pointer must not attempt to release old storage";
  std::free(values);
}

/**
 * @brief Verifies direct page allocation handles zero size, page-query failure, allocator failure,
 * and success.
 */
TEST(StandaloneHostAllocationTest, AllocatesPageAlignedStorage) {
  SCOPED_TRACE("Page allocation must align and zero successful output while preserving output on failures.");
  ResetHostAllocationForTest();
  void* storage = reinterpret_cast<void*>(uintptr_t{1});
  const std::string location = "page-site";

  EXPECT_EQ(occlIbPageAllocDebug(&storage, 0, location.c_str(), 23), ncclSuccess)
      << "A zero-byte page allocation must succeed";
  EXPECT_EQ(storage, nullptr) << "A zero-byte allocation must clear its output pointer";
  EXPECT_EQ(hostAllocationState.pageSizeCalls, 0) << "Zero bytes require no page-size query";

  hostAllocationState.pageSize = static_cast<size_t>(-1);
  storage = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(occlIbPageAllocDebug(&storage, 4097, location.c_str(), 23), ncclSystemError)
      << "Page-size query failure must return ncclSystemError";
  EXPECT_EQ(storage, reinterpret_cast<void*>(uintptr_t{1}))
      << "Page-size failure must preserve the output pointer";
  EXPECT_EQ(hostAllocationState.alignedAllocationCalls, 0)
      << "Page-size failure must not call the aligned allocator";

  ResetHostAllocationForTest();
  hostAllocationState.failAlignedAllocation = true;
  storage = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(occlIbPageAllocDebug(&storage, 4097, location.c_str(), 23), ncclSystemError)
      << "Aligned allocator failure must return ncclSystemError";
  EXPECT_EQ(storage, reinterpret_cast<void*>(uintptr_t{1}))
      << "Aligned allocator failure must preserve the output pointer";

  ResetHostAllocationForTest();
  ASSERT_EQ(occlIbPageAllocDebug(&storage, 4097, location.c_str(), 23), ncclSuccess)
      << "A normal page-aligned allocation must succeed";
  ASSERT_NE(storage, nullptr) << "Successful page allocation must publish storage";
  EXPECT_EQ(hostAllocationState.requestedAlignment, 4096u)
      << "The operating-system page size must be used as alignment";
  EXPECT_EQ(hostAllocationState.requestedAlignedBytes, 8192u)
      << "The allocation size must round up to a complete page";
  EXPECT_EQ(reinterpret_cast<uintptr_t>(storage) % 4096u, 0u)
      << "Published storage must satisfy page alignment";
  const unsigned char* bytes = static_cast<const unsigned char*>(storage);
  EXPECT_TRUE(std::all_of(bytes, bytes + 4097, [](unsigned char value) { return value == 0; }))
      << "Every caller-requested byte must be zero initialized";
  EXPECT_EQ(location, "page-site") << "Page-allocation diagnostics must preserve their input string";
  std::free(storage);
}
