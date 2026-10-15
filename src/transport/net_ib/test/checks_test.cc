/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "checks.h"

#include <gtest/gtest.h>

/**
 * @brief Verifies deterministic host-only GPU-runtime compatibility behavior.
 */
TEST(ChecksGpuRuntimeDisabledTest, ReportsAndClearsErrors) {
  SCOPED_TRACE(
      "The disabled runtime must expose stable text, report no pending error, and return its input.");
  constexpr cudaError_t testError = 17;

  EXPECT_STREQ(occlGpuRuntimeDisabledErrorString(cudaSuccess), "GPU runtime disabled")
      << "The disabled-runtime message must explain why no GPU error text is available";
  EXPECT_STREQ(occlGpuRuntimeDisabledErrorString(testError), "GPU runtime disabled")
      << "The disabled-runtime message must be stable for non-success values";
  EXPECT_EQ(occlGpuRuntimeDisabledGetLastError(), cudaSuccess)
      << "A disabled runtime cannot retain a pending GPU error";
  EXPECT_EQ(cuda_clear(cudaSuccess), cudaSuccess)
      << "Clearing success must preserve success";
  EXPECT_EQ(cuda_clear(testError), testError)
      << "Clearing a GPU error must return the original status to its caller";
  EXPECT_EQ(testError, 17)
      << "Error reporting and clearing must not modify the caller-owned value";
}
