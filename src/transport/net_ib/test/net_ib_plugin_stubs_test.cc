/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

/**
 * @brief Compiles the standalone stubs directly so fallback behavior is tested without
 * full plugin linkage.
 */
#include "../../../plugin/net/net_ib_plugin_stubs.cc"  // NOLINT(build/include)

namespace {

/**
 * @brief Restores one process environment variable after a test scenario.
 */
class ScopedEnvironmentForStubTest {
 public:
  /**
   * @brief Captures the current value of name.
   * @param[in] name Environment-variable name retained for the fixture lifetime.
   * @note Side effects: Reads and stores the current process environment value.
   */
  explicit ScopedEnvironmentForStubTest(const char* name) : name_(name) {
    const char* value = std::getenv(name_.c_str());
    if (value != nullptr) {
      hadValue_ = true;
      value_ = value;
    }
  }

  /**
   * @brief Restores the captured environment value.
   * @note Side effects: Sets or removes the retained process environment variable.
   */
  ~ScopedEnvironmentForStubTest() {
    if (hadValue_) {
      setenv(name_.c_str(), value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string value_;
  bool hadValue_ = false;
};

}  // namespace

/**
 * @brief Verifies standalone plugin stubs expose fixed initialization and stack-size policies.
 */
TEST(NetIbPluginStubsTest, ReturnsStandaloneDefaults) {
  SCOPED_TRACE("Standalone environment stubs must remain successful, uninitialized, and stack-size enabled.");
  EXPECT_EQ(ncclInitEnv(), ncclSuccess)
      << "Standalone environment initialization must be a successful no-op";
  EXPECT_FALSE(ncclEnvPluginInitialized())
      << "The standalone plugin must not claim that the core environment plugin is initialized";
  EXPECT_EQ(ncclParamSetCpuStackSize(), 1)
      << "The standalone plugin must retain the production default that enables CPU stack sizing";
}

/**
 * @brief Verifies standalone environment lookup reports missing and present values without
 * mutation.
 */
TEST(NetIbPluginStubsTest, ReadsProcessEnvironment) {
  SCOPED_TRACE("Standalone environment lookup must delegate to the process environment exactly.");
  const std::string name = "OCCL_NET_IB_STUB_TEST_VALUE";
  ScopedEnvironmentForStubTest restore(name.c_str());
  unsetenv(name.c_str());
  EXPECT_EQ(ncclEnvPluginGetEnv(name.c_str()), nullptr)
      << "A missing process variable must produce a NULL lookup result";

  ASSERT_EQ(setenv(name.c_str(), "configured-value", 1), 0)
      << "The test fixture must install its process environment value";
  const char* value = ncclEnvPluginGetEnv(name.c_str());
  ASSERT_NE(value, nullptr) << "A present process variable must produce a non-NULL result";
  EXPECT_STREQ(value, "configured-value")
      << "Environment lookup must return the complete configured value";
  EXPECT_EQ(name, "OCCL_NET_IB_STUB_TEST_VALUE")
      << "Environment lookup must not modify the caller-owned name";
}
