/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "mlx5/mlx5dvsymbols.h"
#include "mlx5/mlx5dvwrap.h"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

/**
 * @brief Configures fake provider results and records every wrapper-to-provider argument
 * and output interaction.
 */
struct Mlx5ProviderState {
  ncclResult_t buildResult = ncclSuccess;
  int buildCalls = 0;
  struct ncclMlx5dvSymbols* buildOutput = nullptr;
  bool supportResult = false;
  int supportCalls = 0;
  struct ibv_device* supportDevice = nullptr;
  int pathResult = 0;
  int pathCalls = 0;
  struct ibv_context* pathContext = nullptr;
  char* pathBuffer = nullptr;
  size_t pathLength = 0;
  std::string pathOutput = "/mlx5-data-direct";
  struct ibv_mr* registrationResult = nullptr;
  int registrationErrno = 0;
  int registrationCalls = 0;
  struct ibv_pd* registrationDomain = nullptr;
  uint64_t registrationOffset = 0;
  size_t registrationLength = 0;
  uint64_t registrationIova = 0;
  int registrationDescriptor = -1;
  int registrationAccess = 0;
  int registrationMlx5Access = 0;
};

Mlx5ProviderState mlx5ProviderState;

}  // namespace

/**
 * @brief Implements the symbol builder used by the one-time initialization test.
 * @param[in] symbols Valid non-NULL global symbol-table address recorded exactly.
 * @return mlx5ProviderState.buildResult.
 * @note Side effects: Increments buildCalls and replaces buildOutput.
 */
ncclResult_t BuildMlx5DvSymbolsForTest(struct ncclMlx5dvSymbols* symbols) {
  ++mlx5ProviderState.buildCalls;
  mlx5ProviderState.buildOutput = symbols;
  return mlx5ProviderState.buildResult;
}

/**
 * @brief Compiles the wrapper with a recording builder for direct testing of its
 * one-time initialization.
 */
#define buildMlx5dvSymbols BuildMlx5DvSymbolsForTest
#include "../../../misc/mlx5dvwrap.cc"  // NOLINT(build/include)
#undef buildMlx5dvSymbols
#undef MLX5DV_PTR_CHECK_ERRNO
#undef CHECK_NOT_NULL

namespace {

/**
 * @brief Resets fake-provider behavior, call records, and the production symbol table.
 * @note Side effects: Replaces mlx5ProviderState and mlx5dvSymbols.
 */
void ResetMlx5ProviderForTest() {
  mlx5ProviderState = Mlx5ProviderState{};
  mlx5dvSymbols = {};
}

/**
 * @brief Implements deterministic mlx5 device-support detection.
 * @param[in] device Any device pointer recorded exactly.
 * @return mlx5ProviderState.supportResult.
 * @note Side effects: Increments supportCalls and replaces supportDevice.
 */
bool DeviceIsSupportedForTest(struct ibv_device* device) {
  ++mlx5ProviderState.supportCalls;
  mlx5ProviderState.supportDevice = device;
  return mlx5ProviderState.supportResult;
}

/**
 * @brief Implements deterministic Data Direct sysfs-path discovery.
 * @param[in] context Any context pointer recorded exactly.
 * @param[out] buffer NULL when @p length is zero, or storage with @p length bytes; receives
 * pathOutput only on success with sufficient capacity.
 * @param[in] length Buffer capacity, including space for a null terminator; zero permits NULL.
 * @return mlx5ProviderState.pathResult.
 * @note Side effects: Records all arguments and may write @p buffer.
 */
int GetDataDirectPathForTest(struct ibv_context* context, char* buffer, size_t length) {
  ++mlx5ProviderState.pathCalls;
  mlx5ProviderState.pathContext = context;
  mlx5ProviderState.pathBuffer = buffer;
  mlx5ProviderState.pathLength = length;
  if (mlx5ProviderState.pathResult == 0 && mlx5ProviderState.pathOutput.size() + 1 <= length) {
    std::memcpy(buffer, mlx5ProviderState.pathOutput.c_str(), mlx5ProviderState.pathOutput.size() + 1);
  }
  return mlx5ProviderState.pathResult;
}

/**
 * @brief Implements deterministic mlx5 DMA-BUF registration.
 * @param[in] domain Any protection-domain pointer recorded exactly.
 * @param[in] offset DMA-BUF offset in [0, UINT64_MAX] recorded exactly.
 * @param[in] length Registration length in [0, SIZE_MAX] recorded exactly.
 * @param[in] iova I/O virtual address in [0, UINT64_MAX] recorded exactly.
 * @param[in] descriptor Any representable DMA-BUF descriptor recorded exactly.
 * @param[in] access Any representable verbs access mask recorded exactly.
 * @param[in] mlx5Access Any representable mlx5 access mask recorded exactly.
 * @return mlx5ProviderState.registrationResult.
 * @note Side effects: Records every argument, increments registrationCalls, and sets errno to
 * registrationErrno.
 */
struct ibv_mr* RegisterDmaBufForTest(struct ibv_pd* domain, uint64_t offset, size_t length, uint64_t iova,
  int descriptor, int access, int mlx5Access) {
  ++mlx5ProviderState.registrationCalls;
  mlx5ProviderState.registrationDomain = domain;
  mlx5ProviderState.registrationOffset = offset;
  mlx5ProviderState.registrationLength = length;
  mlx5ProviderState.registrationIova = iova;
  mlx5ProviderState.registrationDescriptor = descriptor;
  mlx5ProviderState.registrationAccess = access;
  mlx5ProviderState.registrationMlx5Access = mlx5Access;
  errno = mlx5ProviderState.registrationErrno;
  return mlx5ProviderState.registrationResult;
}

/**
 * @brief Checks the exact arguments of one DMA-BUF provider invocation.
 * @param[in] domain Expected protection-domain pointer.
 * @param[in] offset Expected DMA-BUF offset in [0, UINT64_MAX].
 * @param[in] length Expected registration length in [0, SIZE_MAX].
 * @param[in] iova Expected I/O virtual address in [0, UINT64_MAX].
 * @param[in] descriptor Expected representable descriptor.
 * @param[in] access Expected representable verbs access mask.
 * @param[in] mlx5Access Expected representable mlx5 access mask.
 * @param[in] reason Valid non-NULL explanation shown when an assertion fails.
 * @note Side effects: Records nonfatal GoogleTest failures for incorrect call data.
 */
void ExpectRegistrationCallForTest(struct ibv_pd* domain, uint64_t offset, size_t length, uint64_t iova,
  int descriptor, int access, int mlx5Access, const char* reason) {
  SCOPED_TRACE(reason);
  EXPECT_EQ(mlx5ProviderState.registrationCalls, 1) << "The provider must be called exactly once.";
  EXPECT_EQ(mlx5ProviderState.registrationDomain, domain) << "The protection domain must be forwarded exactly.";
  EXPECT_EQ(mlx5ProviderState.registrationOffset, offset) << "The DMA-BUF offset must be forwarded exactly.";
  EXPECT_EQ(mlx5ProviderState.registrationLength, length) << "The registration length must be forwarded exactly.";
  EXPECT_EQ(mlx5ProviderState.registrationIova, iova) << "The I/O virtual address must be forwarded exactly.";
  EXPECT_EQ(mlx5ProviderState.registrationDescriptor, descriptor) << "The descriptor must be forwarded exactly.";
  EXPECT_EQ(mlx5ProviderState.registrationAccess, access) << "The verbs access mask must be forwarded exactly.";
  EXPECT_EQ(mlx5ProviderState.registrationMlx5Access, mlx5Access)
    << "The mlx5 access mask must be forwarded exactly.";
}

/**
 * @brief Verifies successful one-time initialization inside an isolated process.
 * @return 0 on success, 1 for a wrong result, 2 for a wrong call count, or 3 for a wrong output
 * pointer.
 * @note Side effects: Resets global test state, invokes wrap_mlx5dv_symbols twice, and may write
 * diagnostics to stderr.
 */
int VerifySuccessfulInitializationForTest() {
  ResetMlx5ProviderForTest();
  mlx5ProviderState.buildResult = ncclSuccess;
  if (wrap_mlx5dv_symbols() != ncclSuccess) {
    std::fprintf(stderr, "Successful symbol construction must return ncclSuccess.\n");
    return 1;
  }
  mlx5ProviderState.buildResult = ncclSystemError;
  if (wrap_mlx5dv_symbols() != ncclSuccess) {
    std::fprintf(stderr, "The successful initialization result must be cached.\n");
    return 1;
  }
  if (mlx5ProviderState.buildCalls != 1) {
    std::fprintf(stderr, "The successful symbol builder must be called exactly once.\n");
    return 2;
  }
  if (mlx5ProviderState.buildOutput != &mlx5dvSymbols) {
    std::fprintf(stderr, "The builder must receive the global mlx5 symbol table.\n");
    return 3;
  }
  return 0;
}

/** @brief Provides reset fake symbols and provider state for mlx5 direct-verbs wrappers. */
class Mlx5DvWrapperTest : public ::testing::Test {
 protected:
  /**
   * @brief Restores fake-provider state and removes every configured symbol before each test.
   * @note Side effects: Replaces mlx5ProviderState and mlx5dvSymbols.
   */
  void SetUp() override { ResetMlx5ProviderForTest(); }
};

/**
 * @brief Verifies successful and failed symbol initialization and one-time result caching.
 * @note Side effects: Runs one isolated child and initializes the parent process symbol wrapper
 * once.
 */
TEST_F(Mlx5DvWrapperTest, InitializesSymbolsOnce) {
  SCOPED_TRACE("Symbol initialization must return and cache the builder result without translating it.");

  EXPECT_EXIT({ _exit(VerifySuccessfulInitializationForTest()); }, ::testing::ExitedWithCode(0), "")
    << "Successful initialization must build the global table once and cache ncclSuccess.";

  mlx5ProviderState.buildResult = ncclSystemError;
  EXPECT_EQ(wrap_mlx5dv_symbols(), ncclSystemError)
    << "A symbol-builder failure must be returned unchanged.";
  mlx5ProviderState.buildResult = ncclSuccess;
  EXPECT_EQ(wrap_mlx5dv_symbols(), ncclSystemError)
    << "A completed failed initialization must retain its original result.";
  EXPECT_EQ(mlx5ProviderState.buildCalls, 1) << "The failed symbol builder must be called exactly once.";
  EXPECT_EQ(mlx5ProviderState.buildOutput, &mlx5dvSymbols)
    << "The builder must receive the global mlx5 symbol table.";
}

/**
 * @brief Verifies unavailable, unsupported, and supported device-detection results.
 * @note Side effects: Calls the fake support provider twice and records its input.
 */
TEST_F(Mlx5DvWrapperTest, ReportsDeviceSupport) {
  SCOPED_TRACE("Device support must be false without a symbol and otherwise equal the provider result.");

  struct ibv_device device = {};
  device.node_type = IBV_NODE_CA;
  device.transport_type = IBV_TRANSPORT_IB;
  std::strcpy(device.name, "mlx5-test-device");
  std::strcpy(device.dev_name, "uverbs-test-device");
  std::strcpy(device.dev_path, "/test/verbs/path");
  std::strcpy(device.ibdev_path, "/test/ib/path");
  EXPECT_FALSE(wrap_mlx5dv_is_supported(&device)) << "An unavailable support symbol must report false.";
  EXPECT_EQ(mlx5ProviderState.supportCalls, 0) << "A missing support symbol cannot call the provider.";

  mlx5dvSymbols.mlx5dv_internal_is_supported = DeviceIsSupportedForTest;
  mlx5ProviderState.supportResult = false;
  EXPECT_FALSE(wrap_mlx5dv_is_supported(&device)) << "Provider false must remain false.";
  EXPECT_EQ(mlx5ProviderState.supportCalls, 1) << "The false-result query must call the provider exactly once.";
  EXPECT_EQ(mlx5ProviderState.supportDevice, &device)
    << "The false-result query must forward the exact device pointer.";

  mlx5ProviderState.supportCalls = 0;
  mlx5ProviderState.supportResult = true;
  EXPECT_TRUE(wrap_mlx5dv_is_supported(&device)) << "Provider true must remain true.";
  EXPECT_EQ(mlx5ProviderState.supportCalls, 1) << "The true-result query must call the provider exactly once.";
  EXPECT_EQ(mlx5ProviderState.supportDevice, &device)
    << "The true-result query must forward the exact device pointer.";
  EXPECT_EQ(device.node_type, IBV_NODE_CA) << "Device support queries must not change the node type.";
  EXPECT_EQ(device.transport_type, IBV_TRANSPORT_IB)
    << "Device support queries must not change the transport type.";
  EXPECT_STREQ(device.name, "mlx5-test-device")
    << "Device support queries must not change the provider device name.";
  EXPECT_STREQ(device.dev_name, "uverbs-test-device")
    << "Device support queries must not change the verbs device name.";
  EXPECT_STREQ(device.dev_path, "/test/verbs/path")
    << "Device support queries must not change the verbs device path.";
  EXPECT_STREQ(device.ibdev_path, "/test/ib/path")
    << "Device support queries must not change the IB device path.";
}

/**
 * @brief Verifies every result and caller-owned output behavior of Data Direct path discovery.
 * @note Side effects: Calls the fake path provider three times and writes caller-owned output
 * on success.
 */
TEST_F(Mlx5DvWrapperTest, GetsDataDirectPath) {
  SCOPED_TRACE("Path discovery must distinguish success, a non-Data-Direct device, and provider failure.");

  struct ibv_device device = {};
  int compatibilityToken = 53;
  struct ibv_context context = {};
  context.device = &device;
  context.cmd_fd = 37;
  context.async_fd = 41;
  context.num_comp_vectors = 43;
  context.abi_compat = &compatibilityToken;
  std::array<char, 64> buffer;
  buffer.fill('#');
  const std::array<char, 64> missingSymbolBuffer = buffer;

  EXPECT_EQ(wrap_mlx5dv_get_data_direct_sysfs_path(&context, buffer.data(), buffer.size()), ncclInternalError)
    << "A missing path symbol must return ncclInternalError.";
  EXPECT_EQ(mlx5ProviderState.pathCalls, 0) << "A missing path symbol cannot call the provider.";
  EXPECT_EQ(buffer, missingSymbolBuffer) << "The wrapper must not change output when no provider was called.";

  mlx5dvSymbols.mlx5dv_internal_get_data_direct_sysfs_path = GetDataDirectPathForTest;
  mlx5ProviderState.pathResult = 0;
  EXPECT_EQ(wrap_mlx5dv_get_data_direct_sysfs_path(&context, buffer.data(), buffer.size()), ncclSuccess)
    << "Provider result 0 must return ncclSuccess.";
  EXPECT_STREQ(buffer.data(), mlx5ProviderState.pathOutput.c_str())
    << "Successful path discovery must expose the provider path.";
  EXPECT_EQ(mlx5ProviderState.pathContext, &context) << "The exact context pointer must reach the provider.";
  EXPECT_EQ(mlx5ProviderState.pathBuffer, buffer.data()) << "The exact output pointer must reach the provider.";
  EXPECT_EQ(mlx5ProviderState.pathLength, buffer.size()) << "The exact buffer capacity must reach the provider.";
  EXPECT_EQ(mlx5ProviderState.pathCalls, 1) << "Successful path discovery must call the provider exactly once.";

  errno = EIO;
  mlx5ProviderState.pathCalls = 0;
  mlx5ProviderState.pathResult = ENODEV;
  EXPECT_EQ(wrap_mlx5dv_get_data_direct_sysfs_path(&context, buffer.data(), buffer.size()), ncclInvalidArgument)
    << "Provider ENODEV must identify a valid mlx5 device without Data Direct support.";
  EXPECT_EQ(mlx5ProviderState.pathCalls, 1) << "The ENODEV scenario must call the provider exactly once.";

  errno = ENODEV;
  mlx5ProviderState.pathCalls = 0;
  mlx5ProviderState.pathResult = EIO;
  EXPECT_EQ(wrap_mlx5dv_get_data_direct_sysfs_path(&context, buffer.data(), buffer.size()), ncclSystemError)
    << "A provider error other than ENODEV must return ncclSystemError.";
  EXPECT_EQ(mlx5ProviderState.pathCalls, 1) << "The generic-error scenario must call the provider exactly once.";
  EXPECT_EQ(context.device, &device) << "Path discovery must not replace the context device.";
  EXPECT_EQ(context.cmd_fd, 37) << "Path discovery must not change the context command descriptor.";
  EXPECT_EQ(context.async_fd, 41) << "Path discovery must not change the context event descriptor.";
  EXPECT_EQ(context.num_comp_vectors, 43) << "Path discovery must not change the completion-vector count.";
  EXPECT_EQ(context.abi_compat, &compatibilityToken)
    << "Path discovery must not change the context compatibility pointer.";
}

/**
 * @brief Verifies missing-symbol, successful, and failed checked DMA-BUF registration.
 * @note Side effects: Calls the fake registration provider twice and replaces the output
 * memory-region pointer.
 */
TEST_F(Mlx5DvWrapperTest, RegistersDmaBuf) {
  SCOPED_TRACE("Checked registration must validate capability, forward arguments, and translate a NULL result.");

  struct ibv_context context = {};
  struct ibv_pd domain = {};
  domain.context = &context;
  domain.handle = 0x1234;
  struct ibv_mr sentinelRegion = {};
  struct ibv_mr registeredRegion = {};
  struct ibv_mr* output = &sentinelRegion;

  EXPECT_EQ(wrap_mlx5dv_reg_dmabuf_mr(&output, &domain, 4096, 8192, 0x12345000, 17,
    IBV_ACCESS_REMOTE_WRITE, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT), ncclInternalError)
    << "A missing registration symbol must return ncclInternalError.";
  EXPECT_EQ(output, &sentinelRegion) << "A missing symbol must leave the caller's output pointer unchanged.";
  EXPECT_EQ(mlx5ProviderState.registrationCalls, 0) << "A missing registration symbol cannot call the provider.";

  mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr = RegisterDmaBufForTest;
  mlx5ProviderState.registrationResult = &registeredRegion;
  mlx5ProviderState.registrationCalls = 0;
  output = nullptr;
  EXPECT_EQ(wrap_mlx5dv_reg_dmabuf_mr(&output, &domain, 4096, 8192, 0x12345000, 17,
    IBV_ACCESS_REMOTE_WRITE, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT), ncclSuccess)
    << "A non-NULL provider result must return ncclSuccess.";
  EXPECT_EQ(output, &registeredRegion) << "Successful registration must return the exact provider region.";
  ExpectRegistrationCallForTest(&domain, 4096, 8192, 0x12345000, 17, IBV_ACCESS_REMOTE_WRITE,
    MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT, "Successful checked registration arguments");

  mlx5ProviderState.registrationResult = nullptr;
  mlx5ProviderState.registrationErrno = ENOMEM;
  mlx5ProviderState.registrationCalls = 0;
  output = &sentinelRegion;
  EXPECT_EQ(wrap_mlx5dv_reg_dmabuf_mr(&output, &domain, 12288, 16384, 0x56789000, 19,
    IBV_ACCESS_LOCAL_WRITE, 0), ncclSystemError)
    << "A NULL provider result must return ncclSystemError.";
  EXPECT_EQ(output, nullptr) << "A failed provider call must replace the output with its NULL result.";
  ExpectRegistrationCallForTest(&domain, 12288, 16384, 0x56789000, 19, IBV_ACCESS_LOCAL_WRITE, 0,
    "Failed checked registration arguments");
  EXPECT_EQ(domain.context, &context) << "Registration must not replace the domain context.";
  EXPECT_EQ(domain.handle, 0x1234U)
    << "The wrapper and non-mutating provider must not change observable domain data.";
}

/**
 * @brief Verifies capability errno, provider results, and direct DMA-BUF forwarding.
 * @note Side effects: Calls the fake registration provider twice and updates errno.
 */
TEST_F(Mlx5DvWrapperTest, RegistersDmaBufDirectly) {
  SCOPED_TRACE("Direct registration must expose unsupported capability and otherwise preserve provider behavior.");

  struct ibv_context context = {};
  struct ibv_pd domain = {};
  domain.context = &context;
  domain.handle = 0x5678;
  struct ibv_mr registeredRegion = {};

  errno = 0;
  EXPECT_EQ(wrap_direct_mlx5dv_reg_dmabuf_mr(&domain, 2048, 4096, 0x11112000, 23,
    IBV_ACCESS_REMOTE_READ, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT), nullptr)
    << "A missing direct-registration symbol must return NULL.";
  EXPECT_EQ(errno, EOPNOTSUPP) << "A missing direct-registration symbol must expose EOPNOTSUPP.";
  EXPECT_EQ(mlx5ProviderState.registrationCalls, 0) << "A missing symbol cannot call the provider.";

  mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr = RegisterDmaBufForTest;
  mlx5ProviderState.registrationResult = &registeredRegion;
  mlx5ProviderState.registrationCalls = 0;
  EXPECT_EQ(wrap_direct_mlx5dv_reg_dmabuf_mr(&domain, 2048, 4096, 0x11112000, 23,
    IBV_ACCESS_REMOTE_READ, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT), &registeredRegion)
    << "A successful direct registration must return the exact provider region.";
  ExpectRegistrationCallForTest(&domain, 2048, 4096, 0x11112000, 23, IBV_ACCESS_REMOTE_READ,
    MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT, "Successful direct registration arguments");

  mlx5ProviderState.registrationResult = nullptr;
  mlx5ProviderState.registrationErrno = EPROTONOSUPPORT;
  mlx5ProviderState.registrationCalls = 0;
  errno = 0;
  EXPECT_EQ(wrap_direct_mlx5dv_reg_dmabuf_mr(&domain, 6144, 12288, 0x33334000, 29,
    IBV_ACCESS_LOCAL_WRITE, 0), nullptr) << "A failed direct registration must return provider NULL.";
  EXPECT_EQ(errno, EPROTONOSUPPORT) << "Direct registration must preserve the provider capability errno.";
  ExpectRegistrationCallForTest(&domain, 6144, 12288, 0x33334000, 29, IBV_ACCESS_LOCAL_WRITE, 0,
    "Failed direct registration arguments");
  EXPECT_EQ(domain.context, &context) << "Direct registration must not replace the domain context.";
  EXPECT_EQ(domain.handle, 0x5678U)
    << "The wrapper and non-mutating provider must not change observable domain data.";
}

}  // namespace
