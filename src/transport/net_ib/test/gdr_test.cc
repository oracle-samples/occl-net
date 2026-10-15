/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "test_net_ib_state.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

namespace {

/** @brief Captures one dummy DMA-BUF registration request. */
struct DmaRegistrationCall {
  ibv_pd* protectionDomain = nullptr;
  uint64_t offset = 0;
  size_t length = 0;
  uint64_t iova = 0;
  int descriptor = 0;
  int access = 0;
};

/** @brief Scripts provider outcomes and records GDR probe side effects. */
struct GdrProbeState {
  std::vector<std::string> availablePaths;
  std::vector<std::string> probedPaths;
  std::vector<int> accessModes;
  ncclResult_t allocationResult = ncclSuccess;
  ncclResult_t deallocationResult = ncclSuccess;
  int registrationErrno = EBADF;
  int allocationCalls = 0;
  int registrationCalls = 0;
  int deallocationCalls = 0;
  ibv_context* allocationContext = nullptr;
  ibv_pd* deallocatedProtectionDomain = nullptr;
  DmaRegistrationCall registrationCall;
  ibv_context context = {};
  ibv_pd protectionDomain = {};
};

/** @brief Process-local state shared by the GDR test seams. */
GdrProbeState probeState;

/**
 * @brief Resets every fake kernel-module and DMA-BUF provider observation.
 * @note Side effects: Replaces probeState.
 */
void ResetProbeForTest() {
  probeState = GdrProbeState{};
}

/**
 * @brief Reports whether one kernel module path is configured as present.
 * @param[in] path Valid non-NULL kernel module path.
 * @param[in] mode Access-check mode recorded for verification.
 * @return Zero for a configured path or -1 with ENOENT otherwise.
 * @note Side effects: Records the path and mode and sets errno to ENOENT when absent.
 */
int AccessForTest(const char* path, int mode) {
  probeState.probedPaths.emplace_back(path);
  probeState.accessModes.push_back(mode);
  const bool available =
      std::find(probeState.availablePaths.begin(), probeState.availablePaths.end(), path) !=
          probeState.availablePaths.end();
  if (!available) errno = ENOENT;
  return available ? 0 : -1;
}

}  // namespace

/**
 * @brief Supplies a controlled protection-domain allocation result.
 * @param[out] result Valid storage receiving a borrowed fake domain on success.
 * @param[in] context NULL or any context pointer recorded without modification.
 * @return probeState.allocationResult.
 * @note Side effects: Increments allocationCalls, records context, and returns a borrowed fake
 * domain through result on success.
 */
ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd** result, struct ibv_context* context) {
  ++probeState.allocationCalls;
  probeState.allocationContext = context;
  if (probeState.allocationResult == ncclSuccess) *result = &probeState.protectionDomain;
  return probeState.allocationResult;
}

/**
 * @brief Supplies a controlled protection-domain release result.
 * @param[in] protectionDomain NULL or any domain pointer recorded without modification.
 * @return probeState.deallocationResult.
 * @note Side effects: Increments deallocationCalls and records protectionDomain; ownership is
 * unchanged.
 */
ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd* protectionDomain) {
  ++probeState.deallocationCalls;
  probeState.deallocatedProtectionDomain = protectionDomain;
  return probeState.deallocationResult;
}

/**
 * @brief Supplies a failing DMA-BUF registration with scripted errno.
 * @param[in] protectionDomain NULL or any domain pointer recorded without modification.
 * @param[in] offset DMA-BUF offset recorded exactly.
 * @param[in] length Registration length recorded exactly.
 * @param[in] iova I/O virtual address recorded exactly.
 * @param[in] descriptor File descriptor recorded exactly.
 * @param[in] access Access mask recorded exactly.
 * @return NULL.
 * @note Side effects: Increments registrationCalls, records all arguments, and sets errno.
 */
struct ibv_mr* wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd* protectionDomain, uint64_t offset,
  size_t length, uint64_t iova, int descriptor, int access) {
  ++probeState.registrationCalls;
  probeState.registrationCall = {protectionDomain, offset, length, iova, descriptor, access};
  errno = probeState.registrationErrno;
  return nullptr;
}

/**
 * @brief Compiles gdr.cc with kernel-module existence checks redirected to the scripted
 * access seam.
 */
#define access AccessForTest
#include "../gdr.cc"  // NOLINT(build/include)
#undef access

namespace {

/**
 * @brief Configures one physical device for a direct or public DMA-BUF support probe.
 * @param[in] physicalDevice Valid physical device index selected by the probe.
 * @note Side effects: Resets provider state, sets the physical count and probe selector,
 * and replaces the selected device fields.
 */
void PrepareDmaProbeForTest(int physicalDevice) {
  ResetProbeForTest();
  ncclNIbDevs = physicalDevice + 1;
  ncclIbDevs[physicalDevice].context = &probeState.context;
  ncclIbDevs[physicalDevice].dmaBufSupported = 0;
  ibDmaSupportInitDev = physicalDevice;
}

/**
 * @brief Verifies one cached public module-support outcome in an isolated process.
 * @param[in] peerOnly Selects the nvidia_peermem-only accessor when true.
 * @param[in] available Configures a matching module as present on the first lookup.
 * @return Zero when status and once-only probing match the contract; otherwise one.
 * @note Side effects: Initializes one production once flag in the child process.
 */
int VerifyCachedModuleSupportForTest(bool peerOnly, bool available) {
  ResetProbeForTest();
  const char* selectedPath = peerOnly
      ? "/sys/module/nvidia_peermem/version"
      : "/sys/kernel/mm/memory_peers/nv_mem_nc/version";
  if (available) probeState.availablePaths.emplace_back(selectedPath);
  ncclIbGdrModuleLoaded = 0;
  ncclIbPeerMemModuleLoaded = 0;

  const ncclResult_t first = peerOnly ? ncclIbPeerMemSupport() : ncclIbGdrSupport();
  const size_t firstProbeCount = probeState.probedPaths.size();
  if (available) {
    probeState.availablePaths.clear();
  } else {
    probeState.availablePaths.emplace_back(selectedPath);
  }
  const ncclResult_t second = peerOnly ? ncclIbPeerMemSupport() : ncclIbGdrSupport();
  const ncclResult_t expected = available ? ncclSuccess : ncclSystemError;
  if (first != expected || second != expected || probeState.probedPaths.size() != firstProbeCount) {
    std::fprintf(stderr, "module cache mismatch: peerOnly=%d available=%d first=%d second=%d "
                         "probes=%zu/%zu expected=%d\n",
        peerOnly, available, first, second, firstProbeCount,
        probeState.probedPaths.size(), expected);
    return 1;
  }
  return 0;
}

/**
 * @brief Verifies one public DMA-BUF support outcome in an isolated process.
 * @param[in] registrationError errno produced by the dummy registration probe.
 * @param[in] expected Expected public support status.
 * @return Zero when the public result and cached device state match; otherwise one.
 * @note Side effects: Initializes the device-zero once flag and updates selected-device
 * support state in the child.
 */
int VerifyDmaSupportForTest(int registrationError, ncclResult_t expected) {
  PrepareDmaProbeForTest(0);
  probeState.registrationErrno = registrationError;
  const ncclResult_t first = ncclIbDmaBufSupport(0);
  const int registrationsAfterFirst = probeState.registrationCalls;
  probeState.registrationErrno = registrationError == EBADF ? EOPNOTSUPP : EBADF;
  const ncclResult_t second = ncclIbDmaBufSupport(0);
  if (first != expected || second != expected ||
      probeState.registrationCalls != registrationsAfterFirst) {
    std::fprintf(stderr,
        "DMA-BUF cache mismatch: errno=%d first=%d second=%d registrations=%d/%d expected=%d\n",
        registrationError, first, second, registrationsAfterFirst,
        probeState.registrationCalls, expected);
    return 1;
  }
  return 0;
}

}  // namespace

/**
 * @brief Verifies ordered GDR module probing, nv_mem_nc detection, and the all-missing case.
 * @note Side effects: Replaces fake access state and the production GDR status flag.
 */
TEST(NetIbGdrTest, InitializesGdrModuleSupport) {
  SCOPED_TRACE(
      "GDR initialization must probe supported paths in order and publish the detected state.");
  ResetProbeForTest();
  ncclIbGdrModuleLoaded = 0;
  ibGdrSupportInitOnce();
  EXPECT_EQ(ncclIbGdrModuleLoaded, 0)
      << "Missing kernel modules must leave GDR support disabled";
  EXPECT_EQ(probeState.probedPaths,
      std::vector<std::string>({
          "/sys/kernel/mm/memory_peers/nv_mem/version",
          "/sys/kernel/mm/memory_peers/nv_mem_nc/version",
          "/sys/module/nvidia_peermem/version"}))
      << "The missing case must inspect every supported kernel module path in priority order";
  EXPECT_EQ(probeState.accessModes, std::vector<int>({F_OK, F_OK, F_OK}))
      << "Kernel module probing must use existence checks";

  ResetProbeForTest();
  probeState.availablePaths.emplace_back("/sys/kernel/mm/memory_peers/nv_mem_nc/version");
  ncclIbGdrModuleLoaded = 0;
  ibGdrSupportInitOnce();
  EXPECT_EQ(ncclIbGdrModuleLoaded, 1)
      << "Finding any supported legacy peer-memory module must enable GDR support";
  EXPECT_EQ(probeState.probedPaths,
      std::vector<std::string>({
          "/sys/kernel/mm/memory_peers/nv_mem/version",
          "/sys/kernel/mm/memory_peers/nv_mem_nc/version"}))
      << "Module probing must stop after the first detected implementation";
}

/**
 * @brief Verifies the direct peer-memory initializer accepts only the nvidia_peermem module path.
 * @note Side effects: Replaces fake access state and the production peer-memory status flag.
 */
TEST(NetIbGdrTest, InitializesPeerMemorySupport) {
  SCOPED_TRACE(
      "Peer-memory initialization must reflect the nvidia_peermem module existence result.");
  ResetProbeForTest();
  ncclIbPeerMemModuleLoaded = 0;
  ibPeerMemSupportInitOnce();
  EXPECT_EQ(ncclIbPeerMemModuleLoaded, 0)
      << "A missing nvidia_peermem module must leave peer-memory support disabled";
  EXPECT_EQ(probeState.probedPaths,
      std::vector<std::string>({"/sys/module/nvidia_peermem/version"}))
      << "Peer-memory initialization must inspect only its supported module path";
  EXPECT_EQ(probeState.accessModes, std::vector<int>({F_OK}))
      << "Peer-memory probing must use an existence check";

  ResetProbeForTest();
  probeState.availablePaths.emplace_back("/sys/module/nvidia_peermem/version");
  ncclIbPeerMemModuleLoaded = 0;
  ibPeerMemSupportInitOnce();
  EXPECT_EQ(ncclIbPeerMemModuleLoaded, 1)
      << "An available nvidia_peermem module must enable peer-memory support";
}

/**
 * @brief Verifies the public GDR accessor detects and caches an available legacy module.
 * @note Side effects: Initializes the process-local GDR once flag.
 */
TEST(NetIbGdrTest, PublicGdrSupportReportsAvailableModule) {
  ResetProbeForTest();
  probeState.availablePaths.emplace_back("/sys/kernel/mm/memory_peers/nv_mem/version");
  ncclIbGdrModuleLoaded = 0;

  ASSERT_EQ(ncclIbGdrSupport(), ncclSuccess)
      << "The public accessor must report an available peer-memory module";
  const size_t probesAfterInitialization = probeState.probedPaths.size();
  ASSERT_EQ(probesAfterInitialization, 1u)
      << "Detection must stop after finding the first supported module";

  probeState.availablePaths.clear();
  EXPECT_EQ(ncclIbGdrSupport(), ncclSuccess)
      << "The public accessor must retain its initialized support result";
  EXPECT_EQ(probeState.probedPaths.size(), probesAfterInitialization)
      << "A cached public lookup must not probe kernel paths again";
}

/**
 * @brief Verifies the public peer-memory accessor detects and caches nvidia_peermem.
 * @note Side effects: Initializes the process-local peer-memory once flag.
 */
TEST(NetIbGdrTest, PublicPeerMemorySupportReportsAvailableModule) {
  ResetProbeForTest();
  probeState.availablePaths.emplace_back("/sys/module/nvidia_peermem/version");
  ncclIbPeerMemModuleLoaded = 0;

  ASSERT_EQ(ncclIbPeerMemSupport(), ncclSuccess)
      << "The public accessor must report an available nvidia_peermem module";
  const size_t probesAfterInitialization = probeState.probedPaths.size();
  ASSERT_EQ(probesAfterInitialization, 1u)
      << "Peer-memory detection must inspect exactly its supported module path";

  probeState.availablePaths.clear();
  EXPECT_EQ(ncclIbPeerMemSupport(), ncclSuccess)
      << "The public accessor must retain its initialized support result";
  EXPECT_EQ(probeState.probedPaths.size(), probesAfterInitialization)
      << "A cached public lookup must not probe the kernel path again";
}

/**
 * @brief Verifies the direct DMA-BUF initializer maps provider outcomes and forwards the
 * dummy probe exactly.
 * @note Side effects: Replaces fake provider and physical-device state.
 */
TEST(NetIbGdrTest, InitializesDmaBufSupport) {
  SCOPED_TRACE(
      "DMA-BUF initialization must forward its probe, release its PD, and publish each outcome.");
  PrepareDmaProbeForTest(2);
  probeState.registrationErrno = EBADF;
  ibDmaBufSupportInitOnce();
  EXPECT_EQ(ncclIbDevs[2].dmaBufSupported, 1)
      << "EBADF from fd=-1 means the DMA-BUF registration operation itself is supported";
  EXPECT_EQ(probeState.allocationCalls, 1) << "The probe must allocate one protection domain";
  EXPECT_EQ(probeState.allocationContext, &probeState.context)
      << "The PD must be allocated from the selected physical device context";
  EXPECT_EQ(probeState.registrationCalls, 1)
      << "The probe must issue one dummy DMA-BUF registration";
  EXPECT_EQ(probeState.registrationCall.protectionDomain, &probeState.protectionDomain)
      << "The dummy registration must use the newly allocated protection domain";
  EXPECT_EQ(probeState.registrationCall.offset, 0u) << "The dummy registration offset must be zero";
  EXPECT_EQ(probeState.registrationCall.length, 0u) << "The dummy registration length must be zero";
  EXPECT_EQ(probeState.registrationCall.iova, 0u) << "The dummy registration IOVA must be zero";
  EXPECT_EQ(probeState.registrationCall.descriptor, -1)
      << "The dummy registration must use an intentionally invalid descriptor";
  EXPECT_EQ(probeState.registrationCall.access, 0)
      << "The dummy registration must request no access flags";
  EXPECT_EQ(probeState.deallocationCalls, 1) << "The probe must release its protection domain";
  EXPECT_EQ(probeState.deallocatedProtectionDomain, &probeState.protectionDomain)
      << "The probe must release the same protection domain it allocated";

  for (int unsupportedError : {EOPNOTSUPP, EPROTONOSUPPORT}) {
    SCOPED_TRACE(testing::Message() << "unsupported DMA-BUF errno " << unsupportedError);
    PrepareDmaProbeForTest(2);
    probeState.registrationErrno = unsupportedError;
    ibDmaBufSupportInitOnce();
    EXPECT_EQ(ncclIbDevs[2].dmaBufSupported, -1)
        << "An explicit unsupported errno must publish unavailable DMA-BUF support";
    EXPECT_EQ(probeState.deallocationCalls, 1)
        << "An unsupported dummy registration must still release its protection domain";
  }

  PrepareDmaProbeForTest(2);
  probeState.allocationResult = ncclSystemError;
  ibDmaBufSupportInitOnce();
  EXPECT_EQ(ncclIbDevs[2].dmaBufSupported, -1)
      << "A protection-domain allocation failure must publish unavailable support";
  EXPECT_EQ(probeState.registrationCalls, 0)
      << "A failed protection-domain allocation must skip dummy registration";
  EXPECT_EQ(probeState.deallocationCalls, 0)
      << "A failed protection-domain allocation creates nothing to release";

  PrepareDmaProbeForTest(2);
  probeState.deallocationResult = ncclSystemError;
  ibDmaBufSupportInitOnce();
  EXPECT_EQ(ncclIbDevs[2].dmaBufSupported, -1)
      << "A protection-domain cleanup failure must not publish successful support";
  EXPECT_EQ(probeState.registrationCalls, 1)
      << "Cleanup failure occurs only after the dummy registration probe";
  EXPECT_EQ(probeState.deallocationCalls, 1)
      << "The initializer must expose the attempted protection-domain cleanup";
}

/**
 * @brief Verifies public module-support accessors map both outcomes and cache their first probe.
 * @note Side effects: Sets the death-test style and initializes once flags in child processes.
 */
TEST(NetIbGdrTest, ModuleProbeReturnsStableResultCode) {
  SCOPED_TRACE("Public GDR support accessors must map detected state and probe only once.");
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (bool peerOnly : {false, true}) {
    for (bool available : {false, true}) {
      SCOPED_TRACE(testing::Message() << "peerOnly=" << peerOnly << " available=" << available);
      EXPECT_EXIT(
          { std::exit(VerifyCachedModuleSupportForTest(peerOnly, available)); },
          testing::ExitedWithCode(0), ".*")
          << "The isolated accessor must return and cache the status selected by its module probe";
    }
  }
}

/**
 * @brief Verifies public DMA-BUF status caching and physical-index validation.
 * @note Side effects: Sets the death-test style and updates fake physical-device state.
 */
TEST(NetIbGdrTest, DmaBufUnsupportedWhenKernelReportsOperationNotSupported) {
  SCOPED_TRACE(
      "The public DMA-BUF accessor must map dummy-registration errno and probe device zero once.");
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      { std::exit(VerifyDmaSupportForTest(EBADF, ncclSuccess)); },
      testing::ExitedWithCode(0), ".*")
      << "EBADF from the invalid descriptor must identify a supported DMA-BUF operation";
  EXPECT_EXIT(
      { std::exit(VerifyDmaSupportForTest(EOPNOTSUPP, ncclSystemError)); },
      testing::ExitedWithCode(0), ".*")
      << "EOPNOTSUPP must identify an unsupported DMA-BUF operation";

  PrepareDmaProbeForTest(0);
  EXPECT_EQ(ncclIbDmaBufSupport(-1), ncclSystemError)
      << "A negative physical device index must be rejected before probing";
  EXPECT_EQ(ncclIbDmaBufSupport(ncclNIbDevs), ncclSystemError)
      << "An index at the physical device count must be rejected before probing";
  EXPECT_EQ(probeState.allocationCalls, 0)
      << "Rejected physical device indices must not allocate provider resources";
}
