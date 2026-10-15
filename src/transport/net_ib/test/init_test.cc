/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "test_net_ib_state.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

/** @brief Records one generated parameter accessor lookup. */
struct InitParameterLoadCall {
  std::string environment;
  int64_t defaultValue;
};

/**
 * @brief Holds scripted initialization, discovery, provider, allocation, and ownership state.
 * @note Fields configure dependency outcomes and record calls and resources observed by init.cc.
 */
struct InitTestState {
  std::unordered_map<std::string, int64_t> parameters;
  std::vector<InitParameterLoadCall> parameterLoads;
  bool cacheParameterValues = false;
  bool realpathSucceeds = false;
  std::string resolvedPath;
  std::string requestedPath;
  char* lastRealpathAllocation = nullptr;
  bool allocationFails = false;
  int allocationCalls = 0;
  ncclResult_t gdrSupportResult = ncclSystemError;
  ncclResult_t dmaBufSupportResult = ncclSystemError;
  bool useDeviceDmaBufResults = false;
  bool deviceDmaBufSupport[MAX_IB_DEVS] = {};
  int gdrSupportCalls = 0;
  int dmaBufSupportCalls = 0;
  std::vector<int> dmaBufSupportDevices;
  ncclResult_t relaxedOrderingResult = ncclInternalError;
  int relaxedOrderingCalls = 0;
  ncclResult_t allocPdResult = ncclSuccess;
  ncclResult_t deallocPdResult = ncclSuccess;
  int allocPdCalls = 0;
  int deallocPdCalls = 0;
  ibv_context* allocatedPdContext = nullptr;
  ibv_pd* deallocatedPd = nullptr;
  int genericDmaBufCalls = 0;
  int mlx5DmaBufCalls = 0;
  ibv_pd* genericDmaBufPd = nullptr;
  ibv_pd* mlx5DmaBufPd = nullptr;
  uint64_t genericDmaBufOffset = UINT64_MAX;
  uint64_t genericDmaBufLength = UINT64_MAX;
  uint64_t genericDmaBufIova = UINT64_MAX;
  int genericDmaBufFd = 0;
  int genericDmaBufFlags = -1;
  uint64_t mlx5DmaBufOffset = UINT64_MAX;
  uint64_t mlx5DmaBufLength = UINT64_MAX;
  uint64_t mlx5DmaBufIova = UINT64_MAX;
  int mlx5DmaBufFd = 0;
  int mlx5DmaBufAccessFlags = -1;
  int mlx5DmaBufFlags = -1;
  int genericDmaBufErrno = EBADF;
  int mlx5DmaBufErrno = EBADF;
  ncclResult_t verbsSymbolsResult = ncclSuccess;
  ncclResult_t mlx5SymbolsResult = ncclSuccess;
  ncclResult_t findInterfacesResult = ncclSuccess;
  ncclResult_t getDeviceListResult = ncclSuccess;
  int interfaceCount = 1;
  int deviceCount = 0;
  int verbsSymbolsCalls = 0;
  int mlx5SymbolsCalls = 0;
  int findInterfacesCalls = 0;
  int getDeviceListCalls = 0;
  int freeDeviceListCalls = 0;
  int forkInitCalls = 0;
  ncclResult_t openDeviceResult = ncclSuccess;
  ncclResult_t closeDeviceResult = ncclSuccess;
  ncclResult_t queryDeviceResult = ncclSuccess;
  ncclResult_t queryPortResult = ncclSuccess;
  int openDeviceCalls = 0;
  int closeDeviceCalls = 0;
  int queryDeviceCalls = 0;
  int queryPortCalls = 0;
  bool interfaceMatches = true;
  bool providerSupported = false;
  ncclResult_t dataDirectPathResult = ncclInvalidArgument;
  int dataDirectPathCalls = 0;
  uint8_t activeWidth = 1;
  uint8_t activeSpeed = 32;
  uint32_t activeSpeedEx = 0;
  ibv_device device = {};
  ibv_device* deviceList[2] = {};
  ibv_context context = {};
  ibv_pd protectionDomain = {};
};

/** @brief Process-local script and observation state for init.cc seams. */
InitTestState state;

/**
 * @brief Returns a scripted parameter value or its production default.
 * @param[in] name Non-NULL environment suffix used as the lookup key.
 * @param[in] defaultValue Value returned when no override exists.
 * @return The configured value, or defaultValue.
 * @note Side effects: None.
 */
int64_t ParameterForTest(const char* name, int64_t defaultValue) {
  auto parameter = state.parameters.find(name);
  return parameter == state.parameters.end() ? defaultValue : parameter->second;
}

/**
 * @brief Supplies controlled values to the production-generated initialization parameter accessors.
 * @param[in] environment Full environment-variable name requested by the accessor.
 * @param[in] defaultValue Production default used when the test provides no configured value.
 * @param[in] uninitialized Production cache sentinel; it is not modified.
 * @param[out] cache Accessor-owned cache storage, populated only when cache testing is enabled.
 * @param[out] noCache Accessor-owned policy flag selecting dynamic or cached test behavior.
 * @return The configured value for environment, or defaultValue when it is absent.
 * @note Side effects: Records the lookup and optionally populates the accessor cache.
 */
int64_t LoadInitParameterForTest(const char* environment, int64_t defaultValue,
    int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  (void)uninitialized;
  state.parameterLoads.push_back({environment, defaultValue});
  const int64_t value = ParameterForTest(environment, defaultValue);
  if (state.cacheParameterValues) {
    *cache = value;
    *noCache = 0;
  } else {
    *noCache = 1;
  }
  return value;
}

/**
 * @brief Implements a controllable zero-initializing allocation seam.
 * @param[out] pointer Valid output storage set only on success.
 * @param[in] bytes Number of bytes to allocate.
 * @return ncclSuccess, or ncclSystemError on scripted or allocator failure.
 * @note Side effects: Records the attempt and may allocate caller-owned memory.
 */
ncclResult_t AllocateForTest(void** pointer, size_t bytes) {
  state.allocationCalls++;
  if (state.allocationFails) return ncclSystemError;
  void* allocation = std::calloc(1, bytes);
  if (allocation == nullptr) return ncclSystemError;
  *pointer = allocation;
  return ncclSuccess;
}

/**
 * @brief Reproduces caller-buffer and allocating realpath forms with controlled success.
 * @param[in] path Non-NULL path recorded as the production request.
 * @param[out] resolvedPath NULL to request allocation, or caller-owned storage large enough
 * for the configured resolved path and terminator.
 * @return The caller's buffer, a newly allocated caller-owned path, or NULL on failure.
 * @note Side effects: Records path and may set errno, write resolvedPath, or allocate and
 * record returned storage.
 */
char* RealpathForTest(const char* path, char* resolvedPath) {
  state.requestedPath = path;
  if (!state.realpathSucceeds) {
    errno = ENOENT;
    return nullptr;
  }
  if (resolvedPath != nullptr) {
    std::memcpy(resolvedPath, state.resolvedPath.c_str(), state.resolvedPath.size() + 1);
    return resolvedPath;
  }
  char* allocation = static_cast<char*>(std::malloc(state.resolvedPath.size() + 1));
  if (allocation == nullptr) return nullptr;
  std::memcpy(allocation, state.resolvedPath.c_str(), state.resolvedPath.size() + 1);
  state.lastRealpathAllocation = allocation;
  return allocation;
}

/**
 * @brief Supplies a successful no-op profiler registration callback.
 * @param[out] context Ignored profiler context output.
 * @param[in] event Ignored event identifier.
 * @param[in] data Ignored event data.
 * @param[in] value Ignored event value.
 * @param[in] userData Ignored caller data.
 * @return ncclSuccess.
 * @note Side effects: None.
 */
ncclResult_t ProfilerForTest(
    void** context, int event, void* data, int64_t value, void* userData) {
  (void)context;
  (void)event;
  (void)data;
  (void)value;
  (void)userData;
  return ncclSuccess;
}

}  // namespace

/** @brief Test definition of the transport adaptive-routing threshold. */
int64_t ncclIbArThreshold = 8192;

/** @brief Returns the disabled test value for preposted receive work requests. */
int64_t ncclParamIbPrepostReceiveWorkRequests() {
  return -2;
}

/** @brief Returns the disabled test value for receiver-side matching. */
int64_t ncclParamIbReceiverSideMatchingScheme() {
  return -2;
}

/** @brief Reports successful startup of the stub recovery thread. */
ncclResult_t ncclIbPortRecoveryThreadStart() {
  return ncclSuccess;
}

/** @brief Reports successful shutdown of the stub recovery thread. */
ncclResult_t ncclIbPortRecoveryThreadStop() {
  return ncclSuccess;
}

/**
 * @brief Returns the scripted peer-memory support result.
 * @return state.gdrSupportResult.
 * @note Side effects: Increments gdrSupportCalls.
 */
ncclResult_t ncclIbGdrSupport() {
  state.gdrSupportCalls++;
  return state.gdrSupportResult;
}

/**
 * @brief Returns the configured DMA-BUF result for one physical device.
 * @param[in] dev Valid physical device index queried by production code.
 * @return The per-device result when enabled; otherwise the shared scripted result.
 * @note Side effects: Records the query and increments dmaBufSupportCalls.
 */
ncclResult_t ncclIbDmaBufSupport(int dev) {
  state.dmaBufSupportCalls++;
  state.dmaBufSupportDevices.push_back(dev);
  if (state.useDeviceDmaBufResults) {
    return state.deviceDmaBufSupport[dev] ? ncclSuccess : ncclSystemError;
  }
  return state.dmaBufSupportResult;
}

/**
 * @brief Parses the terminal PCI address component into its numeric identifier.
 * @param[in] path Non-NULL mutable path string; its contents are preserved.
 * @param[out] id Valid storage written only when parsing succeeds.
 * @return ncclSuccess, or ncclInvalidArgument for an invalid address.
 * @note Side effects: Writes id on success.
 */
ncclResult_t pciPathToInt64(char* path, int64_t* id) {
  const char* end = path + std::strlen(path);
  while (end > path && end[-1] == '/') end--;
  const char* component = end;
  while (component > path && component[-1] != '/') component--;
  unsigned int domain = 0;
  unsigned int bus = 0;
  unsigned int device = 0;
  unsigned int function = 0;
  if (std::sscanf(component, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
    return ncclInvalidArgument;
  }
  *id = (static_cast<int64_t>(domain) << 20) | (static_cast<int64_t>(bus) << 12) |
      (static_cast<int64_t>(device) << 4) | function;
  return ncclSuccess;
}

/**
 * @brief Supplies an empty parsed interface filter.
 * @param[in] text Ignored filter text.
 * @param[out] interfaces Ignored interface storage.
 * @param[in] capacity Ignored interface capacity.
 * @return Zero parsed entries.
 * @note Side effects: None.
 */
int parseStringList(const char* text, struct netIf* interfaces, int capacity) {
  (void)text;
  (void)interfaces;
  (void)capacity;
  return 0;
}

/**
 * @brief Returns the scripted interface-filter match result.
 * @param[in] name Ignored interface name.
 * @param[in] port Ignored interface port.
 * @param[in] interfaces Ignored parsed filter entries.
 * @param[in] count Ignored entry count.
 * @param[in] exactMatch Ignored matching mode.
 * @return state.interfaceMatches.
 * @note Side effects: None.
 */
bool matchIfList(
    const char* name, int port, struct netIf* interfaces, int count, bool exactMatch) {
  (void)name;
  (void)port;
  (void)interfaces;
  (void)count;
  (void)exactMatch;
  return state.interfaceMatches;
}

/**
 * @brief Supplies a scripted socket-interface discovery result.
 * @param[out] ifNames Valid output name storage when an interface is returned.
 * @param[out] ifAddrs Valid output address storage when an interface is returned.
 * @param[in] ifNameMaxSize Capacity of each name slot.
 * @param[in] maxIfs Ignored output-entry capacity.
 * @param[out] nIfs Valid discovered-interface count storage.
 * @return state.findInterfacesResult.
 * @note Side effects: Records the call and writes outputs on success.
 */
ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress* ifAddrs,
    int ifNameMaxSize, int maxIfs, int* nIfs) {
  (void)maxIfs;
  state.findInterfacesCalls++;
  if (state.findInterfacesResult != ncclSuccess) return state.findInterfacesResult;
  *nIfs = state.interfaceCount;
  if (state.interfaceCount > 0) {
    std::snprintf(ifNames, ifNameMaxSize, "test-if");
    std::memset(ifAddrs, 0, sizeof(*ifAddrs));
  }
  return ncclSuccess;
}

/**
 * @brief Formats the deterministic test socket address.
 * @param[in] address Ignored socket address.
 * @param[out] buffer Valid storage for the formatted address.
 * @param[in] numericHost Ignored formatting mode.
 * @return buffer.
 * @note Side effects: Writes the fixed loopback address to buffer.
 */
const char* ncclSocketToString(
    const union ncclSocketAddress* address, char* buffer, const int numericHost) {
  (void)address;
  (void)numericHost;
  std::snprintf(buffer, SOCKET_NAME_MAXLEN + 1, "127.0.0.1");
  return buffer;
}

/**
 * @brief Returns the scripted libibverbs loader result.
 * @return state.verbsSymbolsResult.
 * @note Side effects: Increments verbsSymbolsCalls.
 */
ncclResult_t wrap_ibv_symbols() {
  state.verbsSymbolsCalls++;
  return state.verbsSymbolsResult;
}

/**
 * @brief Returns the scripted mlx5 loader result.
 * @return state.mlx5SymbolsResult.
 * @note Side effects: Increments mlx5SymbolsCalls.
 */
ncclResult_t wrap_mlx5dv_symbols() {
  state.mlx5SymbolsCalls++;
  return state.mlx5SymbolsResult;
}

/**
 * @brief Supplies successful fork-safety initialization.
 * @return ncclSuccess.
 * @note Side effects: Increments forkInitCalls.
 */
ncclResult_t wrap_ibv_fork_init() {
  state.forkInitCalls++;
  return ncclSuccess;
}

/**
 * @brief Supplies the scripted verbs device list.
 * @param[out] devices Valid output storage for the borrowed list.
 * @param[out] count Valid output storage for the device count.
 * @return state.getDeviceListResult.
 * @note Side effects: Records the call and writes outputs on success.
 */
ncclResult_t wrap_ibv_get_device_list(struct ibv_device*** devices, int* count) {
  state.getDeviceListCalls++;
  if (state.getDeviceListResult != ncclSuccess) return state.getDeviceListResult;
  state.deviceList[0] = state.deviceCount > 0 ? &state.device : nullptr;
  state.deviceList[1] = nullptr;
  *devices = state.deviceCount > 0 ? state.deviceList : nullptr;
  *count = state.deviceCount;
  return ncclSuccess;
}

/**
 * @brief Records release of the borrowed fake device list.
 * @param[in] devices Ignored borrowed list.
 * @return ncclSuccess.
 * @note Side effects: Increments freeDeviceListCalls.
 */
ncclResult_t wrap_ibv_free_device_list(struct ibv_device** devices) {
  (void)devices;
  state.freeDeviceListCalls++;
  return ncclSuccess;
}

/**
 * @brief Supplies a scripted device-open result and borrowed context.
 * @param[out] context Valid output storage written on success.
 * @param[in] device Ignored borrowed device.
 * @return state.openDeviceResult.
 * @note Side effects: Increments openDeviceCalls and may write context.
 */
ncclResult_t wrap_ibv_open_device(
    struct ibv_context** context, struct ibv_device* device) {
  (void)device;
  state.openDeviceCalls++;
  if (state.openDeviceResult == ncclSuccess) *context = &state.context;
  return state.openDeviceResult;
}

/**
 * @brief Returns the scripted device-close result.
 * @param[in] context Ignored borrowed context.
 * @return state.closeDeviceResult.
 * @note Side effects: Increments closeDeviceCalls.
 */
ncclResult_t wrap_ibv_close_device(struct ibv_context* context) {
  (void)context;
  state.closeDeviceCalls++;
  return state.closeDeviceResult;
}

/**
 * @brief Supplies scripted device attributes.
 * @param[in] context Ignored borrowed context.
 * @param[out] attributes Valid output storage written on success.
 * @return state.queryDeviceResult.
 * @note Side effects: Increments queryDeviceCalls and may write attributes.
 */
ncclResult_t wrap_ibv_query_device(
    struct ibv_context* context, struct ibv_device_attr* attributes) {
  (void)context;
  state.queryDeviceCalls++;
  if (state.queryDeviceResult != ncclSuccess) return state.queryDeviceResult;
  std::memset(attributes, 0, sizeof(*attributes));
  attributes->sys_image_guid = 0x1234;
  attributes->max_qp = 64;
  attributes->phys_port_cnt = 1;
  return ncclSuccess;
}

/**
 * @brief Supplies scripted active-port attributes.
 * @param[in] context Ignored borrowed context.
 * @param[in] port Ignored physical port number.
 * @param[out] attributes Valid output storage written on success.
 * @return state.queryPortResult.
 * @note Side effects: Increments queryPortCalls and may write attributes.
 */
ncclResult_t wrap_ibv_query_port(
    struct ibv_context* context, uint8_t port, struct ibv_port_attr* attributes) {
  (void)context;
  (void)port;
  state.queryPortCalls++;
  if (state.queryPortResult != ncclSuccess) return state.queryPortResult;
  std::memset(attributes, 0, sizeof(*attributes));
  attributes->state = IBV_PORT_ACTIVE;
  attributes->link_layer = IBV_LINK_LAYER_ETHERNET;
  attributes->active_width = state.activeWidth;
  attributes->active_speed = state.activeSpeed;
  attributes->active_speed_ex = state.activeSpeedEx;
  return ncclSuccess;
}

/**
 * @brief Returns the scripted mlx5 provider-support decision.
 * @param[in] device Ignored borrowed device.
 * @return state.providerSupported.
 * @note Side effects: None.
 */
bool wrap_mlx5dv_is_supported(struct ibv_device* device) {
  (void)device;
  return state.providerSupported;
}

/**
 * @brief Supplies a scripted data-direct sysfs path.
 * @param[in] context Ignored borrowed context.
 * @param[out] path Valid output storage written on success.
 * @param[in] size Capacity of path.
 * @return state.dataDirectPathResult.
 * @note Side effects: Records the call and may write path.
 */
ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(
    struct ibv_context* context, char* path, size_t size) {
  (void)context;
  state.dataDirectPathCalls++;
  if (state.dataDirectPathResult == ncclSuccess) std::snprintf(path, size, "/devices/data-direct0");
  return state.dataDirectPathResult;
}

/**
 * @brief Returns the scripted relaxed-ordering registration result.
 * @param[out] memoryRegion Ignored registration output.
 * @param[in] protectionDomain Ignored borrowed protection domain.
 * @param[in] address Ignored buffer address.
 * @param[in] length Ignored buffer length.
 * @param[in] iova Ignored I/O virtual address.
 * @param[in] access Ignored access mask.
 * @return state.relaxedOrderingResult.
 * @note Side effects: Increments relaxedOrderingCalls.
 */
ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr** memoryRegion,
    struct ibv_pd* protectionDomain, void* address, size_t length,
    uint64_t iova, int access) {
  (void)memoryRegion;
  (void)protectionDomain;
  (void)address;
  (void)length;
  (void)iova;
  (void)access;
  state.relaxedOrderingCalls++;
  return state.relaxedOrderingResult;
}

/**
 * @brief Supplies a scripted protection-domain allocation.
 * @param[out] pd Valid output storage written on success.
 * @param[in] context Borrowed context recorded for verification.
 * @return state.allocPdResult.
 * @note Side effects: Records the call and may write pd.
 */
ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd** pd, struct ibv_context* context) {
  state.allocPdCalls++;
  state.allocatedPdContext = context;
  if (state.allocPdResult == ncclSuccess) *pd = &state.protectionDomain;
  return state.allocPdResult;
}

/**
 * @brief Supplies a scripted protection-domain release.
 * @param[in] pd Borrowed protection domain recorded for verification.
 * @return state.deallocPdResult.
 * @note Side effects: Records pd and increments deallocPdCalls.
 */
ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd* pd) {
  state.deallocPdCalls++;
  state.deallocatedPd = pd;
  return state.deallocPdResult;
}

/**
 * @brief Records a generic DMA-BUF capability probe and scripts errno.
 * @param[in] pd Borrowed protection domain.
 * @param[in] offset DMA-BUF offset.
 * @param[in] length Registration length.
 * @param[in] iova I/O virtual address.
 * @param[in] fd DMA-BUF descriptor.
 * @param[in] flags Access flags.
 * @return NULL.
 * @note Side effects: Records all arguments, increments the call count, and sets errno.
 */
struct ibv_mr* wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset,
    size_t length, uint64_t iova, int fd, int flags) {
  state.genericDmaBufCalls++;
  state.genericDmaBufPd = pd;
  state.genericDmaBufOffset = offset;
  state.genericDmaBufLength = length;
  state.genericDmaBufIova = iova;
  state.genericDmaBufFd = fd;
  state.genericDmaBufFlags = flags;
  errno = state.genericDmaBufErrno;
  return nullptr;
}

/**
 * @brief Records an mlx5 DMA-BUF capability probe and scripts errno.
 * @param[in] pd Borrowed protection domain.
 * @param[in] offset DMA-BUF offset.
 * @param[in] length Registration length.
 * @param[in] iova I/O virtual address.
 * @param[in] fd DMA-BUF descriptor.
 * @param[in] accessFlags Registration access flags.
 * @param[in] mlx5Flags Provider-specific flags.
 * @return NULL.
 * @note Side effects: Records all arguments, increments the call count, and sets errno.
 */
struct ibv_mr* wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset,
    size_t length, uint64_t iova, int fd, int accessFlags, int mlx5Flags) {
  state.mlx5DmaBufCalls++;
  state.mlx5DmaBufPd = pd;
  state.mlx5DmaBufOffset = offset;
  state.mlx5DmaBufLength = length;
  state.mlx5DmaBufIova = iova;
  state.mlx5DmaBufFd = fd;
  state.mlx5DmaBufAccessFlags = accessFlags;
  state.mlx5DmaBufFlags = mlx5Flags;
  errno = state.mlx5DmaBufErrno;
  return nullptr;
}

/**
 * @brief Supplies a no-op asynchronous progress thread entry point.
 * @param[in] argument Ignored thread argument.
 * @return NULL.
 * @note Side effects: None.
 */
void* ncclIbAsyncThreadMain(void* argument) {
  (void)argument;
  return nullptr;
}

/**
 * @brief Compiles init.cc with parameter loading, path resolution, and allocation redirected
 * to deterministic seams.
 */
#define ncclLoadParam LoadInitParameterForTest
#define realpath RealpathForTest
#undef ncclCalloc
#define ncclCalloc(pointer, count) \
  AllocateForTest(reinterpret_cast<void**>(pointer), sizeof(**(pointer)) * (count))
#include "../init.cc"  // NOLINT(build/include)
#undef ncclCalloc
#undef realpath
#undef ncclLoadParam

namespace {

/**
 * @brief Restores test controls and production discovery, reference-count, timing, and
 * device globals.
 * @note Side effects: Replaces state and resets process-global initialization fields
 * exercised by the tests.
 */
void ResetGlobalState() {
  state = InitTestState{};
  ncclNIbDevs = 0;
  ncclNMergedIbDevs = 0;
  ncclIbRelaxedOrderingEnabled = 0;
  ncclProfilerFunction = nullptr;
  netRefCount = 0;
  g_inv_cyc_per_us = 0.0;
  g_ncclIbUseCim = false;
  std::memset(ncclIbIfName, 0, sizeof(ncclIbIfName));
  std::memset(&ncclIbIfAddr, 0, sizeof(ncclIbIfAddr));
  std::memset(ncclIbMergedDevs, 0, sizeof(ncclIbMergedDevs));
  for (auto& device : ncclIbDevs) {
    device.device = 0;
    device.guid = 0;
    device.portNum = 0;
    device.link = 0;
    device.speed = 0;
    device.context = nullptr;
    device.pdRefs = 0;
    device.pd = nullptr;
    device.devName[0] = '\0';
    device.fullPciPath[0] = '\0';
    device.pciPath = nullptr;
    device.realPort = 0;
    device.maxQp = 0;
    device.latency = 0.0f;
    device.ar = 0;
    device.portAttr = {};
    device.stats.fatalErrorCount = 0;
    device.dmaBufSupported = 0;
    device.ibProvider = IB_PROVIDER_NONE;
    device.capsProvider.mlx5.dataDirect = 0;
  }
}

/**
 * @brief Checks direct member validation and preservation of caller-owned properties.
 * @param[in] properties Properties to validate, or NULL for the null-input case.
 * @param[in] expected Expected validation result.
 * @param[in] context Non-NULL diagnostic describing the validation scenario.
 * @note Side effects: Records GoogleTest failures and may emit an expected production warning.
 */
void ExpectValidationResult(const ncclNetVDeviceProps_t* properties,
    ncclResult_t expected, const char* context) {
  unsigned char before[sizeof(ncclNetVDeviceProps_t)] = {};
  if (properties != nullptr) std::memcpy(before, properties, sizeof(*properties));
  EXPECT_EQ(occlIbValidateVDeviceMembers(properties), expected)
      << context << ": validation returned an unexpected status";
  if (properties != nullptr) {
    EXPECT_EQ(std::memcmp(properties, before, sizeof(*properties)), 0)
        << context << ": validation modified caller-owned properties";
  }
}

/**
 * @brief Builds the canonical one-device discovery fixture for success and injected-failure
 * scenarios.
 * @note Side effects: Resets global state and configures dependencies to discover one active
 * Ethernet device.
 */
void PrepareSingleDeviceDiscovery() {
  ResetGlobalState();
  ncclNIbDevs = -1;
  ncclNMergedIbDevs = -1;
  state.parameters["NCCL_IB_DISABLE"] = 0;
  state.parameters["NCCL_IB_PCI_RELAXED_ORDERING"] = 0;
  state.deviceCount = 1;
  state.realpathSucceeds = true;
  state.resolvedPath = "/sys/devices/pci0000:00/0000:03:04.7";
  std::snprintf(state.device.name, sizeof(state.device.name), "test0");
  std::snprintf(state.device.dev_name, sizeof(state.device.dev_name), "uverbs0");
}

}  // namespace

/**
 * @brief Verifies every generated initialization parameter accessor's mapping, default, value,
 * and cache contract.
 * @note Side effects: Records fake parameter loads and checks caching in an isolated child process.
 */
TEST(NetIbInitParameterTest, LoadsGeneratedAccessors) {
  ResetGlobalState();

  EXPECT_EQ(ncclParamIbPciRelaxedOrdering(), 2)
      << "IB PCI relaxed ordering must retain its automatic-mode default";
  EXPECT_EQ(ncclParamIbAdaptiveRouting(), -2)
      << "IB adaptive routing must retain its automatic-selection default";
  EXPECT_EQ(ncclParamIbDataDirect(), 1)
      << "IB data-direct support must remain enabled by default";
  EXPECT_EQ(ncclParamOciFrmEnable(), 1)
      << "OCI FRM support must remain enabled by default";
  EXPECT_EQ(ncclParamIbDisable(), 0)
      << "The IB transport must remain enabled by default";
  EXPECT_EQ(ncclParamIbMergeVfs(), 1)
      << "IB virtual-function merging must remain enabled by default";
  EXPECT_EQ(ncclParamIbMergeNics(), 1)
      << "IB NIC merging must remain enabled by default";
  EXPECT_EQ(ncclParamIbDevicePciOrder(), 1)
      << "IB PCI device ordering must remain enabled by default";

  const InitParameterLoadCall expectedLoads[] = {
      {"NCCL_IB_PCI_RELAXED_ORDERING", 2},
      {"NCCL_IB_ADAPTIVE_ROUTING", -2},
      {"NCCL_IB_DATA_DIRECT", 1},
      {"NCCL_OCI_FRM_ENABLE", 1},
      {"NCCL_IB_DISABLE", 0},
      {"NCCL_IB_MERGE_VFS", 1},
      {"NCCL_IB_MERGE_NICS", 1},
      {"NCCL_IB_DEVICE_PCI_ORDER", 1},
  };
  ASSERT_EQ(state.parameterLoads.size(), std::size(expectedLoads))
      << "Every direct accessor call must perform exactly one uncached parameter load";
  for (size_t index = 0; index < std::size(expectedLoads); ++index) {
    SCOPED_TRACE(testing::Message() << "default parameter load " << index);
    EXPECT_EQ(state.parameterLoads[index].environment, expectedLoads[index].environment)
        << "The accessor must request its documented environment variable";
    EXPECT_EQ(state.parameterLoads[index].defaultValue, expectedLoads[index].defaultValue)
        << "The accessor must forward its documented production default";
  }

  state.parameterLoads.clear();
  state.parameters = {
      {"NCCL_IB_PCI_RELAXED_ORDERING", 0},
      {"NCCL_IB_ADAPTIVE_ROUTING", 1},
      {"NCCL_IB_DATA_DIRECT", 2},
      {"NCCL_OCI_FRM_ENABLE", 0},
      {"NCCL_IB_DISABLE", 1},
      {"NCCL_IB_MERGE_VFS", 0},
      {"NCCL_IB_MERGE_NICS", 0},
      {"NCCL_IB_DEVICE_PCI_ORDER", 0},
  };
  EXPECT_EQ(ncclParamIbPciRelaxedOrdering(), 0)
      << "IB PCI relaxed ordering must return its configured value";
  EXPECT_EQ(ncclParamIbAdaptiveRouting(), 1)
      << "IB adaptive routing must return its configured value";
  EXPECT_EQ(ncclParamIbDataDirect(), 2)
      << "IB data-direct support must return its configured value";
  EXPECT_EQ(ncclParamOciFrmEnable(), 0)
      << "OCI FRM support must return its configured value";
  EXPECT_EQ(ncclParamIbDisable(), 1)
      << "IB transport disabling must return its configured value";
  EXPECT_EQ(ncclParamIbMergeVfs(), 0)
      << "IB virtual-function merging must return its configured value";
  EXPECT_EQ(ncclParamIbMergeNics(), 0)
      << "IB NIC merging must return its configured value";
  EXPECT_EQ(ncclParamIbDevicePciOrder(), 0)
      << "IB PCI device ordering must return its configured value";
  ASSERT_EQ(state.parameterLoads.size(), std::size(expectedLoads))
      << "Every configured accessor call must perform exactly one uncached parameter load";
  for (size_t index = 0; index < std::size(expectedLoads); ++index) {
    SCOPED_TRACE(testing::Message() << "configured parameter load " << index);
    EXPECT_EQ(state.parameterLoads[index].environment, expectedLoads[index].environment)
        << "Configured lookup must use the same documented environment variable";
    EXPECT_EQ(state.parameterLoads[index].defaultValue, expectedLoads[index].defaultValue)
        << "Configured lookup must retain the production fallback default";
  }

  EXPECT_EXIT(
      {
        ResetGlobalState();
        state.cacheParameterValues = true;
        state.parameters["NCCL_IB_DISABLE"] = 1;
        const int64_t firstValue = ncclParamIbDisable();
        state.parameters["NCCL_IB_DISABLE"] = 0;
        const int64_t secondValue = ncclParamIbDisable();
        if (firstValue != 1 || secondValue != 1 || state.parameterLoads.size() != 1) {
          std::fprintf(stderr,
              "Cached accessor mismatch: first=%lld second=%lld loads=%zu; expected 1, 1, 1\n",
              static_cast<long long>(firstValue), static_cast<long long>(secondValue),
              state.parameterLoads.size());
          std::_Exit(1);
        }
        std::_Exit(0);
      },
      testing::ExitedWithCode(0), ".*")
      << "A generated accessor must load once and retain its first value when caching is enabled";
}

/**
 * @brief Verifies link-rate tables, PCI path comparison and normalization, and physical-port
 * numbering.
 * @note Side effects: Mutates fake parameter, path-provider, errno, and device state; writes
 * caller outputs and allocates and frees normalized paths.
 */
TEST(NetIbInitTest, LinkAndPathHelpers) {
  SCOPED_TRACE(
      "Link and path helpers must implement tables, outputs, and read-only inputs.");
  ResetGlobalState();

  EXPECT_EQ(firstBitSet(0, 0), 0);
  EXPECT_EQ(firstBitSet(0, 4), 4);
  EXPECT_EQ(firstBitSet(0b10100, 5), 2);
  EXPECT_EQ(ncclIbWidth(1), 1);
  EXPECT_EQ(ncclIbWidth(2), 4);
  EXPECT_EQ(ncclIbWidth(4), 8);
  EXPECT_EQ(ncclIbWidth(8), 12);
  EXPECT_EQ(ncclIbWidth(16), 2);
  EXPECT_EQ(ncclIbWidth(0), 2);
  EXPECT_EQ(ncclIbWidth(9), 1);

  EXPECT_EQ(ncclIbSpeed(1), 2500);
  EXPECT_EQ(ncclIbSpeed(2), 5000);
  EXPECT_EQ(ncclIbSpeed(4), 10000);
  EXPECT_EQ(ncclIbSpeed(8), 10000);
  EXPECT_EQ(ncclIbSpeed(16), 14000);
  EXPECT_EQ(ncclIbSpeed(32), 25000);
  EXPECT_EQ(ncclIbSpeed(64), 50000);
  EXPECT_EQ(ncclIbSpeed(128), 100000);
  EXPECT_EQ(ncclIbSpeed(256), 200000);
  EXPECT_EQ(ncclIbSpeed(0), 200000);
  EXPECT_EQ(ncclIbSpeed(66), 5000);

  char samePhysical1[] = "/sys/devices/0000:01:00.1";
  char samePhysical2[] = "/sys/devices/0000:01:00.7";
  char differentPhysical[] = "/sys/devices/0000:02:00.1";
  const std::string samePhysical1Before = samePhysical1;
  const std::string samePhysical2Before = samePhysical2;
  const std::string differentPhysicalBefore = differentPhysical;
  state.parameters["NCCL_IB_MERGE_VFS"] = 1;
  EXPECT_EQ(ncclIbMatchVfPath(samePhysical1, samePhysical2), 1);
  EXPECT_EQ(ncclIbMatchVfPath(samePhysical1, differentPhysical), 0);
  state.parameters["NCCL_IB_MERGE_VFS"] = 0;
  EXPECT_EQ(ncclIbMatchVfPath(samePhysical1, samePhysical2), 1);
  EXPECT_EQ(ncclIbMatchVfPath(samePhysical1, differentPhysical), 0);
  EXPECT_EQ(samePhysical1, samePhysical1Before)
      << "VF path matching must preserve the first caller-owned path";
  EXPECT_EQ(samePhysical2, samePhysical2Before)
      << "VF path matching must preserve the second caller-owned path";
  EXPECT_EQ(differentPhysical, differentPhysicalBefore)
      << "VF path matching must preserve a nonmatching caller-owned path";

  std::snprintf(ncclIbDevs[0].fullPciPath, PATH_MAX, "/sys/devices/0000:01:00.0");
  std::snprintf(ncclIbDevs[1].fullPciPath, PATH_MAX, "/sys/devices/0000:02:00.0");
  ncclIbDevs[2].fullPciPath[0] = '\0';
  EXPECT_LT(ncclIbCompareDevs(&ncclIbDevs[0], &ncclIbDevs[1]), 0);
  EXPECT_GT(ncclIbCompareDevs(&ncclIbDevs[1], &ncclIbDevs[0]), 0);
  EXPECT_EQ(ncclIbCompareDevs(&ncclIbDevs[0], &ncclIbDevs[0]), 0);
  EXPECT_LT(ncclIbCompareDevs(&ncclIbDevs[0], &ncclIbDevs[2]), 0);
  EXPECT_GT(ncclIbCompareDevs(&ncclIbDevs[2], &ncclIbDevs[0]), 0);
  EXPECT_EQ(ncclIbCompareDevs(&ncclIbDevs[2], &ncclIbDevs[2]), 0);
  EXPECT_STREQ(ncclIbDevs[0].fullPciPath, "/sys/devices/0000:01:00.0")
      << "Device comparison must preserve the first PCI path";
  EXPECT_STREQ(ncclIbDevs[1].fullPciPath, "/sys/devices/0000:02:00.0")
      << "Device comparison must preserve the second PCI path";

  state.parameters["NCCL_IB_MERGE_VFS"] = 1;
  state.realpathSucceeds = true;
  state.resolvedPath = "/sys/devices/pci0000:00/0000:03:04.7";
  char deviceName[] = "mlx5_0";
  const std::string deviceNameBefore = deviceName;
  char* normalizedPath = nullptr;
  char fullPath[PATH_MAX] = {};
  ASSERT_EQ(ncclIbGetPciPath(deviceName, &normalizedPath, fullPath), ncclSuccess);
  ASSERT_NE(normalizedPath, nullptr);
  EXPECT_STREQ(state.requestedPath.c_str(), "/sys/class/infiniband/mlx5_0/device");
  EXPECT_STREQ(fullPath, "/sys/devices/pci0000:00/0000:03:04.7");
  EXPECT_STREQ(normalizedPath, "/sys/devices/pci0000:00/0000:03:00.0");
  std::free(normalizedPath);

  state.parameters["NCCL_IB_MERGE_VFS"] = 0;
  normalizedPath = nullptr;
  ASSERT_EQ(ncclIbGetPciPath(deviceName, &normalizedPath, fullPath), ncclSuccess);
  ASSERT_NE(normalizedPath, nullptr);
  EXPECT_STREQ(fullPath, "/sys/devices/pci0000:00/0000:03:04.7");
  EXPECT_STREQ(normalizedPath, "/sys/devices/pci0000:00/0000:03:04.0");
  std::free(normalizedPath);
  EXPECT_EQ(deviceName, deviceNameBefore)
      << "PCI-path discovery must preserve the caller-owned device name";

  state.realpathSucceeds = false;
  normalizedPath = reinterpret_cast<char*>(1);
  fullPath[0] = 'x';
  char missingDeviceName[] = "missing";
  const std::string missingDeviceNameBefore = missingDeviceName;
  ASSERT_EQ(ncclIbGetPciPath(missingDeviceName, &normalizedPath, fullPath), ncclSuccess);
  EXPECT_EQ(normalizedPath, nullptr);
  EXPECT_STREQ(fullPath, "");
  EXPECT_EQ(missingDeviceName, missingDeviceNameBefore)
      << "Failed PCI-path discovery must preserve the caller-owned device name";

  state.parameters["NCCL_IB_MERGE_VFS"] = 1;
  char previousPath0[] = "/sys/devices/0000:01:00.0";
  char previousPath1[] = "/sys/devices/0000:02:00.0";
  char currentPath[] = "/sys/devices/0000:01:00.7";
  const std::string previousPath0Before = previousPath0;
  const std::string previousPath1Before = previousPath1;
  const std::string currentPathBefore = currentPath;
  ncclIbDevs[0].pciPath = previousPath0;
  ncclIbDevs[1].pciPath = previousPath1;
  ncclNIbDevs = 3;
  int realPort = -1;
  EXPECT_EQ(ncclIbGetRealPort(currentPath, &realPort, 0), ncclSuccess);
  EXPECT_EQ(realPort, 0);
  EXPECT_EQ(ncclIbGetRealPort(currentPath, &realPort, 2), ncclSuccess);
  EXPECT_EQ(realPort, 1);
  ncclIbDevs[1].pciPath = previousPath0;
  EXPECT_EQ(ncclIbGetRealPort(currentPath, &realPort, 2), ncclSuccess);
  EXPECT_EQ(realPort, 2);
  EXPECT_EQ(ncclIbGetRealPort(nullptr, &realPort, 2), ncclSuccess);
  EXPECT_EQ(realPort, 0);
  EXPECT_EQ(previousPath0, previousPath0Before)
      << "Physical-port numbering must preserve the first discovered PCI path";
  EXPECT_EQ(previousPath1, previousPath1Before)
      << "Physical-port numbering must preserve the second discovered PCI path";
  EXPECT_EQ(currentPath, currentPathBefore)
      << "Physical-port numbering must preserve the current caller-owned PCI path";
}

/**
 * @brief Verifies relaxed-ordering and DMA-BUF capability decisions and provider resource handling.
 * @note Side effects: Configures and records fake verbs-provider calls and updates errno.
 */
TEST(NetIbInitTest, ProviderCapabilityProbes) {
  SCOPED_TRACE(
      "Capability probes must classify provider results and release acquired resources.");
  ResetGlobalState();

  state.parameters["NCCL_IB_PCI_RELAXED_ORDERING"] = 0;
  state.relaxedOrderingResult = ncclSuccess;
  EXPECT_EQ(ncclIbRelaxedOrderingCapable(), 0);
  EXPECT_EQ(state.relaxedOrderingCalls, 0);
  state.parameters["NCCL_IB_PCI_RELAXED_ORDERING"] = 2;
  state.relaxedOrderingResult = ncclInternalError;
  EXPECT_EQ(ncclIbRelaxedOrderingCapable(), 0);
  EXPECT_EQ(state.relaxedOrderingCalls, 1);
  state.parameters["NCCL_IB_PCI_RELAXED_ORDERING"] = 1;
  state.relaxedOrderingResult = ncclSuccess;
  EXPECT_EQ(ncclIbRelaxedOrderingCapable(), 1);
  EXPECT_EQ(state.relaxedOrderingCalls, 2);

  state.allocPdResult = ncclSystemError;
  EXPECT_FALSE(ncclMlx5dvDmaBufCapable(&state.context));
  EXPECT_EQ(state.allocPdCalls, 1);
  EXPECT_EQ(state.genericDmaBufCalls, 0);
  EXPECT_EQ(state.mlx5DmaBufCalls, 0);
  EXPECT_EQ(state.deallocPdCalls, 0);

  state.allocPdResult = ncclSuccess;
  state.deallocPdResult = ncclSuccess;
  state.genericDmaBufErrno = EBADF;
  state.mlx5DmaBufErrno = EBADF;
  EXPECT_TRUE(ncclMlx5dvDmaBufCapable(&state.context));
  EXPECT_EQ(state.allocatedPdContext, &state.context)
      << "The capability probe must allocate its protection domain from the supplied context";
  EXPECT_EQ(state.genericDmaBufPd, &state.protectionDomain)
      << "The generic DMA-BUF probe must use the allocated protection domain";
  EXPECT_EQ(state.genericDmaBufOffset, 0u)
      << "The generic capability probe must use a zero dummy offset";
  EXPECT_EQ(state.genericDmaBufLength, 0u)
      << "The generic capability probe must use a zero dummy length";
  EXPECT_EQ(state.genericDmaBufIova, 0u)
      << "The generic capability probe must use a zero dummy IOVA";
  EXPECT_EQ(state.genericDmaBufFd, -1)
      << "The generic capability probe must use the documented invalid descriptor";
  EXPECT_EQ(state.genericDmaBufFlags, 0)
      << "The generic capability probe must not request access flags";
  EXPECT_EQ(state.mlx5DmaBufPd, &state.protectionDomain)
      << "The mlx5 DMA-BUF probe must use the allocated protection domain";
  EXPECT_EQ(state.mlx5DmaBufOffset, 0u)
      << "The mlx5 capability probe must use a zero dummy offset";
  EXPECT_EQ(state.mlx5DmaBufLength, 0u)
      << "The mlx5 capability probe must use a zero dummy length";
  EXPECT_EQ(state.mlx5DmaBufIova, 0u)
      << "The mlx5 capability probe must use a zero dummy IOVA";
  EXPECT_EQ(state.mlx5DmaBufFd, -1)
      << "The mlx5 capability probe must use the documented invalid descriptor";
  EXPECT_EQ(state.mlx5DmaBufAccessFlags, 0)
      << "The mlx5 capability probe must not request access flags";
  EXPECT_EQ(state.mlx5DmaBufFlags, 0)
      << "The mlx5 capability probe must not request provider-specific flags";
  EXPECT_EQ(state.deallocatedPd, &state.protectionDomain)
      << "A successful capability probe must release the allocated protection domain";
  EXPECT_EQ(state.genericDmaBufCalls, 1);
  EXPECT_EQ(state.mlx5DmaBufCalls, 1);
  EXPECT_EQ(state.deallocPdCalls, 1);

  state.genericDmaBufErrno = EOPNOTSUPP;
  state.mlx5DmaBufErrno = EBADF;
  EXPECT_TRUE(ncclMlx5dvDmaBufCapable(&state.context))
      << "mlx5 registration must provide a fallback when generic registration is unsupported";
  EXPECT_EQ(state.allocPdCalls, 3);
  EXPECT_EQ(state.genericDmaBufCalls, 2);
  EXPECT_EQ(state.mlx5DmaBufCalls, 2);
  EXPECT_EQ(state.deallocPdCalls, 2)
      << "A successful mlx5 fallback must release the allocated protection domain";

  state.genericDmaBufErrno = EBADF;
  state.mlx5DmaBufErrno = EOPNOTSUPP;
  EXPECT_FALSE(ncclMlx5dvDmaBufCapable(&state.context))
      << "Generic-only support cannot replace the required mlx5 registration path";
  EXPECT_EQ(state.allocPdCalls, 4);
  EXPECT_EQ(state.genericDmaBufCalls, 3);
  EXPECT_EQ(state.mlx5DmaBufCalls, 3);
  EXPECT_EQ(state.deallocPdCalls, 3);

  state.mlx5DmaBufErrno = EPROTONOSUPPORT;
  EXPECT_FALSE(ncclMlx5dvDmaBufCapable(&state.context));
  EXPECT_EQ(state.deallocPdCalls, 4);

  state.mlx5DmaBufErrno = EBADF;
  state.deallocPdResult = ncclSystemError;
  EXPECT_FALSE(ncclMlx5dvDmaBufCapable(&state.context));
  EXPECT_EQ(state.allocPdCalls, 6);
  EXPECT_EQ(state.genericDmaBufCalls, 5);
  EXPECT_EQ(state.mlx5DmaBufCalls, 5);
  EXPECT_EQ(state.deallocPdCalls, 5);
}

/**
 * @brief Verifies virtual-device validation, capacity, link compatibility, output mutation,
 * and success state.
 * @note Side effects: Rebuilds the fake physical and merged device inventories.
 */
TEST(NetIbInitTest, VirtualDeviceBehavior) {
  SCOPED_TRACE(
      "Virtual-device creation must reject invalid topology and build valid devices.");
  ResetGlobalState();
  ncclNIbDevs = 1;
  int virtualDevice = -1;
  ncclNetVDeviceProps_t properties = {};

  ExpectValidationResult(nullptr, ncclInvalidUsage, "Missing properties");

  properties.ndevs = -1;
  ExpectValidationResult(&properties, ncclInvalidUsage, "Negative member count");
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&virtualDevice, &properties), ncclInvalidUsage)
      << "a negative physical-device count must be rejected";
  EXPECT_EQ(virtualDevice, -1) << "a rejected negative count must preserve the output device index";
  EXPECT_EQ(ncclNMergedIbDevs, 0) << "a rejected negative count must not publish a virtual device";

  ResetGlobalState();
  ncclNIbDevs = 1;
  virtualDevice = -1;
  properties = {};
  properties.ndevs = 1;
  properties.devs[0] = -1;
  ExpectValidationResult(&properties, ncclInvalidUsage, "Negative member index");
  properties.devs[0] = ncclNIbDevs;
  ExpectValidationResult(&properties, ncclInvalidUsage, "Upper-bound member index");
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&virtualDevice, &properties), ncclInvalidUsage)
      << "the first device index must be rejected when it is outside the discovered-device range";
  EXPECT_EQ(virtualDevice, -1) << "a rejected first device must preserve the output device index";
  EXPECT_EQ(ncclNMergedIbDevs, 0) << "a rejected first device must not publish a virtual device";

  ResetGlobalState();
  ncclNIbDevs = 1;
  virtualDevice = -1;
  properties = {};
  properties.ndevs = 0;
  ExpectValidationResult(&properties, ncclInvalidUsage, "Empty member set");
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&virtualDevice, &properties), ncclInvalidUsage);
  properties.ndevs = NCCL_IB_MAX_DEVS_PER_NIC + 1;
  ExpectValidationResult(&properties, ncclInvalidUsage, "Excess member count");
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&virtualDevice, &properties), ncclInvalidUsage);
  EXPECT_EQ(virtualDevice, -1);
  EXPECT_EQ(ncclNMergedIbDevs, 0);

  ResetGlobalState();
  ncclNIbDevs = NCCL_IB_MAX_DEVS_PER_NIC;
  properties = {};
  properties.ndevs = NCCL_IB_MAX_DEVS_PER_NIC;
  for (int member = 0; member < properties.ndevs; ++member) {
    properties.devs[member] = member;
    ncclIbDevs[member].link = IBV_LINK_LAYER_INFINIBAND;
  }
  ExpectValidationResult(&properties, ncclSuccess, "Maximum valid member set");

  ResetGlobalState();
  ncclNIbDevs = 1;
  virtualDevice = -1;

  properties = {};
  properties.ndevs = 2;
  properties.devs[0] = 0;
  properties.devs[1] = 0;
  ExpectValidationResult(&properties, ncclInvalidUsage, "Duplicate members");
  state.parameters["NCCL_IB_MERGE_NICS"] = 0;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&virtualDevice, &properties), ncclInvalidUsage);

  state.parameters["NCCL_IB_MERGE_NICS"] = 1;
  ncclNMergedIbDevs = MAX_IB_VDEVS;
  properties.ndevs = 1;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&virtualDevice, &properties), ncclInvalidUsage);

  ResetGlobalState();
  ncclNIbDevs = 2;
  ncclIbDevs[0].link = IBV_LINK_LAYER_INFINIBAND;
  ncclIbDevs[1].link = IBV_LINK_LAYER_ETHERNET;
  std::snprintf(ncclIbDevs[0].devName, MAXNAMESIZE, "rail0");
  std::snprintf(ncclIbDevs[1].devName, MAXNAMESIZE, "rail1");
  properties = {};
  properties.ndevs = 2;
  properties.devs[0] = 0;
  properties.devs[1] = 1;
  ExpectValidationResult(&properties, ncclInvalidUsage, "Mixed link layers");
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&virtualDevice, &properties), ncclInvalidUsage);
  EXPECT_EQ(virtualDevice, -1);
  EXPECT_EQ(ncclNMergedIbDevs, 0);

  ResetGlobalState();
  ncclNIbDevs = 1;
  ncclIbDevs[0].speed = 100;
  ncclIbDevs[0].link = IBV_LINK_LAYER_ETHERNET;
  std::snprintf(ncclIbDevs[0].devName, MAXNAMESIZE, "rail0");
  state.parameters["NCCL_IB_MERGE_NICS"] = 0;
  properties = {};
  properties.ndevs = 1;
  properties.devs[0] = 0;
  unsigned char singleDeviceInputBefore[sizeof(properties)];
  std::memcpy(singleDeviceInputBefore, &properties, sizeof(properties));
  ExpectValidationResult(&properties, ncclSuccess, "Single valid member");
  ASSERT_EQ(ncclIbMakeVDevice(&virtualDevice, &properties), ncclSuccess);
  EXPECT_EQ(std::memcmp(&properties, singleDeviceInputBefore, sizeof(properties)), 0)
      << "Successful single-device creation must preserve caller-owned topology input";
  EXPECT_EQ(virtualDevice, 0);
  EXPECT_EQ(ncclNMergedIbDevs, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.ndevs, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.devs[0], 0);
  EXPECT_EQ(ncclIbMergedDevs[0].publicVProps.ndevs, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].publicVProps.devs[0], 0);
  EXPECT_EQ(ncclIbMergedDevs[0].anchorDev, 0);
  EXPECT_EQ(ncclIbMergedDevs[0].firstAlias, 0);
  EXPECT_EQ(ncclIbMergedDevs[0].aliasCount, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].isForcedAlias, 0);
  EXPECT_STREQ(ncclIbMergedDevs[0].devName, "rail0");
  EXPECT_EQ(ncclIbMergedDevs[0].speed, 100);

  ResetGlobalState();
  ncclNIbDevs = 3;
  ncclIbDevs[0].speed = 100;
  ncclIbDevs[2].speed = 300;
  ncclIbDevs[0].link = IBV_LINK_LAYER_ETHERNET;
  ncclIbDevs[2].link = IBV_LINK_LAYER_ETHERNET;
  std::snprintf(ncclIbDevs[0].devName, MAXNAMESIZE, "rail0");
  std::snprintf(ncclIbDevs[2].devName, MAXNAMESIZE, "rail2");
  properties = {};
  properties.ndevs = 2;
  properties.devs[0] = 2;
  properties.devs[1] = 0;
  unsigned char mergedInputBefore[sizeof(properties)];
  std::memcpy(mergedInputBefore, &properties, sizeof(properties));
  ExpectValidationResult(&properties, ncclSuccess, "Valid same-link members");
  ASSERT_EQ(ncclIbMakeVDevice(&virtualDevice, &properties), ncclSuccess);
  EXPECT_EQ(std::memcmp(&properties, mergedInputBefore, sizeof(properties)), 0)
      << "Successful multi-device creation must preserve caller-owned topology input";
  EXPECT_EQ(virtualDevice, 0);
  EXPECT_EQ(ncclNMergedIbDevs, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.ndevs, 2);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.devs[0], 2);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.devs[1], 0);
  EXPECT_EQ(ncclIbMergedDevs[0].publicVProps.ndevs, 2);
  EXPECT_EQ(ncclIbMergedDevs[0].publicVProps.devs[0], 2);
  EXPECT_EQ(ncclIbMergedDevs[0].publicVProps.devs[1], 0);
  EXPECT_EQ(ncclIbMergedDevs[0].anchorDev, 2);
  EXPECT_EQ(ncclIbMergedDevs[0].firstAlias, 0);
  EXPECT_EQ(ncclIbMergedDevs[0].aliasCount, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].isForcedAlias, 0);
  EXPECT_STREQ(ncclIbMergedDevs[0].devName, "rail2+rail0");
  EXPECT_EQ(ncclIbMergedDevs[0].speed, 400);
}

/**
 * @brief Verifies device-count, net-attribute, physical-property, and merged-property contracts.
 * @note Side effects: Populates fake device inventories and records capability-provider calls.
 */
TEST(NetIbInitTest, PropertyBehavior) {
  SCOPED_TRACE(
      "Property APIs must return exact data and preserve outputs on rejection.");
  ResetGlobalState();

  int deviceCount = 17;
  ncclNMergedIbDevs = -1;
  EXPECT_EQ(ncclIbDevices(&deviceCount), ncclSuccess);
  EXPECT_EQ(deviceCount, -1);
  int contextToken = 7;
  ncclNetAttr_t attributes;
  ncclNetAttr_t expectedAttributes;
  std::memset(&attributes, 0xa5, sizeof(attributes));
  std::memcpy(&expectedAttributes, &attributes, sizeof(attributes));
  EXPECT_EQ(ncclIbSetNetAttr(&contextToken, &attributes), ncclSuccess);
  EXPECT_EQ(contextToken, 7)
      << "The no-op net-attribute setter must preserve caller-owned context storage";
  EXPECT_EQ(std::memcmp(&attributes, &expectedAttributes, sizeof(attributes)), 0);

  ncclNIbDevs = 1;
  ncclNMergedIbDevs = 1;
  std::snprintf(ncclIbDevs[0].devName, MAXNAMESIZE, "physical0");
  char pciPath[] = "/sys/devices/0000:01:00.0";
  ncclIbDevs[0].pciPath = pciPath;
  ncclIbDevs[0].speed = 100000;
  ncclIbDevs[0].guid = 42;
  ncclIbDevs[0].portNum = 2;
  ncclIbDevs[0].realPort = 3;
  ncclIbDevs[0].maxQp = 64;
  ncclIbDevs[0].capsProvider.mlx5.dataDirect = 1;
  ncclIbMergedDevs[0].vProps.ndevs = 1;
  ncclIbMergedDevs[0].vProps.devs[0] = 0;
  ncclIbMergedDevs[0].publicVProps = ncclIbMergedDevs[0].vProps;
  ncclIbMergedDevs[0].anchorDev = 0;
  ncclIbMergedDevs[0].speed = 125000;
  std::snprintf(ncclIbMergedDevs[0].devName, MAX_MERGED_DEV_NAME, "virtual0");
  state.gdrSupportResult = ncclSystemError;
  state.dmaBufSupportResult = ncclSuccess;

  ncclNetProperties_t physicalProperties = {};
  physicalProperties.vProps.ndevs = 3;
  physicalProperties.vProps.devs[0] = 9;
  ASSERT_EQ(ncclIbGetPhysProperties(0, &physicalProperties), ncclSuccess);
  EXPECT_STREQ(physicalProperties.name, "physical0");
  EXPECT_EQ(physicalProperties.pciPath, pciPath);
  EXPECT_EQ(physicalProperties.speed, 100000);
  EXPECT_EQ(physicalProperties.guid, 42U);
  EXPECT_EQ(physicalProperties.ptrSupport, NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF);
  EXPECT_EQ(physicalProperties.regIsGlobal, 1);
  EXPECT_EQ(physicalProperties.forceFlush, 1);
  EXPECT_FLOAT_EQ(physicalProperties.latency, 0.0f);
  EXPECT_EQ(physicalProperties.port, 5);
  EXPECT_EQ(physicalProperties.maxComms, 64);
  EXPECT_EQ(physicalProperties.maxRecvs, NCCL_NET_IB_MAX_RECVS);
  EXPECT_EQ(physicalProperties.netDeviceType, NCCL_NET_DEVICE_HOST);
  EXPECT_EQ(physicalProperties.netDeviceVersion, NCCL_NET_DEVICE_INVALID_VERSION);
  EXPECT_EQ(physicalProperties.maxP2pBytes, NCCL_MAX_NET_SIZE_BYTES);
  EXPECT_EQ(physicalProperties.maxCollBytes, MAX_COLLNET_SIZE);
  EXPECT_EQ(physicalProperties.maxMultiRequestSize, 1);
  EXPECT_EQ(physicalProperties.railId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(physicalProperties.planeId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(physicalProperties.vProps.ndevs, 3);
  EXPECT_EQ(physicalProperties.vProps.devs[0], 9);

  ncclNetProperties_t mergedProperties = {};
  ASSERT_EQ(ncclIbGetProperties(0, &mergedProperties), ncclSuccess);
  EXPECT_STREQ(mergedProperties.name, "virtual0");
  EXPECT_EQ(mergedProperties.speed, 125000);
  EXPECT_EQ(mergedProperties.vProps.ndevs, 1);
  EXPECT_EQ(mergedProperties.vProps.devs[0], 0);
  EXPECT_EQ(mergedProperties.railId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(mergedProperties.planeId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(state.gdrSupportCalls, 2);
  EXPECT_EQ(state.dmaBufSupportCalls, 2);

  ncclIbDevs[0].capsProvider.mlx5.dataDirect = 0;
  state.gdrSupportResult = ncclSystemError;
  state.dmaBufSupportResult = ncclSystemError;
  ncclNetProperties_t hostOnlyProperties = {};
  ASSERT_EQ(ncclIbGetPhysProperties(0, &hostOnlyProperties), ncclSuccess);
  EXPECT_EQ(hostOnlyProperties.ptrSupport, NCCL_PTR_HOST);
  EXPECT_EQ(hostOnlyProperties.forceFlush, 0);

  state.gdrSupportResult = ncclSuccess;
  state.dmaBufSupportResult = ncclSystemError;
  ncclNetProperties_t gdrProperties = {};
  ASSERT_EQ(ncclIbGetPhysProperties(0, &gdrProperties), ncclSuccess);
  EXPECT_EQ(gdrProperties.ptrSupport, NCCL_PTR_HOST | NCCL_PTR_CUDA);
  EXPECT_EQ(gdrProperties.ptrSupport & NCCL_PTR_DMABUF, 0);
  EXPECT_EQ(state.gdrSupportCalls, 4);
  EXPECT_EQ(state.dmaBufSupportCalls, 4);

  std::memset(&mergedProperties, 0xa5, sizeof(mergedProperties));
  unsigned char rejectedPropertiesBefore[sizeof(mergedProperties)];
  std::memcpy(rejectedPropertiesBefore, &mergedProperties, sizeof(mergedProperties));
  EXPECT_EQ(ncclIbGetProperties(1, &mergedProperties), ncclInvalidUsage);
  EXPECT_EQ(std::memcmp(&mergedProperties, rejectedPropertiesBefore, sizeof(mergedProperties)), 0)
      << "Rejected merged-device lookup must preserve every byte of the output structure";
  EXPECT_EQ(ncclIbDevices(&deviceCount), ncclSuccess);
  EXPECT_EQ(deviceCount, 1);
}

/**
 * @brief Verifies selected initialization and finalization results, discovery branches,
 * and ownership.
 * @note Side effects: Repeatedly rebuilds fake global discovery state and provider call records.
 */
TEST(NetIbInitTest, InitializationLifecycle) {
  SCOPED_TRACE(
      "Tested initialization results must preserve output and resource ownership contracts.");

  // A disabled transport must propagate the initialization error without allocating or replacing
  // caller output.
  ResetGlobalState();
  ncclNIbDevs = -1;
  ncclNMergedIbDevs = -1;
  state.parameters["NCCL_IB_DISABLE"] = 1;
  ncclNetCommConfig_t rejectedConfig = {};
  rejectedConfig.trafficClass = 31;
  unsigned char rejectedConfigBefore[sizeof(rejectedConfig)];
  std::memcpy(rejectedConfigBefore, &rejectedConfig, sizeof(rejectedConfig));
  void* rejectedContext = reinterpret_cast<void*>(uintptr_t{0x1234});
  EXPECT_EQ(ncclIbInit(
                &rejectedContext, 7, &rejectedConfig, nullptr, ProfilerForTest),
      ncclInternalError)
      << "Top-level initialization must propagate the disabled-transport result";
  EXPECT_EQ(rejectedContext, reinterpret_cast<void*>(uintptr_t{0x1234}))
      << "Failed top-level initialization must preserve the caller's context output";
  EXPECT_EQ(std::memcmp(&rejectedConfig, &rejectedConfigBefore, sizeof(rejectedConfig)), 0)
      << "Failed top-level initialization must preserve the caller-owned configuration";
  EXPECT_EQ(state.allocationCalls, 0)
      << "Top-level initialization must not allocate a context after device initialization fails";
  EXPECT_EQ(netRefCount, 1)
      << "A failed first device initialization must retain one reference for explicit finalization";
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess)
      << "The caller must be able to release the reference retained by failed initialization";
  EXPECT_EQ(netRefCount, 0)
      << "Finalizing the failed initialization must restore the reference count";

  ResetGlobalState();
  ncclNIbDevs = -1;
  ncclNMergedIbDevs = -1;
  state.verbsSymbolsResult = ncclSystemError;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclInternalError);
  EXPECT_EQ(netRefCount, 1);
  EXPECT_EQ(state.verbsSymbolsCalls, 1);
  EXPECT_EQ(state.mlx5SymbolsCalls, 0);
  EXPECT_EQ(state.findInterfacesCalls, 0);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);
  EXPECT_EQ(netRefCount, 0);

  PrepareSingleDeviceDiscovery();
  state.openDeviceResult = ncclSystemError;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclSuccess);
  EXPECT_EQ(state.openDeviceCalls, 1);
  EXPECT_EQ(state.queryDeviceCalls, 0);
  EXPECT_EQ(state.closeDeviceCalls, 0);
  EXPECT_EQ(state.freeDeviceListCalls, 1);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);

  PrepareSingleDeviceDiscovery();
  state.queryDeviceResult = ncclSystemError;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclSuccess);
  EXPECT_EQ(state.openDeviceCalls, 1);
  EXPECT_EQ(state.queryDeviceCalls, 1);
  EXPECT_EQ(state.closeDeviceCalls, 1);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);

  PrepareSingleDeviceDiscovery();
  state.queryDeviceResult = ncclSystemError;
  state.closeDeviceResult = ncclSystemError;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclInternalError);
  EXPECT_EQ(state.closeDeviceCalls, 1);
  EXPECT_EQ(state.freeDeviceListCalls, 0);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);

  PrepareSingleDeviceDiscovery();
  state.queryPortResult = ncclSystemError;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclSuccess);
  EXPECT_EQ(state.queryPortCalls, 1);
  EXPECT_EQ(state.closeDeviceCalls, 1);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);

  PrepareSingleDeviceDiscovery();
  state.interfaceMatches = false;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclSuccess);
  EXPECT_EQ(state.queryPortCalls, 1);
  EXPECT_EQ(state.closeDeviceCalls, 1);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);

  ResetGlobalState();
  ncclNIbDevs = -1;
  ncclNMergedIbDevs = -1;
  state.interfaceCount = 0;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclInternalError);
  EXPECT_EQ(netRefCount, 1);
  EXPECT_EQ(state.findInterfacesCalls, 1);
  EXPECT_EQ(state.getDeviceListCalls, 0);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);
  EXPECT_EQ(netRefCount, 0);

  ResetGlobalState();
  ncclNIbDevs = -1;
  ncclNMergedIbDevs = -1;
  state.getDeviceListResult = ncclSystemError;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclInternalError);
  EXPECT_EQ(netRefCount, 1);
  EXPECT_EQ(state.findInterfacesCalls, 1);
  EXPECT_EQ(state.getDeviceListCalls, 1);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);
  EXPECT_EQ(netRefCount, 0);

  ResetGlobalState();
  ncclNIbDevs = -1;
  ncclNMergedIbDevs = -1;
  state.parameters["NCCL_IB_DISABLE"] = 1;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclInternalError);
  EXPECT_EQ(netRefCount, 1);
  EXPECT_EQ(ncclProfilerFunction, ProfilerForTest);
  EXPECT_EQ(state.verbsSymbolsCalls, 0);
  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(netRefCount, 2);
  EXPECT_EQ(ncclProfilerFunction, ProfilerForTest);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);
  EXPECT_EQ(netRefCount, 1);
  EXPECT_EQ(ncclIbFinalize(nullptr), ncclSuccess);
  EXPECT_EQ(netRefCount, 0);

  ResetGlobalState();
  ncclNIbDevs = -1;
  ncclNMergedIbDevs = -1;
  state.parameters["NCCL_IB_DISABLE"] = 0;
  state.parameters["NCCL_IB_PCI_RELAXED_ORDERING"] = 0;
  state.parameters["NCCL_OCI_FRM_ENABLE"] = 1;
  state.allocationFails = true;
  ncclNetCommConfig_t config = {};
  config.trafficClass = 23;
  unsigned char failedConfigBefore[sizeof(config)];
  std::memcpy(failedConfigBefore, &config, sizeof(config));
  void* context = reinterpret_cast<void*>(1);
  ASSERT_EQ(ncclIbInit(&context, UINT64_MAX, &config, nullptr, ProfilerForTest), ncclSystemError);
  EXPECT_EQ(context, reinterpret_cast<void*>(1));
  EXPECT_EQ(std::memcmp(&config, &failedConfigBefore, sizeof(config)), 0)
      << "Allocation failure must preserve the caller-owned initialization configuration";
  EXPECT_TRUE(g_ncclIbUseCim);
  EXPECT_EQ(netRefCount, 1);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(ncclNMergedIbDevs, 0);
  EXPECT_EQ(state.verbsSymbolsCalls, 1);
  EXPECT_EQ(state.mlx5SymbolsCalls, 1);
  EXPECT_EQ(state.findInterfacesCalls, 1);
  EXPECT_EQ(state.getDeviceListCalls, 1);
  EXPECT_EQ(ncclIbFinalize(nullptr), ncclSuccess);
  EXPECT_EQ(netRefCount, 0);

  PrepareSingleDeviceDiscovery();
  state.parameters["NCCL_OCI_FRM_ENABLE"] = 0;
  state.mlx5SymbolsResult = ncclSystemError;
  config.trafficClass = -1;
  unsigned char successfulConfigBefore[sizeof(config)];
  std::memcpy(successfulConfigBefore, &config, sizeof(config));
  context = nullptr;
  ASSERT_EQ(ncclIbInit(&context, 0, &config, nullptr, ProfilerForTest), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(std::memcmp(&config, &successfulConfigBefore, sizeof(config)), 0)
      << "Successful initialization must preserve the caller-owned configuration";
  EXPECT_EQ(static_cast<ncclNetCommConfig_t*>(context)->trafficClass, -1);
  EXPECT_FALSE(g_ncclIbUseCim);
  EXPECT_EQ(state.mlx5SymbolsCalls, 1);
  EXPECT_TRUE(std::isfinite(g_inv_cyc_per_us));
  EXPECT_GE(g_inv_cyc_per_us, 0.0);
  EXPECT_EQ(netRefCount, 1);
  ASSERT_EQ(ncclNIbDevs, 1);
  EXPECT_EQ(ncclNMergedIbDevs, 1);
  EXPECT_STREQ(ncclIbDevs[0].devName, "test0");
  EXPECT_STREQ(ncclIbDevs[0].fullPciPath, "/sys/devices/pci0000:00/0000:03:04.7");
  ASSERT_NE(ncclIbDevs[0].pciPath, nullptr);
  EXPECT_STREQ(ncclIbDevs[0].pciPath, "/sys/devices/pci0000:00/0000:03:00.0");
  EXPECT_EQ(ncclIbDevs[0].guid, 0x1234U);
  EXPECT_EQ(ncclIbDevs[0].portNum, 1);
  EXPECT_EQ(ncclIbDevs[0].realPort, 0);
  EXPECT_EQ(ncclIbDevs[0].speed, 25000);
  EXPECT_EQ(ncclIbDevs[0].maxQp, 64);
  EXPECT_EQ(ncclIbDevs[0].link, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.ndevs, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.devs[0], 0);
  EXPECT_EQ(ncclIbMergedDevs[0].publicVProps.ndevs, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].publicVProps.devs[0], 0);
  EXPECT_EQ(ncclIbMergedDevs[0].anchorDev, 0);
  EXPECT_EQ(ncclIbMergedDevs[0].firstAlias, 0);
  EXPECT_EQ(ncclIbMergedDevs[0].aliasCount, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].isForcedAlias, 0);
  EXPECT_STREQ(ncclIbMergedDevs[0].devName, "test0");
  EXPECT_EQ(ncclIbMergedDevs[0].speed, 25000);
  EXPECT_EQ(state.freeDeviceListCalls, 1);
  int deviceCount = -1;
  EXPECT_EQ(ncclIbDevices(&deviceCount), ncclSuccess);
  EXPECT_EQ(deviceCount, 1);
  char* discoveredPciPath = ncclIbDevs[0].pciPath;
  EXPECT_EQ(ncclIbFinalize(context), ncclSuccess);
  EXPECT_EQ(netRefCount, 0);
  std::free(discoveredPciPath);
  ncclIbDevs[0].pciPath = nullptr;

  PrepareSingleDeviceDiscovery();
  state.providerSupported = true;
  state.dataDirectPathResult = ncclSystemError;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclSystemError);
  EXPECT_EQ(state.dataDirectPathCalls, 1);
  EXPECT_EQ(state.freeDeviceListCalls, 0);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);

  PrepareSingleDeviceDiscovery();
  state.parameters["NCCL_IB_DATA_DIRECT"] = 2;
  state.providerSupported = true;
  state.dataDirectPathResult = ncclSuccess;
  state.activeSpeedEx = 256;
  ASSERT_EQ(ncclIbInitDevices(nullptr, ProfilerForTest), ncclSuccess);
  ASSERT_EQ(ncclNIbDevs, 2);
  EXPECT_EQ(ncclNMergedIbDevs, 2);
  int directDevice = ncclIbDevs[0].capsProvider.mlx5.dataDirect ? 0 : 1;
  int regularDevice = 1 - directDevice;
  EXPECT_STREQ(ncclIbDevs[directDevice].devName, "test0_dma");
  EXPECT_STREQ(ncclIbDevs[directDevice].pciPath, "/sys/devices/data-direct0");
  EXPECT_EQ(ncclIbDevs[directDevice].speed, 200000);
  EXPECT_EQ(ncclIbDevs[regularDevice].capsProvider.mlx5.dataDirect, 0);
  EXPECT_STREQ(ncclIbDevs[regularDevice].pciPath, "/sys/devices/pci0000:00/0000:03:00.0");
  EXPECT_EQ(ncclIbDevs[regularDevice].speed, 200000);
  EXPECT_EQ(state.dataDirectPathCalls, 1);
  EXPECT_EQ(state.freeDeviceListCalls, 1);
  char* directPciPath = ncclIbDevs[directDevice].pciPath;
  char* regularPciPath = ncclIbDevs[regularDevice].pciPath;
  char* discardedRealpath = state.lastRealpathAllocation;
  ASSERT_NE(discardedRealpath, nullptr);
  EXPECT_NE(discardedRealpath, directPciPath);
  EXPECT_NE(discardedRealpath, regularPciPath);
  EXPECT_EQ(ncclIbFinalizeDevices(), ncclSuccess);
  std::free(directPciPath);
  std::free(regularPciPath);
  std::free(discardedRealpath);
  ncclIbDevs[directDevice].pciPath = nullptr;
  ncclIbDevs[regularDevice].pciPath = nullptr;
}
namespace {

/** @brief Preserves the force-merge environment while isolating each alias scenario. */
class ForcedMergeInitTest : public ::testing::Test {
 protected:
  /**
   * @brief Saves and clears the force-merge environment before each test.
   * @note Side effects: Resets production/test globals and updates process environment.
   */
  void SetUp() override {
    const char* value = std::getenv("NCCL_NET_FORCE_MERGE");
    hadOriginalValue_ = value != nullptr;
    if (value != nullptr) originalValue_ = value;
    ASSERT_EQ(unsetenv("NCCL_NET_FORCE_MERGE"), 0);
    ResetGlobalState();
  }

  /**
   * @brief Restores the force-merge environment after each test.
   * @note Side effects: Updates process environment.
   */
  void TearDown() override {
    if (hadOriginalValue_) {
      EXPECT_EQ(setenv("NCCL_NET_FORCE_MERGE", originalValue_.c_str(), 1), 0);
    } else {
      EXPECT_EQ(unsetenv("NCCL_NET_FORCE_MERGE"), 0);
    }
  }

  /**
   * @brief Configures one deterministic physical device.
   * @param[in] index Valid physical device index to replace.
   * @param[in] name Non-NULL device name copied into global state.
   * @param[in] speed Link speed assigned to the device.
   * @note Side effects: Updates selected ncclIbDevs[index] fields.
   */
  void ConfigureDevice(int index, const char* name, int speed) {
    ncclIbDev& device = ncclIbDevs[index];
    std::snprintf(device.devName, sizeof(device.devName), "%s", name);
    std::snprintf(device.fullPciPath, sizeof(device.fullPciPath),
                  "/sys/devices/pci0000:00/0000:%02x:00.0", static_cast<unsigned int>(index));
    device.pciPath = device.fullPciPath;
    device.speed = speed;
    device.link = IBV_LINK_LAYER_INFINIBAND;
    device.portNum = static_cast<uint8_t>(index + 1);
    device.realPort = index + 10;
    device.guid = 0x1000 + index;
    device.maxQp = 100 + index;
  }

  /**
   * @brief Replaces one device's normalized PCI locality.
   * @param[in] index Valid physical device index to update.
   * @param[in] path NULL, empty, or non-empty PCI path to publish.
   * @note Side effects: Replaces the device's fullPciPath and pciPath fields.
   */
  void SetPciPath(int index, const char* path) {
    ncclIbDev& device = ncclIbDevs[index];
    std::snprintf(device.fullPciPath, sizeof(device.fullPciPath), "%s",
        path == nullptr ? "" : path);
    device.pciPath = path == nullptr ? nullptr : device.fullPciPath;
  }

 private:
  bool hadOriginalValue_ = false;
  std::string originalValue_;
};

}  // namespace

/** @brief Verifies all known and fallback width and speed table entries. */
TEST(NetIbInitTest, WidthAndSpeedDecodeKnownValues) {
  EXPECT_EQ(ncclIbWidth(1), 1);
  EXPECT_EQ(ncclIbWidth(2), 4);
  EXPECT_EQ(ncclIbWidth(4), 8);
  EXPECT_EQ(ncclIbWidth(8), 12);
  EXPECT_EQ(ncclIbWidth(16), 2);
  EXPECT_EQ(ncclIbWidth(0), 2);

  EXPECT_EQ(ncclIbSpeed(1), 2500);
  EXPECT_EQ(ncclIbSpeed(2), 5000);
  EXPECT_EQ(ncclIbSpeed(4), 10000);
  EXPECT_EQ(ncclIbSpeed(8), 10000);
  EXPECT_EQ(ncclIbSpeed(16), 14000);
  EXPECT_EQ(ncclIbSpeed(32), 25000);
  EXPECT_EQ(ncclIbSpeed(64), 50000);
  EXPECT_EQ(ncclIbSpeed(128), 100000);
  EXPECT_EQ(ncclIbSpeed(0), 200000);
}

/**
 * @brief Verifies virtual-device member-count and capacity rejection.
 * @note Side effects: Rebuilds the fake physical and merged inventories.
 */
TEST(NetIbInitTest, MakeVirtualDeviceRejectsInvalidInputs) {
  ResetGlobalState();
  ncclNIbDevs = 1;
  int dev = -1;
  ncclNetVDeviceProps_t props = {};

  props.ndevs = 0;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&dev, &props), ncclInvalidUsage);

  props.ndevs = NCCL_IB_MAX_DEVS_PER_NIC + 1;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&dev, &props), ncclInvalidUsage);

  ncclNMergedIbDevs = MAX_IB_VDEVS;
  props = {};
  props.ndevs = 1;
  props.devs[0] = 0;
  unsigned char propsBefore[sizeof(props)] = {};
  std::memcpy(propsBefore, &props, sizeof(props));
  std::vector<unsigned char> inventoryBefore(sizeof(ncclIbMergedDevs));
  std::memcpy(inventoryBefore.data(), ncclIbMergedDevs, sizeof(ncclIbMergedDevs));
  EXPECT_EQ(occlIbMakeLegacyVDevice(&dev, &props), ncclInvalidUsage)
      << "Direct legacy creation must reject exhausted virtual-device capacity";
  EXPECT_EQ(dev, -1)
      << "Legacy capacity rejection must preserve the output device index";
  EXPECT_EQ(ncclNMergedIbDevs, MAX_IB_VDEVS)
      << "Legacy capacity rejection must preserve the published-device count";
  EXPECT_EQ(std::memcmp(&props, propsBefore, sizeof(props)), 0)
      << "Legacy capacity rejection must preserve caller-owned properties";
  EXPECT_EQ(std::memcmp(ncclIbMergedDevs, inventoryBefore.data(),
                sizeof(ncclIbMergedDevs)), 0)
      << "Legacy capacity rejection must preserve the merged-device inventory";
}

/**
 * @brief Verifies direct legacy creation publishes the requested aggregate exactly once.
 * @note Side effects: Resets and configures fake device inventories, then publishes one device.
 */
TEST(NetIbInitTest, MakeLegacyVirtualDevicePublishesRequestedAggregate) {
  ResetGlobalState();
  ncclNIbDevs = 3;
  ncclIbDevs[0].speed = 100;
  ncclIbDevs[1].speed = 200;
  ncclIbDevs[2].speed = 300;
  std::snprintf(ncclIbDevs[0].devName, MAXNAMESIZE, "mlx5_0");
  std::snprintf(ncclIbDevs[2].devName, MAXNAMESIZE, "mlx5_2");

  int dev = -1;
  ncclNetVDeviceProps_t props = {};
  props.ndevs = 2;
  props.devs[0] = 2;
  props.devs[1] = 0;
  unsigned char propsBefore[sizeof(props)] = {};
  std::memcpy(propsBefore, &props, sizeof(props));

  ASSERT_EQ(occlIbMakeLegacyVDevice(&dev, &props), ncclSuccess)
      << "Direct legacy creation must accept a valid aggregate";

  EXPECT_EQ(dev, 0)
      << "Legacy creation must return the newly published index";
  ASSERT_EQ(ncclNMergedIbDevs, 1)
      << "Legacy creation must publish exactly one device";
  EXPECT_EQ(std::memcmp(&props, propsBefore, sizeof(props)), 0)
      << "Legacy creation must preserve caller-owned properties";
  EXPECT_EQ(std::memcmp(&ncclIbMergedDevs[0].vProps, &props, sizeof(props)), 0)
      << "The private member view must match the request";
  EXPECT_EQ(std::memcmp(&ncclIbMergedDevs[0].publicVProps, &props, sizeof(props)), 0)
      << "The public member view must match the request";
  EXPECT_EQ(ncclIbMergedDevs[0].anchorDev, 2)
      << "The first requested member must anchor the aggregate";
  EXPECT_EQ(ncclIbMergedDevs[0].firstAlias, dev)
      << "A legacy aggregate must identify itself as its first alias";
  EXPECT_EQ(ncclIbMergedDevs[0].aliasCount, 1)
      << "A legacy aggregate must expose one alias";
  EXPECT_EQ(ncclIbMergedDevs[0].isForcedAlias, 0)
      << "Legacy creation must not mark the aggregate as forced";
  EXPECT_EQ(ncclIbMergedDevs[0].speed, 400)
      << "Aggregate speed must equal the sum of member speeds";
  EXPECT_STREQ(ncclIbMergedDevs[0].devName, "mlx5_2+mlx5_0")
      << "Aggregate naming must preserve requested member order";
}

/**
 * @brief Verifies the longest valid physical names still fit in one legacy aggregate name.
 * @note Side effects: Rebuilds the physical inventory and publishes one maximum-width aggregate.
 */
TEST(NetIbInitTest, MakeLegacyVirtualDeviceAcceptsLongestValidNames) {
  constexpr size_t maximumAggregateLength =
      NCCL_IB_MAX_DEVS_PER_NIC * (MAXNAMESIZE - 1) + (NCCL_IB_MAX_DEVS_PER_NIC - 1);
  static_assert(maximumAggregateLength < MAX_MERGED_DEV_NAME,
      "Every valid legacy member name must fit in the merged-name buffer");

  ResetGlobalState();
  ncclNIbDevs = NCCL_IB_MAX_DEVS_PER_NIC;
  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = NCCL_IB_MAX_DEVS_PER_NIC;
  std::string expectedName;
  for (int member = 0; member < requested.ndevs; member++) {
    const std::string physicalName(MAXNAMESIZE - 1, static_cast<char>('a' + member));
    std::snprintf(ncclIbDevs[member].devName,
        sizeof(ncclIbDevs[member].devName), "%s", physicalName.c_str());
    ncclIbDevs[member].speed = member + 1;
    requested.devs[member] = member;
    if (member != 0) expectedName += '+';
    expectedName += physicalName;
  }

  int aggregate = -1;
  ASSERT_EQ(occlIbMakeLegacyVDevice(&aggregate, &requested), ncclSuccess)
      << "All maximum-width valid member names must fit in the declared aggregate buffer";
  ASSERT_EQ(aggregate, 0);
  ASSERT_EQ(ncclNMergedIbDevs, 1);
  EXPECT_EQ(expectedName.size(), maximumAggregateLength);
  EXPECT_LT(expectedName.size(), sizeof(ncclIbMergedDevs[aggregate].devName));
  EXPECT_STREQ(ncclIbMergedDevs[aggregate].devName, expectedName.c_str());
}

/**
 * @brief Verifies merged-property lookup rejects the first out-of-range index.
 * @note Side effects: Replaces the fake merged-device count.
 */
TEST(NetIbInitTest, PropertiesRejectInvalidMergedDevice) {
  ResetGlobalState();
  ncclNMergedIbDevs = 1;
  ncclNetProperties_t props = {};

  EXPECT_EQ(ncclIbGetProperties(1, &props), ncclInvalidUsage);
}

/**
 * @brief Verifies a forced non-local group publishes one singleton alias per anchor.
 * @note Side effects: Publishes forced aliases and their public properties.
 */
TEST_F(ForcedMergeInitTest, PublishesOneSingletonAliasPerAnchor) {
  ncclNIbDevs = 3;
  ConfigureDevice(0, "mlx5_0", 100);
  ConfigureDevice(1, "mlx5_1", 200);
  ConfigureDevice(2, "mlx5_2", 300);
  ASSERT_EQ(setenv("NCCL_NET_FORCE_MERGE", "mlx5_2,mlx5_0,mlx5_1", 1), 0);
  EXPECT_TRUE(occlIbForceMergeAliasesEnabled())
      << "A non-empty force-merge value must enable forced aliases";
  EXPECT_STREQ(std::getenv("NCCL_NET_FORCE_MERGE"), "mlx5_2,mlx5_0,mlx5_1")
      << "Force-merge detection must preserve the configured environment value";

  const int members[] = {2, 0, 1};
  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = 3;
  for (int i = 0; i < requested.ndevs; i++) requested.devs[i] = members[i];

  int firstAlias = -1;
  ASSERT_EQ(ncclIbMakeVDeviceInternal(&firstAlias, &requested), ncclSuccess);
  ASSERT_EQ(firstAlias, 0);
  ASSERT_EQ(ncclNMergedIbDevs, requested.ndevs);

  for (int alias = 0; alias < requested.ndevs; alias++) {
    const int anchor = members[alias];
    const ncclIbMergedDev& merged = ncclIbMergedDevs[firstAlias + alias];
    EXPECT_EQ(merged.publicVProps.ndevs, 1);
    EXPECT_EQ(merged.publicVProps.devs[0], anchor);
    EXPECT_EQ(merged.anchorDev, anchor);
    EXPECT_EQ(merged.firstAlias, firstAlias);
    EXPECT_EQ(merged.aliasCount, requested.ndevs);
    EXPECT_EQ(merged.isForcedAlias, 1);
    EXPECT_EQ(merged.speed, ncclIbDevs[anchor].speed);
    EXPECT_STREQ(merged.devName, ncclIbDevs[anchor].devName);

    ASSERT_EQ(merged.vProps.ndevs, requested.ndevs);
    for (int member = 0; member < requested.ndevs; member++) {
      EXPECT_EQ(merged.vProps.devs[member], members[(alias + member) % requested.ndevs]);
    }

    ncclNetProperties_t properties = {};
    ASSERT_EQ(ncclIbGetProperties(firstAlias + alias, &properties), ncclSuccess);
    EXPECT_EQ(properties.name, ncclIbDevs[anchor].devName);
    EXPECT_EQ(properties.speed, ncclIbDevs[anchor].speed);
    EXPECT_EQ(properties.pciPath, ncclIbDevs[anchor].pciPath);
    EXPECT_EQ(properties.guid, ncclIbDevs[anchor].guid);
    EXPECT_EQ(properties.port, ncclIbDevs[anchor].portNum + ncclIbDevs[anchor].realPort);
    EXPECT_EQ(properties.maxComms, ncclIbDevs[anchor].maxQp);
    EXPECT_EQ(properties.vProps.ndevs, 1);
    EXPECT_EQ(properties.vProps.devs[0], anchor);
  }
}

/**
 * @brief Verifies forced merging forms disjoint pairs from shared PCI localities.
 * @note Side effects: Publishes two forced aliases and queries their properties.
 */
TEST_F(ForcedMergeInitTest, PublishesDisjointLocalPairsFromNonAdjacentMembers) {
  SCOPED_TRACE("Direct forced creation must publish exact disjoint PCI-local pairs.");
  ncclNIbDevs = 4;
  for (int dev = 0; dev < ncclNIbDevs; dev++) {
    const std::string name = "mlx5_" + std::to_string(dev);
    ConfigureDevice(dev, name.c_str(), 100 * (dev + 1));
    SetPciPath(dev, dev % 2 == 0 ? "/sys/devices/pci0000:00/0000:01:00.0"
                               : "/sys/devices/pci0000:00/0000:02:00.0");
  }
  ASSERT_EQ(setenv("NCCL_NET_FORCE_MERGE", "mlx5_3,mlx5_0,mlx5_1,mlx5_2", 1), 0);

  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = 4;
  const int members[] = {3, 0, 1, 2};
  for (int member = 0; member < requested.ndevs; member++) requested.devs[member] = members[member];
  unsigned char requestedBefore[sizeof(requested)] = {};
  std::memcpy(requestedBefore, &requested, sizeof(requested));
  int firstAlias = -1;
  ASSERT_EQ(occlIbMakeForcedAliases(&firstAlias, &requested), ncclSuccess)
      << "Direct forced creation must accept valid distinct same-link members";
  EXPECT_EQ(std::memcmp(&requested, requestedBefore, sizeof(requested)), 0)
      << "Forced creation must preserve caller-owned properties";
  ASSERT_EQ(firstAlias, 0);
  ASSERT_EQ(ncclNMergedIbDevs, 2);

  const int privateMembers[][4] = {{3, 1, 0, 2}, {0, 2, 3, 1}};
  const int activeSpeeds[] = {600, 400};
  const char* activeNames[] = {"mlx5_3+mlx5_1", "mlx5_0+mlx5_2"};
  bool publiclyAdvertised[4] = {};
  for (int alias = 0; alias < ncclNMergedIbDevs; alias++) {
    const ncclIbMergedDev& merged = ncclIbMergedDevs[alias];
    ASSERT_EQ(merged.publicVProps.ndevs, 2);
    EXPECT_EQ(merged.anchorDev, privateMembers[alias][0]);
    EXPECT_EQ(merged.firstAlias, firstAlias);
    EXPECT_EQ(merged.aliasCount, 2);
    EXPECT_EQ(merged.isForcedAlias, 1);
    EXPECT_EQ(merged.speed, activeSpeeds[alias]);
    EXPECT_STREQ(merged.devName, activeNames[alias]);
    ASSERT_EQ(merged.vProps.ndevs, 4);
    for (int member = 0; member < merged.vProps.ndevs; member++) {
      EXPECT_EQ(merged.vProps.devs[member], privateMembers[alias][member]);
    }
    for (int member = 0; member < merged.publicVProps.ndevs; member++) {
      const int dev = merged.publicVProps.devs[member];
      EXPECT_EQ(dev, privateMembers[alias][member]);
      EXPECT_FALSE(publiclyAdvertised[dev]);
      publiclyAdvertised[dev] = true;
    }

    ncclNetProperties_t properties = {};
    ASSERT_EQ(ncclIbGetProperties(alias, &properties), ncclSuccess);
    EXPECT_STREQ(properties.name, activeNames[alias]);
    EXPECT_EQ(properties.speed, activeSpeeds[alias]);
    EXPECT_EQ(properties.pciPath, ncclIbDevs[merged.anchorDev].pciPath);
    EXPECT_EQ(properties.guid, ncclIbDevs[merged.anchorDev].guid);
    EXPECT_EQ(properties.port,
        ncclIbDevs[merged.anchorDev].portNum + ncclIbDevs[merged.anchorDev].realPort);
    EXPECT_EQ(properties.vProps.ndevs, 2);
    for (int member = 0; member < properties.vProps.ndevs; member++) {
      EXPECT_EQ(properties.vProps.devs[member], privateMembers[alias][member]);
    }
  }
  for (bool advertised : publiclyAdvertised) EXPECT_TRUE(advertised);
}

/**
 * @brief Verifies the longest valid names fit in a forced two-port active alias.
 * @note Side effects: Configures one local pair and publishes its forced alias.
 */
TEST_F(ForcedMergeInitTest, LongestValidPairNamesRemainWithinAliasCapacity) {
  constexpr size_t maximumPairNameLength = 2 * (MAXNAMESIZE - 1) + 1;
  static_assert(maximumPairNameLength < MAX_MERGED_DEV_NAME,
      "Every valid forced-pair name must fit in the merged-name buffer");

  ncclNIbDevs = 2;
  const std::string firstName(MAXNAMESIZE - 1, 'a');
  const std::string secondName(MAXNAMESIZE - 1, 'b');
  ConfigureDevice(0, firstName.c_str(), 100);
  ConfigureDevice(1, secondName.c_str(), 200);
  SetPciPath(1, ncclIbDevs[0].pciPath);

  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = 2;
  requested.devs[0] = 0;
  requested.devs[1] = 1;
  int firstAlias = -1;
  ASSERT_EQ(occlIbMakeForcedAliases(&firstAlias, &requested), ncclSuccess)
      << "A maximum-width valid local pair must fit in the declared alias-name buffer";
  ASSERT_EQ(firstAlias, 0);
  ASSERT_EQ(ncclNMergedIbDevs, 1);

  const std::string expectedName = firstName + "+" + secondName;
  EXPECT_EQ(expectedName.size(), maximumPairNameLength);
  EXPECT_LT(expectedName.size(), sizeof(ncclIbMergedDevs[firstAlias].devName));
  EXPECT_STREQ(ncclIbMergedDevs[firstAlias].devName, expectedName.c_str());
}

/**
 * @brief Verifies unmatched or unknown PCI localities remain singleton aliases.
 * @note Side effects: Publishes forced aliases and queries their properties.
 */
TEST_F(ForcedMergeInitTest, UnmatchedAndUnknownLocalitiesRemainSingletons) {
  ncclNIbDevs = 7;
  for (int dev = 0; dev < ncclNIbDevs; dev++) {
    const std::string name = "mlx5_" + std::to_string(dev);
    ConfigureDevice(dev, name.c_str(), 100);
    SetPciPath(dev, dev < 3 ? "/sys/devices/pci0000:00/0000:01:00.0" : dev < 5 ? nullptr : "");
  }
  ASSERT_EQ(setenv("NCCL_NET_FORCE_MERGE",
                "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6", 1),
      0);
  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = ncclNIbDevs;
  for (int member = 0; member < requested.ndevs; member++) requested.devs[member] = member;

  int firstAlias = -1;
  ASSERT_EQ(ncclIbMakeVDeviceInternal(&firstAlias, &requested), ncclSuccess);
  ASSERT_EQ(ncclNMergedIbDevs, 6);
  bool publiclyAdvertised[7] = {};
  for (int alias = 0; alias < ncclNMergedIbDevs; alias++) {
    const ncclIbMergedDev& merged = ncclIbMergedDevs[alias];
    ASSERT_EQ(merged.publicVProps.ndevs, alias == 0 ? 2 : 1);
    EXPECT_EQ(merged.anchorDev, alias == 0 ? 0 : alias + 1);
    EXPECT_EQ(merged.aliasCount, 6);
    ASSERT_EQ(merged.vProps.ndevs, requested.ndevs);
    bool privatelyRegistered[7] = {};
    for (int member = 0; member < merged.vProps.ndevs; member++) {
      const int dev = merged.vProps.devs[member];
      ASSERT_GE(dev, 0);
      ASSERT_LT(dev, requested.ndevs);
      EXPECT_FALSE(privatelyRegistered[dev]);
      privatelyRegistered[dev] = true;
    }
    for (bool registered : privatelyRegistered) EXPECT_TRUE(registered);
    for (int member = 0; member < merged.publicVProps.ndevs; member++) {
      const int dev = merged.publicVProps.devs[member];
      EXPECT_EQ(merged.vProps.devs[member], dev);
      EXPECT_FALSE(publiclyAdvertised[dev]);
      publiclyAdvertised[dev] = true;
    }
    ncclNetProperties_t properties = {};
    ASSERT_EQ(ncclIbGetProperties(alias, &properties), ncclSuccess);
    EXPECT_EQ(properties.speed, alias == 0 ? 200 : 100);
    if (alias != 0) {
      EXPECT_EQ(properties.name, ncclIbDevs[merged.anchorDev].devName);
      EXPECT_EQ(properties.maxComms, ncclIbDevs[merged.anchorDev].maxQp);
    }
  }
  for (bool advertised : publiclyAdvertised) EXPECT_TRUE(advertised);
}

/**
 * @brief Verifies forced-alias capacity checks and reordered group reuse are atomic.
 * @note Side effects: Fills the fake merged inventory to capacity.
 */
TEST_F(ForcedMergeInitTest, PairCapacityAndReorderedReuseAreAtomic) {
  ncclNIbDevs = 6;
  for (int dev = 0; dev < ncclNIbDevs; dev++) {
    const std::string name = "mlx5_" + std::to_string(dev);
    ConfigureDevice(dev, name.c_str(), 100);
    SetPciPath(dev, dev < 2 ? "/sys/devices/pci0000:00/0000:01:00.0"
                   : dev < 4 ? "/sys/devices/pci0000:00/0000:02:00.0"
                             : "/sys/devices/pci0000:00/0000:03:00.0");
  }
  ncclNMergedIbDevs = MAX_IB_VDEVS - 2;
  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = 4;
  for (int member = 0; member < requested.ndevs; member++) requested.devs[member] = member;
  unsigned char requestedBefore[sizeof(requested)] = {};
  std::memcpy(requestedBefore, &requested, sizeof(requested));
  int firstAlias = -1;
  ASSERT_EQ(occlIbMakeForcedAliases(&firstAlias, &requested), ncclSuccess)
      << "Direct forced creation must use the two available alias slots";
  EXPECT_EQ(firstAlias, MAX_IB_VDEVS - 2)
      << "Forced creation must return the first newly published alias";
  ASSERT_EQ(ncclNMergedIbDevs, MAX_IB_VDEVS)
      << "Two local pairs must fill the virtual-device inventory";
  EXPECT_EQ(std::memcmp(&requested, requestedBefore, sizeof(requested)), 0)
      << "Forced creation must preserve caller-owned properties";
  std::vector<unsigned char> inventoryBefore(sizeof(ncclIbMergedDevs));
  std::memcpy(inventoryBefore.data(), ncclIbMergedDevs, sizeof(ncclIbMergedDevs));

  const ncclNetVDeviceProps_t reordered = {4, {3, 1, 2, 0}};
  unsigned char reorderedBefore[sizeof(reordered)] = {};
  std::memcpy(reorderedBefore, &reordered, sizeof(reordered));
  EXPECT_TRUE(occlIbSameMemberSet(&ncclIbMergedDevs[firstAlias].vProps, &reordered))
      << "Member-set equality must ignore member order";
  EXPECT_FALSE(occlIbSameMemberSet(nullptr, &reordered))
      << "A missing left member set must not compare equal";
  EXPECT_FALSE(occlIbSameMemberSet(&reordered, nullptr))
      << "A missing right member set must not compare equal";
  int repeatedAlias = 71;
  EXPECT_EQ(occlIbMakeForcedAliases(&repeatedAlias, &reordered), ncclSuccess)
      << "A reordered request must reuse its existing forced alias group";
  EXPECT_EQ(repeatedAlias, firstAlias)
      << "Reordered reuse must return the existing first alias";
  EXPECT_EQ(ncclNMergedIbDevs, MAX_IB_VDEVS)
      << "Reordered reuse must not publish additional aliases";
  EXPECT_EQ(std::memcmp(&reordered, reorderedBefore, sizeof(reordered)), 0)
      << "Reordered reuse must preserve caller-owned properties";
  EXPECT_EQ(std::memcmp(ncclIbMergedDevs, inventoryBefore.data(),
                sizeof(ncclIbMergedDevs)), 0)
      << "Reordered reuse must preserve the merged-device inventory";

  const ncclNetVDeviceProps_t different = {4, {0, 1, 4, 5}};
  unsigned char differentBefore[sizeof(different)] = {};
  std::memcpy(differentBefore, &different, sizeof(different));
  EXPECT_FALSE(occlIbSameMemberSet(&ncclIbMergedDevs[firstAlias].vProps, &different))
      << "Equal-sized sets with different members must not compare equal";
  EXPECT_TRUE(occlIbMemberSetsOverlap(&ncclIbMergedDevs[firstAlias].vProps, &different))
      << "Sets sharing physical members must overlap";
  EXPECT_EQ(std::memcmp(&different, differentBefore, sizeof(different)), 0)
      << "Set comparisons must preserve caller-owned properties";

  ncclNetVDeviceProps_t rejected = {};
  rejected.ndevs = 2;
  rejected.devs[0] = 4;
  rejected.devs[1] = 5;
  unsigned char rejectedBefore[sizeof(rejected)] = {};
  std::memcpy(rejectedBefore, &rejected, sizeof(rejected));
  EXPECT_FALSE(occlIbSameMemberSet(&ncclIbMergedDevs[firstAlias].vProps, &rejected))
      << "Sets with different member counts must not compare equal";
  EXPECT_EQ(std::memcmp(&rejected, rejectedBefore, sizeof(rejected)), 0)
      << "Unequal-count comparison must preserve caller-owned properties";
  EXPECT_FALSE(occlIbMemberSetsOverlap(&ncclIbMergedDevs[firstAlias].vProps, &rejected))
      << "Disjoint physical-member sets must not overlap";
  EXPECT_FALSE(occlIbMemberSetsOverlap(nullptr, &rejected))
      << "A missing left member set must not overlap another set";
  EXPECT_FALSE(occlIbMemberSetsOverlap(&rejected, nullptr))
      << "A missing right member set must not overlap another set";
  int rejectedAlias = 73;
  EXPECT_EQ(occlIbMakeForcedAliases(&rejectedAlias, &rejected), ncclInvalidUsage)
      << "Forced creation must reject an exhausted alias inventory";
  EXPECT_EQ(rejectedAlias, 73)
      << "Capacity rejection must preserve the caller's output index";
  EXPECT_EQ(ncclNMergedIbDevs, MAX_IB_VDEVS)
      << "Capacity rejection must preserve the published-device count";
  EXPECT_EQ(std::memcmp(&rejected, rejectedBefore, sizeof(rejected)), 0)
      << "Capacity rejection must preserve caller-owned properties";
  EXPECT_EQ(std::memcmp(ncclIbMergedDevs, inventoryBefore.data(),
                sizeof(ncclIbMergedDevs)), 0)
      << "Capacity rejection must preserve the merged-device inventory";
}

/**
 * @brief Verifies pair properties intersect constraints across private standby members.
 * @note Side effects: Publishes aliases and records per-device DMA-BUF queries.
 */
TEST_F(ForcedMergeInitTest, PairPropertiesIncludePrivateRegistrationConstraints) {
  state.useDeviceDmaBufResults = true;
  ncclNIbDevs = 4;
  for (int dev = 0; dev < ncclNIbDevs; dev++) {
    const std::string name = "mlx5_" + std::to_string(dev);
    ConfigureDevice(dev, name.c_str(), 100);
    SetPciPath(dev, dev < 2 ? "/sys/devices/pci0000:00/0000:01:00.0"
                          : "/sys/devices/pci0000:00/0000:02:00.0");
    state.deviceDmaBufSupport[dev] = true;
  }
  state.deviceDmaBufSupport[3] = false;
  ncclIbDevs[3].maxQp = 7;
  ncclIbDevs[3].capsProvider.mlx5.dataDirect = 1;
  ASSERT_EQ(setenv("NCCL_NET_FORCE_MERGE", "mlx5_0,mlx5_1,mlx5_2,mlx5_3", 1), 0);
  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = ncclNIbDevs;
  for (int member = 0; member < requested.ndevs; member++) requested.devs[member] = member;
  int firstAlias = -1;
  ASSERT_EQ(ncclIbMakeVDeviceInternal(&firstAlias, &requested), ncclSuccess);

  ncclNetProperties_t properties = {};
  ASSERT_EQ(ncclIbGetProperties(firstAlias, &properties), ncclSuccess);
  EXPECT_EQ(properties.ptrSupport, NCCL_PTR_HOST);
  EXPECT_EQ(properties.regIsGlobal, 1);
  EXPECT_EQ(properties.forceFlush, 1);
  EXPECT_EQ(properties.maxComms, 7);
  EXPECT_EQ(properties.maxRecvs, NCCL_NET_IB_MAX_RECVS);
  EXPECT_EQ(properties.maxP2pBytes, NCCL_MAX_NET_SIZE_BYTES);
  EXPECT_EQ(properties.maxCollBytes, MAX_COLLNET_SIZE);
  EXPECT_EQ(properties.maxMultiRequestSize, 1);
  EXPECT_EQ(properties.speed, 200);
  EXPECT_STREQ(properties.name, "mlx5_0+mlx5_1");
  EXPECT_EQ(properties.guid, ncclIbDevs[0].guid);
  EXPECT_EQ(properties.pciPath, ncclIbDevs[0].pciPath);
  EXPECT_EQ(state.dmaBufSupportDevices, std::vector<int>({0, 1, 2, 3}))
      << "Property intersection must query every private physical member";

  state.deviceDmaBufSupport[3] = true;
  state.dmaBufSupportDevices.clear();
  ASSERT_EQ(ncclIbGetProperties(firstAlias, &properties), ncclSuccess);
  EXPECT_EQ(properties.ptrSupport, NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF);
  EXPECT_EQ(state.dmaBufSupportDevices, std::vector<int>({0, 1, 2, 3}))
      << "Repeated property lookup must retain physical-device DMA-BUF indexing";
}

/**
 * @brief Verifies pair-speed overflow rejects the group without publishing aliases.
 * @note Side effects: Configures two maximum-speed physical devices.
 */
TEST_F(ForcedMergeInitTest, PairSpeedOverflowDoesNotPublishAliases) {
  ncclNIbDevs = 2;
  ConfigureDevice(0, "mlx5_0", std::numeric_limits<int>::max());
  ConfigureDevice(1, "mlx5_1", 1);
  SetPciPath(1, ncclIbDevs[0].pciPath);
  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = 2;
  requested.devs[0] = 0;
  requested.devs[1] = 1;
  unsigned char requestedBefore[sizeof(requested)] = {};
  std::vector<unsigned char> inventoryBefore(sizeof(ncclIbMergedDevs));
  std::memcpy(requestedBefore, &requested, sizeof(requested));
  std::memcpy(inventoryBefore.data(), ncclIbMergedDevs, sizeof(ncclIbMergedDevs));
  int firstAlias = 79;
  EXPECT_EQ(occlIbMakeForcedAliases(&firstAlias, &requested), ncclInvalidUsage)
      << "Direct forced creation must reject an active speed outside the public int range";
  EXPECT_EQ(firstAlias, 79)
      << "Speed rejection must preserve the caller's output index";
  EXPECT_EQ(ncclNMergedIbDevs, 0)
      << "Speed rejection must not publish an alias";
  EXPECT_EQ(std::memcmp(&requested, requestedBefore, sizeof(requested)), 0)
      << "Speed rejection must preserve caller-owned properties";
  EXPECT_EQ(std::memcmp(ncclIbMergedDevs, inventoryBefore.data(),
                sizeof(ncclIbMergedDevs)), 0)
      << "Speed rejection must preserve the merged-device inventory";
}

/**
 * @brief Verifies repeated creation of the same forced group is idempotent.
 * @note Side effects: Publishes one forced alias group.
 */
TEST_F(ForcedMergeInitTest, RepeatedGroupIsIdempotent) {
  ncclNIbDevs = 3;
  ConfigureDevice(0, "mlx5_0", 100);
  ConfigureDevice(1, "mlx5_1", 200);
  ConfigureDevice(2, "mlx5_2", 300);
  ASSERT_EQ(setenv("NCCL_NET_FORCE_MERGE", "mlx5_0,mlx5_1,mlx5_2", 1), 0);

  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = 3;
  requested.devs[0] = 0;
  requested.devs[1] = 1;
  requested.devs[2] = 2;

  int firstAlias = -1;
  ASSERT_EQ(ncclIbMakeVDeviceInternal(&firstAlias, &requested), ncclSuccess);
  ASSERT_EQ(firstAlias, 0);
  ASSERT_EQ(ncclNMergedIbDevs, 3);

  int repeatedAlias = -1;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&repeatedAlias, &requested), ncclSuccess);
  EXPECT_EQ(repeatedAlias, firstAlias);
  EXPECT_EQ(ncclNMergedIbDevs, 3);
}

/**
 * @brief Verifies direct overlap and duplicate-member rejection preserve existing aliases.
 * @note Side effects: Publishes one group and attempts two rejected groups.
 */
TEST_F(ForcedMergeInitTest, InvalidGroupsDoNotMutatePublishedAliases) {
  ncclNIbDevs = 4;
  ConfigureDevice(0, "mlx5_0", 100);
  ConfigureDevice(1, "mlx5_1", 200);
  ConfigureDevice(2, "mlx5_2", 300);
  ConfigureDevice(3, "mlx5_3", 400);
  ASSERT_EQ(setenv("NCCL_NET_FORCE_MERGE", "mlx5_0,mlx5_1,mlx5_2", 1), 0);

  ncclNetVDeviceProps_t initial = {};
  initial.ndevs = 3;
  initial.devs[0] = 0;
  initial.devs[1] = 1;
  initial.devs[2] = 2;
  int firstAlias = -1;
  ASSERT_EQ(ncclIbMakeVDeviceInternal(&firstAlias, &initial), ncclSuccess);
  ASSERT_EQ(ncclNMergedIbDevs, 3);
  std::vector<unsigned char> inventoryBefore(sizeof(ncclIbMergedDevs));
  std::memcpy(inventoryBefore.data(), ncclIbMergedDevs, sizeof(ncclIbMergedDevs));

  ncclNetVDeviceProps_t overlapping = {};
  overlapping.ndevs = 2;
  overlapping.devs[0] = 2;
  overlapping.devs[1] = 3;
  unsigned char overlappingBefore[sizeof(overlapping)] = {};
  std::memcpy(overlappingBefore, &overlapping, sizeof(overlapping));
  int rejectedAlias = 83;
  EXPECT_EQ(occlIbMakeForcedAliases(&rejectedAlias, &overlapping), ncclInvalidUsage)
      << "Direct forced creation must reject overlap while alias capacity remains";
  EXPECT_EQ(rejectedAlias, 83)
      << "Overlap rejection must preserve the caller's output index";
  EXPECT_EQ(ncclNMergedIbDevs, 3)
      << "Overlap rejection must preserve the published-device count";
  EXPECT_EQ(std::memcmp(&overlapping, overlappingBefore, sizeof(overlapping)), 0)
      << "Overlap rejection must preserve caller-owned properties";
  EXPECT_EQ(std::memcmp(ncclIbMergedDevs, inventoryBefore.data(),
                sizeof(ncclIbMergedDevs)), 0)
      << "Overlap rejection must preserve the merged-device inventory";

  ncclNetVDeviceProps_t duplicate = {};
  duplicate.ndevs = 2;
  duplicate.devs[0] = 3;
  duplicate.devs[1] = 3;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&rejectedAlias, &duplicate), ncclInvalidUsage);
  EXPECT_EQ(ncclNMergedIbDevs, 3);
  EXPECT_EQ(ncclIbMergedDevs[firstAlias].publicVProps.devs[0], 0);
  EXPECT_EQ(ncclIbMergedDevs[firstAlias + 1].publicVProps.devs[0], 1);
  EXPECT_EQ(ncclIbMergedDevs[firstAlias + 2].publicVProps.devs[0], 2);
}

/**
 * @brief Verifies an absent force-merge environment retains one legacy aggregate.
 * @note Side effects: Temporarily updates and restores the force-merge environment, publishes
 * one legacy merged device, and queries its properties.
 */
TEST_F(ForcedMergeInitTest, MissingForceEnvironmentKeepsLegacyAggregate) {
  ncclNIbDevs = 3;
  ConfigureDevice(0, "mlx5_0", 100);
  ConfigureDevice(1, "mlx5_1", 200);
  ConfigureDevice(2, "mlx5_2", 300);
  SetPciPath(1, ncclIbDevs[0].pciPath);
  EXPECT_FALSE(occlIbForceMergeAliasesEnabled())
      << "An absent force-merge value must retain legacy aggregation";
  ASSERT_EQ(setenv("NCCL_NET_FORCE_MERGE", "", 1), 0)
      << "The test must install an empty force-merge value";
  EXPECT_FALSE(occlIbForceMergeAliasesEnabled())
      << "An empty force-merge value must retain legacy aggregation";
  EXPECT_STREQ(std::getenv("NCCL_NET_FORCE_MERGE"), "")
      << "Force-merge detection must preserve an empty environment value";
  ASSERT_EQ(unsetenv("NCCL_NET_FORCE_MERGE"), 0)
      << "The test must restore the absent environment before legacy creation";

  ncclNetVDeviceProps_t requested = {};
  requested.ndevs = 3;
  requested.devs[0] = 2;
  requested.devs[1] = 0;
  requested.devs[2] = 1;

  int aggregate = -1;
  ASSERT_EQ(ncclIbMakeVDeviceInternal(&aggregate, &requested), ncclSuccess);
  ASSERT_EQ(aggregate, 0);
  ASSERT_EQ(ncclNMergedIbDevs, 1);

  const ncclIbMergedDev& merged = ncclIbMergedDevs[aggregate];
  EXPECT_EQ(merged.isForcedAlias, 0);
  EXPECT_EQ(merged.anchorDev, 2);
  EXPECT_EQ(merged.firstAlias, aggregate);
  EXPECT_EQ(merged.aliasCount, 1);
  EXPECT_EQ(merged.vProps.ndevs, requested.ndevs);
  EXPECT_EQ(merged.publicVProps.ndevs, requested.ndevs);
  for (int member = 0; member < requested.ndevs; member++) {
    EXPECT_EQ(merged.vProps.devs[member], requested.devs[member]);
    EXPECT_EQ(merged.publicVProps.devs[member], requested.devs[member]);
  }
  EXPECT_EQ(merged.speed, 600);
  EXPECT_STREQ(merged.devName, "mlx5_2+mlx5_0+mlx5_1");

  ncclNetProperties_t properties = {};
  ASSERT_EQ(ncclIbGetProperties(aggregate, &properties), ncclSuccess);
  EXPECT_STREQ(properties.name, "mlx5_2+mlx5_0+mlx5_1");
  EXPECT_EQ(properties.speed, 600);
  EXPECT_EQ(properties.vProps.ndevs, requested.ndevs);
  for (int member = 0; member < requested.ndevs; member++) {
    EXPECT_EQ(properties.vProps.devs[member], requested.devs[member]);
  }
}
