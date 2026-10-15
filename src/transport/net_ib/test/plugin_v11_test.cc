/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "transport/net_ib/common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>

namespace {

/** @brief Identifies the backend operation recorded by the plugin adapter seam. */
enum class BackendOperation {
  None,
  Init,
  Finalize,
  Devices,
  GetProperties,
  Listen,
  Connect,
  Accept,
  RegMr,
  RegMrDmaBuf,
  DeregMr,
  Isend,
  Irecv,
  Iflush,
  Test,
  CloseSend,
  CloseRecv,
  CloseListen,
  MakeVDevice,
  SetNetAttr,
};

/**
 * @brief Configures backend outcomes and records the latest operation, forwarded
 * arguments, aliasing, and output interactions across all ABIs.
 */
struct PluginBackendState {
  BackendOperation operation = BackendOperation::None;
  ncclResult_t result = ncclSuccess;
  int calls = 0;
  bool writeOutputs = true;
  void** contextOutput = nullptr;
  void* contextArgument = nullptr;
  void* contextResult = nullptr;
  uint64_t commId = 0;
  ncclNetCommConfig_t* configArgument = nullptr;
  ncclNetCommConfig_t configValue = {};
  bool captureConnectConfig = false;
  bool connectContextWasNull = false;
  ncclNetCommConfig_t connectConfigValue = {};
  ncclDebugLogger_t logger = nullptr;
  ncclProfilerCallback_t profiler = nullptr;
  int dev = 0;
  int deviceCount = 0;
  void* handle = nullptr;
  void* communicator = nullptr;
  void** communicatorOutput = nullptr;
  void* communicatorResult = nullptr;
  ncclNetDeviceHandle_t** deviceHandleOutput = nullptr;
  bool deviceHandleOutputWasNull = false;
  ncclNetDeviceHandle_t* deviceHandleResult = nullptr;
  void* data = nullptr;
  size_t size = 0;
  int type = 0;
  uint64_t offset = 0;
  int fd = 0;
  int tag = 0;
  void* memoryHandle = nullptr;
  void* profilingHandle = nullptr;
  void** requestOutput = nullptr;
  void* requestResult = nullptr;
  int n = 0;
  void** dataArray = nullptr;
  size_t* sizeArray = nullptr;
  const void* forbiddenSizeArray = nullptr;
  bool sizeArrayAliasesForbidden = false;
  int* intSizeArray = nullptr;
  int* tagArray = nullptr;
  void** memoryHandleArray = nullptr;
  void** profilingHandleArray = nullptr;
  std::array<size_t, NCCL_NET_MAX_REQUESTS> copiedSizes = {};
  void* request = nullptr;
  int* doneOutput = nullptr;
  int doneResult = 0;
  int* completedSizes = nullptr;
  int completedSizeResult = 0;
  int* virtualDeviceOutput = nullptr;
  int virtualDeviceResult = 0;
  ncclNetVDeviceProps_t virtualProperties = {};
  ncclNetProperties_t* propertiesOutput = nullptr;
  ncclNetProperties_t propertiesResult = {};
  ncclNetAttr_t* attributes = nullptr;
  bool attributesWereNull = false;
  ncclNetAttr_t attributesValue = {};
};

PluginBackendState backend;
int contextToken;
int listenToken;
int sendToken;
int receiveToken;
int memoryToken;
int requestToken;
ncclNetDeviceHandle_t deviceHandleToken = {};
char deviceName[] = "adapter-test-device";
char pciPath[] = "/sys/devices/adapter-test";

constexpr std::array<ncclResult_t, 6> kDelegatedErrors = {
    ncclUnhandledCudaError,
    ncclSystemError,
    ncclInternalError,
    ncclInvalidArgument,
    ncclInvalidUsage,
    ncclRemoteError,
};

/**
 * @brief Restores the standard successful backend scenario with distinct sentinel outputs.
 * @note Side effects: Replaces every field in backend and installs sentinel result
 * pointers and values.
 */
void ResetBackend() {
  backend = PluginBackendState{};
  backend.contextResult = &contextToken;
  backend.communicatorResult = &listenToken;
  backend.deviceHandleResult = &deviceHandleToken;
  backend.requestResult = &requestToken;
  backend.deviceCount = 7;
  backend.doneResult = 1;
  backend.completedSizeResult = 1234;
  backend.virtualDeviceResult = 9;
}

/**
 * @brief Supplies distinct current-version properties for projection tests.
 * @param[in] ndevs Device count to encode in [0, NCCL_NET_MAX_DEVS_PER_NIC].
 * @return A fully initialized current-version property value with ndevs active device indices.
 * @note This function has no side effects.
 */
ncclNetProperties_t MakeProperties(int ndevs) {
  ncclNetProperties_t properties = {};
  properties.name = deviceName;
  properties.pciPath = pciPath;
  properties.guid = 0x123456789abcdef0ULL;
  properties.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  properties.regIsGlobal = 1;
  properties.forceFlush = 0;
  properties.speed = 700000;
  properties.port = 11;
  properties.latency = 12.5f;
  properties.maxComms = 13;
  properties.maxRecvs = 8;
  properties.netDeviceType = NCCL_NET_DEVICE_UNPACK;
  properties.netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
  properties.vProps.ndevs = ndevs;
  for (int i = 0; i < ndevs; ++i) properties.vProps.devs[i] = NCCL_NET_MAX_DEVS_PER_NIC - 1 - i;
  properties.maxP2pBytes = 23;
  properties.maxCollBytes = 29;
  properties.maxMultiRequestSize = 31;
  return properties;
}

/**
 * @brief Provides a stable logger callback address for forwarding tests.
 * @param[in] level Log level in the ncclDebugLogLevel domain; ignored.
 * @param[in] flags Log subsystem flags in [0, ULONG_MAX]; ignored.
 * @param[in] file NULL or a source-file name; ignored.
 * @param[in] line Source line in [INT_MIN, INT_MAX]; ignored.
 * @param[in] format NULL or a printf-compatible format string; ignored with all variadic arguments.
 * @note This function has no side effects.
 */
void LoggerForTest(ncclDebugLogLevel level, unsigned long flags, const char* file, int line, const char* format, ...) {
  (void)level;
  (void)flags;
  (void)file;
  (void)line;
  (void)format;
}

/**
 * @brief Provides a stable profiler callback address for forwarding tests.
 * @param[in] eventHandle NULL or valid event-handle storage; ignored.
 * @param[in] type Event type in [INT_MIN, INT_MAX]; ignored.
 * @param[in] parentHandle NULL or an opaque parent handle; ignored.
 * @param[in] pluginId Plugin identifier in [INT64_MIN, INT64_MAX]; ignored.
 * @param[in] externalData NULL or opaque event data; ignored.
 * @return ncclSuccess.
 * @note This function has no side effects.
 */
ncclResult_t ProfilerForTest(void** eventHandle, int type, void* parentHandle, int64_t pluginId,
                            void* externalData) {
  (void)eventHandle;
  (void)type;
  (void)parentHandle;
  (void)pluginId;
  (void)externalData;
  return ncclSuccess;
}

/**
 * @brief Verifies the property fields shared by every supported legacy ABI.
 * @tparam Properties Legacy property structure accepted by the checked ABI.
 * @param[in] actual Legacy property value produced by the adapter.
 * @param[in] expected Current property value supplied by the backend.
 * @param[in] version Non-NULL ABI label used in failure diagnostics.
 * @note Side effects: Records GoogleTest failures for fields that were not projected exactly.
 */
template <typename Properties>
void ExpectBaseProperties(const Properties& actual, const ncclNetProperties_t& expected, const char* version) {
  SCOPED_TRACE(version);
  EXPECT_EQ(actual.name, expected.name) << "The device-name pointer was not preserved";
  EXPECT_EQ(actual.pciPath, expected.pciPath) << "The PCI-path pointer was not preserved";
  EXPECT_EQ(actual.guid, expected.guid) << "The GUID was not copied";
  EXPECT_EQ(actual.ptrSupport, expected.ptrSupport) << "The pointer-support flags were not copied";
  EXPECT_EQ(actual.speed, expected.speed) << "The port speed was not copied";
  EXPECT_EQ(actual.port, expected.port) << "The port number was not copied";
  EXPECT_FLOAT_EQ(actual.latency, expected.latency) << "The latency was not copied";
  EXPECT_EQ(actual.maxComms, expected.maxComms) << "The communicator limit was not copied";
  EXPECT_EQ(actual.maxRecvs, expected.maxRecvs) << "The grouped-receive limit was not copied";
}

/**
 * @brief Verifies fields introduced in the v7 property ABI.
 * @tparam Properties v7-compatible property structure.
 * @param[in] actual v7-compatible property value produced by the adapter.
 * @param[in] expected Current property value supplied by the backend.
 * @param[in] version Non-NULL ABI label used in failure diagnostics.
 * @note Side effects: Records GoogleTest failures for fields that were not projected exactly.
 */
template <typename Properties>
void ExpectV7Properties(const Properties& actual, const ncclNetProperties_t& expected, const char* version) {
  ExpectBaseProperties(actual, expected, version);
  SCOPED_TRACE(version);
  EXPECT_EQ(actual.netDeviceType, expected.netDeviceType) << "The network-device type was not copied";
  EXPECT_EQ(actual.netDeviceVersion, expected.netDeviceVersion) << "The network-device version was not copied";
}

/**
 * @brief Verifies fields introduced in the v8 property ABI.
 * @tparam Properties v8-compatible property structure.
 * @param[in] actual v8-compatible property value produced by the adapter.
 * @param[in] expected Current property value supplied by the backend.
 * @param[in] version Non-NULL ABI label used in failure diagnostics.
 * @note Side effects: Records GoogleTest failures for fields that were not projected exactly.
 */
template <typename Properties>
void ExpectV8Properties(const Properties& actual, const ncclNetProperties_t& expected, const char* version) {
  ExpectV7Properties(actual, expected, version);
  SCOPED_TRACE(version);
  EXPECT_EQ(actual.regIsGlobal, expected.regIsGlobal) << "The global-registration flag was not copied";
}

/**
 * @brief Verifies fields shared by the v9 and v10 property ABIs.
 * @tparam Properties v9- or v10-compatible property structure.
 * @param[in] actual v9- or v10-compatible property value produced by the adapter.
 * @param[in] expected Current property value supplied by the backend.
 * @param[in] version Non-NULL ABI label used in failure diagnostics.
 * @note Side effects: Records GoogleTest failures for fields that were not projected exactly.
 */
template <typename Properties>
void ExpectV10Properties(const Properties& actual, const ncclNetProperties_t& expected, const char* version) {
  ExpectV8Properties(actual, expected, version);
  SCOPED_TRACE(version);
  EXPECT_EQ(actual.forceFlush, expected.forceFlush) << "The force-flush flag was not copied";
  EXPECT_EQ(actual.vProps.ndevs, expected.vProps.ndevs) << "The active virtual-device count was not copied";
  for (int i = 0; i < expected.vProps.ndevs; ++i) {
    EXPECT_EQ(actual.vProps.devs[i], expected.vProps.devs[i]) << "Virtual-device index " << i << " was not copied";
  }
  EXPECT_EQ(actual.maxP2pBytes, expected.maxP2pBytes) << "The point-to-point size limit was not copied";
  EXPECT_EQ(actual.maxCollBytes, expected.maxCollBytes) << "The collective size limit was not copied";
}

/**
 * @brief Verifies that a direct adapter returns every delegated error without translating it.
 * @tparam Callable Callable type accepting no arguments and returning ncclResult_t.
 * @param[in] operation Non-NULL operation label used in failure diagnostics.
 * @param[in] invoke Callable accepting no arguments and invoking one adapter function.
 * @note Side effects: Leaves backend.result at the last delegated error and
 * backend.writeOutputs false, updates backend call records, and records GoogleTest failures.
 */
template <typename Callable>
void ExpectDelegatedErrors(const char* operation, Callable invoke) {
  for (ncclResult_t error : kDelegatedErrors) {
    SCOPED_TRACE(std::string(operation) + " result=" + std::to_string(static_cast<int>(error)));
    backend.result = error;
    backend.writeOutputs = false;
    int callsBefore = backend.calls;
    EXPECT_EQ(invoke(), error) << operation << " did not return the backend error unchanged";
    EXPECT_EQ(backend.calls, callsBefore + 1) << operation << " did not call the backend exactly once";
  }
}

}  // namespace

/**
 * @brief Records current network initialization and supplies its configured context result.
 * @param[out] ctx Valid non-NULL context storage written on configured success.
 * @param[in] commId Communicator identifier in [0, UINT64_MAX].
 * @param[in] config NULL or a communication configuration copied for inspection.
 * @param[in] logFunction NULL or the logger callback forwarded by the adapter.
 * @param[in] profFunction NULL or the profiler callback forwarded by the adapter.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write *ctx.
 */
ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config,
                          ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  backend.operation = BackendOperation::Init;
  backend.calls++;
  backend.contextOutput = ctx;
  backend.commId = commId;
  backend.configArgument = config;
  if (config != nullptr) backend.configValue = *config;
  backend.logger = logFunction;
  backend.profiler = profFunction;
  if (backend.result == ncclSuccess && backend.writeOutputs) *ctx = backend.contextResult;
  return backend.result;
}

/**
 * @brief Records network finalization.
 * @param[in] ctx NULL or the context supplied by the adapter.
 * @return The configured backend result.
 * @note Side effects: Updates backend.
 */
ncclResult_t ncclIbFinalize(void* ctx) {
  backend.operation = BackendOperation::Finalize;
  backend.calls++;
  backend.contextArgument = ctx;
  return backend.result;
}

/**
 * @brief Records device discovery and supplies the configured device count.
 * @param[out] ndev Valid non-NULL device-count storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write *ndev.
 */
ncclResult_t ncclIbDevices(int* ndev) {
  backend.operation = BackendOperation::Devices;
  backend.calls++;
  if (backend.result == ncclSuccess && backend.writeOutputs) *ndev = backend.deviceCount;
  return backend.result;
}

/**
 * @brief Records property discovery and supplies the configured current-version properties.
 * @param[in] dev Device index in [INT_MIN, INT_MAX] forwarded by the adapter.
 * @param[out] properties Valid non-NULL property storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write *properties.
 */
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* properties) {
  backend.operation = BackendOperation::GetProperties;
  backend.calls++;
  backend.dev = dev;
  backend.propertiesOutput = properties;
  if (backend.result == ncclSuccess && backend.writeOutputs) *properties = backend.propertiesResult;
  return backend.result;
}

/**
 * @brief Records listener creation and supplies the configured listener communicator.
 * @param[in] ctx NULL or the communicator context forwarded by the adapter.
 * @param[in] dev Device index in [INT_MIN, INT_MAX].
 * @param[in] handle Valid opaque handle storage recorded unchanged.
 * @param[out] listenComm Valid non-NULL listener storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write *listenComm.
 */
ncclResult_t ncclIbListen(void* ctx, int dev, void* handle, void** listenComm) {
  backend.operation = BackendOperation::Listen;
  backend.calls++;
  backend.contextArgument = ctx;
  backend.dev = dev;
  backend.handle = handle;
  backend.communicatorOutput = listenComm;
  if (backend.result == ncclSuccess && backend.writeOutputs) *listenComm = backend.communicatorResult;
  return backend.result;
}

/**
 * @brief Records sender connection and supplies configured host and device communicators.
 * @param[in] ctx NULL or the communicator context forwarded by the adapter.
 * @param[in] dev Device index in [INT_MIN, INT_MAX].
 * @param[in] handle Valid opaque connection state recorded unchanged.
 * @param[out] sendComm Valid non-NULL sender storage written on configured success.
 * @param[out] sendDevComm NULL or valid device-communicator storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write sender outputs.
 */
ncclResult_t ncclIbConnect(void* ctx, int dev, void* handle, void** sendComm,
                             ncclNetDeviceHandle_t** sendDevComm) {
  backend.operation = BackendOperation::Connect;
  backend.calls++;
  backend.contextArgument = ctx;
  backend.connectContextWasNull = ctx == nullptr;
  if (backend.captureConnectConfig && ctx != nullptr) {
    backend.connectConfigValue = *static_cast<ncclNetCommConfig_t*>(ctx);
  }
  backend.dev = dev;
  backend.handle = handle;
  backend.communicatorOutput = sendComm;
  backend.deviceHandleOutput = sendDevComm;
  backend.deviceHandleOutputWasNull = sendDevComm == nullptr;
  if (backend.result == ncclSuccess && backend.writeOutputs) {
    *sendComm = backend.communicatorResult;
    if (sendDevComm != nullptr) *sendDevComm = backend.deviceHandleResult;
  }
  return backend.result;
}

/**
 * @brief Records receiver acceptance and supplies configured host and device communicators.
 * @param[in] listenComm Valid listener communicator forwarded by the adapter.
 * @param[out] recvComm Valid non-NULL receiver storage written on configured success.
 * @param[out] recvDevComm NULL or valid device-communicator storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write receiver outputs.
 */
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  backend.operation = BackendOperation::Accept;
  backend.calls++;
  backend.communicator = listenComm;
  backend.communicatorOutput = recvComm;
  backend.deviceHandleOutput = recvDevComm;
  backend.deviceHandleOutputWasNull = recvDevComm == nullptr;
  if (backend.result == ncclSuccess && backend.writeOutputs) {
    *recvComm = backend.communicatorResult;
    if (recvDevComm != nullptr) *recvDevComm = backend.deviceHandleResult;
  }
  return backend.result;
}

/**
 * @brief Records ordinary memory registration and supplies a fixed fake handle.
 * @param[in] comm Valid communicator forwarded by the adapter.
 * @param[in] data Valid buffer address forwarded by the adapter.
 * @param[in] size Buffer size in [0, SIZE_MAX].
 * @param[in] type Registration-type flags in [INT_MIN, INT_MAX].
 * @param[out] mhandle Valid non-NULL memory-handle storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write *mhandle.
 */
ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  backend.operation = BackendOperation::RegMr;
  backend.calls++;
  backend.communicator = comm;
  backend.data = data;
  backend.size = size;
  backend.type = type;
  if (backend.result == ncclSuccess && backend.writeOutputs) *mhandle = &memoryToken;
  return backend.result;
}

/**
 * @brief Records DMA-BUF memory registration and supplies a fixed fake handle.
 * @param[in] comm Valid communicator forwarded by the adapter.
 * @param[in] data Valid buffer address forwarded by the adapter.
 * @param[in] size Buffer size in [0, SIZE_MAX].
 * @param[in] type Registration-type flags in [INT_MIN, INT_MAX].
 * @param[in] offset DMA-BUF offset in [0, UINT64_MAX].
 * @param[in] fd DMA-BUF descriptor in [INT_MIN, INT_MAX].
 * @param[out] mhandle Valid non-NULL memory-handle storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write *mhandle.
 */
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                void** mhandle) {
  backend.operation = BackendOperation::RegMrDmaBuf;
  backend.calls++;
  backend.communicator = comm;
  backend.data = data;
  backend.size = size;
  backend.type = type;
  backend.offset = offset;
  backend.fd = fd;
  if (backend.result == ncclSuccess && backend.writeOutputs) *mhandle = &memoryToken;
  return backend.result;
}

/**
 * @brief Records memory deregistration.
 * @param[in] comm Valid communicator forwarded by the adapter.
 * @param[in] mhandle Valid registered-memory handle forwarded by the adapter.
 * @return The configured backend result.
 * @note Side effects: Updates backend.
 */
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  backend.operation = BackendOperation::DeregMr;
  backend.calls++;
  backend.communicator = comm;
  backend.memoryHandle = mhandle;
  return backend.result;
}

/**
 * @brief Records asynchronous sending and supplies its configured request.
 * @param[in] sendComm Valid sender communicator forwarded by the adapter.
 * @param[in] data Valid payload address forwarded by the adapter.
 * @param[in] size Payload size in [0, SIZE_MAX].
 * @param[in] tag Match tag in [INT_MIN, INT_MAX].
 * @param[in] mhandle Valid registered-memory handle forwarded by the adapter.
 * @param[in] phandle NULL or profiling handle forwarded by the adapter.
 * @param[out] request Valid non-NULL request storage written when output writing is enabled and
 * the result is successful.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write *request when output writing is enabled and
 * the result is successful.
 */
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                          void** request) {
  backend.operation = BackendOperation::Isend;
  backend.calls++;
  backend.communicator = sendComm;
  backend.data = data;
  backend.size = size;
  backend.tag = tag;
  backend.memoryHandle = mhandle;
  backend.profilingHandle = phandle;
  backend.requestOutput = request;
  if (backend.result == ncclSuccess && backend.writeOutputs) *request = backend.requestResult;
  return backend.result;
}

/**
 * @brief Records grouped receiving, including copied size values, and supplies its configured
 * request.
 * @param[in] recvComm Valid receiver communicator forwarded by the adapter.
 * @param[in] n Array length in [1, NCCL_NET_MAX_REQUESTS].
 * @param[in] data Valid non-NULL array of n buffer addresses.
 * @param[in] sizes Valid non-NULL array of n sizes in [0, SIZE_MAX].
 * @param[in] tags Valid non-NULL array of n tags in [INT_MIN, INT_MAX].
 * @param[in] mhandles Valid non-NULL array of n registered-memory handles.
 * @param[in] phandles NULL or an array of n profiling handles.
 * @param[out] request Valid non-NULL request storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend, copies n sizes for inspection, and may write *request.
 */
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles,
                          void** phandles, void** request) {
  backend.operation = BackendOperation::Irecv;
  backend.calls++;
  backend.communicator = recvComm;
  backend.n = n;
  backend.dataArray = data;
  backend.sizeArray = sizes;
  backend.sizeArrayAliasesForbidden = static_cast<const void*>(sizes) == backend.forbiddenSizeArray;
  backend.tagArray = tags;
  backend.memoryHandleArray = mhandles;
  backend.profilingHandleArray = phandles;
  backend.requestOutput = request;
  int copied = std::max(0, std::min(n, NCCL_NET_MAX_REQUESTS));
  for (int i = 0; i < copied; ++i) backend.copiedSizes[i] = sizes[i];
  if (backend.result == ncclSuccess && backend.writeOutputs) *request = backend.requestResult;
  return backend.result;
}

/**
 * @brief Records receive flushing and supplies its configured request.
 * @param[in] recvComm Valid receiver communicator forwarded by the adapter.
 * @param[in] n Array length in [0, INT_MAX].
 * @param[in] data NULL or an array of n buffer addresses.
 * @param[in] sizes NULL or an array of n byte counts.
 * @param[in] mhandles NULL or an array of n registered-memory handles.
 * @param[out] request Valid non-NULL request storage written when output writing is enabled,
 * the result is successful, and n > 0.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write *request when output writing is enabled,
 * the result is successful, and n > 0.
 */
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  backend.operation = BackendOperation::Iflush;
  backend.calls++;
  backend.communicator = recvComm;
  backend.n = n;
  backend.dataArray = data;
  backend.intSizeArray = sizes;
  backend.memoryHandleArray = mhandles;
  backend.requestOutput = request;
  if (backend.result == ncclSuccess && backend.writeOutputs && n > 0) *request = backend.requestResult;
  return backend.result;
}

/**
 * @brief Records request polling and supplies configured completion outputs.
 * @param[in] request Valid request forwarded by the adapter.
 * @param[out] done Valid non-NULL completion storage written on configured success.
 * @param[out] sizes NULL or valid completed-size storage written on configured success.
 * @return The configured backend result.
 * @note Side effects: Updates backend and may write completion outputs.
 */
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  backend.operation = BackendOperation::Test;
  backend.calls++;
  backend.request = request;
  backend.doneOutput = done;
  backend.completedSizes = sizes;
  if (backend.result == ncclSuccess && backend.writeOutputs) {
    *done = backend.doneResult;
    if (sizes != nullptr) sizes[0] = backend.completedSizeResult;
  }
  return backend.result;
}

/**
 * @brief Records sender closure.
 * @param[in] sendComm Valid sender communicator forwarded by the adapter.
 * @return The configured backend result.
 * @note Side effects: Updates backend.
 */
ncclResult_t ncclIbCloseSend(void* sendComm) {
  backend.operation = BackendOperation::CloseSend;
  backend.calls++;
  backend.communicator = sendComm;
  return backend.result;
}

/**
 * @brief Records receiver closure.
 * @param[in] recvComm Valid receiver communicator forwarded by the adapter.
 * @return The configured backend result.
 * @note Side effects: Updates backend.
 */
ncclResult_t ncclIbCloseRecv(void* recvComm) {
  backend.operation = BackendOperation::CloseRecv;
  backend.calls++;
  backend.communicator = recvComm;
  return backend.result;
}

/**
 * @brief Records listener closure.
 * @param[in] listenComm Valid listener communicator forwarded by the adapter.
 * @return The configured backend result.
 * @note Side effects: Updates backend.
 */
ncclResult_t ncclIbCloseListen(void* listenComm) {
  backend.operation = BackendOperation::CloseListen;
  backend.calls++;
  backend.communicator = listenComm;
  return backend.result;
}

/**
 * @brief Records virtual-device creation and supplies its configured index.
 * @param[out] d Valid non-NULL virtual-device index storage written on configured success.
 * @param[in] properties Valid non-NULL properties whose ndevs is in [0, NCCL_NET_MAX_DEVS_PER_NIC].
 * @return The configured backend result.
 * @note Side effects: Updates backend, copies the properties, and may write *d.
 */
ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* properties) {
  backend.operation = BackendOperation::MakeVDevice;
  backend.calls++;
  backend.virtualDeviceOutput = d;
  backend.virtualProperties = *properties;
  if (backend.result == ncclSuccess && backend.writeOutputs) *d = backend.virtualDeviceResult;
  return backend.result;
}

/**
 * @brief Records current network-attribute calls made directly or through the v11 bridge.
 * @param[in] ctx NULL or the communicator context supplied by the caller.
 * @param[in] attributes NULL or current-version network attributes copied for inspection.
 * @return The configured backend result.
 * @note Side effects: Updates backend and copies @p attributes when present.
 */
ncclResult_t ncclIbSetNetAttr(void* ctx, ncclNetAttr_t* attributes) {
  backend.operation = BackendOperation::SetNetAttr;
  backend.calls++;
  backend.contextArgument = ctx;
  backend.attributes = attributes;
  backend.attributesWereNull = attributes == nullptr;
  if (attributes != nullptr) backend.attributesValue = *attributes;
  return backend.result;
}

/**
 * @brief Compiles the adapter beside its recording backend so internal legacy state is
 * directly testable.
 */
#include "../../../plugin/net/net_ib_plugin_v11.cc"  // NOLINT(build/include)

namespace {

/** @brief Provides isolated backend and legacy-adapter state for plugin ABI tests. */
class PluginAdapterTest : public ::testing::Test {
 protected:
  /**
   * @brief Restores the standard backend scenario and clears translation-unit-local
   * legacy adapter state.
   * @note Side effects: Replaces backend and clears gLegacyCtx and gLegacyRefCount.
   */
  void SetUp() override {
    ResetBackend();
    gLegacyCtx = nullptr;
    gLegacyRefCount = 0;
  }
};

/** @brief Groups exported-table and initialization contracts. */
class PluginAbiTest : public PluginAdapterTest {};

/** @brief Groups cross-version property and virtual-device conversions. */
class PluginCompatibilityTest : public PluginAdapterTest {};

/** @brief Groups listener, connector, and acceptor contracts. */
class PluginConnectionTest : public PluginAdapterTest {};

/** @brief Groups memory, transfer, completion, and close contracts. */
class PluginDataPathTest : public PluginAdapterTest {};

/**
 * @brief Verifies every v6-v12 table slot against the implementation selected for that ABI.
 * @note Side effects: Records GoogleTest failures; production and backend state remain unchanged.
 */
TEST_F(PluginAbiTest, VersionTables) {
  // Every table must advertise the same transport name.
  EXPECT_STREQ(ncclNetPlugin_v12.name, "IB") << "The v12 plugin name is not IB";
  EXPECT_STREQ(ncclNetPlugin_v11.name, "IB") << "The v11 plugin name is not IB";
  EXPECT_STREQ(ncclNetPlugin_v10.name, "IB") << "The v10 plugin name is not IB";
  EXPECT_STREQ(ncclNetPlugin_v9.name, "IB") << "The v9 plugin name is not IB";
  EXPECT_STREQ(ncclNetPlugin_v8.name, "IB") << "The v8 plugin name is not IB";
  EXPECT_STREQ(ncclNetPlugin_v7.name, "IB") << "The v7 plugin name is not IB";
  EXPECT_STREQ(ncclNetPlugin_v6.name, "IB") << "The v6 plugin name is not IB";

#define EXPECT_ADAPTER_SLOT(table, slot, expected) \
  EXPECT_EQ((table).slot, (expected)) << #table "." #slot " is wired incorrectly"

  // The v12 table exposes the current API and only its supported optional operations.
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, init, pluginInit);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, devices, pluginDevices);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, getProperties, occlPluginGetPropertiesV12);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, listen, pluginListen);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, connect, pluginConnect);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, accept, pluginAccept);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, regMr, pluginRegMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, regMrDmaBuf, pluginRegMrDmaBuf);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, deregMr, pluginDeregMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, isend, pluginIsend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, irecv, pluginIrecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, iflush, pluginIflush);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, test, pluginTest);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, closeSend, pluginCloseSend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, closeRecv, pluginCloseRecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, closeListen, pluginCloseListen);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, getDeviceMr, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, irecvConsumed, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, makeVDevice, occlPluginMakeVDeviceV12);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, finalize, pluginFinalize);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v12, setNetAttr, ncclIbSetNetAttr);

  // The v11 table uses conversion bridges where v12 changed a public structure.
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, init, occlPluginInitV11);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, devices, pluginDevices);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, getProperties, occlPluginGetPropertiesV11);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, listen, pluginListen);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, connect, pluginConnect);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, accept, pluginAccept);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, regMr, pluginRegMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, regMrDmaBuf, pluginRegMrDmaBuf);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, deregMr, pluginDeregMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, isend, pluginIsend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, irecv, pluginIrecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, iflush, pluginIflush);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, test, pluginTest);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, closeSend, pluginCloseSend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, closeRecv, pluginCloseRecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, closeListen, pluginCloseListen);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, getDeviceMr, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, irecvConsumed, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, makeVDevice, occlPluginMakeVDeviceV11);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, finalize, pluginFinalize);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v11, setNetAttr, occlPluginSetNetAttrV11);

  // The v10 table must preserve its config-aware connect and virtual-device adapters.
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, init, pluginInitV10);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, devices, pluginDevices);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, getProperties, pluginGetPropertiesV10);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, listen, pluginListenLegacy);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, connect, pluginConnectV10);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, accept, pluginAcceptV10);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, regMr, pluginRegMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, regMrDmaBuf, pluginRegMrDmaBuf);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, deregMr, pluginDeregMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, isend, pluginIsend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, irecv, pluginIrecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, iflush, pluginIflush);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, test, pluginTest);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, closeSend, pluginCloseSend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, closeRecv, pluginCloseRecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, closeListen, pluginCloseListen);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, getDeviceMr, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, irecvConsumed, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v10, makeVDevice, pluginMakeVDeviceV10);

  // The v9 table must remove connection config and profiling handles without changing shared slots.
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, init, pluginInitV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, devices, pluginDevices);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, getProperties, pluginGetPropertiesV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, listen, pluginListenLegacy);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, connect, pluginConnectV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, accept, pluginAcceptV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, regMr, pluginRegMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, regMrDmaBuf, pluginRegMrDmaBuf);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, deregMr, pluginDeregMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, isend, pluginIsendV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, irecv, pluginIrecvV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, iflush, pluginIflush);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, test, pluginTest);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, closeSend, pluginCloseSend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, closeRecv, pluginCloseRecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, closeListen, pluginCloseListen);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, getDeviceMr, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, irecvConsumed, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v9, makeVDevice, pluginMakeVDeviceV9);

  // The v8 table must select signed-size data adapters and expose no virtual-device operation.
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, init, pluginInitV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, devices, pluginDevices);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, getProperties, pluginGetPropertiesV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, listen, pluginListenLegacy);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, connect, pluginConnectV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, accept, pluginAcceptV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, regMr, pluginRegMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, regMrDmaBuf, pluginRegMrDmaBuf);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, deregMr, pluginDeregMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, isend, pluginIsendV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, irecv, pluginIrecvV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, iflush, pluginIflush);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, test, pluginTest);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, closeSend, pluginCloseSend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, closeRecv, pluginCloseRecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, closeListen, pluginCloseListen);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, getDeviceMr, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v8, irecvConsumed, nullptr);

  // The v7 table must additionally use signed-size memory registration.
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, init, pluginInitV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, devices, pluginDevices);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, getProperties, pluginGetPropertiesV7);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, listen, pluginListenLegacy);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, connect, pluginConnectV7);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, accept, pluginAcceptV7);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, regMr, pluginRegMrV7);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, regMrDmaBuf, pluginRegMrDmaBuf);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, deregMr, pluginDeregMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, isend, pluginIsendV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, irecv, pluginIrecvV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, iflush, pluginIflush);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, test, pluginTest);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, closeSend, pluginCloseSend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, closeRecv, pluginCloseRecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, closeListen, pluginCloseListen);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, getDeviceMr, nullptr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v7, irecvConsumed, nullptr);

  // The v6 table must contain exactly the oldest sixteen operations.
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, init, pluginInitV9);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, devices, pluginDevices);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, getProperties, pluginGetPropertiesV6);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, listen, pluginListenLegacy);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, connect, pluginConnectV6);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, accept, pluginAcceptV6);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, regMr, pluginRegMrV7);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, regMrDmaBuf, pluginRegMrDmaBuf);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, deregMr, pluginDeregMr);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, isend, pluginIsendV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, irecv, pluginIrecvV8);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, iflush, pluginIflush);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, test, pluginTest);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, closeSend, pluginCloseSend);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, closeRecv, pluginCloseRecv);
  EXPECT_ADAPTER_SLOT(ncclNetPlugin_v6, closeListen, pluginCloseListen);

#undef EXPECT_ADAPTER_SLOT
}

/**
 * @brief Verifies current and legacy initialization, retry, caching, callbacks, outputs, and
 * delegated statuses.
 * @note Side effects: Mutates the recording backend and translation-unit-local legacy adapter
 * state.
 */
TEST_F(PluginAbiTest, Initialization) {
  // The local legacy-config helper must reject a missing output and initialize a valid output
  // explicitly.
  EXPECT_EQ(makeLegacyConfig(nullptr), ncclInvalidArgument) << "A missing legacy-config output was accepted";
  ncclNetCommConfig_t legacyConfig = {};
  legacyConfig.trafficClass = 41;
  ASSERT_EQ(makeLegacyConfig(&legacyConfig), ncclSuccess) << "A valid legacy config was rejected";
  EXPECT_EQ(legacyConfig.trafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF)
      << "Legacy initialization did not select the undefined traffic class";

  // Current initialization must forward every argument and expose the backend-owned context output.
  ncclNetCommConfig_t currentConfig = {};
  currentConfig.trafficClass = 23;
  const ncclNetCommConfig_t currentConfigBefore = currentConfig;
  void* context = nullptr;
  ASSERT_EQ(pluginInit(&context, UINT64_MAX, &currentConfig, LoggerForTest, ProfilerForTest), ncclSuccess)
      << "Current initialization did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Init) << "Current initialization called the wrong backend operation";
  EXPECT_EQ(backend.contextOutput, &context) << "Current initialization forwarded the wrong output storage";
  EXPECT_EQ(backend.commId, UINT64_MAX) << "Current initialization changed the communicator identifier";
  EXPECT_EQ(backend.configArgument, &currentConfig) << "Current initialization forwarded the wrong config pointer";
  EXPECT_EQ(backend.configValue.trafficClass, 23) << "Current initialization changed the traffic class";
  EXPECT_EQ(backend.logger, LoggerForTest) << "Current initialization changed the logger callback";
  EXPECT_EQ(backend.profiler, ProfilerForTest) << "Current initialization changed the profiler callback";
  EXPECT_EQ(context, &contextToken) << "Current initialization lost the backend context output";
  EXPECT_EQ(currentConfig.trafficClass, currentConfigBefore.trafficClass)
      << "Current initialization modified its input config";

  // Every concrete backend error must pass through current initialization without changing an
  // untouched output.
  context = &receiveToken;
  ExpectDelegatedErrors("pluginInit", [&] {
    return pluginInit(&context, 5, &currentConfig, LoggerForTest, ProfilerForTest);
  });
  EXPECT_EQ(context, &receiveToken) << "Current initialization changed the context after backend errors";
  EXPECT_EQ(currentConfig.trafficClass, currentConfigBefore.trafficClass)
      << "Current initialization modified its input config while returning a backend error";

  // The v11 bridge must copy its legacy traffic class into a current configuration.
  ResetBackend();
  ncclNetCommConfig_v11_t configV11 = {};
  configV11.trafficClass = 106;
  const ncclNetCommConfig_v11_t configV11Before = configV11;
  context = nullptr;
  ASSERT_EQ(occlPluginInitV11(&context, 42, &configV11, LoggerForTest, ProfilerForTest),
            ncclSuccess)
      << "v11 initialization did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Init)
      << "v11 initialization called the wrong backend operation";
  EXPECT_EQ(backend.commId, 42u) << "v11 initialization changed the communicator identifier";
  EXPECT_EQ(backend.configValue.trafficClass, configV11.trafficClass)
      << "v11 initialization changed the traffic class during conversion";
  EXPECT_EQ(backend.logger, LoggerForTest) << "v11 initialization changed the logger callback";
  EXPECT_EQ(backend.profiler, ProfilerForTest)
      << "v11 initialization changed the profiler callback";
  EXPECT_EQ(context, &contextToken) << "v11 initialization lost the backend context output";
  EXPECT_EQ(std::memcmp(&configV11, &configV11Before, sizeof(configV11)), 0)
      << "v11 initialization modified its legacy input configuration";

  // A null v11 configuration must remain null at the backend boundary.
  ResetBackend();
  context = nullptr;
  ASSERT_EQ(occlPluginInitV11(&context, 43, nullptr, LoggerForTest, ProfilerForTest),
            ncclSuccess)
      << "v11 initialization rejected a null configuration";
  EXPECT_EQ(backend.configArgument, nullptr)
      << "v11 initialization replaced a null configuration";
  EXPECT_EQ(context, &contextToken)
      << "v11 initialization lost the context returned for a null configuration";

  // Backend errors must pass through the v11 conversion without modifying caller-owned input.
  context = &receiveToken;
  ExpectDelegatedErrors("occlPluginInitV11", [&] {
    return occlPluginInitV11(&context, 44, &configV11, LoggerForTest, ProfilerForTest);
  });
  EXPECT_EQ(context, &receiveToken) << "v11 initialization errors modified the context output";
  EXPECT_EQ(std::memcmp(&configV11, &configV11Before, sizeof(configV11)), 0)
      << "v11 initialization errors modified the legacy input configuration";

  // Finalization must forward its exact context and every backend error.
  ResetBackend();
  ASSERT_EQ(pluginFinalize(&contextToken), ncclSuccess) << "Current finalization did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Finalize) << "Current finalization called the wrong backend operation";
  EXPECT_EQ(backend.contextArgument, &contextToken) << "Current finalization forwarded the wrong context";
  ExpectDelegatedErrors("pluginFinalize", [&] { return pluginFinalize(&contextToken); });

  // A failed first v10 initialization must remain retryable and must not publish legacy state.
  for (ncclResult_t error : kDelegatedErrors) {
    SCOPED_TRACE(std::string("pluginInitV10 result=") + std::to_string(static_cast<int>(error)));
    ResetBackend();
    gLegacyCtx = nullptr;
    gLegacyRefCount = 0;
    backend.result = error;
    EXPECT_EQ(pluginInitV10(LoggerForTest, ProfilerForTest), error)
        << "v10 initialization did not return the backend error";
    EXPECT_EQ(backend.calls, 1) << "Failed v10 initialization did not call the backend exactly once";
    EXPECT_EQ(gLegacyCtx, nullptr) << "Failed v10 initialization published a legacy context";
    EXPECT_EQ(gLegacyRefCount, 0) << "Failed v10 initialization incorrectly populated the legacy cache";
  }

  // A successful v10 initialization must use the default config once and cache the resulting
  // context.
  ResetBackend();
  gLegacyCtx = nullptr;
  gLegacyRefCount = 0;
  ASSERT_EQ(pluginInitV10(LoggerForTest, ProfilerForTest), ncclSuccess)
      << "Valid v10 initialization did not succeed";
  EXPECT_EQ(backend.calls, 1) << "v10 initialization did not call the backend exactly once";
  EXPECT_EQ(backend.commId, 0U) << "v10 initialization did not use the legacy communicator identifier";
  EXPECT_EQ(backend.configValue.trafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF)
      << "v10 initialization did not create the default legacy config";
  EXPECT_EQ(backend.logger, LoggerForTest) << "v10 initialization changed the logger callback";
  EXPECT_EQ(backend.profiler, ProfilerForTest) << "v10 initialization changed the profiler callback";
  EXPECT_EQ(gLegacyCtx, &contextToken) << "v10 initialization did not cache the backend context";
  EXPECT_EQ(gLegacyRefCount, 1) << "v10 initialization stored the wrong initial reference count";

  // Repeated legacy initialization must reuse the cached context without touching the backend.
  backend.result = ncclSystemError;
  ASSERT_EQ(pluginInitV9(LoggerForTest), ncclSuccess) << "Cached v9 initialization did not report success";
  EXPECT_EQ(backend.calls, 1) << "Cached v9 initialization called the backend again";
  EXPECT_EQ(gLegacyRefCount, 2) << "Cached v9 initialization did not increase the reference count";

  // A fresh v9 initialization must propagate every backend error without publishing legacy state.
  for (ncclResult_t error : kDelegatedErrors) {
    SCOPED_TRACE(std::string("pluginInitV9 result=") + std::to_string(static_cast<int>(error)));
    ResetBackend();
    gLegacyCtx = nullptr;
    gLegacyRefCount = 0;
    backend.result = error;
    EXPECT_EQ(pluginInitV9(LoggerForTest), error) << "Fresh v9 initialization did not return the backend error";
    EXPECT_EQ(backend.calls, 1) << "Fresh v9 initialization did not call the backend exactly once";
    EXPECT_EQ(backend.profiler, nullptr) << "Fresh v9 initialization supplied a profiler absent from its ABI";
    EXPECT_EQ(gLegacyCtx, nullptr) << "Failed v9 initialization published a legacy context";
    EXPECT_EQ(gLegacyRefCount, 0) << "Failed v9 initialization changed the legacy reference count";
  }

  // A fresh v9 initialization must supply a null profiler because that ABI has no profiler
  // parameter.
  ResetBackend();
  gLegacyCtx = nullptr;
  gLegacyRefCount = 0;
  ASSERT_EQ(pluginInitV9(LoggerForTest), ncclSuccess) << "Fresh v9 initialization did not succeed";
  EXPECT_EQ(backend.logger, LoggerForTest) << "v9 initialization changed the logger callback";
  EXPECT_EQ(backend.profiler, nullptr) << "v9 initialization supplied a profiler absent from its ABI";
  EXPECT_EQ(gLegacyRefCount, 1) << "Fresh v9 initialization stored the wrong reference count";
}

/**
 * @brief Verifies exact current forwarding and field-by-field property projection for every legacy
 * ABI.
 * @note Side effects: Mutates the recording backend and output property values.
 */
TEST_F(PluginCompatibilityTest, Properties) {
  // Device discovery must forward success, its output, and every concrete backend error.
  int deviceCount = -1;
  ASSERT_EQ(pluginDevices(&deviceCount), ncclSuccess) << "Device discovery did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Devices) << "Device discovery called the wrong backend operation";
  EXPECT_EQ(deviceCount, backend.deviceCount) << "Device discovery lost the backend device count";
  ExpectDelegatedErrors("pluginDevices", [&] { return pluginDevices(&deviceCount); });
  EXPECT_EQ(deviceCount, backend.deviceCount) << "Device discovery errors modified the device-count output";

  // Current property discovery must forward the exact device and output without conversion.
  ResetBackend();
  backend.propertiesResult = MakeProperties(1);
  ncclNetProperties_t current = {};
  ASSERT_EQ(occlPluginGetPropertiesV12(2, &current), ncclSuccess)
      << "Current property discovery did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::GetProperties)
      << "Current property discovery called the wrong backend operation";
  EXPECT_EQ(backend.dev, 2) << "Current property discovery changed the device index";
  EXPECT_EQ(backend.propertiesOutput, &current) << "Current property discovery changed the output pointer";
  EXPECT_EQ(current.maxMultiRequestSize, backend.propertiesResult.maxMultiRequestSize)
      << "Current property discovery lost a current-only field";
  ExpectV10Properties(current, backend.propertiesResult, "v12");
  const ncclNetProperties_t currentExpected = backend.propertiesResult;
  ExpectDelegatedErrors("occlPluginGetPropertiesV12",
                        [&] { return occlPluginGetPropertiesV12(2, &current); });
  ExpectV10Properties(current, currentExpected, "v12 after errors");
  EXPECT_EQ(current.maxMultiRequestSize, currentExpected.maxMultiRequestSize)
      << "Current property errors modified the current-only output field";

  // v10 and v9 must copy every field their ABIs represent at the maximum virtual-device count.
  ResetBackend();
  backend.propertiesResult = MakeProperties(NCCL_NET_MAX_DEVS_PER_NIC_V11);
  ncclNetProperties_v10_t propertiesV10 = {};
  propertiesV10.forceFlush = 1;
  ASSERT_EQ(pluginGetPropertiesV10(3, &propertiesV10), ncclSuccess)
      << "v10 property conversion did not return success";
  ExpectV10Properties(propertiesV10, backend.propertiesResult, "v10");
  ncclNetProperties_v9_t propertiesV9 = {};
  propertiesV9.forceFlush = 1;
  ASSERT_EQ(pluginGetPropertiesV9(4, &propertiesV9), ncclSuccess)
      << "v9 property conversion did not return success";
  ExpectV10Properties(propertiesV9, backend.propertiesResult, "v9");
  const ncclNetProperties_t modernLegacyExpected = backend.propertiesResult;

  // v8, v7, and v6 must copy exactly the progressively smaller field sets in their ABIs.
  backend.propertiesResult = MakeProperties(1);
  ncclNetProperties_v8_t propertiesV8 = {};
  ASSERT_EQ(pluginGetPropertiesV8(5, &propertiesV8), ncclSuccess)
      << "v8 property conversion did not return success";
  ExpectV8Properties(propertiesV8, backend.propertiesResult, "v8");
  ncclNetProperties_v7_t propertiesV7 = {};
  ASSERT_EQ(pluginGetPropertiesV7(6, &propertiesV7), ncclSuccess)
      << "v7 property conversion did not return success";
  ExpectV7Properties(propertiesV7, backend.propertiesResult, "v7");
  ncclNetProperties_v6_t propertiesV6 = {};
  ASSERT_EQ(pluginGetPropertiesV6(7, &propertiesV6), ncclSuccess)
      << "v6 property conversion did not return success";
  ExpectBaseProperties(propertiesV6, backend.propertiesResult, "v6");
  const ncclNetProperties_t oldLegacyExpected = backend.propertiesResult;

  // Conversion adapters must propagate each synchronous error before writing the caller's
  // destination.
  for (ncclResult_t error : kDelegatedErrors) {
    SCOPED_TRACE(std::string("legacy property result=") + std::to_string(static_cast<int>(error)));
    ResetBackend();
    backend.result = error;
    EXPECT_EQ(pluginGetPropertiesV10(0, &propertiesV10), error) << "v10 did not propagate a property error";
    EXPECT_EQ(pluginGetPropertiesV9(0, &propertiesV9), error) << "v9 did not propagate a property error";
    EXPECT_EQ(pluginGetPropertiesV8(0, &propertiesV8), error) << "v8 did not propagate a property error";
    EXPECT_EQ(pluginGetPropertiesV7(0, &propertiesV7), error) << "v7 did not propagate a property error";
    EXPECT_EQ(pluginGetPropertiesV6(0, &propertiesV6), error) << "v6 did not propagate a property error";
    ExpectV10Properties(propertiesV10, modernLegacyExpected, "v10 after error");
    ExpectV10Properties(propertiesV9, modernLegacyExpected, "v9 after error");
    ExpectV8Properties(propertiesV8, oldLegacyExpected, "v8 after error");
    ExpectV7Properties(propertiesV7, oldLegacyExpected, "v7 after error");
    ExpectBaseProperties(propertiesV6, oldLegacyExpected, "v6 after error");
  }
}

/**
 * @brief Verifies v11 property projection, validation, error propagation, and atomic output.
 * @note Side effects: Mutates the recording backend and writes the caller output only on success.
 */
TEST_F(PluginCompatibilityTest, V11PropertyProjectionAndErrors) {
  // A null destination is rejected before property discovery begins.
  EXPECT_EQ(occlPluginGetPropertiesV11(3, nullptr), ncclInvalidArgument)
      << "v11 property discovery accepted a null output";
  EXPECT_EQ(backend.calls, 0) << "A null v11 property output reached the backend";

  // The largest v11 device list and every representable field must be projected exactly.
  ResetBackend();
  backend.propertiesResult = MakeProperties(NCCL_NET_MAX_DEVS_PER_NIC_V11);
  ncclNetProperties_v11_t properties = {};
  ASSERT_EQ(occlPluginGetPropertiesV11(3, &properties), ncclSuccess)
      << "Valid v11 property discovery did not succeed";
  EXPECT_EQ(backend.operation, BackendOperation::GetProperties)
      << "v11 property discovery called the wrong backend operation";
  EXPECT_EQ(backend.calls, 1) << "v11 property discovery did not call the backend exactly once";
  EXPECT_EQ(backend.dev, 3) << "v11 property discovery changed the device index";
  ExpectV10Properties(properties, backend.propertiesResult, "v11");
  EXPECT_EQ(properties.maxMultiRequestSize, backend.propertiesResult.maxMultiRequestSize)
      << "v11 property projection lost the multi-request limit";

  // Invalid device counts must leave a previously valid destination byte-for-byte unchanged.
  const ncclNetProperties_v11_t validProperties = properties;
  for (int ndevs : {0, NCCL_NET_MAX_DEVS_PER_NIC_V11 + 1}) {
    SCOPED_TRACE(::testing::Message() << "ndevs=" << ndevs);
    ResetBackend();
    backend.propertiesResult = MakeProperties(ndevs);
    properties = validProperties;
    EXPECT_EQ(occlPluginGetPropertiesV11(3, &properties), ncclInvalidUsage)
        << "v11 property discovery accepted an invalid device count";
    EXPECT_EQ(backend.calls, 1)
        << "v11 property validation did not perform exactly one backend query";
    EXPECT_EQ(std::memcmp(&properties, &validProperties, sizeof(properties)), 0)
        << "v11 property validation modified the caller output";
  }

  // Every backend failure must be returned unchanged without publishing partial properties.
  for (ncclResult_t error : kDelegatedErrors) {
    SCOPED_TRACE(::testing::Message() << "backend result=" << static_cast<int>(error));
    ResetBackend();
    backend.result = error;
    backend.propertiesResult = MakeProperties(1);
    properties = validProperties;
    EXPECT_EQ(occlPluginGetPropertiesV11(3, &properties), error)
        << "v11 property discovery changed the backend error";
    EXPECT_EQ(backend.calls, 1)
        << "Failed v11 property discovery did not call the backend exactly once";
    EXPECT_EQ(std::memcmp(&properties, &validProperties, sizeof(properties)), 0)
        << "Failed v11 property discovery modified the caller output";
  }
}

/**
 * @brief Verifies current forwarding and valid v9-v11 virtual-device conversion boundaries.
 * @note Side effects: Mutates the recording backend and virtual-device outputs.
 */
TEST_F(PluginCompatibilityTest, VirtualDevices) {
  // The current adapter must pass the original properties directly and return the backend-created
  // index.
  ncclNetVDeviceProps_t current = {};
  current.ndevs = 1;
  current.devs[0] = 1;
  const ncclNetVDeviceProps_t currentBefore = current;
  int virtualDevice = -1;
  ASSERT_EQ(occlPluginMakeVDeviceV12(&virtualDevice, &current), ncclSuccess)
      << "Current virtual-device creation did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::MakeVDevice)
      << "Current virtual-device creation called the wrong backend operation";
  EXPECT_EQ(virtualDevice, backend.virtualDeviceResult) << "Current virtual-device creation lost the output index";
  EXPECT_EQ(backend.virtualProperties.ndevs, 1) << "Current virtual-device creation changed ndevs";
  EXPECT_EQ(backend.virtualProperties.devs[0], 1) << "Current virtual-device creation changed the device index";
  EXPECT_EQ(current.ndevs, currentBefore.ndevs) << "Current virtual-device creation modified its input count";
  EXPECT_EQ(current.devs[0], currentBefore.devs[0]) << "Current virtual-device creation modified its input array";
  ExpectDelegatedErrors("occlPluginMakeVDeviceV12",
                        [&] { return occlPluginMakeVDeviceV12(&virtualDevice, &current); });
  EXPECT_EQ(virtualDevice, backend.virtualDeviceResult)
      << "Current virtual-device errors modified the caller's output index";
  EXPECT_EQ(current.ndevs, currentBefore.ndevs) << "Current virtual-device errors modified the input count";
  EXPECT_EQ(current.devs[0], currentBefore.devs[0])
      << "Current virtual-device errors modified the input device";

  // v11 conversion must copy the maximum legacy device list without modifying its input.
  ResetBackend();
  ncclNetVDeviceProps_v11_t propertiesV11 = {};
  propertiesV11.ndevs = NCCL_NET_MAX_DEVS_PER_NIC_V11;
  for (int i = 0; i < propertiesV11.ndevs; ++i) {
    propertiesV11.devs[i] = NCCL_NET_MAX_DEVS_PER_NIC_V11 - 1 - i;
  }
  const ncclNetVDeviceProps_v11_t propertiesV11Before = propertiesV11;
  virtualDevice = -1;
  ASSERT_EQ(occlPluginMakeVDeviceV11(&virtualDevice, &propertiesV11), ncclSuccess)
      << "v11 virtual-device conversion did not return backend success";
  EXPECT_EQ(virtualDevice, backend.virtualDeviceResult)
      << "v11 virtual-device conversion lost the output index";
  EXPECT_EQ(backend.virtualProperties.ndevs, propertiesV11.ndevs)
      << "v11 virtual-device conversion changed ndevs";
  for (int i = 0; i < propertiesV11.ndevs; ++i) {
    EXPECT_EQ(backend.virtualProperties.devs[i], propertiesV11.devs[i])
        << "v11 conversion changed device index " << i;
    EXPECT_EQ(propertiesV11.devs[i], propertiesV11Before.devs[i])
        << "v11 conversion modified input device index " << i;
  }
  ExpectDelegatedErrors("occlPluginMakeVDeviceV11", [&] {
    return occlPluginMakeVDeviceV11(&virtualDevice, &propertiesV11);
  });
  EXPECT_EQ(virtualDevice, backend.virtualDeviceResult)
      << "v11 virtual-device errors modified the caller's output index";
  EXPECT_EQ(std::memcmp(&propertiesV11, &propertiesV11Before, sizeof(propertiesV11)), 0)
      << "v11 virtual-device errors modified the legacy input";

  // v10 conversion must copy the maximum valid list without modifying its legacy input.
  ResetBackend();
  ncclNetVDeviceProps_v10_t propertiesV10 = {};
  propertiesV10.ndevs = NCCL_NET_MAX_DEVS_PER_NIC_V11;
  for (int i = 0; i < propertiesV10.ndevs; ++i) {
    propertiesV10.devs[i] = NCCL_NET_MAX_DEVS_PER_NIC_V11 - 1 - i;
  }
  const ncclNetVDeviceProps_v10_t propertiesV10Before = propertiesV10;
  virtualDevice = -1;
  ASSERT_EQ(pluginMakeVDeviceV10(&virtualDevice, &propertiesV10), ncclSuccess)
      << "v10 virtual-device conversion did not return backend success";
  EXPECT_EQ(virtualDevice, backend.virtualDeviceResult) << "v10 virtual-device conversion lost the output index";
  EXPECT_EQ(backend.virtualProperties.ndevs, propertiesV10.ndevs) << "v10 conversion changed ndevs";
  for (int i = 0; i < propertiesV10.ndevs; ++i) {
    EXPECT_EQ(backend.virtualProperties.devs[i], propertiesV10.devs[i])
        << "v10 conversion changed device index " << i;
    EXPECT_EQ(propertiesV10.devs[i], propertiesV10Before.devs[i])
        << "v10 conversion modified input device index " << i;
  }

  // v9 conversion must preserve the minimum valid one-device list and propagate backend errors.
  ResetBackend();
  ncclNetVDeviceProps_v9_t propertiesV9 = {};
  propertiesV9.ndevs = 1;
  propertiesV9.devs[0] = 1;
  const ncclNetVDeviceProps_v9_t propertiesV9Before = propertiesV9;
  virtualDevice = -1;
  ASSERT_EQ(pluginMakeVDeviceV9(&virtualDevice, &propertiesV9), ncclSuccess)
      << "v9 virtual-device conversion did not return backend success";
  EXPECT_EQ(backend.virtualProperties.ndevs, 1) << "v9 conversion changed ndevs";
  EXPECT_EQ(backend.virtualProperties.devs[0], 1) << "v9 conversion changed the device index";
  EXPECT_EQ(propertiesV9.ndevs, propertiesV9Before.ndevs) << "v9 conversion modified its input count";
  EXPECT_EQ(propertiesV9.devs[0], propertiesV9Before.devs[0]) << "v9 conversion modified its input array";
  ExpectDelegatedErrors("pluginMakeVDeviceV9", [&] { return pluginMakeVDeviceV9(&virtualDevice, &propertiesV9); });
  ExpectDelegatedErrors("pluginMakeVDeviceV10", [&] {
    return pluginMakeVDeviceV10(&virtualDevice, &propertiesV10);
  });
  EXPECT_EQ(virtualDevice, backend.virtualDeviceResult)
      << "Legacy virtual-device errors modified the caller's output index";
  EXPECT_EQ(propertiesV10.ndevs, propertiesV10Before.ndevs)
      << "v10 virtual-device errors modified the input count";
  for (int i = 0; i < propertiesV10.ndevs; ++i) {
    EXPECT_EQ(propertiesV10.devs[i], propertiesV10Before.devs[i])
        << "v10 virtual-device errors modified input device " << i;
  }
  EXPECT_EQ(propertiesV9.ndevs, propertiesV9Before.ndevs)
      << "v9 virtual-device errors modified the input count";
  EXPECT_EQ(propertiesV9.devs[0], propertiesV9Before.devs[0])
      << "v9 virtual-device errors modified the input device";

  // The v9 adapter rejects an empty list before reading an inactive array entry or calling the
  // backend.
  ResetBackend();
  backend.result = ncclInvalidUsage;
  propertiesV9.ndevs = 0;
  virtualDevice = -1;
  EXPECT_EQ(pluginMakeVDeviceV9(&virtualDevice, &propertiesV9), ncclInvalidArgument)
      << "v9 virtual-device conversion accepted an empty list";
  EXPECT_EQ(backend.operation, BackendOperation::None) << "v9 empty-list rejection called the backend";
  EXPECT_EQ(virtualDevice, -1) << "v9 empty-list rejection changed the output";
  EXPECT_EQ(propertiesV9.ndevs, 0) << "v9 conversion modified the empty input count";

  // The v10 adapter follows the same local validation path without changing its output.
  ResetBackend();
  backend.result = ncclInvalidUsage;
  propertiesV10.ndevs = 0;
  virtualDevice = -1;
  EXPECT_EQ(pluginMakeVDeviceV10(&virtualDevice, &propertiesV10), ncclInvalidArgument)
      << "v10 virtual-device conversion accepted an empty list";
  EXPECT_EQ(backend.operation, BackendOperation::None) << "v10 empty-list rejection called the backend";
  EXPECT_EQ(virtualDevice, -1) << "v10 empty-list rejection changed the output";
  EXPECT_EQ(propertiesV10.ndevs, 0) << "v10 conversion modified the empty input count";
}

/**
 * @brief Verifies the legacy virtual-device bridge's bounds, conversion, and delegated results.
 * @note Side effects: Mutates the recording backend and writes the device output only on success.
 */
TEST_F(PluginCompatibilityTest, LegacyVirtualDeviceBridge) {
  ncclNetVDeviceProps_v11_t properties = {};
  properties.ndevs = NCCL_NET_MAX_DEVS_PER_NIC_V11;
  for (int i = 0; i < properties.ndevs; ++i) properties.devs[i] = properties.ndevs - 1 - i;
  const ncclNetVDeviceProps_v11_t propertiesBefore = properties;

  // Null outputs and inputs are rejected locally without changing caller-owned state.
  int virtualDevice = -7;
  EXPECT_EQ(occlPluginMakeLegacyVDevice(nullptr, &properties), ncclInvalidArgument)
      << "The legacy bridge accepted a null device output";
  EXPECT_EQ(occlPluginMakeLegacyVDevice(
                &virtualDevice, static_cast<ncclNetVDeviceProps_v11_t*>(nullptr)),
            ncclInvalidArgument)
      << "The legacy bridge accepted null properties";
  EXPECT_EQ(backend.calls, 0) << "Invalid legacy pointers reached the backend";
  EXPECT_EQ(virtualDevice, -7) << "Null legacy properties modified the device output";
  EXPECT_EQ(std::memcmp(&properties, &propertiesBefore, sizeof(properties)), 0)
      << "Null-output validation modified the legacy properties";

  // Empty and oversized lists are rejected before conversion or backend dispatch.
  for (int ndevs : {0, NCCL_NET_MAX_DEVS_PER_NIC_V11 + 1}) {
    SCOPED_TRACE(::testing::Message() << "ndevs=" << ndevs);
    ResetBackend();
    ncclNetVDeviceProps_v11_t invalidProperties = propertiesBefore;
    invalidProperties.ndevs = ndevs;
    const ncclNetVDeviceProps_v11_t invalidBefore = invalidProperties;
    virtualDevice = -7;
    EXPECT_EQ(occlPluginMakeLegacyVDevice(&virtualDevice, &invalidProperties),
              ncclInvalidArgument)
        << "The legacy bridge accepted an invalid device count";
    EXPECT_EQ(backend.calls, 0) << "An invalid legacy device count reached the backend";
    EXPECT_EQ(virtualDevice, -7) << "Legacy count validation modified the device output";
    EXPECT_EQ(std::memcmp(&invalidProperties, &invalidBefore, sizeof(invalidProperties)), 0)
        << "Legacy count validation modified the input properties";
  }

  // The maximum valid list is copied exactly and the backend result is published.
  ResetBackend();
  virtualDevice = -7;
  ASSERT_EQ(occlPluginMakeLegacyVDevice(&virtualDevice, &properties), ncclSuccess)
      << "The maximum valid legacy device list did not succeed";
  EXPECT_EQ(backend.operation, BackendOperation::MakeVDevice)
      << "The legacy bridge called the wrong backend operation";
  EXPECT_EQ(backend.calls, 1) << "The legacy bridge did not call the backend exactly once";
  EXPECT_EQ(backend.virtualDeviceOutput, &virtualDevice)
      << "The legacy bridge changed the device-output pointer";
  EXPECT_EQ(virtualDevice, backend.virtualDeviceResult)
      << "The legacy bridge lost the backend device index";
  EXPECT_EQ(backend.virtualProperties.ndevs, properties.ndevs)
      << "The legacy bridge changed the device count";
  for (int i = 0; i < properties.ndevs; ++i) {
    EXPECT_EQ(backend.virtualProperties.devs[i], properties.devs[i])
        << "The legacy bridge changed device index " << i;
  }
  EXPECT_EQ(std::memcmp(&properties, &propertiesBefore, sizeof(properties)), 0)
      << "Successful legacy conversion modified the input properties";

  // Every backend failure is propagated without changing the output or legacy input.
  for (ncclResult_t error : kDelegatedErrors) {
    SCOPED_TRACE(::testing::Message() << "backend result=" << static_cast<int>(error));
    ResetBackend();
    backend.result = error;
    virtualDevice = -7;
    EXPECT_EQ(occlPluginMakeLegacyVDevice(&virtualDevice, &properties), error)
        << "The legacy bridge changed the backend error";
    EXPECT_EQ(backend.calls, 1) << "The failed legacy bridge did not call the backend exactly once";
    EXPECT_EQ(virtualDevice, -7) << "A failed legacy bridge modified the device output";
    EXPECT_EQ(std::memcmp(&properties, &propertiesBefore, sizeof(properties)), 0)
        << "A failed legacy bridge modified the input properties";
  }
}

/**
 * @brief Verifies field-by-field v11 network-attribute conversion and null forwarding.
 * @note Side effects: Mutates the recording backend; caller-owned attributes remain unchanged.
 */
TEST_F(PluginCompatibilityTest, NetworkAttributes) {
  ncclNetAttr_v11_t attributes = {};
  attributes.sendCommAttr.maxConcurrentPeers = 1;
  attributes.sendCommAttr.minConcurrentPeers = 2;
  attributes.sendCommAttr.maxFlowsPerPeer = 3;
  attributes.sendCommAttr.minFlowsPerPeer = 4;
  attributes.recvCommAttr.maxConcurrentPeers = 5;
  attributes.recvCommAttr.minConcurrentPeers = 6;
  attributes.recvCommAttr.maxFlowsPerPeer = 7;
  attributes.recvCommAttr.minFlowsPerPeer = 8;
  attributes.op = 9;
  attributes.algo = 10;
  attributes.proto = 11;
  const ncclNetAttr_v11_t attributesBefore = attributes;

  ASSERT_EQ(occlPluginSetNetAttrV11(&contextToken, &attributes), ncclSuccess)
      << "v11 network-attribute conversion did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::SetNetAttr)
      << "v11 network-attribute conversion called the wrong backend operation";
  EXPECT_EQ(backend.contextArgument, &contextToken)
      << "v11 network-attribute conversion changed the context";
  EXPECT_FALSE(backend.attributesWereNull)
      << "v11 network-attribute conversion supplied a null backend value";
  EXPECT_EQ(backend.attributesValue.sendCommAttr.maxConcurrentPeers, 1u);
  EXPECT_EQ(backend.attributesValue.sendCommAttr.minConcurrentPeers, 2u);
  EXPECT_EQ(backend.attributesValue.sendCommAttr.maxFlowsPerPeer, 3u);
  EXPECT_EQ(backend.attributesValue.sendCommAttr.minFlowsPerPeer, 4u);
  EXPECT_EQ(backend.attributesValue.recvCommAttr.maxConcurrentPeers, 5u);
  EXPECT_EQ(backend.attributesValue.recvCommAttr.minConcurrentPeers, 6u);
  EXPECT_EQ(backend.attributesValue.recvCommAttr.maxFlowsPerPeer, 7u);
  EXPECT_EQ(backend.attributesValue.recvCommAttr.minFlowsPerPeer, 8u);
  EXPECT_EQ(backend.attributesValue.op, 9u);
  EXPECT_EQ(backend.attributesValue.algo, 10u);
  EXPECT_EQ(backend.attributesValue.proto, 11u);
  EXPECT_EQ(std::memcmp(&attributes, &attributesBefore, sizeof(attributes)), 0)
      << "v11 network-attribute conversion modified its legacy input";

  ResetBackend();
  ASSERT_EQ(occlPluginSetNetAttrV11(&contextToken, nullptr), ncclSuccess)
      << "v11 network-attribute conversion rejected a null input";
  EXPECT_TRUE(backend.attributesWereNull)
      << "v11 network-attribute conversion replaced a null backend value";

  ExpectDelegatedErrors("occlPluginSetNetAttrV11", [&] {
    return occlPluginSetNetAttrV11(&contextToken, &attributes);
  });
  EXPECT_EQ(std::memcmp(&attributes, &attributesBefore, sizeof(attributes)), 0)
      << "v11 network-attribute errors modified the legacy input";
}

/**
 * @brief Verifies listener, connector, and acceptor adapters for current and legacy ABIs.
 * @note Side effects: Mutates the recording backend, legacy context, and communicator outputs.
 */
TEST_F(PluginConnectionTest, ConnectionLifecycle) {
  std::array<unsigned char, NCCL_NET_HANDLE_MAXSIZE> handle = {};
  for (size_t i = 0; i < handle.size(); ++i) handle[i] = static_cast<unsigned char>(i);
  const auto handleBefore = handle;
  gLegacyCtx = &contextToken;
  gLegacyRefCount = 1;

  // Current listen must forward its context, device, handle, and output exactly.
  void* listenComm = nullptr;
  ASSERT_EQ(pluginListen(&contextToken, 3, handle.data(), &listenComm), ncclSuccess)
      << "Current listen did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Listen) << "Current listen called the wrong backend operation";
  EXPECT_EQ(backend.contextArgument, &contextToken) << "Current listen changed the context";
  EXPECT_EQ(backend.dev, 3) << "Current listen changed the device index";
  EXPECT_EQ(backend.handle, handle.data()) << "Current listen changed the handle pointer";
  EXPECT_EQ(listenComm, &listenToken) << "Current listen lost the backend communicator output";
  EXPECT_EQ(handle, handleBefore) << "Current listen wrapper modified the opaque handle";
  ExpectDelegatedErrors("pluginListen", [&] {
    return pluginListen(&contextToken, 1, handle.data(), &listenComm);
  });
  EXPECT_EQ(listenComm, &listenToken) << "Current listen errors modified the listener output";

  // Legacy listen must replace only the missing ABI context with the cached legacy context.
  ResetBackend();
  listenComm = &receiveToken;
  ASSERT_EQ(pluginListenLegacy(2, handle.data(), &listenComm), ncclSuccess)
      << "Legacy listen did not return backend success";
  EXPECT_EQ(backend.contextArgument, gLegacyCtx) << "Legacy listen did not use the cached context";
  EXPECT_EQ(backend.dev, 2) << "Legacy listen changed the device index";
  EXPECT_EQ(listenComm, &listenToken) << "Legacy listen lost the backend communicator output";
  ExpectDelegatedErrors("pluginListenLegacy", [&] { return pluginListenLegacy(2, handle.data(), &listenComm); });
  EXPECT_EQ(listenComm, &listenToken) << "Legacy listen errors modified the listener output";

  // Current connect must forward every input and expose both backend outputs.
  ResetBackend();
  backend.communicatorResult = &sendToken;
  void* sendComm = nullptr;
  ncclNetDeviceHandle_t* sendDevice = nullptr;
  ASSERT_EQ(pluginConnect(&contextToken, 3, handle.data(), &sendComm, &sendDevice), ncclSuccess)
      << "Current connect did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Connect) << "Current connect called the wrong backend operation";
  EXPECT_EQ(backend.contextArgument, &contextToken) << "Current connect changed the context";
  EXPECT_EQ(backend.dev, 3) << "Current connect changed the device index";
  EXPECT_EQ(backend.handle, handle.data()) << "Current connect changed the handle pointer";
  EXPECT_EQ(sendComm, &sendToken) << "Current connect lost the backend communicator output";
  EXPECT_EQ(sendDevice, &deviceHandleToken) << "Current connect lost the backend device output";
  ExpectDelegatedErrors("pluginConnect", [&] {
    return pluginConnect(&contextToken, 3, handle.data(), &sendComm, &sendDevice);
  });
  EXPECT_EQ(sendComm, &sendToken) << "Current connect errors modified the sender output";
  EXPECT_EQ(sendDevice, &deviceHandleToken) << "Current connect errors modified the device output";

  // Success with a null communicator is the documented pending result and must remain successful.
  ResetBackend();
  backend.communicatorResult = nullptr;
  sendComm = &sendToken;
  ASSERT_EQ(pluginConnect(&contextToken, 3, handle.data(), &sendComm, &sendDevice), ncclSuccess)
      << "Current connect rejected the documented pending result";
  EXPECT_EQ(sendComm, nullptr) << "Current connect did not expose the pending null communicator";

  // v10 accepts its retired per-call config field but uses only the cached legacy context.
  ResetBackend();
  backend.captureConnectConfig = true;
  backend.communicatorResult = &sendToken;
  ncclNetCommConfig_t legacyContextConfig = {};
  legacyContextConfig.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  gLegacyCtx = &legacyContextConfig;
  ncclNetCommConfig_v10_t connectConfig = {};
  connectConfig.trafficClass = 37;
  const ncclNetCommConfig_v10_t connectConfigBefore = connectConfig;
  sendComm = &receiveToken;
  ncclNetDeviceHandle_v10_t* sendDeviceV10 = nullptr;
  ASSERT_EQ(pluginConnectV10(4, &connectConfig, handle.data(), &sendComm, &sendDeviceV10), ncclSuccess)
      << "v10 connect did not return backend success";
  ASSERT_FALSE(backend.connectContextWasNull) << "v10 connect did not supply a transport context";
  EXPECT_EQ(backend.contextArgument, gLegacyCtx) << "v10 connect did not use the cached context";
  EXPECT_EQ(backend.connectConfigValue.trafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF)
      << "v10 connect forwarded its retired per-call config instead of the cached context";
  EXPECT_EQ(sendComm, &sendToken) << "v10 connect lost the backend communicator output";
  EXPECT_EQ(connectConfig.trafficClass, connectConfigBefore.trafficClass) << "v10 connect modified its input config";
  EXPECT_EQ(legacyContextConfig.trafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF)
      << "v10 connect modified the shared legacy context";
  EXPECT_EQ(sendDeviceV10, &deviceHandleToken) << "v10 connect lost the backend device output";
  ExpectDelegatedErrors("pluginConnectV10", [&] {
    return pluginConnectV10(4, &connectConfig, handle.data(), &sendComm, &sendDeviceV10);
  });
  EXPECT_EQ(sendComm, &sendToken) << "v10 connect errors modified the sender output";
  EXPECT_EQ(sendDeviceV10, &deviceHandleToken) << "v10 connect errors modified the device output";
  EXPECT_EQ(connectConfig.trafficClass, connectConfigBefore.trafficClass)
      << "v10 connect errors modified the input config";

  // v9-v7 must pass the cached context and preserve the versioned device-handle output.
  ResetBackend();
  backend.communicatorResult = &sendToken;
  gLegacyCtx = &contextToken;
  sendComm = &receiveToken;
  ncclNetDeviceHandle_v9_t* sendDeviceV9 = nullptr;
  ASSERT_EQ(pluginConnectV9(5, handle.data(), &sendComm, &sendDeviceV9), ncclSuccess)
      << "v9 connect did not return backend success";
  EXPECT_EQ(backend.contextArgument, gLegacyCtx) << "v9 connect did not use the cached context";
  EXPECT_EQ(sendComm, &sendToken) << "v9 connect lost the backend communicator output";
  EXPECT_EQ(sendDeviceV9, &deviceHandleToken) << "v9 connect lost the backend device output";
  ExpectDelegatedErrors("pluginConnectV9", [&] {
    return pluginConnectV9(5, handle.data(), &sendComm, &sendDeviceV9);
  });
  EXPECT_EQ(sendComm, &sendToken) << "v9 connect errors modified the sender output";
  EXPECT_EQ(sendDeviceV9, &deviceHandleToken) << "v9 connect errors modified the device output";
  ResetBackend();
  backend.communicatorResult = &sendToken;
  sendComm = &receiveToken;
  ncclNetDeviceHandle_v8_t* sendDeviceV8 = nullptr;
  ASSERT_EQ(pluginConnectV8(6, handle.data(), &sendComm, &sendDeviceV8), ncclSuccess)
      << "v8 connect did not return backend success";
  EXPECT_EQ(backend.contextArgument, gLegacyCtx) << "v8 connect did not use the cached context";
  EXPECT_EQ(sendComm, &sendToken) << "v8 connect lost the backend communicator output";
  EXPECT_EQ(sendDeviceV8, &deviceHandleToken) << "v8 connect lost the backend device output";
  ExpectDelegatedErrors("pluginConnectV8", [&] {
    return pluginConnectV8(6, handle.data(), &sendComm, &sendDeviceV8);
  });
  EXPECT_EQ(sendComm, &sendToken) << "v8 connect errors modified the sender output";
  EXPECT_EQ(sendDeviceV8, &deviceHandleToken) << "v8 connect errors modified the device output";
  ResetBackend();
  backend.communicatorResult = &sendToken;
  sendComm = &receiveToken;
  ncclNetDeviceHandle_v7_t* sendDeviceV7 = nullptr;
  ASSERT_EQ(pluginConnectV7(7, handle.data(), &sendComm, &sendDeviceV7), ncclSuccess)
      << "v7 connect did not return backend success";
  EXPECT_EQ(backend.contextArgument, gLegacyCtx) << "v7 connect did not use the cached context";
  EXPECT_EQ(sendComm, &sendToken) << "v7 connect lost the backend communicator output";
  EXPECT_EQ(sendDeviceV7, &deviceHandleToken) << "v7 connect lost the backend device output";
  ExpectDelegatedErrors("pluginConnectV7", [&] {
    return pluginConnectV7(7, handle.data(), &sendComm, &sendDeviceV7);
  });
  EXPECT_EQ(sendComm, &sendToken) << "v7 connect errors modified the sender output";
  EXPECT_EQ(sendDeviceV7, &deviceHandleToken) << "v7 connect errors modified the device output";

  // v6 must supply backend-only device-handle storage and discard the produced handle safely.
  ResetBackend();
  backend.communicatorResult = &sendToken;
  sendComm = &receiveToken;
  ASSERT_EQ(pluginConnectV6(8, handle.data(), &sendComm), ncclSuccess)
      << "v6 connect did not return backend success";
  EXPECT_EQ(backend.contextArgument, gLegacyCtx) << "v6 connect did not use the cached context";
  EXPECT_EQ(sendComm, &sendToken) << "v6 connect lost the backend communicator output";
  EXPECT_FALSE(backend.deviceHandleOutputWasNull) << "v6 connect did not supply backend device-handle storage";
  ExpectDelegatedErrors("pluginConnectV6", [&] { return pluginConnectV6(8, handle.data(), &sendComm); });
  EXPECT_EQ(sendComm, &sendToken) << "v6 connect errors modified the sender output";

  // Current accept must forward its listener and expose both backend outputs.
  ResetBackend();
  backend.communicatorResult = &receiveToken;
  void* recvComm = nullptr;
  ncclNetDeviceHandle_t* recvDevice = nullptr;
  ASSERT_EQ(pluginAccept(&listenToken, &recvComm, &recvDevice), ncclSuccess)
      << "Current accept did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Accept) << "Current accept called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &listenToken) << "Current accept changed the listener communicator";
  EXPECT_EQ(recvComm, &receiveToken) << "Current accept lost the backend communicator output";
  EXPECT_EQ(recvDevice, &deviceHandleToken) << "Current accept lost the backend device output";
  ExpectDelegatedErrors("pluginAccept", [&] { return pluginAccept(&listenToken, &recvComm, &recvDevice); });
  EXPECT_EQ(recvComm, &receiveToken) << "Current accept errors modified the receiver output";
  EXPECT_EQ(recvDevice, &deviceHandleToken) << "Current accept errors modified the device output";

  // Success with a null receiver is the documented pending result and must remain successful.
  ResetBackend();
  backend.communicatorResult = nullptr;
  recvComm = &receiveToken;
  ASSERT_EQ(pluginAccept(&listenToken, &recvComm, &recvDevice), ncclSuccess)
      << "Current accept rejected the documented pending result";
  EXPECT_EQ(recvComm, nullptr) << "Current accept did not expose the pending null communicator";

  // v10-v7 accept must preserve versioned device outputs while v6 supplies and discards
  // backend-only storage.
  ResetBackend();
  backend.communicatorResult = &receiveToken;
  recvComm = &sendToken;
  ncclNetDeviceHandle_v10_t* recvDeviceV10 = nullptr;
  ASSERT_EQ(pluginAcceptV10(&listenToken, &recvComm, &recvDeviceV10), ncclSuccess)
      << "v10 accept did not return backend success";
  EXPECT_EQ(recvComm, &receiveToken) << "v10 accept lost the backend communicator output";
  EXPECT_EQ(recvDeviceV10, &deviceHandleToken) << "v10 accept lost the backend device output";
  ExpectDelegatedErrors("pluginAcceptV10", [&] {
    return pluginAcceptV10(&listenToken, &recvComm, &recvDeviceV10);
  });
  EXPECT_EQ(recvComm, &receiveToken) << "v10 accept errors modified the receiver output";
  EXPECT_EQ(recvDeviceV10, &deviceHandleToken) << "v10 accept errors modified the device output";
  ResetBackend();
  backend.communicatorResult = &receiveToken;
  recvComm = &sendToken;
  ncclNetDeviceHandle_v9_t* recvDeviceV9 = nullptr;
  ASSERT_EQ(pluginAcceptV9(&listenToken, &recvComm, &recvDeviceV9), ncclSuccess)
      << "v9 accept did not return backend success";
  EXPECT_EQ(recvComm, &receiveToken) << "v9 accept lost the backend communicator output";
  EXPECT_EQ(recvDeviceV9, &deviceHandleToken) << "v9 accept lost the backend device output";
  ExpectDelegatedErrors("pluginAcceptV9", [&] {
    return pluginAcceptV9(&listenToken, &recvComm, &recvDeviceV9);
  });
  EXPECT_EQ(recvComm, &receiveToken) << "v9 accept errors modified the receiver output";
  EXPECT_EQ(recvDeviceV9, &deviceHandleToken) << "v9 accept errors modified the device output";
  ResetBackend();
  backend.communicatorResult = &receiveToken;
  recvComm = &sendToken;
  ncclNetDeviceHandle_v8_t* recvDeviceV8 = nullptr;
  ASSERT_EQ(pluginAcceptV8(&listenToken, &recvComm, &recvDeviceV8), ncclSuccess)
      << "v8 accept did not return backend success";
  EXPECT_EQ(recvComm, &receiveToken) << "v8 accept lost the backend communicator output";
  EXPECT_EQ(recvDeviceV8, &deviceHandleToken) << "v8 accept lost the backend device output";
  ExpectDelegatedErrors("pluginAcceptV8", [&] {
    return pluginAcceptV8(&listenToken, &recvComm, &recvDeviceV8);
  });
  EXPECT_EQ(recvComm, &receiveToken) << "v8 accept errors modified the receiver output";
  EXPECT_EQ(recvDeviceV8, &deviceHandleToken) << "v8 accept errors modified the device output";
  ResetBackend();
  backend.communicatorResult = &receiveToken;
  recvComm = &sendToken;
  ncclNetDeviceHandle_v7_t* recvDeviceV7 = nullptr;
  ASSERT_EQ(pluginAcceptV7(&listenToken, &recvComm, &recvDeviceV7), ncclSuccess)
      << "v7 accept did not return backend success";
  EXPECT_EQ(recvComm, &receiveToken) << "v7 accept lost the backend communicator output";
  EXPECT_EQ(recvDeviceV7, &deviceHandleToken) << "v7 accept lost the backend device output";
  ExpectDelegatedErrors("pluginAcceptV7", [&] {
    return pluginAcceptV7(&listenToken, &recvComm, &recvDeviceV7);
  });
  EXPECT_EQ(recvComm, &receiveToken) << "v7 accept errors modified the receiver output";
  EXPECT_EQ(recvDeviceV7, &deviceHandleToken) << "v7 accept errors modified the device output";
  ResetBackend();
  backend.communicatorResult = &receiveToken;
  recvComm = &sendToken;
  ASSERT_EQ(pluginAcceptV6(&listenToken, &recvComm), ncclSuccess)
      << "v6 accept did not return backend success";
  EXPECT_EQ(recvComm, &receiveToken) << "v6 accept lost the backend communicator output";
  EXPECT_FALSE(backend.deviceHandleOutputWasNull) << "v6 accept did not supply backend device-handle storage";
  ExpectDelegatedErrors("pluginAcceptV6", [&] { return pluginAcceptV6(&listenToken, &recvComm); });
  EXPECT_EQ(recvComm, &receiveToken) << "v6 accept errors modified the receiver output";

  // No adapter may modify the caller-owned opaque handle when the backend leaves it untouched.
  EXPECT_EQ(handle, handleBefore) << "A connection adapter modified the opaque handle outside the backend";
}

/**
 * @brief Verifies memory, data-path, completion, and close adapters including legacy size
 * conversion.
 * @note Side effects: Mutates the recording backend and operation outputs.
 */
TEST_F(PluginDataPathTest, DataPathOperations) {
  std::array<unsigned char, 16> payload = {};
  for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<unsigned char>(0xa0 + i);
  const auto payloadBefore = payload;
  void* memoryHandle = nullptr;

  // Current registration must preserve a valid buffer extent and expose the backend memory handle.
  ASSERT_EQ(pluginRegMr(&sendToken, payload.data(), payload.size(), NCCL_PTR_CUDA, &memoryHandle), ncclSuccess)
      << "Current registration did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::RegMr) << "Current registration called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &sendToken) << "Current registration changed the communicator";
  EXPECT_EQ(backend.data, payload.data()) << "Current registration changed the data pointer";
  EXPECT_EQ(backend.size, payload.size()) << "Current registration changed the size";
  EXPECT_EQ(backend.type, NCCL_PTR_CUDA) << "Current registration changed the memory type";
  EXPECT_EQ(memoryHandle, &memoryToken) << "Current registration lost the backend memory handle";
  ExpectDelegatedErrors("pluginRegMr", [&] {
    return pluginRegMr(&sendToken, payload.data(), payload.size(), NCCL_PTR_HOST, &memoryHandle);
  });
  EXPECT_EQ(memoryHandle, &memoryToken) << "Registration errors modified the memory-handle output";

  // DMA-BUF registration must preserve size, type, offset, descriptor, and output.
  ResetBackend();
  memoryHandle = nullptr;
  ASSERT_EQ(pluginRegMrDmaBuf(&sendToken, payload.data(), payload.size(), NCCL_PTR_DMABUF, 4096, 17,
                              &memoryHandle), ncclSuccess)
      << "DMA-BUF registration did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::RegMrDmaBuf)
      << "DMA-BUF registration called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &sendToken) << "DMA-BUF registration changed the communicator";
  EXPECT_EQ(backend.data, payload.data()) << "DMA-BUF registration changed the data pointer";
  EXPECT_EQ(backend.size, payload.size()) << "DMA-BUF registration changed the size";
  EXPECT_EQ(backend.type, NCCL_PTR_DMABUF) << "DMA-BUF registration changed the memory type";
  EXPECT_EQ(backend.offset, 4096U) << "DMA-BUF registration changed the offset";
  EXPECT_EQ(backend.fd, 17) << "DMA-BUF registration changed the descriptor";
  EXPECT_EQ(memoryHandle, &memoryToken) << "DMA-BUF registration lost the backend memory handle";
  ExpectDelegatedErrors("pluginRegMrDmaBuf", [&] {
    return pluginRegMrDmaBuf(&sendToken, payload.data(), payload.size(), NCCL_PTR_DMABUF, 7, 8, &memoryHandle);
  });
  EXPECT_EQ(memoryHandle, &memoryToken) << "DMA-BUF errors modified the memory-handle output";

  // Legacy registration must widen a valid signed size exactly without modifying the payload.
  ResetBackend();
  memoryHandle = &listenToken;
  ASSERT_EQ(pluginRegMrV7(&sendToken, payload.data(), static_cast<int>(payload.size()), NCCL_PTR_HOST,
                           &memoryHandle), ncclSuccess)
      << "Legacy registration did not return backend success";
  EXPECT_EQ(backend.communicator, &sendToken) << "Legacy registration changed the communicator";
  EXPECT_EQ(backend.data, payload.data()) << "Legacy registration changed the data pointer";
  EXPECT_EQ(backend.size, payload.size()) << "Legacy registration changed the widened size";
  EXPECT_EQ(backend.type, NCCL_PTR_HOST) << "Legacy registration changed the memory type";
  EXPECT_EQ(memoryHandle, &memoryToken) << "Legacy registration lost the backend memory handle";
  ExpectDelegatedErrors("pluginRegMrV7", [&] {
    return pluginRegMrV7(&sendToken, payload.data(), 1, NCCL_PTR_HOST, &memoryHandle);
  });
  EXPECT_EQ(memoryHandle, &memoryToken) << "Legacy registration errors modified the memory-handle output";

  // Deregistration must forward both opaque inputs and every backend error.
  ResetBackend();
  ASSERT_EQ(pluginDeregMr(&sendToken, &memoryToken), ncclSuccess)
      << "Deregistration did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::DeregMr) << "Deregistration called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &sendToken) << "Deregistration changed the communicator";
  EXPECT_EQ(backend.memoryHandle, &memoryToken) << "Deregistration changed the memory handle";
  ExpectDelegatedErrors("pluginDeregMr", [&] { return pluginDeregMr(&sendToken, &memoryToken); });

  // Current send must preserve its profiling handle and expose the backend request.
  ResetBackend();
  backend.communicatorResult = &sendToken;
  void* request = nullptr;
  ASSERT_EQ(pluginIsend(&sendToken, payload.data(), payload.size(), INT_MIN, &memoryToken, &contextToken, &request),
            ncclSuccess)
      << "Current send did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Isend) << "Current send called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &sendToken) << "Current send changed the communicator";
  EXPECT_EQ(backend.data, payload.data()) << "Current send changed the data pointer";
  EXPECT_EQ(backend.size, payload.size()) << "Current send changed the size";
  EXPECT_EQ(backend.tag, INT_MIN) << "Current send changed the tag";
  EXPECT_EQ(backend.memoryHandle, &memoryToken) << "Current send changed the memory handle";
  EXPECT_EQ(backend.profilingHandle, &contextToken) << "Current send changed the profiling handle";
  EXPECT_EQ(backend.requestOutput, &request) << "Current send changed the request-output pointer";
  EXPECT_EQ(request, &requestToken) << "Current send lost the backend request";
  ExpectDelegatedErrors("pluginIsend", [&] {
    return pluginIsend(&sendToken, payload.data(), payload.size(), 3, &memoryToken, &contextToken, &request);
  });
  EXPECT_EQ(request, &requestToken) << "Current send errors modified the request output";

  // v9 send must omit the profiling handle while preserving its size and request output.
  ResetBackend();
  request = &listenToken;
  ASSERT_EQ(pluginIsendV9(&sendToken, payload.data(), payload.size(), INT_MAX, &memoryToken, &request), ncclSuccess)
      << "v9 send did not return backend success";
  EXPECT_EQ(backend.communicator, &sendToken) << "v9 send changed the communicator";
  EXPECT_EQ(backend.data, payload.data()) << "v9 send changed the data pointer";
  EXPECT_EQ(backend.size, payload.size()) << "v9 send changed the size";
  EXPECT_EQ(backend.tag, INT_MAX) << "v9 send changed the tag";
  EXPECT_EQ(backend.memoryHandle, &memoryToken) << "v9 send changed the memory handle";
  EXPECT_EQ(backend.profilingHandle, nullptr) << "v9 send supplied a profiling handle absent from its ABI";
  EXPECT_EQ(request, &requestToken) << "v9 send lost the backend request";
  ExpectDelegatedErrors("pluginIsendV9", [&] {
    return pluginIsendV9(&sendToken, payload.data(), payload.size(), 4, &memoryToken, &request);
  });
  EXPECT_EQ(request, &requestToken) << "v9 send errors modified the request output";

  // v8 send must widen its valid signed size and omit the profiling handle.
  ResetBackend();
  request = &listenToken;
  ASSERT_EQ(pluginIsendV8(&sendToken, payload.data(), static_cast<int>(payload.size()), 5, &memoryToken, &request),
            ncclSuccess)
      << "v8 send did not return backend success";
  EXPECT_EQ(backend.communicator, &sendToken) << "v8 send changed the communicator";
  EXPECT_EQ(backend.data, payload.data()) << "v8 send changed the data pointer";
  EXPECT_EQ(backend.size, payload.size()) << "v8 send changed the widened size";
  EXPECT_EQ(backend.tag, 5) << "v8 send changed the tag";
  EXPECT_EQ(backend.memoryHandle, &memoryToken) << "v8 send changed the memory handle";
  EXPECT_EQ(backend.profilingHandle, nullptr) << "v8 send supplied a profiling handle absent from its ABI";
  EXPECT_EQ(request, &requestToken) << "v8 send lost the backend request";
  ExpectDelegatedErrors("pluginIsendV8", [&] {
    return pluginIsendV8(&sendToken, payload.data(), 1, 5, &memoryToken, &request);
  });
  EXPECT_EQ(request, &requestToken) << "v8 send errors modified the request output";

  // Current receive must forward every array and preserve the profiling-handle array.
  constexpr int receiveCount = NCCL_NET_IB_MAX_RECVS;
  std::array<void*, receiveCount> data = {};
  std::array<size_t, receiveCount> sizes = {};
  std::array<int, receiveCount> tags = {};
  std::array<void*, receiveCount> memoryHandles = {};
  std::array<void*, receiveCount> profilingHandles = {};
  for (int i = 0; i < receiveCount; ++i) {
    data[i] = payload.data() + i;
    sizes[i] = i == 0 ? 0 : payload.size() - i;
    tags[i] = INT_MIN + i;
    memoryHandles[i] = &memoryToken;
    profilingHandles[i] = &contextToken;
  }
  const auto dataBefore = data;
  const auto sizesBefore = sizes;
  const auto tagsBefore = tags;
  const auto memoryHandlesBefore = memoryHandles;
  const auto profilingHandlesBefore = profilingHandles;
  ResetBackend();
  request = &listenToken;
  ASSERT_EQ(pluginIrecv(&receiveToken, receiveCount, data.data(), sizes.data(), tags.data(),
                         memoryHandles.data(), profilingHandles.data(), &request), ncclSuccess)
      << "Current receive did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Irecv) << "Current receive called the wrong backend operation";
  EXPECT_EQ(backend.n, receiveCount) << "Current receive changed the receive count";
  EXPECT_EQ(backend.dataArray, data.data()) << "Current receive changed the data-array pointer";
  EXPECT_EQ(backend.sizeArray, sizes.data()) << "Current receive changed the size-array pointer";
  EXPECT_EQ(backend.tagArray, tags.data()) << "Current receive changed the tag-array pointer";
  EXPECT_EQ(backend.memoryHandleArray, memoryHandles.data()) << "Current receive changed the handle-array pointer";
  EXPECT_EQ(backend.profilingHandleArray, profilingHandles.data())
      << "Current receive changed the profiling-array pointer";
  EXPECT_EQ(request, &requestToken) << "Current receive lost the backend request";
  ExpectDelegatedErrors("pluginIrecv", [&] {
    return pluginIrecv(&receiveToken, 1, data.data(), sizes.data(), tags.data(), memoryHandles.data(),
                       profilingHandles.data(), &request);
  });
  EXPECT_EQ(request, &requestToken) << "Current receive errors modified the request output";
  EXPECT_EQ(sizes, sizesBefore) << "Current receive modified the size array";

  // v9 receive must preserve size_t values while omitting the profiling-handle array.
  ResetBackend();
  sizes[receiveCount - 1] = payload.size();
  const auto sizesV9Before = sizes;
  request = &listenToken;
  ASSERT_EQ(pluginIrecvV9(&receiveToken, receiveCount, data.data(), sizes.data(), tags.data(),
                           memoryHandles.data(), &request), ncclSuccess)
      << "v9 receive did not return backend success";
  EXPECT_EQ(backend.communicator, &receiveToken) << "v9 receive changed the communicator";
  EXPECT_EQ(backend.dataArray, data.data()) << "v9 receive changed the data-array pointer";
  EXPECT_EQ(backend.sizeArray, sizes.data()) << "v9 receive changed the size-array pointer";
  EXPECT_EQ(backend.tagArray, tags.data()) << "v9 receive changed the tag-array pointer";
  EXPECT_EQ(backend.memoryHandleArray, memoryHandles.data()) << "v9 receive changed the handle-array pointer";
  EXPECT_EQ(backend.copiedSizes[receiveCount - 1], payload.size()) << "v9 receive changed the final size";
  EXPECT_EQ(backend.profilingHandleArray, nullptr) << "v9 receive supplied a profiling array absent from its ABI";
  EXPECT_EQ(request, &requestToken) << "v9 receive lost the backend request";
  ExpectDelegatedErrors("pluginIrecvV9", [&] {
    return pluginIrecvV9(&receiveToken, 1, data.data(), sizes.data(), tags.data(), memoryHandles.data(), &request);
  });
  EXPECT_EQ(request, &requestToken) << "v9 receive errors modified the request output";
  EXPECT_EQ(sizes, sizesV9Before) << "v9 receive modified the size array";

  // v8 receive must convert every valid signed size without changing any caller-owned array.
  std::array<int, receiveCount> legacySizes = {};
  for (int i = 0; i < receiveCount; ++i) {
    legacySizes[i] = i == 0 ? 0 : static_cast<int>(payload.size() - i);
  }
  const auto legacySizesBefore = legacySizes;
  ResetBackend();
  backend.forbiddenSizeArray = legacySizes.data();
  request = &listenToken;
  ASSERT_EQ(pluginIrecvV8(&receiveToken, receiveCount, data.data(), legacySizes.data(), tags.data(),
                           memoryHandles.data(), &request), ncclSuccess)
      << "v8 receive did not return backend success";
  EXPECT_EQ(backend.communicator, &receiveToken) << "v8 receive changed the communicator";
  EXPECT_EQ(backend.dataArray, data.data()) << "v8 receive changed the data-array pointer";
  EXPECT_EQ(backend.tagArray, tags.data()) << "v8 receive changed the tag-array pointer";
  EXPECT_EQ(backend.memoryHandleArray, memoryHandles.data()) << "v8 receive changed the handle-array pointer";
  EXPECT_FALSE(backend.sizeArrayAliasesForbidden)
      << "v8 receive forwarded the legacy int array instead of a converted size_t array";
  for (int i = 0; i < receiveCount; ++i) {
    EXPECT_EQ(backend.copiedSizes[i], static_cast<size_t>(legacySizes[i]))
        << "v8 receive converted size " << i << " incorrectly";
  }
  EXPECT_EQ(backend.profilingHandleArray, nullptr) << "v8 receive supplied a profiling array absent from its ABI";
  EXPECT_EQ(request, &requestToken) << "v8 receive lost the backend request output";
  EXPECT_EQ(legacySizes, legacySizesBefore) << "v8 receive modified the legacy size array";
  ExpectDelegatedErrors("pluginIrecvV8", [&] {
    return pluginIrecvV8(&receiveToken, 1, data.data(), legacySizes.data(), tags.data(), memoryHandles.data(),
                         &request);
  });
  EXPECT_EQ(request, &requestToken) << "v8 receive errors modified the request output";

  // Explicitly rejected counts must not call the backend or modify inputs and outputs.
  for (int invalidCount : {-1, NCCL_NET_MAX_REQUESTS + 1}) {
    SCOPED_TRACE(std::string("pluginIrecvV8 n=") + std::to_string(invalidCount));
    ResetBackend();
    request = &listenToken;
    EXPECT_EQ(pluginIrecvV8(&receiveToken, invalidCount, data.data(), legacySizes.data(), tags.data(),
                             memoryHandles.data(), &request), ncclInvalidArgument)
        << "v8 receive accepted an explicitly rejected count";
    EXPECT_EQ(backend.calls, 0) << "v8 receive called the backend for an explicitly rejected count";
    EXPECT_EQ(request, &listenToken) << "v8 receive modified the request for a rejected count";
    EXPECT_EQ(legacySizes, legacySizesBefore) << "v8 receive modified sizes for a rejected count";
  }

  // Flush must forward all arrays and expose the backend request.
  ResetBackend();
  request = nullptr;
  ASSERT_EQ(pluginIflush(&receiveToken, receiveCount, data.data(), legacySizes.data(), memoryHandles.data(), &request),
            ncclSuccess)
      << "Flush did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Iflush) << "Flush called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &receiveToken) << "Flush changed the communicator";
  EXPECT_EQ(backend.n, receiveCount) << "Flush changed the array length";
  EXPECT_EQ(backend.dataArray, data.data()) << "Flush changed the data-array pointer";
  EXPECT_EQ(backend.intSizeArray, legacySizes.data()) << "Flush changed the size-array pointer";
  EXPECT_EQ(backend.memoryHandleArray, memoryHandles.data()) << "Flush changed the handle-array pointer";
  EXPECT_EQ(request, &requestToken) << "Flush lost the backend request";
  ExpectDelegatedErrors("pluginIflush", [&] {
    return pluginIflush(&receiveToken, receiveCount, data.data(), legacySizes.data(), memoryHandles.data(), &request);
  });
  EXPECT_EQ(request, &requestToken) << "Flush errors modified the request output";

  // An empty flush must forward null arrays and preserve the caller-owned request output.
  ResetBackend();
  request = &listenToken;
  ASSERT_EQ(pluginIflush(&receiveToken, 0, nullptr, nullptr, nullptr, &request), ncclSuccess)
      << "An empty flush did not return backend success";
  EXPECT_EQ(backend.n, 0) << "An empty flush changed its zero count";
  EXPECT_EQ(backend.dataArray, nullptr) << "An empty flush changed its null data array";
  EXPECT_EQ(backend.intSizeArray, nullptr) << "An empty flush changed its null size array";
  EXPECT_EQ(backend.memoryHandleArray, nullptr) << "An empty flush changed its null memory-handle array";
  EXPECT_EQ(request, &listenToken) << "An empty flush unexpectedly created a request";

  // Completion testing must expose both provider-supplied values through caller-owned outputs.
  ResetBackend();
  int done = 0;
  int completedSize = -1;
  ASSERT_EQ(pluginTest(&requestToken, &done, &completedSize), ncclSuccess)
      << "Request testing did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::Test) << "Request testing called the wrong backend operation";
  EXPECT_EQ(backend.request, &requestToken) << "Request testing changed the request pointer";
  EXPECT_EQ(backend.doneOutput, &done) << "Request testing changed the completion-output pointer";
  EXPECT_EQ(backend.completedSizes, &completedSize) << "Request testing changed the size-output pointer";
  EXPECT_EQ(done, backend.doneResult) << "Request testing lost the backend completion flag";
  EXPECT_EQ(completedSize, backend.completedSizeResult) << "Request testing lost the backend completed size";
  ExpectDelegatedErrors("pluginTest", [&] { return pluginTest(&requestToken, &done, &completedSize); });
  EXPECT_EQ(done, backend.doneResult) << "Request-testing errors modified the completion flag";
  EXPECT_EQ(completedSize, backend.completedSizeResult) << "Request-testing errors modified the completed size";

  // A null optional size output must be forwarded without affecting the completion flag.
  ResetBackend();
  done = 0;
  ASSERT_EQ(pluginTest(&requestToken, &done, nullptr), ncclSuccess)
      << "Request testing rejected a null optional size output";
  EXPECT_EQ(backend.completedSizes, nullptr) << "Request testing changed the null optional size output";
  EXPECT_EQ(done, backend.doneResult) << "Request testing lost completion with a null size output";

  // Every close adapter must select the matching backend operation and propagate all errors.
  ResetBackend();
  ASSERT_EQ(pluginCloseSend(&sendToken), ncclSuccess) << "Sender close did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::CloseSend) << "Sender close called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &sendToken) << "Sender close changed the communicator";
  ExpectDelegatedErrors("pluginCloseSend", [&] { return pluginCloseSend(&sendToken); });
  ResetBackend();
  ASSERT_EQ(pluginCloseRecv(&receiveToken), ncclSuccess) << "Receiver close did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::CloseRecv) << "Receiver close called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &receiveToken) << "Receiver close changed the communicator";
  ExpectDelegatedErrors("pluginCloseRecv", [&] { return pluginCloseRecv(&receiveToken); });
  ResetBackend();
  ASSERT_EQ(pluginCloseListen(&listenToken), ncclSuccess) << "Listener close did not return backend success";
  EXPECT_EQ(backend.operation, BackendOperation::CloseListen) << "Listener close called the wrong backend operation";
  EXPECT_EQ(backend.communicator, &listenToken) << "Listener close changed the communicator";
  ExpectDelegatedErrors("pluginCloseListen", [&] { return pluginCloseListen(&listenToken); });

  // Every input-only data buffer and array must remain unchanged when the backend leaves it
  // untouched.
  EXPECT_EQ(payload, payloadBefore) << "A data-path adapter modified the payload";
  EXPECT_EQ(data, dataBefore) << "A receive adapter modified the data-pointer array";
  EXPECT_EQ(tags, tagsBefore) << "A receive adapter modified the tag array";
  EXPECT_EQ(memoryHandles, memoryHandlesBefore) << "A receive adapter modified the memory-handle array";
  EXPECT_EQ(profilingHandles, profilingHandlesBefore) << "A receive adapter modified the profiling-handle array";
  EXPECT_EQ(legacySizes, legacySizesBefore) << "A receive adapter modified the legacy size array";
}

}  // namespace
