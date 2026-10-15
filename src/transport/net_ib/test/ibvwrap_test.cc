/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "ibvsymbols.h"
#include "ibvwrap.h"
#include "param.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

extern struct ncclIbvSymbols ibvSymbols;

namespace {

/** @brief Records one normalized call to the fake verbs provider. */
struct ProviderCall {
  std::string operation;
  std::array<uintptr_t, 7> arguments = {};
  size_t argumentCount = 0;
};

/** @brief Records one generated verbs-wrapper parameter lookup. */
struct ParameterLoadCall {
  std::string environment;
  int64_t defaultValue = 0;
};

/**
 * @brief Owns stable fake-provider result objects and records a normalized call trace.
 */
struct ProviderState {
  ncclResult_t buildResult = ncclSuccess;
  int buildCalls = 0;
  struct ncclIbvSymbols buildSymbols = {};
  int integerResult = 0;
  bool returnNull = false;
  int retryAll = 0;
  int retryCount = 2;
  int retryTimeout = 0;
  int deviceCount = 1;
  int extendedPortResult = -1;
  std::vector<int> modifyResults;
  size_t nextModifyResult = 0;
  std::vector<ProviderCall> calls;
  std::vector<ParameterLoadCall> parameterLoads;
  struct ibv_device device = {};
  struct ibv_device* deviceList[2] = {};
  struct ibv_context context = {};
  struct ibv_pd protectionDomain = {};
  struct ibv_mr memoryRegion = {};
  struct ibv_cq completionQueue = {};
  struct ibv_qp queuePair = {};
  struct ibv_device_attr deviceAttributes = {};
  struct ibv_port_attr portAttributes = {};
  union ibv_gid gid = {};
  struct ibv_qp_attr queuePairAttributes = {};
  struct ibv_qp_init_attr queuePairInitAttributes = {};
  struct ibv_ece eceResult = {};
  enum ibv_event_type asyncEventType = IBV_EVENT_PORT_ACTIVE;
  const char* deviceName = "provider-device";
  const char* eventName = "provider-event";
};

ProviderState providerState;

uintptr_t PointerValueForTest(const void* pointer);
void RecordProviderCallForTest(const char* operation, std::initializer_list<uintptr_t> arguments);
void ResetProviderForTest();
const ProviderCall& LastProviderCallForTest();
int ForkInitForTest();
struct ibv_device** GetDeviceListForTest(int* count);
void FreeDeviceListForTest(struct ibv_device** list);
const char* GetDeviceNameForTest(struct ibv_device* device);
struct ibv_context* OpenDeviceForTest(struct ibv_device* device);
int CloseDeviceForTest(struct ibv_context* context);
int GetAsyncEventForTest(struct ibv_context* context, struct ibv_async_event* event);
void AckAsyncEventForTest(struct ibv_async_event* event);
int QueryDeviceForTest(struct ibv_context* context, struct ibv_device_attr* attributes);
int QueryPortForTest(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attributes);
int QueryExtendedPortForTest(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attributes, size_t size);
int QueryGidForTest(struct ibv_context* context, uint8_t port, int index, union ibv_gid* gid);
int QueryQueuePairForTest(struct ibv_qp* queuePair, struct ibv_qp_attr* attributes, int mask,
  struct ibv_qp_init_attr* initialAttributes);
struct ibv_pd* AllocateProtectionDomainForTest(struct ibv_context* context);
int DeallocateProtectionDomainForTest(struct ibv_pd* protectionDomain);
struct ibv_mr* RegisterMemoryForTest(struct ibv_pd* protectionDomain, void* address, size_t length, int access);
struct ibv_mr* RegisterMemoryIovaForTest(struct ibv_pd* protectionDomain, void* address, size_t length,
  uint64_t iova, unsigned int access);
struct ibv_mr* RegisterDmaBufForTest(struct ibv_pd* protectionDomain, uint64_t offset, size_t length,
  uint64_t iova, int descriptor, int access);
int DeregisterMemoryForTest(struct ibv_mr* memoryRegion);
struct ibv_cq* CreateCompletionQueueForTest(struct ibv_context* context, int entries, void* queueContext,
  struct ibv_comp_channel* channel, int vector);
int DestroyCompletionQueueForTest(struct ibv_cq* completionQueue);
struct ibv_qp* CreateQueuePairForTest(struct ibv_pd* protectionDomain, struct ibv_qp_init_attr* attributes);
int DestroyQueuePairForTest(struct ibv_qp* queuePair);
int ModifyQueuePairForTest(struct ibv_qp* queuePair, struct ibv_qp_attr* attributes, int mask);
int QueryEceForTest(struct ibv_qp* queuePair, struct ibv_ece* ece);
int SetEceForTest(struct ibv_qp* queuePair, struct ibv_ece* ece);
const char* EventTypeStringForTest(enum ibv_event_type event);
struct ncclIbvSymbols MakeSymbolTableForTest();
void ExpectLastProviderCallForTest(const char* operation, std::initializer_list<uintptr_t> arguments,
  const char* reason);
void ExpectBytesEqualForTest(const void* actual, const void* expected, size_t size, const char* reason);
size_t CountProviderCallsForTest(const char* operation);

template <typename Symbol, typename Call>
void ExpectIntegerDispatchForTest(Symbol& slot, Symbol implementation, int failure, Call call, const char* reason,
  const std::function<void()>& checkMissingOutput = {});

template <typename Symbol, typename Output, typename Call>
void ExpectPointerDispatchForTest(Symbol& slot, Symbol implementation, Output& output,
  Output successfulOutput, Call call, const char* reason);

template <typename Symbol, typename Call>
void ExpectVoidDispatchForTest(Symbol& slot, Symbol implementation, Call call, const char* reason);

template <typename Symbol, typename Call>
void ExpectOptionalEceDispatchForTest(Symbol& slot, Symbol implementation, Call call, const char* reason,
  const std::function<void()>& checkMissingOutput = {});

/** @brief Provides an isolated complete fake provider for verbs-wrapper tests. */
class IbvTestBase : public ::testing::Test {
 protected:
  void SetUp() override;
};

/** @brief Groups verbs symbol-table initialization scenarios. */
class IbvSymbolTest : public IbvTestBase {};
/** @brief Groups verbs device and port query scenarios. */
class IbvQueryTest : public IbvTestBase {};
/** @brief Groups verbs resource ownership scenarios. */
class IbvResourceTest : public IbvTestBase {};
/** @brief Groups verbs queue-pair and diagnostic scenarios. */
class IbvQueuePairTest : public IbvTestBase {};

}  // namespace

/**
 * @brief Supplies the controlled symbol table used by wrap_ibv_symbols in this test target.
 * @param[out] symbols Valid non-NULL storage receiving providerState.buildSymbols.
 * @return providerState.buildResult.
 * @note Side effects: Writes symbols, increments buildCalls, and records the output pointer.
 */
ncclResult_t buildIbvSymbols(struct ncclIbvSymbols* symbols) {
  ++providerState.buildCalls;
  RecordProviderCallForTest("build_symbols", {PointerValueForTest(symbols)});
  *symbols = providerState.buildSymbols;
  return providerState.buildResult;
}

/**
 * @brief Resolves retry parameters for the production-generated parameter accessors.
 * @param[in] environment Valid non-NULL environment-variable name.
 * @param[in] defaultValue Production default returned for an unknown parameter name.
 * @param[in] uninitialized Production cache sentinel, accepted without modification.
 * @param[in] cache Valid non-NULL production cache storage left unmodified to keep the test
 * seam dynamic.
 * @param[out] noCache Valid non-NULL cache-policy storage set to one.
 * @return The configured retry value selected by environment, or defaultValue for an unknown name.
 * @note Side effects: Records the parameter name and default value and disables caching for
 * this test target.
 */
int64_t LoadIbvParameterForTest(const char* environment, int64_t defaultValue, int64_t uninitialized,
  int64_t* cache, int8_t* noCache) {
  (void)uninitialized;
  (void)cache;
  providerState.parameterLoads.push_back({environment, defaultValue});
  *noCache = 1;
  if (std::strcmp(environment, "NCCL_IB_MQP_RETRY_ALL") == 0) return providerState.retryAll;
  if (std::strcmp(environment, "NCCL_IB_MQP_RETRY_CNT") == 0) return providerState.retryCount;
  if (std::strcmp(environment, "NCCL_IB_MQP_RETRY_SLEEP_MSEC") == 0) return providerState.retryTimeout;
  return defaultValue;
}

/**
 * @brief Compiles ibvwrap.cc with uncached retry parameters that remain scenario-configurable.
 */
#define ncclLoadParam LoadIbvParameterForTest
#include "../../../misc/ibvwrap.cc"  // NOLINT(build/include)
#undef ncclLoadParam

namespace {

/**
 * @brief Verifies one-time symbol loading and the complete device lifecycle dispatch contract.
 * @note Side effects: Runs two isolated child processes and calls the fake device provider.
 */
TEST_F(IbvSymbolTest, InitializesAndManagesDevices) {
  SCOPED_TRACE("Symbol loading must be cached, and device wrappers must preserve provider results and arguments.");

  EXPECT_EXIT(
    {
      ResetProviderForTest();
      providerState.buildSymbols = MakeSymbolTableForTest();
      ibvSymbols = {};
      const ncclResult_t firstResult = wrap_ibv_symbols();
      const ncclResult_t secondResult = wrap_ibv_symbols();
      if (firstResult != ncclSuccess || secondResult != ncclSuccess) {
        std::fprintf(stderr, "symbol loader did not preserve successful results: %d, %d\n",
          firstResult, secondResult);
        _exit(1);
      }
      if (providerState.buildCalls != 1) {
        std::fprintf(stderr, "cached symbol load called builder %d times\n", providerState.buildCalls);
        _exit(2);
      }
      if (ibvSymbols.ibv_internal_open_device != OpenDeviceForTest) {
        std::fprintf(stderr, "successful symbol load did not install the returned table\n");
        _exit(3);
      }
      _exit(0);
    },
    ::testing::ExitedWithCode(0), "") << "A successful symbol load must run exactly once and install the table.";

  EXPECT_EXIT(
    {
      ResetProviderForTest();
      providerState.buildResult = ncclSystemError;
      providerState.buildSymbols = MakeSymbolTableForTest();
      ibvSymbols = {};
      const ncclResult_t firstResult = wrap_ibv_symbols();
      const ncclResult_t secondResult = wrap_ibv_symbols();
      if (firstResult != ncclSystemError || secondResult != ncclSystemError) {
        std::fprintf(stderr, "symbol loader did not preserve failed results: %d, %d\n",
          firstResult, secondResult);
        _exit(1);
      }
      if (providerState.buildCalls != 1 || providerState.calls.size() != 1) {
        std::fprintf(stderr, "failed symbol load was not cached: builds=%d calls=%zu\n",
          providerState.buildCalls, providerState.calls.size());
        _exit(2);
      }
      const ProviderCall& call = providerState.calls.back();
      if (call.operation != "build_symbols" || call.argumentCount != 1 ||
          call.arguments[0] != PointerValueForTest(&ibvSymbols)) {
        std::fprintf(stderr, "symbol builder did not receive the production table\n");
        _exit(3);
      }
      _exit(0);
    },
    ::testing::ExitedWithCode(0), "")
    << "A failed symbol load must be returned unchanged and cached without leaking process-wide state.";

  ibvSymbols = MakeSymbolTableForTest();
  const struct ibv_device deviceBefore = providerState.device;
  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_fork_init, ForkInitForTest, -1,
    [] { return wrap_ibv_fork_init(); }, "Fork initialization must translate its three defined outcomes.");
  ExpectLastProviderCallForTest("fork_init", {}, "Fork initialization must dispatch without arguments.");

  struct ibv_device** list = nullptr;
  int deviceCount = -1;
  providerState.calls.clear();
  providerState.returnNull = false;
  EXPECT_EQ(wrap_ibv_get_device_list(&list, &deviceCount), ncclSuccess)
    << "A provider device list must be returned successfully.";
  EXPECT_EQ(list, providerState.deviceList) << "The wrapper must return the provider-owned device list.";
  EXPECT_EQ(deviceCount, 1) << "The wrapper must preserve the provider's device count.";
  ExpectLastProviderCallForTest("get_device_list", {PointerValueForTest(&deviceCount)},
    "Device-list discovery must forward the count output pointer.");

  providerState.returnNull = true;
  list = reinterpret_cast<struct ibv_device**>(static_cast<uintptr_t>(1));
  deviceCount = 17;
  EXPECT_EQ(wrap_ibv_get_device_list(&list, &deviceCount), ncclSuccess)
    << "An empty provider device list is a successful discovery result.";
  EXPECT_EQ(list, nullptr) << "The wrapper must preserve the provider's NULL list result.";
  EXPECT_EQ(deviceCount, 0) << "A NULL device list must force the wrapper-owned count to zero.";
  providerState.returnNull = false;

  EXPECT_EXIT(
    {
      ibvSymbols.ibv_internal_get_device_name = nullptr;
      wrap_ibv_get_device_name(&providerState.device);
      _exit(0);
    },
    ::testing::ExitedWithCode(255), "")
    << "Device-name lookup must terminate with exit(-1) when its mandatory symbol is absent.";

  providerState.calls.clear();
  EXPECT_STREQ(wrap_ibv_get_device_name(&providerState.device), providerState.deviceName)
    << "The wrapper must return the provider's device-name pointer.";
  ExpectLastProviderCallForTest("get_device_name", {PointerValueForTest(&providerState.device)},
    "Device-name lookup must forward the exact device.");

  struct ibv_context* context = nullptr;
  ExpectPointerDispatchForTest(ibvSymbols.ibv_internal_open_device, OpenDeviceForTest, context,
    &providerState.context,
    [&] { return wrap_ibv_open_device(&context, &providerState.device); },
    "Device opening must translate missing, NULL, and successful provider results.");
  ExpectLastProviderCallForTest("open_device", {PointerValueForTest(&providerState.device)},
    "Device opening must forward the exact device.");

  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_close_device, CloseDeviceForTest, -1,
    [&] { return wrap_ibv_close_device(context); },
    "Device closing must translate missing, failing, and successful provider results.");
  ExpectLastProviderCallForTest("close_device", {PointerValueForTest(context)},
    "Device closing must forward the exact context.");
  ExpectBytesEqualForTest(&providerState.device, &deviceBefore, sizeof(deviceBefore),
    "Device lookup and opening must not change the input device through the fake provider.");

  ExpectVoidDispatchForTest(ibvSymbols.ibv_internal_free_device_list, FreeDeviceListForTest,
    [&] { return wrap_ibv_free_device_list(providerState.deviceList); },
    "Device-list release must reject a missing symbol and dispatch an installed symbol.");
  ExpectLastProviderCallForTest("free_device_list", {PointerValueForTest(providerState.deviceList)},
    "Device-list release must forward the exact provider-owned list after device use is complete.");
}

/**
 * @brief Verifies asynchronous-event retrieval and acknowledgement without changing
 * acknowledgement input.
 * @note Side effects: Calls the fake provider and writes retrieval output on success.
 */
TEST_F(IbvSymbolTest, HandlesAsyncEvents) {
  SCOPED_TRACE(
      "Async-event wrappers must translate provider results, expose successful output, forward "
      "arguments, and preserve input.");

  const struct ibv_context contextBefore = providerState.context;
  struct ibv_async_event event = {};
  event.event_type = IBV_EVENT_DEVICE_FATAL;
  const struct ibv_async_event eventBeforeMissingSymbol = event;
  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_get_async_event, GetAsyncEventForTest, -1,
    [&] { return wrap_ibv_get_async_event(&providerState.context, &event); },
    "Event retrieval must translate missing, failing, and successful provider results.", [&] {
      ExpectBytesEqualForTest(&event, &eventBeforeMissingSymbol, sizeof(event),
        "Missing event-retrieval symbols must leave event output storage unchanged.");
    });
  EXPECT_EQ(event.event_type, providerState.asyncEventType)
    << "Successful event retrieval must expose the provider event type.";
  ExpectLastProviderCallForTest("get_async_event",
    {PointerValueForTest(&providerState.context), PointerValueForTest(&event)},
    "Event retrieval must forward the exact context and output storage.");

  const struct ibv_async_event eventBefore = event;
  ExpectVoidDispatchForTest(ibvSymbols.ibv_internal_ack_async_event, AckAsyncEventForTest,
    [&] { return wrap_ibv_ack_async_event(&event); },
    "Event acknowledgement must reject a missing symbol and dispatch an installed symbol.");
  ExpectBytesEqualForTest(&event, &eventBefore, sizeof(event),
    "Acknowledgement must not change the caller's event through the non-mutating provider.");
  ExpectLastProviderCallForTest("ack_async_event", {PointerValueForTest(&event)},
    "Event acknowledgement must forward the exact event.");
  ExpectBytesEqualForTest(&providerState.context, &contextBefore, sizeof(contextBefore),
    "Event retrieval and acknowledgement must not change the provider context.");
}

/**
 * @brief Verifies attribute queries, exact outputs, regular-port fallback, and extended-port
 * preference.
 * @note Side effects: Calls the fake query provider and writes local output structures.
 */
TEST_F(IbvQueryTest, QueriesAttributesAndPorts) {
  SCOPED_TRACE("Query wrappers must translate status, forward arguments, and return exact successful provider data.");

  const struct ibv_context contextBefore = providerState.context;
  const struct ibv_qp queuePairBefore = providerState.queuePair;

  struct ibv_device_attr deviceAttributes = {};
  deviceAttributes.max_qp = 707;
  const struct ibv_device_attr deviceAttributesBeforeMissingSymbol = deviceAttributes;
  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_query_device, QueryDeviceForTest, EIO,
    [&] { return wrap_ibv_query_device(&providerState.context, &deviceAttributes); },
    "Device queries must translate missing, failing, and successful provider results.", [&] {
      ExpectBytesEqualForTest(&deviceAttributes, &deviceAttributesBeforeMissingSymbol,
        sizeof(deviceAttributes), "Missing device-query symbols must leave output storage unchanged.");
    });
  ExpectBytesEqualForTest(&deviceAttributes, &providerState.deviceAttributes, sizeof(deviceAttributes),
    "A successful device query must return every provider attribute.");
  ExpectLastProviderCallForTest("query_device",
    {PointerValueForTest(&providerState.context), PointerValueForTest(&deviceAttributes)},
    "Device queries must forward the exact context and output storage.");

  union ibv_gid gid = {};
  gid.raw[0] = 0xa5;
  const union ibv_gid gidBeforeMissingSymbol = gid;
  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_query_gid, QueryGidForTest, EIO,
    [&] { return wrap_ibv_query_gid(&providerState.context, 7, 11, &gid); },
    "GID queries must translate missing, failing, and successful provider results.", [&] {
      ExpectBytesEqualForTest(&gid, &gidBeforeMissingSymbol, sizeof(gid),
        "Missing GID-query symbols must leave output storage unchanged.");
    });
  ExpectBytesEqualForTest(&gid, &providerState.gid, sizeof(gid),
    "A successful GID query must return the complete provider GID.");
  ExpectLastProviderCallForTest("query_gid",
    {PointerValueForTest(&providerState.context), 7, 11, PointerValueForTest(&gid)},
    "GID queries must preserve the requested port and table index.");

  struct ibv_qp_attr queuePairAttributes = {};
  queuePairAttributes.port_num = 17;
  struct ibv_qp_init_attr initialAttributes = {};
  initialAttributes.cap.max_send_wr = 19;
  const struct ibv_qp_attr queuePairAttributesBeforeMissingSymbol = queuePairAttributes;
  const struct ibv_qp_init_attr initialAttributesBeforeMissingSymbol = initialAttributes;
  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_query_qp, QueryQueuePairForTest, EIO,
    [&] {
      return wrap_ibv_query_qp(&providerState.queuePair, &queuePairAttributes,
        IBV_QP_STATE | IBV_QP_PORT, &initialAttributes);
    },
    "Queue-pair queries must translate missing, failing, and successful provider results.", [&] {
      ExpectBytesEqualForTest(&queuePairAttributes, &queuePairAttributesBeforeMissingSymbol,
        sizeof(queuePairAttributes), "Missing QP-query symbols must preserve attribute output storage.");
      ExpectBytesEqualForTest(&initialAttributes, &initialAttributesBeforeMissingSymbol,
        sizeof(initialAttributes), "Missing QP-query symbols must preserve initialization output storage.");
    });
  ExpectBytesEqualForTest(&queuePairAttributes, &providerState.queuePairAttributes,
    sizeof(queuePairAttributes), "A successful queue-pair query must return every provider attribute.");
  ExpectBytesEqualForTest(&initialAttributes, &providerState.queuePairInitAttributes,
    sizeof(initialAttributes), "A successful queue-pair query must return every provider initialization field.");
  ExpectLastProviderCallForTest("query_qp", {PointerValueForTest(&providerState.queuePair),
    PointerValueForTest(&queuePairAttributes), IBV_QP_STATE | IBV_QP_PORT,
    PointerValueForTest(&initialAttributes)}, "Queue-pair queries must preserve both outputs and the mask.");

  struct ibv_port_attr portAttributes;
  struct ibv_port_attr zeroPortAttributes = {};
  std::memset(&portAttributes, 0xa5, sizeof(portAttributes));
#ifdef NCCL_BUILD_RDMA_CORE
  const struct ibv_port_attr portAttributesBeforeMissingSymbol = portAttributes;
#endif
  ibvSymbols.ibv_internal_query_port = nullptr;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_query_port(&providerState.context, 9, &portAttributes), ncclInternalError)
    << "A missing regular query symbol after extended-query fallback must return ncclInternalError.";
#ifdef NCCL_BUILD_RDMA_CORE
  ExpectBytesEqualForTest(&portAttributes, &portAttributesBeforeMissingSymbol, sizeof(portAttributes),
    "A missing regular port symbol must leave output storage unchanged in the direct rdma-core path.");
#else
  ExpectBytesEqualForTest(&portAttributes, &zeroPortAttributes, sizeof(portAttributes),
    "Regular-port fallback must clear the wrapper-owned output before discovering a missing symbol.");
#endif
  EXPECT_TRUE(providerState.calls.empty()) << "A missing regular port operation cannot reach the provider.";

  ibvSymbols.ibv_internal_query_port = QueryPortForTest;
  providerState.integerResult = EIO;
  std::memset(&portAttributes, 0xa5, sizeof(portAttributes));
  EXPECT_EQ(wrap_ibv_query_port(&providerState.context, 9, &portAttributes), ncclSystemError)
    << "A regular provider port-query failure must return ncclSystemError.";
#ifndef NCCL_BUILD_RDMA_CORE
  ExpectBytesEqualForTest(&portAttributes, &zeroPortAttributes, sizeof(portAttributes),
    "Regular-port fallback must clear the wrapper-owned output before the provider call.");
#endif

  providerState.integerResult = 0;
  EXPECT_EQ(wrap_ibv_query_port(&providerState.context, 9, &portAttributes), ncclSuccess)
    << "A successful regular port query must return ncclSuccess.";
  ExpectBytesEqualForTest(&portAttributes, &providerState.portAttributes, sizeof(portAttributes),
    "A successful regular port query must return every provider attribute.");
  ExpectLastProviderCallForTest("query_port",
    {PointerValueForTest(&providerState.context), 9, PointerValueForTest(&portAttributes)},
    "Regular-port fallback must preserve the context, port, and output storage.");

#ifndef NCCL_BUILD_RDMA_CORE
  struct verbs_context extendedContext = {};
  extendedContext.query_port = QueryExtendedPortForTest;
  extendedContext.sz = sizeof(extendedContext);
  extendedContext.context.device = &providerState.device;
  extendedContext.context.abi_compat = __VERBS_ABI_IS_EXTENDED;
  providerState.extendedPortResult = 0;
  ibvSymbols.ibv_internal_query_port = nullptr;
  providerState.calls.clear();
  std::memset(&portAttributes, 0xa5, sizeof(portAttributes));
  EXPECT_EQ(wrap_ibv_query_port(&extendedContext.context, 13, &portAttributes), ncclSuccess)
    << "A successful extended port query must not require the fallback symbol.";
  ExpectBytesEqualForTest(&portAttributes, &providerState.portAttributes, sizeof(portAttributes),
    "A successful extended port query must return every provider attribute.");
  EXPECT_EQ(providerState.calls.size(), 1U) << "A successful extended query must skip the regular provider query.";
  ExpectLastProviderCallForTest("query_port_ex",
    {PointerValueForTest(&extendedContext.context), 13, PointerValueForTest(&portAttributes),
      sizeof(portAttributes)}, "The extended query must receive the exact context, port, output, and output size.");

  providerState.extendedPortResult = EIO;
  providerState.integerResult = 0;
  ibvSymbols.ibv_internal_query_port = QueryPortForTest;
  providerState.calls.clear();
  std::memset(&portAttributes, 0xa5, sizeof(portAttributes));
  EXPECT_EQ(wrap_ibv_query_port(&extendedContext.context, 15, &portAttributes), ncclSuccess)
    << "A failed extended query must fall back to a successful regular provider query.";
  ExpectBytesEqualForTest(&portAttributes, &providerState.portAttributes, sizeof(portAttributes),
    "The regular fallback must replace output left by a failed extended query.");
  ASSERT_EQ(providerState.calls.size(), 2U) << "Extended failure must be followed by one regular query.";
  EXPECT_EQ(providerState.calls[0].operation, "query_port_ex")
    << "The extended operation must remain the first port-query attempt.";
  EXPECT_EQ(providerState.calls[1].operation, "query_port")
    << "The regular operation must be the fallback port-query attempt.";
#endif

  ExpectBytesEqualForTest(&providerState.context, &contextBefore, sizeof(contextBefore),
    "Read-only query dispatch must not change the provider context.");
  ExpectBytesEqualForTest(&providerState.queuePair, &queuePairBefore, sizeof(queuePairBefore),
    "Read-only query dispatch must not change the provider queue pair.");
}

/**
 * @brief Verifies protection-domain and memory-region creation, direct registration, and
 * release contracts.
 * @note Side effects: Calls the fake resource provider and writes returned object pointers.
 */
TEST_F(IbvResourceTest, ManagesProtectionDomainsAndMemory) {
  SCOPED_TRACE("Memory wrappers must translate provider results, preserve arguments, and avoid changing input bytes.");

  const struct ibv_context contextBefore = providerState.context;
  const struct ibv_pd protectionDomainBefore = providerState.protectionDomain;
  std::array<unsigned char, 32> buffer = {};
  for (size_t index = 0; index < buffer.size(); ++index) buffer[index] = static_cast<unsigned char>(index + 1);
  const std::array<unsigned char, 32> bufferBefore = buffer;

  struct ibv_pd* protectionDomain = nullptr;
  ExpectPointerDispatchForTest(ibvSymbols.ibv_internal_alloc_pd, AllocateProtectionDomainForTest,
    protectionDomain, &providerState.protectionDomain,
    [&] { return wrap_ibv_alloc_pd(&protectionDomain, &providerState.context); },
    "Protection-domain allocation must translate missing, NULL, and successful provider results.");
  ExpectLastProviderCallForTest("alloc_pd", {PointerValueForTest(&providerState.context)},
    "Protection-domain allocation must forward the exact context.");

  struct ibv_mr* memoryRegion = nullptr;
  ExpectPointerDispatchForTest(ibvSymbols.ibv_internal_reg_mr, RegisterMemoryForTest, memoryRegion,
    &providerState.memoryRegion,
    [&] {
      return wrap_ibv_reg_mr(&memoryRegion, protectionDomain, buffer.data(), buffer.size(),
        IBV_ACCESS_LOCAL_WRITE);
    }, "Memory registration must translate missing, NULL, and successful provider results.");
  ExpectLastProviderCallForTest("reg_mr", {PointerValueForTest(protectionDomain),
    PointerValueForTest(buffer.data()), buffer.size(), IBV_ACCESS_LOCAL_WRITE},
    "Memory registration must preserve the domain, address, size, and access mask.");

  ibvSymbols.ibv_internal_reg_mr = nullptr;
  providerState.calls.clear();
  EXPECT_EQ(wrap_direct_ibv_reg_mr(protectionDomain, buffer.data(), buffer.size(), IBV_ACCESS_REMOTE_READ), nullptr)
    << "Direct registration must return NULL when the mandatory provider symbol is absent.";
  EXPECT_TRUE(providerState.calls.empty()) << "A missing direct-registration operation cannot reach the provider.";
  ibvSymbols.ibv_internal_reg_mr = RegisterMemoryForTest;
  providerState.returnNull = true;
  EXPECT_EQ(wrap_direct_ibv_reg_mr(protectionDomain, buffer.data(), buffer.size(), IBV_ACCESS_REMOTE_READ), nullptr)
    << "Direct registration must preserve the provider's NULL result.";
  EXPECT_EQ(CountProviderCallsForTest("reg_mr"), 1U)
    << "A provider NULL result must still come from exactly one direct registration call.";
  providerState.returnNull = false;
  EXPECT_EQ(wrap_direct_ibv_reg_mr(protectionDomain, buffer.data(), buffer.size(), IBV_ACCESS_REMOTE_READ),
    &providerState.memoryRegion) << "Direct registration must return the provider memory region.";
  ExpectLastProviderCallForTest("reg_mr", {PointerValueForTest(protectionDomain),
    PointerValueForTest(buffer.data()), buffer.size(), IBV_ACCESS_REMOTE_READ},
    "Direct registration must preserve every registration argument.");

  ibvSymbols.ibv_internal_reg_mr_iova2 = nullptr;
  EXPECT_EQ(wrap_ibv_reg_mr_iova2(nullptr, protectionDomain, buffer.data(), buffer.size(), 0x12340000,
    IBV_ACCESS_REMOTE_WRITE), ncclInternalError)
    << "A dummy IOVA registration still requires the provider capability symbol.";
  ibvSymbols.ibv_internal_reg_mr_iova2 = RegisterMemoryIovaForTest;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_reg_mr_iova2(nullptr, protectionDomain, buffer.data(), buffer.size(), 0x12340000,
    IBV_ACCESS_REMOTE_WRITE), ncclSuccess) << "A supported dummy IOVA registration must skip the provider call.";
  EXPECT_TRUE(providerState.calls.empty()) << "A dummy IOVA registration must not call the provider.";
  ExpectPointerDispatchForTest(ibvSymbols.ibv_internal_reg_mr_iova2, RegisterMemoryIovaForTest,
    memoryRegion, &providerState.memoryRegion,
    [&] {
      return wrap_ibv_reg_mr_iova2(&memoryRegion, protectionDomain, buffer.data(), buffer.size(),
        0x12340000, IBV_ACCESS_REMOTE_WRITE);
    }, "IOVA registration must translate missing, NULL, and successful provider results.");
  ExpectLastProviderCallForTest("reg_mr_iova2", {PointerValueForTest(protectionDomain),
    PointerValueForTest(buffer.data()), buffer.size(), 0x12340000, IBV_ACCESS_REMOTE_WRITE},
    "IOVA registration must preserve every registration argument.");

  ExpectPointerDispatchForTest(ibvSymbols.ibv_internal_reg_dmabuf_mr, RegisterDmaBufForTest,
    memoryRegion, &providerState.memoryRegion,
    [&] {
      return wrap_ibv_reg_dmabuf_mr(&memoryRegion, protectionDomain, 4096, buffer.size(),
        0x56780000, 19, IBV_ACCESS_REMOTE_READ);
    }, "DMA-BUF registration must translate missing, NULL, and successful provider results.");
  ExpectLastProviderCallForTest("reg_dmabuf_mr", {PointerValueForTest(protectionDomain), 4096,
    buffer.size(), 0x56780000, 19, IBV_ACCESS_REMOTE_READ},
    "DMA-BUF registration must preserve every registration argument.");

  ibvSymbols.ibv_internal_reg_dmabuf_mr = nullptr;
  providerState.calls.clear();
  errno = 0;
  EXPECT_EQ(wrap_direct_ibv_reg_dmabuf_mr(protectionDomain, 8192, buffer.size(), 0x9abc0000, 23,
    IBV_ACCESS_LOCAL_WRITE), nullptr) << "Unsupported direct DMA-BUF registration must return NULL.";
  EXPECT_EQ(errno, EOPNOTSUPP) << "Unsupported direct DMA-BUF registration must expose EOPNOTSUPP.";
  EXPECT_TRUE(providerState.calls.empty()) << "An unsupported direct DMA-BUF operation cannot reach the provider.";
  ibvSymbols.ibv_internal_reg_dmabuf_mr = RegisterDmaBufForTest;
  providerState.returnNull = true;
  EXPECT_EQ(wrap_direct_ibv_reg_dmabuf_mr(protectionDomain, 8192, buffer.size(), 0x9abc0000, 23,
    IBV_ACCESS_LOCAL_WRITE), nullptr) << "Direct DMA-BUF registration must preserve a provider NULL result.";
  EXPECT_EQ(CountProviderCallsForTest("reg_dmabuf_mr"), 1U)
    << "A provider NULL result must still come from exactly one direct DMA-BUF call.";
  providerState.returnNull = false;
  EXPECT_EQ(wrap_direct_ibv_reg_dmabuf_mr(protectionDomain, 8192, buffer.size(), 0x9abc0000, 23,
    IBV_ACCESS_LOCAL_WRITE), &providerState.memoryRegion)
    << "Supported direct DMA-BUF registration must return the provider memory region.";
  ExpectLastProviderCallForTest("reg_dmabuf_mr", {PointerValueForTest(protectionDomain), 8192,
    buffer.size(), 0x9abc0000, 23, IBV_ACCESS_LOCAL_WRITE},
    "Direct DMA-BUF registration must preserve every registration argument.");

  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_dereg_mr, DeregisterMemoryForTest, EIO,
    [&] { return wrap_ibv_dereg_mr(memoryRegion); },
    "Memory deregistration must translate missing, failing, and successful provider results.");
  ExpectLastProviderCallForTest("dereg_mr", {PointerValueForTest(memoryRegion)},
    "Memory deregistration must forward the exact memory region.");

  ExpectBytesEqualForTest(buffer.data(), bufferBefore.data(), buffer.size(),
    "Registration wrappers and the configured provider must not modify registered bytes.");
  ExpectBytesEqualForTest(&providerState.context, &contextBefore, sizeof(contextBefore),
    "Resource wrappers and the configured provider must not modify the input context.");
  ExpectBytesEqualForTest(&providerState.protectionDomain, &protectionDomainBefore,
    sizeof(protectionDomainBefore), "Resource wrappers and the configured provider must not modify the input domain.");

  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_dealloc_pd, DeallocateProtectionDomainForTest, EIO,
    [&] { return wrap_ibv_dealloc_pd(protectionDomain); },
    "Protection-domain release must translate missing, failing, and successful provider results.");
  ExpectLastProviderCallForTest("dealloc_pd", {PointerValueForTest(protectionDomain)},
    "Protection-domain release must forward the exact domain after all registrations are released.");
}

/**
 * @brief Verifies completion-queue and queue-pair construction and destruction contracts.
 * @note Side effects: Calls the fake resource provider and writes returned object pointers.
 */
TEST_F(IbvResourceTest, ManagesQueues) {
  SCOPED_TRACE("Queue wrappers must translate provider results and preserve every construction argument.");

  struct ibv_comp_channel channel = {};
  int queueContext = 37;
  const struct ibv_context contextBefore = providerState.context;
  const struct ibv_pd protectionDomainBefore = providerState.protectionDomain;
  const struct ibv_comp_channel channelBefore = channel;
  const int queueContextBefore = queueContext;
  struct ibv_cq* completionQueue = nullptr;
  ExpectPointerDispatchForTest(ibvSymbols.ibv_internal_create_cq, CreateCompletionQueueForTest,
    completionQueue, &providerState.completionQueue,
    [&] {
      return wrap_ibv_create_cq(&completionQueue, &providerState.context, 127, &queueContext, &channel, 3);
    }, "Completion-queue creation must translate missing, NULL, and successful provider results.");
  ExpectLastProviderCallForTest("create_cq", {PointerValueForTest(&providerState.context), 127,
    PointerValueForTest(&queueContext), PointerValueForTest(&channel), 3},
    "Completion-queue creation must preserve context, size, owner, channel, and vector.");

  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_destroy_cq, DestroyCompletionQueueForTest, EIO,
    [&] { return wrap_ibv_destroy_cq(completionQueue); },
    "Completion-queue destruction must translate missing, failing, and successful provider results.");
  ExpectLastProviderCallForTest("destroy_cq", {PointerValueForTest(completionQueue)},
    "Completion-queue destruction must forward the exact queue.");

  struct ibv_qp_init_attr initialAttributes = {};
  initialAttributes.cap.max_send_wr = 61;
  initialAttributes.cap.max_recv_wr = 67;
  const struct ibv_qp_init_attr initialAttributesBefore = initialAttributes;
  struct ibv_qp* queuePair = nullptr;
  ExpectPointerDispatchForTest(ibvSymbols.ibv_internal_create_qp, CreateQueuePairForTest, queuePair,
    &providerState.queuePair,
    [&] { return wrap_ibv_create_qp(&queuePair, &providerState.protectionDomain, &initialAttributes); },
    "Queue-pair creation must translate missing, NULL, and successful provider results.");
  ExpectLastProviderCallForTest("create_qp",
    {PointerValueForTest(&providerState.protectionDomain), PointerValueForTest(&initialAttributes)},
    "Queue-pair creation must preserve the protection domain and initialization storage.");
  ExpectBytesEqualForTest(&initialAttributes, &initialAttributesBefore, sizeof(initialAttributes),
    "Queue-pair creation must not change initialization data through the non-mutating provider.");

  ExpectIntegerDispatchForTest(ibvSymbols.ibv_internal_destroy_qp, DestroyQueuePairForTest, EIO,
    [&] { return wrap_ibv_destroy_qp(queuePair); },
    "Queue-pair destruction must translate missing, failing, and successful provider results.");
  ExpectLastProviderCallForTest("destroy_qp", {PointerValueForTest(queuePair)},
    "Queue-pair destruction must forward the exact queue pair.");
  ExpectBytesEqualForTest(&providerState.context, &contextBefore, sizeof(contextBefore),
    "Queue creation and destruction must not change the provider context through the fake provider.");
  ExpectBytesEqualForTest(&providerState.protectionDomain, &protectionDomainBefore,
    sizeof(protectionDomainBefore), "Queue-pair creation must not change the input protection domain.");
  ExpectBytesEqualForTest(&channel, &channelBefore, sizeof(channelBefore),
    "Completion-queue creation must not change the input completion channel.");
  EXPECT_EQ(queueContext, queueContextBefore)
    << "Completion-queue creation must not change the caller-owned queue context value.";
}

/**
 * @brief Verifies queue-pair state formatting and diagnostic data selection.
 * @note Side effects: Calls the fake query provider and writes local diagnostic strings.
 */
TEST_F(IbvQueuePairTest, FormatsDiagnostics) {
  SCOPED_TRACE("QP diagnostics must name every state and report queried or caller-supplied address data.");

  /** @brief Maps one verbs QP state to its expected diagnostic text. */
  struct StateText {
    enum ibv_qp_state state;
    const char* text;
  };
  const StateText states[] = {{IBV_QPS_RESET, "RESET"}, {IBV_QPS_INIT, "INIT"}, {IBV_QPS_RTR, "RTR"},
    {IBV_QPS_RTS, "RTS"}, {IBV_QPS_SQD, "SQD"}, {IBV_QPS_SQE, "SQE"}, {IBV_QPS_ERR, "ERR"},
    {IBV_QPS_UNKNOWN, "UNKNOWN"}};
  for (const StateText& state : states) {
    char text[32] = {};
    ibvQpStateName(state.state, text, sizeof(text));
    EXPECT_STREQ(text, state.text) << "Each public verbs QP state must have its exact diagnostic name.";
  }
  char unknownText[32] = {};
  ibvQpStateName(static_cast<enum ibv_qp_state>(99), unknownText, sizeof(unknownText));
  EXPECT_STREQ(unknownText, "NOT RECOGNIZED (99)") << "Unknown QP states must retain their numeric value.";
  char zeroLengthText = 'x';
  ibvQpStateName(IBV_QPS_RESET, &zeroLengthText, 0);
  EXPECT_EQ(zeroLengthText, 'x') << "A zero-length destination must not be written.";
  std::array<char, 5> shortText = {'x', 'x', 'x', 'x', 'x'};
  ibvQpStateName(IBV_QPS_RESET, shortText.data(), 4);
  EXPECT_STREQ(shortText.data(), "RES") << "A short destination must contain a terminated prefix.";
  EXPECT_EQ(shortText[4], 'x') << "QP-state formatting must not write beyond the declared destination length.";

  providerState.queuePair.state = IBV_QPS_RESET;
  providerState.queuePairAttributes.port_num = 4;
  providerState.queuePairAttributes.ah_attr.is_global = 1;
  providerState.queuePairAttributes.ah_attr.grh.sgid_index = 5;
  providerState.queuePairAttributes.ah_attr.grh.dgid.raw[15] = 9;
  providerState.gid.raw[15] = 3;
  providerState.calls.clear();
  char message[256] = {};
  struct ibv_qp_attr userAttributes = {};
  const struct ibv_qp queuePairBeforeDiagnostics = providerState.queuePair;
  const struct ibv_qp_attr initialUserAttributes = userAttributes;
  ibvModifyQpLog(&providerState.queuePair, IBV_QPS_RTS, &userAttributes, 0, message, sizeof(message));
  EXPECT_STREQ(message, "on dev provider-device:4, curr state RESET, next state RTS, local GID index 5, "
    "local GID ::3, remote GID ::9")
    << "Successful queries must populate device, state, port, and both global identifiers.";
  EXPECT_EQ(CountProviderCallsForTest("query_qp"), 1U) << "Diagnostics must query current QP attributes once.";
  EXPECT_EQ(CountProviderCallsForTest("query_gid"), 1U) << "Global diagnostics must query the local GID once.";

  providerState.integerResult = EIO;
  providerState.calls.clear();
  ibvModifyQpLog(&providerState.queuePair, IBV_QPS_ERR, &userAttributes, 0, message, sizeof(message));
  EXPECT_STREQ(message, "on dev provider-device:-1, curr state RESET, next state ERR, local GID index -1, "
    "local GID N/A, remote GID N/A")
    << "A failed QP query must produce safe unavailable markers rather than uninitialized fields.";

  providerState.integerResult = 0;
  providerState.queuePairAttributes.ah_attr.is_global = 1;
  ibvSymbols.ibv_internal_query_gid = nullptr;
  providerState.calls.clear();
  ibvModifyQpLog(&providerState.queuePair, IBV_QPS_RTR, &userAttributes, 0, message, sizeof(message));
  EXPECT_STREQ(message, "on dev provider-device:4, curr state RESET, next state RTR, local GID index 5, "
    "local GID N/A, remote GID ::9")
    << "A failed local-GID query must preserve remote data and mark only the local GID unavailable.";
  EXPECT_EQ(CountProviderCallsForTest("query_gid"), 0U)
    << "A missing GID symbol must be rejected before a provider call is recorded.";
  ibvSymbols.ibv_internal_query_gid = QueryGidForTest;
  ExpectBytesEqualForTest(&userAttributes, &initialUserAttributes, sizeof(initialUserAttributes),
    "Queried and failed diagnostic paths must not change caller-supplied attributes.");

  userAttributes.port_num = 8;
  userAttributes.ah_attr.is_global = 0;
  const struct ibv_qp_attr flaggedUserAttributes = userAttributes;
  providerState.integerResult = EIO;
  providerState.calls.clear();
  ibvModifyQpLog(&providerState.queuePair, IBV_QPS_INIT, &userAttributes,
    IBV_QP_PORT | IBV_QP_AV, message, sizeof(message));
  EXPECT_STREQ(message, "on dev provider-device:8, curr state RESET, next state INIT, local GID index -1, "
    "local GID N/A, remote GID N/A")
    << "Caller-supplied flagged attributes must override a failed current-attribute query.";
  EXPECT_EQ(CountProviderCallsForTest("query_gid"), 0U)
    << "A non-global caller address must not trigger a local GID query.";
  ExpectBytesEqualForTest(&userAttributes, &flaggedUserAttributes, sizeof(flaggedUserAttributes),
    "Caller-supplied diagnostic attributes must remain read-only.");

  providerState.integerResult = 0;
  providerState.calls.clear();
  ibvModifyQpLog(&providerState.queuePair, IBV_QPS_RTR, &userAttributes,
    IBV_QP_PORT, message, sizeof(message));
  EXPECT_STREQ(message, "on dev provider-device:8, curr state RESET, next state RTR, local GID index 5, "
    "local GID ::3, remote GID ::9")
    << "A caller port override must combine with queried address data and select the matching local GID.";
  ASSERT_EQ(CountProviderCallsForTest("query_gid"), 1U)
    << "A queried global address must request exactly one local GID.";
  const ProviderCall& mixedGidCall = LastProviderCallForTest();
  EXPECT_EQ(mixedGidCall.operation, "query_gid")
    << "The final provider call for a global address must be the local-GID query.";
  ASSERT_EQ(mixedGidCall.argumentCount, 4U)
    << "The local-GID query must receive context, port, index, and output storage.";
  EXPECT_EQ(mixedGidCall.arguments[1], 8U)
    << "The local-GID query must use the caller-overridden port number.";

  struct ibv_context* context = providerState.protectionDomain.context;
  providerState.protectionDomain.context = nullptr;
  providerState.integerResult = 0;
  providerState.queuePairAttributes.ah_attr.is_global = 1;
  providerState.calls.clear();
  const struct ibv_qp_attr userAttributesBefore = userAttributes;
  ibvModifyQpLog(&providerState.queuePair, IBV_QPS_RTS, &userAttributes, 0, message, sizeof(message));
  EXPECT_STREQ(message, "on dev N/A:4, curr state RESET, next state RTS, local GID index -1, "
    "local GID N/A, remote GID ::9")
    << "A QP without a provider context must still report queried port and remote-GID data safely.";
  EXPECT_EQ(CountProviderCallsForTest("get_device_name"), 0U)
    << "Diagnostics cannot request a device name when the protection domain has no context.";
  EXPECT_EQ(CountProviderCallsForTest("query_qp"), 1U)
    << "Context-free diagnostics must still query the current QP attributes once.";
  EXPECT_EQ(CountProviderCallsForTest("query_gid"), 0U)
    << "Diagnostics cannot query a local GID when the protection domain has no context.";
  ExpectBytesEqualForTest(&providerState.queuePair, &queuePairBeforeDiagnostics,
    sizeof(queuePairBeforeDiagnostics),
    "Diagnostic formatting and the fake provider must not change the queue pair.");
  ExpectBytesEqualForTest(&userAttributes, &userAttributesBefore, sizeof(userAttributesBefore),
    "Diagnostic formatting must not change caller-supplied attributes.");
  providerState.protectionDomain.context = context;
}

/**
 * @brief Verifies immediate QP modification, selective retries, retry-all mode, and retry
 * exhaustion.
 * @note Side effects: Calls the fake QP modification and diagnostic query providers.
 */
TEST_F(IbvQueuePairTest, ModifiesWithRetries) {
  SCOPED_TRACE("QP modification must retry configured errors, stop on success, and preserve caller attributes.");

  providerState.retryAll = 1;
  providerState.retryCount = 7;
  providerState.retryTimeout = 13;
  providerState.parameterLoads.clear();
  EXPECT_EQ(ncclParamIbMQpRetryAll(), 1)
    << "The retry-all accessor must return the value supplied by its parameter dependency.";
  EXPECT_EQ(ncclParamIbMQpRetryCnt(), 7)
    << "The retry-count accessor must return the value supplied by its parameter dependency.";
  EXPECT_EQ(ncclParamIbMQpRetryTimeout(), 13)
    << "The retry-timeout accessor must return the value supplied by its parameter dependency.";
  ASSERT_EQ(providerState.parameterLoads.size(), 3U)
    << "Each retry accessor must request exactly one production parameter.";
  EXPECT_EQ(providerState.parameterLoads[0].environment, "NCCL_IB_MQP_RETRY_ALL")
    << "The retry-all accessor must request its documented environment variable.";
  EXPECT_EQ(providerState.parameterLoads[0].defaultValue, 0)
    << "The retry-all accessor must retain its disabled-by-default contract.";
  EXPECT_EQ(providerState.parameterLoads[1].environment, "NCCL_IB_MQP_RETRY_CNT")
    << "The retry-count accessor must request its documented environment variable.";
  EXPECT_EQ(providerState.parameterLoads[1].defaultValue, 34)
    << "The retry-count accessor must retain its production default.";
  EXPECT_EQ(providerState.parameterLoads[2].environment, "NCCL_IB_MQP_RETRY_SLEEP_MSEC")
    << "The retry-timeout accessor must request its documented environment variable.";
  EXPECT_EQ(providerState.parameterLoads[2].defaultValue, 100)
    << "The retry-timeout accessor must retain its production millisecond default.";
  providerState.retryAll = 0;
  providerState.retryCount = 2;
  providerState.retryTimeout = 0;

  struct ibv_qp_attr attributes = {};
  attributes.qp_state = IBV_QPS_INIT;
  attributes.port_num = 7;
  const struct ibv_qp_attr attributesBefore = attributes;
  /**
   * @brief Verifies the count and arguments of recorded QP modification calls.
   * @param[in] expectedCount Required number of matching provider calls.
   * @param[in] reason Valid non-NULL trace context.
   * @note Side effects: Records GoogleTest failures for an incorrect count or forwarded argument.
   */
  const auto expectModifyCalls = [&](size_t expectedCount, const char* reason) {
    SCOPED_TRACE(reason);
    size_t observedCount = 0;
    for (const ProviderCall& call : providerState.calls) {
      if (call.operation != "modify_qp") continue;
      ASSERT_EQ(call.argumentCount, 3U) << "Each QP modification call must receive all three arguments.";
      EXPECT_EQ(call.arguments[0], PointerValueForTest(&providerState.queuePair))
        << "Every QP modification attempt must receive the original queue pair.";
      EXPECT_EQ(call.arguments[1], PointerValueForTest(&attributes))
        << "Every QP modification attempt must receive the original attribute storage.";
      EXPECT_EQ(call.arguments[2], static_cast<uintptr_t>(IBV_QP_STATE))
        << "Every QP modification attempt must receive the original attribute mask.";
      ++observedCount;
    }
    EXPECT_EQ(observedCount, expectedCount) << "QP modification used an unexpected number of provider attempts.";
  };

  ibvSymbols.ibv_internal_modify_qp = nullptr;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_modify_qp(&providerState.queuePair, &attributes, IBV_QP_STATE), ncclInternalError)
    << "A missing mandatory modification symbol must return ncclInternalError.";
  EXPECT_TRUE(providerState.calls.empty()) << "A missing QP modification operation cannot reach the provider.";
  ibvSymbols.ibv_internal_modify_qp = ModifyQueuePairForTest;

  providerState.integerResult = 0;
  providerState.modifyResults.clear();
  providerState.nextModifyResult = 0;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_modify_qp(&providerState.queuePair, &attributes, IBV_QP_STATE), ncclSuccess)
    << "An immediately successful modification must return ncclSuccess.";
  expectModifyCalls(1, "Immediate success must use one exact provider call.");
  EXPECT_EQ(CountProviderCallsForTest("query_qp"), 0U)
    << "Immediate success must not produce failure diagnostics.";
  ExpectLastProviderCallForTest("modify_qp", {PointerValueForTest(&providerState.queuePair),
    PointerValueForTest(&attributes), IBV_QP_STATE},
    "QP modification must forward the exact queue pair, attributes, and mask.");

  providerState.modifyResults = {-ETIMEDOUT, 0};
  providerState.nextModifyResult = 0;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_modify_qp(&providerState.queuePair, &attributes, IBV_QP_STATE), ncclSuccess)
    << "A timeout followed by success must return ncclSuccess.";
  expectModifyCalls(2, "A signed timeout must repeat the exact provider call once before success.");
  EXPECT_EQ(CountProviderCallsForTest("query_qp"), 1U)
    << "One recovered timeout must produce exactly one pre-retry diagnostic query.";

  providerState.modifyResults = {EINVAL, 0};
  providerState.nextModifyResult = 0;
  providerState.retryAll = 0;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_modify_qp(&providerState.queuePair, &attributes, IBV_QP_STATE), ncclSystemError)
    << "A non-timeout provider error must fail when retry-all mode is disabled.";
  expectModifyCalls(1, "A non-retryable error must stop after the first exact provider call.");
  EXPECT_EQ(CountProviderCallsForTest("query_qp"), 1U)
    << "A terminal non-retryable error must produce exactly one final diagnostic query.";

  providerState.modifyResults = {EINVAL, 0};
  providerState.nextModifyResult = 0;
  providerState.retryAll = 1;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_modify_qp(&providerState.queuePair, &attributes, IBV_QP_STATE), ncclSuccess)
    << "Retry-all mode must retry a general provider error and return the later success.";
  expectModifyCalls(2, "Retry-all mode must repeat the exact provider call once.");
  EXPECT_EQ(CountProviderCallsForTest("query_qp"), 1U)
    << "A recovered retry-all error must produce exactly one pre-retry diagnostic query.";

  providerState.modifyResults = {ETIMEDOUT, 0};
  providerState.nextModifyResult = 0;
  providerState.retryAll = 0;
  providerState.retryCount = 0;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_modify_qp(&providerState.queuePair, &attributes, IBV_QP_STATE), ncclSystemError)
    << "A zero retry budget must return the first timeout as ncclSystemError.";
  expectModifyCalls(1, "Zero configured retries must permit only the initial exact provider call.");
  EXPECT_EQ(CountProviderCallsForTest("query_qp"), 1U)
    << "An unretried timeout must produce exactly one final diagnostic query.";

  providerState.modifyResults = {ETIMEDOUT, ETIMEDOUT, ETIMEDOUT};
  providerState.nextModifyResult = 0;
  providerState.retryAll = 0;
  providerState.retryCount = 2;
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_modify_qp(&providerState.queuePair, &attributes, IBV_QP_STATE), ncclSystemError)
    << "Exhausting the configured timeout retries must return ncclSystemError.";
  expectModifyCalls(3, "Two configured retries must repeat the exact provider call three times in total.");
  EXPECT_EQ(CountProviderCallsForTest("query_qp"), 3U)
    << "Retry exhaustion must diagnose both retries and the final failure exactly once each.";
  ASSERT_FALSE(providerState.calls.empty()) << "Retry exhaustion must emit final QP diagnostics.";
  const ProviderCall& finalCall = LastProviderCallForTest();
  EXPECT_EQ(finalCall.operation, "query_qp") << "Final failure diagnostics must end with a current-QP query.";
  ASSERT_EQ(finalCall.argumentCount, 4U) << "The diagnostic QP query must receive all four provider arguments.";
  EXPECT_EQ(finalCall.arguments[0], PointerValueForTest(&providerState.queuePair))
    << "Final failure diagnostics must query the modified queue pair.";
  EXPECT_NE(finalCall.arguments[1], 0U) << "Final failure diagnostics must provide attribute output storage.";
  EXPECT_EQ(finalCall.arguments[2], static_cast<uintptr_t>(IBV_QP_PORT | IBV_QP_AV))
    << "Final failure diagnostics must request port and address-vector attributes.";
  EXPECT_NE(finalCall.arguments[3], 0U) << "Final failure diagnostics must provide initialization output storage.";

  ExpectBytesEqualForTest(&attributes, &attributesBefore, sizeof(attributes),
    "QP modification and the configured provider must not change caller attributes.");
}

/**
 * @brief Verifies optional ECE outcomes, provider event text, and completion-status,
 * completion-opcode, and request-opcode names.
 * @note Side effects: Calls the fake optional and event-name providers.
 */
TEST_F(IbvQueuePairTest, HandlesOptionalFeaturesAndText) {
  SCOPED_TRACE(
      "ECE handling must expose query output and support status while preserving configuration "
      "input; enum text must cover every defined mapping.");

  struct ibv_ece ece = {};
  ece.vendor_id = 0x10203;
  ece.options = 0x40506;
  ece.comp_mask = 0x70809;
  const struct ibv_ece eceBefore = ece;
  ExpectOptionalEceDispatchForTest(ibvSymbols.ibv_internal_query_ece, QueryEceForTest,
    [&](int* supported) { return wrap_ibv_query_ece(&providerState.queuePair, &ece, supported); },
    "ECE query must distinguish absent, unsupported, failed, and successful provider operations.", [&] {
      ExpectBytesEqualForTest(&ece, &eceBefore, sizeof(ece),
        "An absent optional ECE query must leave output storage unchanged.");
    });
  ExpectLastProviderCallForTest("query_ece",
    {PointerValueForTest(&providerState.queuePair), PointerValueForTest(&ece)},
    "ECE query must preserve the queue pair and ECE output storage.");
  ExpectBytesEqualForTest(&ece, &providerState.eceResult, sizeof(ece),
    "A successful ECE query must expose every provider output field.");
  ece = eceBefore;
  ExpectOptionalEceDispatchForTest(ibvSymbols.ibv_internal_set_ece, SetEceForTest,
    [&](int* supported) { return wrap_ibv_set_ece(&providerState.queuePair, &ece, supported); },
    "ECE configuration must distinguish absent, unsupported, failed, and successful provider operations.");
  ExpectLastProviderCallForTest("set_ece",
    {PointerValueForTest(&providerState.queuePair), PointerValueForTest(&ece)},
    "ECE configuration must preserve the queue pair and ECE input storage.");
  ExpectBytesEqualForTest(&ece, &eceBefore, sizeof(ece),
    "The wrappers and configured fake provider must not modify ECE storage.");

  char* eventName = reinterpret_cast<char*>(static_cast<uintptr_t>(1));
  providerState.calls.clear();
  EXPECT_EQ(wrap_ibv_event_type_str(&eventName, IBV_EVENT_PORT_ACTIVE), ncclSuccess)
    << "Event-name lookup must return ncclSuccess.";
  EXPECT_EQ(eventName, providerState.eventName) << "Event-name lookup must return the provider string pointer.";
  ExpectLastProviderCallForTest("event_type_str", {IBV_EVENT_PORT_ACTIVE},
    "Event-name lookup must preserve the event value.");

  /** @brief Maps one completion status to its expected symbolic text. */
  struct CompletionStatusText {
    enum ibv_wc_status value;
    const char* text;
  };
  const CompletionStatusText completionStatuses[] = {
    {IBV_WC_SUCCESS, "IBV_WC_SUCCESS"}, {IBV_WC_LOC_LEN_ERR, "IBV_WC_LOC_LEN_ERR"},
    {IBV_WC_LOC_QP_OP_ERR, "IBV_WC_LOC_QP_OP_ERR"}, {IBV_WC_LOC_EEC_OP_ERR, "IBV_WC_LOC_EEC_OP_ERR"},
    {IBV_WC_LOC_PROT_ERR, "IBV_WC_LOC_PROT_ERR"}, {IBV_WC_WR_FLUSH_ERR, "IBV_WC_WR_FLUSH_ERR"},
    {IBV_WC_MW_BIND_ERR, "IBV_WC_MW_BIND_ERR"}, {IBV_WC_BAD_RESP_ERR, "IBV_WC_BAD_RESP_ERR"},
    {IBV_WC_LOC_ACCESS_ERR, "IBV_WC_LOC_ACCESS_ERR"}, {IBV_WC_REM_INV_REQ_ERR, "IBV_WC_REM_INV_REQ_ERR"},
    {IBV_WC_REM_ACCESS_ERR, "IBV_WC_REM_ACCESS_ERR"}, {IBV_WC_REM_OP_ERR, "IBV_WC_REM_OP_ERR"},
    {IBV_WC_RETRY_EXC_ERR, "IBV_WC_RETRY_EXC_ERR"}, {IBV_WC_RNR_RETRY_EXC_ERR, "IBV_WC_RNR_RETRY_EXC_ERR"},
    {IBV_WC_LOC_RDD_VIOL_ERR, "IBV_WC_LOC_RDD_VIOL_ERR"},
    {IBV_WC_REM_INV_RD_REQ_ERR, "IBV_WC_REM_INV_RD_REQ_ERR"},
    {IBV_WC_REM_ABORT_ERR, "IBV_WC_REM_ABORT_ERR"}, {IBV_WC_INV_EECN_ERR, "IBV_WC_INV_EECN_ERR"},
    {IBV_WC_INV_EEC_STATE_ERR, "IBV_WC_INV_EEC_STATE_ERR"}, {IBV_WC_FATAL_ERR, "IBV_WC_FATAL_ERR"},
    {IBV_WC_RESP_TIMEOUT_ERR, "IBV_WC_RESP_TIMEOUT_ERR"}, {IBV_WC_GENERAL_ERR, "IBV_WC_GENERAL_ERR"}};
  for (const CompletionStatusText& item : completionStatuses) {
    EXPECT_STREQ(ibvWcStatusStr(item.value), item.text)
      << "Every supported completion status must have its exact symbolic name; value=" << item.value;
  }
  EXPECT_STREQ(ibvWcStatusStr(static_cast<enum ibv_wc_status>(999)), "UNKNOWN_STATUS")
    << "Unknown completion statuses must use the documented fallback name.";

  /** @brief Maps one completion opcode to its expected symbolic text. */
  struct CompletionOpcodeText {
    enum ibv_wc_opcode value;
    const char* text;
  };
  const CompletionOpcodeText completionOpcodes[] = {{IBV_WC_SEND, "IBV_WC_SEND"},
    {IBV_WC_RDMA_WRITE, "IBV_WC_RDMA_WRITE"}, {IBV_WC_RDMA_READ, "IBV_WC_RDMA_READ"},
    {IBV_WC_COMP_SWAP, "IBV_WC_COMP_SWAP"}, {IBV_WC_FETCH_ADD, "IBV_WC_FETCH_ADD"},
    {IBV_WC_BIND_MW, "IBV_WC_BIND_MW"}, {IBV_WC_RECV, "IBV_WC_RECV"},
    {IBV_WC_RECV_RDMA_WITH_IMM, "IBV_WC_RECV_RDMA_WITH_IMM"}};
  for (const CompletionOpcodeText& item : completionOpcodes) {
    EXPECT_STREQ(ibvWcOpcodeStr(item.value), item.text)
      << "Every supported completion opcode must have its exact symbolic name; value=" << item.value;
  }
  EXPECT_STREQ(ibvWcOpcodeStr(static_cast<enum ibv_wc_opcode>(999)), "UNKNOWN_OPCODE")
    << "Unknown completion opcodes must use the documented fallback name.";

  /** @brief Maps one work-request opcode to its expected symbolic text. */
  struct RequestOpcodeText {
    enum ibv_wr_opcode value;
    const char* text;
  };
  const RequestOpcodeText requestOpcodes[] = {{IBV_WR_RDMA_WRITE, "IBV_WR_RDMA_WRITE"},
    {IBV_WR_RDMA_WRITE_WITH_IMM, "IBV_WR_RDMA_WRITE_WITH_IMM"}, {IBV_WR_SEND, "IBV_WR_SEND"},
    {IBV_WR_SEND_WITH_IMM, "IBV_WR_SEND_WITH_IMM"}, {IBV_WR_RDMA_READ, "IBV_WR_RDMA_READ"},
    {IBV_WR_ATOMIC_CMP_AND_SWP, "IBV_WR_ATOMIC_CMP_AND_SWP"},
    {IBV_WR_ATOMIC_FETCH_AND_ADD, "IBV_WR_ATOMIC_FETCH_AND_ADD"}};
  for (const RequestOpcodeText& item : requestOpcodes) {
    EXPECT_STREQ(ibvWrOpcodeStr(item.value), item.text)
      << "Every supported work-request opcode must have its exact symbolic name; value=" << item.value;
  }
  EXPECT_STREQ(ibvWrOpcodeStr(static_cast<enum ibv_wr_opcode>(999)), "UNKNOWN_OPCODE")
    << "Unknown work-request opcodes must use the documented fallback name.";
}

/**
 * @brief Builds a complete fake verbs symbol table.
 * @return A symbol table whose entries call the local fake provider.
 * @note Side effects: None.
 */
struct ncclIbvSymbols MakeSymbolTableForTest() {
  struct ncclIbvSymbols symbols = {};
  symbols.ibv_internal_fork_init = ForkInitForTest;
  symbols.ibv_internal_get_device_list = GetDeviceListForTest;
  symbols.ibv_internal_free_device_list = FreeDeviceListForTest;
  symbols.ibv_internal_get_device_name = GetDeviceNameForTest;
  symbols.ibv_internal_open_device = OpenDeviceForTest;
  symbols.ibv_internal_close_device = CloseDeviceForTest;
  symbols.ibv_internal_get_async_event = GetAsyncEventForTest;
  symbols.ibv_internal_ack_async_event = AckAsyncEventForTest;
  symbols.ibv_internal_query_device = QueryDeviceForTest;
  symbols.ibv_internal_query_port = QueryPortForTest;
  symbols.ibv_internal_query_gid = QueryGidForTest;
  symbols.ibv_internal_query_qp = QueryQueuePairForTest;
  symbols.ibv_internal_alloc_pd = AllocateProtectionDomainForTest;
  symbols.ibv_internal_dealloc_pd = DeallocateProtectionDomainForTest;
  symbols.ibv_internal_reg_mr = RegisterMemoryForTest;
  symbols.ibv_internal_reg_mr_iova2 = RegisterMemoryIovaForTest;
  symbols.ibv_internal_reg_dmabuf_mr = RegisterDmaBufForTest;
  symbols.ibv_internal_dereg_mr = DeregisterMemoryForTest;
  symbols.ibv_internal_create_cq = CreateCompletionQueueForTest;
  symbols.ibv_internal_destroy_cq = DestroyCompletionQueueForTest;
  symbols.ibv_internal_create_qp = CreateQueuePairForTest;
  symbols.ibv_internal_modify_qp = ModifyQueuePairForTest;
  symbols.ibv_internal_destroy_qp = DestroyQueuePairForTest;
  symbols.ibv_internal_event_type_str = EventTypeStringForTest;
  symbols.ibv_internal_query_ece = QueryEceForTest;
  symbols.ibv_internal_set_ece = SetEceForTest;
  return symbols;
}

/**
 * @brief Verifies the final provider call name and every recorded argument.
 * @param[in] operation Valid non-NULL expected operation name.
 * @param[in] arguments Zero to seven expected pointer-sized argument values.
 * @param[in] reason Valid non-NULL contract explanation printed on failure.
 * @note Side effects: Records GoogleTest failures when the call differs.
 */
void ExpectLastProviderCallForTest(const char* operation, std::initializer_list<uintptr_t> arguments,
  const char* reason) {
  SCOPED_TRACE(reason);
  ASSERT_FALSE(providerState.calls.empty()) << "The expected provider call was not recorded.";
  const ProviderCall& call = LastProviderCallForTest();
  EXPECT_EQ(call.operation, operation) << "The final provider operation has the wrong name.";
  ASSERT_EQ(call.argumentCount, arguments.size()) << "The provider operation received the wrong argument count.";
  size_t index = 0;
  for (uintptr_t argument : arguments) {
    EXPECT_EQ(call.arguments[index], argument) << "Provider argument " << index << " was not forwarded unchanged.";
    ++index;
  }
}

/**
 * @brief Verifies that two valid byte ranges are identical.
 * @param[in] actual Valid non-NULL observed storage of size bytes.
 * @param[in] expected Valid non-NULL expected storage of size bytes.
 * @param[in] size Compared byte count in [1, SIZE_MAX].
 * @param[in] reason Valid non-NULL contract explanation printed on failure.
 * @note Side effects: Records a GoogleTest failure when the ranges differ.
 */
void ExpectBytesEqualForTest(const void* actual, const void* expected, size_t size, const char* reason) {
  SCOPED_TRACE(reason);
  EXPECT_EQ(std::memcmp(actual, expected, size), 0) << "The compared byte ranges differ.";
}

/**
 * @brief Checks missing-symbol, provider-error, and successful integer-return dispatch.
 * @tparam Symbol Provider function-pointer type stored in slot.
 * @tparam Call Callable type that invokes the wrapper and returns ncclResult_t.
 * @param[in,out] slot Function-pointer slot temporarily cleared and restored to implementation.
 * @param[in] implementation Valid non-NULL fake provider function.
 * @param[in] failure Provider result that the wrapper must classify as ncclSystemError.
 * @param[in] call Callable invoking the wrapper with valid arguments.
 * @param[in] reason Valid non-NULL contract explanation printed on failure.
 * @param[in] checkMissingOutput Optional callback verifying outputs after missing-symbol rejection.
 * @note Side effects: Calls the wrapper three times and updates providerState result and call
 * records.
 */
template <typename Symbol, typename Call>
void ExpectIntegerDispatchForTest(Symbol& slot, Symbol implementation, int failure, Call call, const char* reason,
  const std::function<void()>& checkMissingOutput) {
  SCOPED_TRACE(reason);
  providerState.calls.clear();
  slot = nullptr;
  EXPECT_EQ(call(), ncclInternalError) << "A missing mandatory provider symbol must be rejected.";
  if (checkMissingOutput) checkMissingOutput();
  EXPECT_TRUE(providerState.calls.empty()) << "A missing provider symbol cannot dispatch a provider call.";
  slot = implementation;
  providerState.integerResult = failure;
  EXPECT_EQ(call(), ncclSystemError) << "A provider failure must be translated to ncclSystemError.";
  EXPECT_EQ(providerState.calls.size(), 1U) << "A provider failure must dispatch exactly one call.";
  providerState.integerResult = 0;
  EXPECT_EQ(call(), ncclSuccess) << "A successful provider operation must return ncclSuccess.";
  EXPECT_EQ(providerState.calls.size(), 2U) << "The failure and success paths must dispatch one call each.";
}

/**
 * @brief Checks missing-symbol, NULL-provider-result, and successful pointer-return dispatch.
 * @tparam Symbol Provider function-pointer type stored in slot.
 * @tparam Output Pointer-output type under test.
 * @tparam Call Callable type that invokes the wrapper and returns ncclResult_t.
 * @param[in,out] slot Function-pointer slot temporarily cleared and restored to implementation.
 * @param[in] implementation Valid non-NULL fake provider function.
 * @param[in,out] output Pointer output checked for preservation, NULL, and the provider result.
 * @param[in] successfulOutput Valid non-NULL pointer expected from the successful provider call.
 * @param[in] call Callable invoking the wrapper with valid arguments.
 * @param[in] reason Valid non-NULL contract explanation printed on failure.
 * @note Side effects: Calls the wrapper three times and updates output, providerState result,
 * and call records.
 */
template <typename Symbol, typename Output, typename Call>
void ExpectPointerDispatchForTest(Symbol& slot, Symbol implementation, Output& output,
  Output successfulOutput, Call call, const char* reason) {
  SCOPED_TRACE(reason);
  const Output sentinel = reinterpret_cast<Output>(static_cast<uintptr_t>(1));
  providerState.calls.clear();
  slot = nullptr;
  output = sentinel;
  EXPECT_EQ(call(), ncclInternalError) << "A missing mandatory provider symbol must be rejected.";
  EXPECT_EQ(output, sentinel) << "A rejected operation must leave its output pointer unchanged.";
  EXPECT_TRUE(providerState.calls.empty()) << "A missing provider symbol cannot dispatch a provider call.";
  slot = implementation;
  providerState.returnNull = true;
  output = sentinel;
  EXPECT_EQ(call(), ncclSystemError) << "A NULL provider result must be translated to ncclSystemError.";
  EXPECT_EQ(output, nullptr) << "A provider NULL result must replace the caller's output pointer.";
  EXPECT_EQ(providerState.calls.size(), 1U) << "A NULL provider result must come from exactly one call.";
  providerState.returnNull = false;
  output = sentinel;
  EXPECT_EQ(call(), ncclSuccess) << "A non-NULL provider result must return ncclSuccess.";
  EXPECT_EQ(output, successfulOutput) << "A successful operation must expose the exact provider pointer.";
  EXPECT_EQ(providerState.calls.size(), 2U) << "The NULL and successful paths must dispatch one call each.";
}

/**
 * @brief Checks missing-symbol and successful void-provider dispatch.
 * @tparam Symbol Provider function-pointer type stored in slot.
 * @tparam Call Callable type that invokes the wrapper and returns ncclResult_t.
 * @param[in,out] slot Function-pointer slot temporarily cleared and restored to implementation.
 * @param[in] implementation Valid non-NULL fake provider function.
 * @param[in] call Callable invoking the wrapper with valid arguments.
 * @param[in] reason Valid non-NULL contract explanation printed on failure.
 * @note Side effects: Calls the wrapper twice and updates providerState call records.
 */
template <typename Symbol, typename Call>
void ExpectVoidDispatchForTest(Symbol& slot, Symbol implementation, Call call, const char* reason) {
  SCOPED_TRACE(reason);
  providerState.calls.clear();
  slot = nullptr;
  EXPECT_EQ(call(), ncclInternalError) << "A missing mandatory provider symbol must be rejected.";
  EXPECT_TRUE(providerState.calls.empty()) << "A missing provider symbol cannot dispatch a provider call.";
  slot = implementation;
  EXPECT_EQ(call(), ncclSuccess) << "A dispatched void provider operation must return ncclSuccess.";
  EXPECT_EQ(providerState.calls.size(), 1U) << "The installed void operation must dispatch exactly one call.";
}

/**
 * @brief Checks every defined result of an optional ECE provider operation.
 * @tparam Symbol Provider function-pointer type stored in slot.
 * @tparam Call Callable type that invokes the wrapper and accepts supported-output storage.
 * @param[in,out] slot Function-pointer slot temporarily cleared and restored to implementation.
 * @param[in] implementation Valid non-NULL fake provider function.
 * @param[in] call Callable invoking the wrapper with valid arguments and supported output storage.
 * @param[in] reason Valid non-NULL contract explanation printed on failure.
 * @param[in] checkMissingOutput Optional callback verifying ECE storage after the
 * absent-symbol path.
 * @note Side effects: Calls the wrapper five times and updates providerState result and call
 * records.
 */
template <typename Symbol, typename Call>
void ExpectOptionalEceDispatchForTest(Symbol& slot, Symbol implementation, Call call, const char* reason,
  const std::function<void()>& checkMissingOutput) {
  SCOPED_TRACE(reason);
  int supported = -1;
  providerState.calls.clear();
  slot = nullptr;
  EXPECT_EQ(call(&supported), ncclSuccess) << "An absent optional ECE operation must be a successful no-op.";
  EXPECT_EQ(supported, 0) << "An absent optional ECE operation must report unsupported.";
  if (checkMissingOutput) checkMissingOutput();
  EXPECT_TRUE(providerState.calls.empty()) << "An absent provider operation cannot be called.";
  slot = implementation;
  for (int unavailable : {ENOTSUP, EOPNOTSUPP}) {
    providerState.integerResult = unavailable;
    supported = -1;
    EXPECT_EQ(call(&supported), ncclSuccess) << "A provider unsupported result must be a successful no-op.";
    EXPECT_EQ(supported, 0) << "A provider unsupported result must clear the supported flag.";
  }
  providerState.integerResult = EIO;
  supported = -1;
  EXPECT_EQ(call(&supported), ncclSystemError) << "A real provider failure must return ncclSystemError.";
  EXPECT_EQ(supported, 1) << "A failing implemented operation is still supported.";
  providerState.integerResult = 0;
  supported = -1;
  EXPECT_EQ(call(&supported), ncclSuccess) << "A successful provider operation must return ncclSuccess.";
  EXPECT_EQ(supported, 1) << "A successful provider operation must report supported.";
  EXPECT_EQ(providerState.calls.size(), 4U) << "Only installed optional operations may reach the provider.";
}

/**
 * @brief Counts recorded calls having one operation name.
 * @param[in] operation Valid non-NULL operation name.
 * @return The number of matching entries in providerState.calls.
 * @note Side effects: None.
 */
size_t CountProviderCallsForTest(const char* operation) {
  return static_cast<size_t>(std::count_if(providerState.calls.begin(), providerState.calls.end(),
    [operation](const ProviderCall& call) { return call.operation == operation; }));
}

/**
 * @brief Restores a complete fake provider before each test.
 * @note Side effects: Resets providerState and the production symbol table.
 */
void IbvTestBase::SetUp() {
  ResetProviderForTest();
  providerState.buildSymbols = MakeSymbolTableForTest();
  ibvSymbols = providerState.buildSymbols;
}

/**
 * @brief Converts one pointer to an integer suitable for an argument record.
 * @param[in] pointer Any pointer value, including NULL.
 * @return The pointer representation as uintptr_t.
 * @note Side effects: None.
 */
uintptr_t PointerValueForTest(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer);
}

/**
 * @brief Records one fake-provider call and its ordered arguments.
 * @param[in] operation Valid non-NULL operation name.
 * @param[in] arguments Zero to seven pointer-sized argument values.
 * @note Side effects: Appends one entry to providerState.calls.
 */
void RecordProviderCallForTest(const char* operation, std::initializer_list<uintptr_t> arguments) {
  ProviderCall call;
  call.operation = operation;
  call.argumentCount = arguments.size();
  std::copy(arguments.begin(), arguments.end(), call.arguments.begin());
  providerState.calls.push_back(call);
}

/**
 * @brief Restores the fake provider and all returned verbs objects to deterministic defaults.
 * @note Side effects: Replaces providerState and reconnects its object pointers.
 */
void ResetProviderForTest() {
  providerState = ProviderState{};
  std::snprintf(providerState.device.name, sizeof(providerState.device.name), "%s", providerState.deviceName);
  providerState.deviceList[0] = &providerState.device;
  providerState.deviceList[1] = nullptr;
  providerState.context.device = &providerState.device;
  providerState.protectionDomain.context = &providerState.context;
  providerState.memoryRegion.context = &providerState.context;
  providerState.memoryRegion.pd = &providerState.protectionDomain;
  providerState.completionQueue.context = &providerState.context;
  providerState.queuePair.context = &providerState.context;
  providerState.queuePair.pd = &providerState.protectionDomain;
  providerState.queuePair.state = IBV_QPS_RESET;
  providerState.deviceAttributes.max_qp = 101;
  providerState.portAttributes.lid = 202;
  providerState.gid.raw[15] = 3;
  providerState.queuePairAttributes.port_num = 4;
  providerState.queuePairInitAttributes.cap.max_send_wr = 303;
  providerState.eceResult.vendor_id = 0xabc;
  providerState.eceResult.options = 0xdef;
  providerState.eceResult.comp_mask = 0x123;
}

/**
 * @brief Returns the last recorded fake-provider call.
 * @return The final entry in providerState.calls, which must be nonempty.
 * @note Side effects: None.
 */
const ProviderCall& LastProviderCallForTest() {
  return providerState.calls.back();
}

/**
 * @brief Implements the fake fork-initialization operation.
 * @return providerState.integerResult.
 * @note Side effects: Records the operation.
 */
int ForkInitForTest() {
  RecordProviderCallForTest("fork_init", {});
  return providerState.integerResult;
}

/**
 * @brief Returns the configured fake device list.
 * @param[out] count Valid non-NULL storage receiving providerState.deviceCount when the list
 * exists.
 * @return NULL when returnNull is true, otherwise providerState.deviceList.
 * @note Side effects: Records the output pointer and may write count.
 */
struct ibv_device** GetDeviceListForTest(int* count) {
  RecordProviderCallForTest("get_device_list", {PointerValueForTest(count)});
  if (providerState.returnNull) return nullptr;
  *count = providerState.deviceCount;
  return providerState.deviceList;
}

/**
 * @brief Implements fake device-list release.
 * @param[in] list Any device-list pointer.
 * @note Side effects: Records list.
 */
void FreeDeviceListForTest(struct ibv_device** list) {
  RecordProviderCallForTest("free_device_list", {PointerValueForTest(list)});
}

/**
 * @brief Returns the configured fake device name.
 * @param[in] device Any device pointer recorded without modification.
 * @return providerState.deviceName.
 * @note Side effects: Records device.
 */
const char* GetDeviceNameForTest(struct ibv_device* device) {
  RecordProviderCallForTest("get_device_name", {PointerValueForTest(device)});
  return providerState.deviceName;
}

/**
 * @brief Returns the configured fake device context.
 * @param[in] device Any device pointer recorded without modification.
 * @return NULL when returnNull is true, otherwise the fake context.
 * @note Side effects: Records device.
 */
struct ibv_context* OpenDeviceForTest(struct ibv_device* device) {
  RecordProviderCallForTest("open_device", {PointerValueForTest(device)});
  return providerState.returnNull ? nullptr : &providerState.context;
}

/**
 * @brief Implements fake context closure.
 * @param[in] context Any context pointer recorded without modification.
 * @return providerState.integerResult.
 * @note Side effects: Records context.
 */
int CloseDeviceForTest(struct ibv_context* context) {
  RecordProviderCallForTest("close_device", {PointerValueForTest(context)});
  return providerState.integerResult;
}

/**
 * @brief Implements fake asynchronous-event retrieval.
 * @param[in] context Any context pointer recorded without modification.
 * @param[out] event Valid non-NULL storage written on success.
 * @return providerState.integerResult.
 * @note Side effects: Records both pointers and writes event on success.
 */
int GetAsyncEventForTest(struct ibv_context* context, struct ibv_async_event* event) {
  RecordProviderCallForTest("get_async_event", {PointerValueForTest(context), PointerValueForTest(event)});
  if (providerState.integerResult == 0) event->event_type = providerState.asyncEventType;
  return providerState.integerResult;
}

/**
 * @brief Implements fake asynchronous-event acknowledgement.
 * @param[in] event Any event pointer recorded without modification.
 * @note Side effects: Records event.
 */
void AckAsyncEventForTest(struct ibv_async_event* event) {
  RecordProviderCallForTest("ack_async_event", {PointerValueForTest(event)});
}

/**
 * @brief Implements fake device-attribute query.
 * @param[in] context Any context pointer recorded without modification.
 * @param[out] attributes Valid non-NULL storage written on success.
 * @return providerState.integerResult.
 * @note Side effects: Records both pointers and writes attributes on success.
 */
int QueryDeviceForTest(struct ibv_context* context, struct ibv_device_attr* attributes) {
  RecordProviderCallForTest("query_device", {PointerValueForTest(context), PointerValueForTest(attributes)});
  if (providerState.integerResult == 0) *attributes = providerState.deviceAttributes;
  return providerState.integerResult;
}

/**
 * @brief Implements the fallback fake port query.
 * @param[in] context Any context pointer recorded without modification.
 * @param[in] port Port number in [0, UINT8_MAX].
 * @param[out] attributes Valid non-NULL storage written on success.
 * @return providerState.integerResult.
 * @note Side effects: Records all arguments and writes attributes on success.
 */
int QueryPortForTest(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attributes) {
  RecordProviderCallForTest("query_port",
    {PointerValueForTest(context), static_cast<uintptr_t>(port), PointerValueForTest(attributes)});
  if (providerState.integerResult == 0) *attributes = providerState.portAttributes;
  return providerState.integerResult;
}

/**
 * @brief Implements the extended fake port query.
 * @param[in] context Any context pointer recorded without modification.
 * @param[in] port Port number in [0, UINT8_MAX].
 * @param[out] attributes Valid non-NULL storage written on success.
 * @param[in] size Size of attributes in bytes.
 * @return providerState.extendedPortResult.
 * @note Side effects: Records all arguments and writes attributes on success.
 */
int QueryExtendedPortForTest(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attributes,
  size_t size) {
  RecordProviderCallForTest("query_port_ex",
    {PointerValueForTest(context), static_cast<uintptr_t>(port), PointerValueForTest(attributes), size});
  if (providerState.extendedPortResult == 0) *attributes = providerState.portAttributes;
  return providerState.extendedPortResult;
}

/**
 * @brief Implements fake GID query.
 * @param[in] context Any context pointer recorded without modification.
 * @param[in] port Port number in [0, UINT8_MAX].
 * @param[in] index Any representable GID-table index.
 * @param[out] gid Valid non-NULL storage written on success.
 * @return providerState.integerResult.
 * @note Side effects: Records all arguments and writes gid on success.
 */
int QueryGidForTest(struct ibv_context* context, uint8_t port, int index, union ibv_gid* gid) {
  RecordProviderCallForTest("query_gid", {PointerValueForTest(context), static_cast<uintptr_t>(port),
    static_cast<uintptr_t>(index), PointerValueForTest(gid)});
  if (providerState.integerResult == 0) *gid = providerState.gid;
  return providerState.integerResult;
}

/**
 * @brief Implements fake queue-pair query.
 * @param[in] queuePair Any queue-pair pointer recorded without modification.
 * @param[out] attributes Valid non-NULL storage written on success.
 * @param[in] mask Any representable verbs attribute mask.
 * @param[out] initialAttributes Valid non-NULL storage written on success.
 * @return providerState.integerResult.
 * @note Side effects: Records all arguments and writes both outputs on success.
 */
int QueryQueuePairForTest(struct ibv_qp* queuePair, struct ibv_qp_attr* attributes, int mask,
  struct ibv_qp_init_attr* initialAttributes) {
  RecordProviderCallForTest("query_qp", {PointerValueForTest(queuePair), PointerValueForTest(attributes),
    static_cast<uintptr_t>(mask), PointerValueForTest(initialAttributes)});
  if (providerState.integerResult == 0) {
    *attributes = providerState.queuePairAttributes;
    *initialAttributes = providerState.queuePairInitAttributes;
  }
  return providerState.integerResult;
}

/**
 * @brief Returns the configured fake protection domain.
 * @param[in] context Any context pointer recorded without modification.
 * @return NULL when returnNull is true, otherwise the fake protection domain.
 * @note Side effects: Records context.
 */
struct ibv_pd* AllocateProtectionDomainForTest(struct ibv_context* context) {
  RecordProviderCallForTest("alloc_pd", {PointerValueForTest(context)});
  return providerState.returnNull ? nullptr : &providerState.protectionDomain;
}

/**
 * @brief Implements fake protection-domain release.
 * @param[in] protectionDomain Any protection-domain pointer recorded without modification.
 * @return providerState.integerResult.
 * @note Side effects: Records protectionDomain.
 */
int DeallocateProtectionDomainForTest(struct ibv_pd* protectionDomain) {
  RecordProviderCallForTest("dealloc_pd", {PointerValueForTest(protectionDomain)});
  return providerState.integerResult;
}

/**
 * @brief Returns the configured fake registered memory region.
 * @param[in] protectionDomain Any protection-domain pointer recorded without modification.
 * @param[in] address Any buffer pointer recorded without modification.
 * @param[in] length Registration size in [0, SIZE_MAX].
 * @param[in] access Any representable verbs access mask.
 * @return NULL when returnNull is true, otherwise the fake memory region.
 * @note Side effects: Records all arguments.
 */
struct ibv_mr* RegisterMemoryForTest(struct ibv_pd* protectionDomain, void* address, size_t length, int access) {
  RecordProviderCallForTest("reg_mr", {PointerValueForTest(protectionDomain), PointerValueForTest(address), length,
    static_cast<uintptr_t>(access)});
  return providerState.returnNull ? nullptr : &providerState.memoryRegion;
}

/**
 * @brief Returns the configured fake IOVA memory region.
 * @param[in] protectionDomain Any protection-domain pointer recorded without modification.
 * @param[in] address Any buffer pointer recorded without modification.
 * @param[in] length Registration size in [0, SIZE_MAX].
 * @param[in] iova I/O virtual address in [0, UINT64_MAX].
 * @param[in] access Any representable verbs access mask.
 * @return NULL when returnNull is true, otherwise the fake memory region.
 * @note Side effects: Records all arguments.
 */
struct ibv_mr* RegisterMemoryIovaForTest(struct ibv_pd* protectionDomain, void* address, size_t length,
  uint64_t iova, unsigned int access) {
  RecordProviderCallForTest("reg_mr_iova2", {PointerValueForTest(protectionDomain), PointerValueForTest(address),
    length, static_cast<uintptr_t>(iova), static_cast<uintptr_t>(access)});
  return providerState.returnNull ? nullptr : &providerState.memoryRegion;
}

/**
 * @brief Returns the configured fake DMA-BUF memory region.
 * @param[in] protectionDomain Any protection-domain pointer recorded without modification.
 * @param[in] offset DMA-BUF offset in [0, UINT64_MAX].
 * @param[in] length Registration size in [0, SIZE_MAX].
 * @param[in] iova I/O virtual address in [0, UINT64_MAX].
 * @param[in] descriptor Any representable file descriptor.
 * @param[in] access Any representable verbs access mask.
 * @return NULL when returnNull is true, otherwise the fake memory region.
 * @note Side effects: Records all arguments.
 */
struct ibv_mr* RegisterDmaBufForTest(struct ibv_pd* protectionDomain, uint64_t offset, size_t length,
  uint64_t iova, int descriptor, int access) {
  RecordProviderCallForTest("reg_dmabuf_mr", {PointerValueForTest(protectionDomain), static_cast<uintptr_t>(offset),
    length, static_cast<uintptr_t>(iova), static_cast<uintptr_t>(descriptor), static_cast<uintptr_t>(access)});
  return providerState.returnNull ? nullptr : &providerState.memoryRegion;
}

/**
 * @brief Implements fake memory deregistration.
 * @param[in] memoryRegion Any memory-region pointer recorded without modification.
 * @return providerState.integerResult.
 * @note Side effects: Records memoryRegion.
 */
int DeregisterMemoryForTest(struct ibv_mr* memoryRegion) {
  RecordProviderCallForTest("dereg_mr", {PointerValueForTest(memoryRegion)});
  return providerState.integerResult;
}

/**
 * @brief Returns the configured fake completion queue.
 * @param[in] context Any context pointer recorded without modification.
 * @param[in] entries Requested completion count in [0, INT_MAX].
 * @param[in] queueContext Any caller context pointer.
 * @param[in] channel Any completion-channel pointer.
 * @param[in] vector Any representable completion vector.
 * @return NULL when returnNull is true, otherwise the fake completion queue.
 * @note Side effects: Records all arguments.
 */
struct ibv_cq* CreateCompletionQueueForTest(struct ibv_context* context, int entries, void* queueContext,
  struct ibv_comp_channel* channel, int vector) {
  RecordProviderCallForTest("create_cq", {PointerValueForTest(context), static_cast<uintptr_t>(entries),
    PointerValueForTest(queueContext), PointerValueForTest(channel), static_cast<uintptr_t>(vector)});
  return providerState.returnNull ? nullptr : &providerState.completionQueue;
}

/**
 * @brief Implements fake completion-queue destruction.
 * @param[in] completionQueue Any completion-queue pointer recorded without modification.
 * @return providerState.integerResult.
 * @note Side effects: Records completionQueue.
 */
int DestroyCompletionQueueForTest(struct ibv_cq* completionQueue) {
  RecordProviderCallForTest("destroy_cq", {PointerValueForTest(completionQueue)});
  return providerState.integerResult;
}

/**
 * @brief Returns the configured fake queue pair.
 * @param[in] protectionDomain Any protection-domain pointer recorded without modification.
 * @param[in] attributes Valid non-NULL initialization attributes recorded without modification.
 * @return NULL when returnNull is true, otherwise the fake queue pair.
 * @note Side effects: Records both pointers.
 */
struct ibv_qp* CreateQueuePairForTest(struct ibv_pd* protectionDomain, struct ibv_qp_init_attr* attributes) {
  RecordProviderCallForTest("create_qp",
    {PointerValueForTest(protectionDomain), PointerValueForTest(attributes)});
  return providerState.returnNull ? nullptr : &providerState.queuePair;
}

/**
 * @brief Implements fake queue-pair destruction.
 * @param[in] queuePair Any queue-pair pointer recorded without modification.
 * @return providerState.integerResult.
 * @note Side effects: Records queuePair.
 */
int DestroyQueuePairForTest(struct ibv_qp* queuePair) {
  RecordProviderCallForTest("destroy_qp", {PointerValueForTest(queuePair)});
  return providerState.integerResult;
}

/**
 * @brief Returns the next configured queue-pair modification result.
 * @param[in] queuePair Any queue-pair pointer recorded without modification.
 * @param[in] attributes Valid non-NULL modification attributes recorded without modification.
 * @param[in] mask Any representable verbs attribute mask.
 * @return The next modifyResults entry or providerState.integerResult when the sequence is
 * exhausted.
 * @note Side effects: Records all arguments and advances nextModifyResult when a scripted
 * result exists.
 */
int ModifyQueuePairForTest(struct ibv_qp* queuePair, struct ibv_qp_attr* attributes, int mask) {
  RecordProviderCallForTest("modify_qp", {PointerValueForTest(queuePair), PointerValueForTest(attributes),
    static_cast<uintptr_t>(mask)});
  if (providerState.nextModifyResult < providerState.modifyResults.size()) {
    return providerState.modifyResults[providerState.nextModifyResult++];
  }
  return providerState.integerResult;
}

/**
 * @brief Implements fake optional ECE query.
 * @param[in] queuePair Any queue-pair pointer recorded without modification.
 * @param[out] ece Valid non-NULL storage receiving eceResult on success.
 * @return providerState.integerResult.
 * @note Side effects: Records both pointers and writes ece on success.
 */
int QueryEceForTest(struct ibv_qp* queuePair, struct ibv_ece* ece) {
  RecordProviderCallForTest("query_ece", {PointerValueForTest(queuePair), PointerValueForTest(ece)});
  if (providerState.integerResult == 0) *ece = providerState.eceResult;
  return providerState.integerResult;
}

/**
 * @brief Implements fake optional ECE configuration.
 * @param[in] queuePair Any queue-pair pointer recorded without modification.
 * @param[in] ece Valid non-NULL ECE data recorded without modification.
 * @return providerState.integerResult.
 * @note Side effects: Records both pointers.
 */
int SetEceForTest(struct ibv_qp* queuePair, struct ibv_ece* ece) {
  RecordProviderCallForTest("set_ece", {PointerValueForTest(queuePair), PointerValueForTest(ece)});
  return providerState.integerResult;
}

/**
 * @brief Returns the configured fake asynchronous-event name.
 * @param[in] event Any ibv_event_type value.
 * @return providerState.eventName.
 * @note Side effects: Records event.
 */
const char* EventTypeStringForTest(enum ibv_event_type event) {
  RecordProviderCallForTest("event_type_str", {static_cast<uintptr_t>(event)});
  return providerState.eventName;
}

/**
 * @brief Captures arguments and results shared by inline poll, send, and receive wrappers.
 */
struct InlineVerbsState {
  int result = 0;
  int calls = 0;
  void* object = nullptr;
  int count = 0;
  void* work = nullptr;
  void* badWorkOutput = nullptr;
};

InlineVerbsState inlineVerbsState;

/**
 * @brief Supplies a controlled completion count or negative polling error.
 * @param[in] completionQueue Exact completion queue forwarded by the wrapper.
 * @param[in] count Maximum number of output completions.
 * @param[out] completions Receives one recognizable completion when result is positive.
 * @return inlineVerbsState.result.
 * @note Side effects: Records every argument, increments calls, and writes the first
 * completion on success.
 */
int PollCompletionQueueForHeaderTest(ibv_cq* completionQueue, int count, ibv_wc* completions) {
  ++inlineVerbsState.calls;
  inlineVerbsState.object = completionQueue;
  inlineVerbsState.count = count;
  inlineVerbsState.work = completions;
  if (inlineVerbsState.result > 0) {
    completions[0].wr_id = 0x1234u;
    completions[0].status = IBV_WC_SUCCESS;
  }
  return inlineVerbsState.result;
}

/**
 * @brief Supplies a controlled send result and corresponding bad-work-request output.
 * @param[in] queuePair Exact queue pair forwarded by the wrapper.
 * @param[in] workRequest Exact first send work request forwarded by the wrapper.
 * @param[out] badWorkRequest Receives NULL on success or workRequest on failure.
 * @return inlineVerbsState.result.
 * @note Side effects: Records every argument, increments calls, and writes badWorkRequest.
 */
int PostSendForHeaderTest(ibv_qp* queuePair, ibv_send_wr* workRequest,
    ibv_send_wr** badWorkRequest) {
  ++inlineVerbsState.calls;
  inlineVerbsState.object = queuePair;
  inlineVerbsState.work = workRequest;
  inlineVerbsState.badWorkOutput = badWorkRequest;
  *badWorkRequest = inlineVerbsState.result == 0 ? nullptr : workRequest;
  return inlineVerbsState.result;
}

/**
 * @brief Supplies a controlled receive result and corresponding bad-work-request output.
 * @param[in] queuePair Exact queue pair forwarded by the wrapper.
 * @param[in] workRequest Exact receive work request forwarded by the wrapper.
 * @param[out] badWorkRequest Receives NULL on success or workRequest on failure.
 * @return inlineVerbsState.result.
 * @note Side effects: Records every argument, increments calls, and writes badWorkRequest.
 */
int PostReceiveForHeaderTest(ibv_qp* queuePair, ibv_recv_wr* workRequest,
    ibv_recv_wr** badWorkRequest) {
  ++inlineVerbsState.calls;
  inlineVerbsState.object = queuePair;
  inlineVerbsState.work = workRequest;
  inlineVerbsState.badWorkOutput = badWorkRequest;
  *badWorkRequest = inlineVerbsState.result == 0 ? nullptr : workRequest;
  return inlineVerbsState.result;
}

/**
 * @brief Verifies direct CQ polling maps counts and errors while preserving the error-output
 * contract.
 */
TEST(IbvHeaderWrapperTest, PollsCompletionQueue) {
  SCOPED_TRACE("CQ polling must forward every argument, publish nonnegative counts, and map negative results.");
  inlineVerbsState = {};
  ibv_context context = {};
  context.ops.poll_cq = PollCompletionQueueForHeaderTest;
  ibv_cq completionQueue = {};
  completionQueue.context = &context;
  ibv_wc completion = {};
  completion.wr_id = 0xfeedu;
  int completed = -1;

  inlineVerbsState.result = 1;
  EXPECT_EQ(wrap_ibv_poll_cq(&completionQueue, 4, &completion, &completed), ncclSuccess)
      << "A positive provider count must produce ncclSuccess";
  EXPECT_EQ(inlineVerbsState.calls, 1) << "A poll must call its provider exactly once";
  EXPECT_EQ(inlineVerbsState.object, &completionQueue)
      << "The provider must receive the caller's completion queue";
  EXPECT_EQ(inlineVerbsState.count, 4) << "The provider must receive the requested completion capacity";
  EXPECT_EQ(inlineVerbsState.work, &completion)
      << "The provider must receive the caller's completion storage";
  EXPECT_EQ(completed, 1) << "The wrapper must publish the provider's completion count";
  EXPECT_EQ(completion.wr_id, 0x1234u) << "The provider completion must remain visible to the caller";

  inlineVerbsState.result = 0;
  completed = -1;
  EXPECT_EQ(wrap_ibv_poll_cq(&completionQueue, 4, &completion, &completed), ncclSuccess)
      << "A zero-completion poll is a successful operation";
  EXPECT_EQ(completed, 0) << "A zero provider count must replace the caller's output";

  inlineVerbsState.result = -EIO;
  completed = 77;
  const ibv_wc completionSnapshot = completion;
  EXPECT_EQ(wrap_ibv_poll_cq(&completionQueue, 4, &completion, &completed), ncclSystemError)
      << "A negative provider result must map to ncclSystemError";
  EXPECT_EQ(completed, 77) << "A failed poll must preserve the completion-count output";
  EXPECT_EQ(std::memcmp(&completion, &completionSnapshot, sizeof(completion)), 0)
      << "A failed non-writing provider must leave caller completion storage unchanged";
}

/**
 * @brief Verifies direct send and receive wrappers forward inputs, outputs, and both provider
 * result classes.
 */
TEST(IbvHeaderWrapperTest, PostsSendAndReceiveWorkRequests) {
  SCOPED_TRACE("Work-request posting must map provider status and forward all pointer arguments exactly.");
  ibv_context context = {};
  context.ops.post_send = PostSendForHeaderTest;
  context.ops.post_recv = PostReceiveForHeaderTest;
  ibv_qp queuePair = {};
  queuePair.context = &context;
  ibv_send_wr send = {};
  send.wr_id = 11;
  ibv_recv_wr receive = {};
  receive.wr_id = 12;
  const ibv_send_wr sendSnapshot = send;
  const ibv_recv_wr receiveSnapshot = receive;

  inlineVerbsState = {};
  ibv_send_wr* badSend = reinterpret_cast<ibv_send_wr*>(uintptr_t{1});
  EXPECT_EQ(wrap_ibv_post_send(&queuePair, &send, &badSend), ncclSuccess)
      << "A zero send-provider result must produce ncclSuccess";
  EXPECT_EQ(inlineVerbsState.object, &queuePair) << "Send must forward the queue pair";
  EXPECT_EQ(inlineVerbsState.work, &send) << "Send must forward the first work request";
  EXPECT_EQ(inlineVerbsState.badWorkOutput, &badSend) << "Send must forward its out-argument address";
  EXPECT_EQ(badSend, nullptr) << "A successful provider must clear the bad-send output";

  inlineVerbsState.result = EIO;
  EXPECT_EQ(wrap_ibv_post_send(&queuePair, &send, &badSend), ncclSystemError)
      << "A nonzero send-provider result must map to ncclSystemError";
  EXPECT_EQ(inlineVerbsState.calls, 2) << "The failing send must make one additional provider call";
  EXPECT_EQ(inlineVerbsState.object, &queuePair) << "The failing send must forward the queue pair";
  EXPECT_EQ(inlineVerbsState.work, &send) << "The failing send must forward the work request";
  EXPECT_EQ(inlineVerbsState.badWorkOutput, &badSend)
      << "The failing send must forward its out-argument address";
  EXPECT_EQ(badSend, &send) << "A failing send provider must identify the rejected request";

  inlineVerbsState = {};
  ibv_recv_wr* badReceive = reinterpret_cast<ibv_recv_wr*>(uintptr_t{1});
  EXPECT_EQ(wrap_ibv_post_recv(&queuePair, &receive, &badReceive), ncclSuccess)
      << "A zero receive-provider result must produce ncclSuccess";
  EXPECT_EQ(inlineVerbsState.object, &queuePair) << "Receive must forward the queue pair";
  EXPECT_EQ(inlineVerbsState.work, &receive) << "Receive must forward the work request";
  EXPECT_EQ(inlineVerbsState.badWorkOutput, &badReceive)
      << "Receive must forward its out-argument address";
  EXPECT_EQ(badReceive, nullptr) << "A successful provider must clear the bad-receive output";

  inlineVerbsState.result = EIO;
  EXPECT_EQ(wrap_ibv_post_recv(&queuePair, &receive, &badReceive), ncclSystemError)
      << "A nonzero receive-provider result must map to ncclSystemError";
  EXPECT_EQ(inlineVerbsState.calls, 2) << "The failing receive must make one additional provider call";
  EXPECT_EQ(inlineVerbsState.object, &queuePair) << "The failing receive must forward the queue pair";
  EXPECT_EQ(inlineVerbsState.work, &receive) << "The failing receive must forward the work request";
  EXPECT_EQ(inlineVerbsState.badWorkOutput, &badReceive)
      << "The failing receive must forward its out-argument address";
  EXPECT_EQ(badReceive, &receive) << "A failing receive provider must identify the rejected request";
  EXPECT_EQ(std::memcmp(&send, &sendSnapshot, sizeof(send)), 0)
      << "A non-writing provider and both wrappers must preserve the send request";
  EXPECT_EQ(std::memcmp(&receive, &receiveSnapshot, sizeof(receive)), 0)
      << "A non-writing provider and both wrappers must preserve the receive request";
}

/**
 * @brief Verifies GID conversion succeeds at its exact buffer boundary and reports
 * insufficient storage.
 */
TEST(IbvHeaderWrapperTest, FormatsGidWithExactCapacity) {
  SCOPED_TRACE("GID formatting must preserve its binary input and report exact-capacity success or ENOSPC.");
  union ibv_gid gid = {};
  ASSERT_EQ(inet_pton(AF_INET6, "::1", gid.raw), 1)
      << "The test fixture must create a valid IPv6 loopback GID";
  const union ibv_gid gidSnapshot = gid;
  char exact[4] = {'x', 'x', 'x', 'x'};

  EXPECT_EQ(ibvGetGidStr(&gid, exact, sizeof(exact)), exact)
      << "A buffer containing exactly the text and terminator must be accepted";
  EXPECT_STREQ(exact, "::1") << "The GID must use canonical IPv6 text";
  EXPECT_EQ(std::memcmp(&gid, &gidSnapshot, sizeof(gid)), 0)
      << "Formatting must not modify the caller-owned binary GID";

  char tooSmall[3] = {'@', '@', '@'};
  errno = 0;
  EXPECT_EQ(ibvGetGidStr(&gid, tooSmall, sizeof(tooSmall)), nullptr)
      << "A buffer one byte below the exact requirement must be rejected";
  EXPECT_EQ(errno, ENOSPC) << "Insufficient GID output capacity must set errno to ENOSPC";
  EXPECT_EQ(std::memcmp(&gid, &gidSnapshot, sizeof(gid)), 0)
      << "A failed conversion must preserve the caller-owned binary GID";
}

}  // namespace
