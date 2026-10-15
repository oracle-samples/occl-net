/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "common.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct ncclIbResiliency;
struct ncclIbNetCommBase;

namespace {

bool failScratchAllocation = false;
ncclResult_t resiliencyInitResult = ncclSuccess;
ncclIbResiliency* resiliencyInitContext = nullptr;
int64_t splitDataOnQpsParameter = 0;
int64_t prepostReceiveWorkRequestsParameter = 0;
int64_t asyncEventsParameter = 1;

/** @brief Captures one parameter-loader invocation for metadata assertions. */
struct CommonParameterLoadCall {
  std::string environment;
  int64_t defaultValue;
};

std::vector<CommonParameterLoadCall> commonParameterLoads;
std::vector<ibv_async_event> queuedAsyncEvents;
std::vector<ibv_event_type> decodedAsyncEventTypes;
std::vector<ibv_event_type> acknowledgedAsyncEventTypes;
size_t nextAsyncEvent = 0;
int asyncEventGetCalls = 0;
int asyncEventDecodeCalls = 0;
int asyncEventAckCalls = 0;
bool failAsyncEventDecode = false;
bool failAsyncEventAck = false;
ibv_context* observedAsyncEventContext = nullptr;
char asyncEventName[] = "test-event";

/** @brief Holds two nonzero QP numbers that map to the same hash bucket. */
struct QpCollisionNumbers {
  uint32_t first;
  uint32_t second;
};

/**
 * @brief Finds two QP numbers that exercise the production hash collision path.
 * @return Two colliding numbers, or two zeros if the search unexpectedly fails.
 */
QpCollisionNumbers FindTwoCollidingQpNumbers() {
  uint32_t firstByBucket[QP_MAP_SIZE] = {};

  for (uint32_t candidate = 1; candidate <= QP_MAP_SIZE + 1u; ++candidate) {
    const uint32_t bucket = qpHash32(candidate) & QP_MAP_MASK;
    if (0 == firstByBucket[bucket]) {
      firstByBucket[bucket] = candidate;
    } else {
      return {firstByBucket[bucket], candidate};
    }
  }
  return {0, 0};
}

/**
 * @brief Fills a QP hash map using test-side open addressing.
 * @param[out] hashMap Valid, initially empty map populated to its capacity.
 */
void FillFullQpHashMapForLookup(ncclIbQpNumHash* hashMap) {
  for (uint32_t qpNumber = 1; qpNumber <= QP_MAP_SIZE; ++qpNumber) {
    uint32_t slot = qpHash32(qpNumber) & QP_MAP_MASK;
    for (uint32_t probes = 0; probes < QP_MAP_SIZE; ++probes) {
      if (0 == hashMap->keys[slot]) {
        hashMap->keys[slot] = qpNumber;
        break;
      }
      slot = (slot + 1u) & QP_MAP_MASK;
    }
  }
}

/**
 * @brief Restores async-event observations and injected failures to their defaults.
 * @note Side effects: Clears queued events and recorded dependency calls.
 */
void ResetAsyncEventTestState() {
  queuedAsyncEvents.clear();
  decodedAsyncEventTypes.clear();
  acknowledgedAsyncEventTypes.clear();
  nextAsyncEvent = 0;
  asyncEventGetCalls = 0;
  asyncEventDecodeCalls = 0;
  asyncEventAckCalls = 0;
  failAsyncEventDecode = false;
  failAsyncEventAck = false;
  observedAsyncEventContext = nullptr;
}

/**
 * @brief Allocates multi-send scratch storage or injects an allocation failure.
 * @param[out] pointer Valid storage receiving the allocation or NULL on failure.
 * @param[in] bytes Number of bytes requested by the production caller.
 * @return ncclSuccess on success, otherwise the injected or allocator error.
 * @note Side effects: Transfers allocation ownership through @p pointer on success.
 */
ncclResult_t AllocateScratchForTest(void** pointer, size_t bytes) {
  if (failScratchAllocation) {
    *pointer = nullptr;
    return ncclSystemError;
  }
#ifdef OCCL_DISABLE_CUDA_RUNTIME
  return occlIbPageAllocDebug(pointer, bytes, __FILE__, __LINE__);
#else
  return ncclIbMallocDebug(pointer, bytes, __FILE__, __LINE__);
#endif
}

/**
 * @brief Resolves common parameters through controlled no-cache values.
 * @param[in] environment Valid environment-variable name requested by an accessor.
 * @param[in] defaultValue Production default for the parameter.
 * @param[in] uninitialized Production cache sentinel, accepted without modification.
 * @param[in] cache Optional production cache storage, left unmodified.
 * @param[out] noCache Valid storage set to one to request another load.
 * @return The configured test value, or @p defaultValue for an unknown key.
 * @note Side effects: Records the parameter name and default value.
 */
int64_t LoadCommonParameterForTest(const char* environment, int64_t defaultValue,
    int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  (void)uninitialized;
  (void)cache;
  commonParameterLoads.push_back({environment, defaultValue});
  *noCache = 1;
  if (std::strcmp(environment, "NCCL_IB_SPLIT_DATA_ON_QPS") == 0) {
    return splitDataOnQpsParameter;
  }
  if (std::strcmp(environment, "NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS") == 0) {
    return prepostReceiveWorkRequestsParameter;
  }
  if (std::strcmp(environment, "NCCL_IB_RETURN_ASYNC_EVENTS") == 0) {
    return asyncEventsParameter;
  }
  return defaultValue;
}

}  // namespace

/**
 * @brief Supplies the disabled out-of-order receive-queue policy required by common.cc.
 * @return Zero to disable the policy.
 */
int ncclParamIbOooRq() {
  return 0;
}

/**
 * @brief Supplies the disabled resiliency policy required by common.cc.
 * @return Zero to disable the policy.
 */
int ncclParamIbResiliencyPortFailover() {
  return 0;
}

/**
 * @brief Supplies the default receiver-side matching policy required by common.cc.
 * @return Negative two, the production default.
 */
int ncclParamIbReceiverSideMatchingScheme() {
  return -2;
}

/**
 * @brief Supplies a controlled resiliency initialization result and context.
 * @param[in] base Optional communicator base, accepted without modification.
 * @param[out] resCtx Valid output storage on configured success.
 * @return The configured resiliency initialization result.
 * @note Side effects: Writes the configured context through @p resCtx on success.
 */
ncclResult_t ncclIbResiliencyInit(
    struct ncclIbNetCommBase* base, struct ncclIbResiliency** resCtx) {
  (void)base;
  if (resiliencyInitResult != ncclSuccess) return resiliencyInitResult;
  *resCtx = resiliencyInitContext;
  return ncclSuccess;
}

/**
 * @brief Delivers the next queued asynchronous event or reports exhaustion.
 * @param[in] context Optional device context recorded without modification.
 * @param[out] event Valid storage populated while an event remains.
 * @return ncclSuccess while an event remains, otherwise ncclSystemError.
 * @note Side effects: Records context and consumes one queued event on success.
 */
ncclResult_t wrap_ibv_get_async_event(struct ibv_context* context, struct ibv_async_event* event) {
  ++asyncEventGetCalls;
  observedAsyncEventContext = context;
  if (nextAsyncEvent == queuedAsyncEvents.size()) return ncclSystemError;
  *event = queuedAsyncEvents[nextAsyncEvent++];
  return ncclSuccess;
}

/**
 * @brief Records event decoding and supplies a stable diagnostic name.
 * @param[out] result Valid storage receiving a borrowed name on success.
 * @param[in] eventType Event type recorded for ordering verification.
 * @return ncclSuccess or the injected decoding error.
 * @note Side effects: Records the call and writes @p result on success.
 */
ncclResult_t wrap_ibv_event_type_str(char** result, enum ibv_event_type eventType) {
  ++asyncEventDecodeCalls;
  decodedAsyncEventTypes.push_back(eventType);
  if (failAsyncEventDecode) return ncclSystemError;
  *result = asyncEventName;
  return ncclSuccess;
}

/**
 * @brief Records asynchronous-event acknowledgment and supplies a controlled result.
 * @param[in] event Valid event whose type is recorded without modification.
 * @return ncclSuccess or the injected acknowledgment error.
 * @note Side effects: Records the call and acknowledged event type.
 */
ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event* event) {
  ++asyncEventAckCalls;
  acknowledgedAsyncEventTypes.push_back(event->event_type);
  return failAsyncEventAck ? ncclSystemError : ncclSuccess;
}

/**
 * @brief Compiles common.cc with allocation and parameter dependencies redirected to test seams.
 */
#undef ncclIbMalloc
#define ncclIbMalloc(pointer, bytes) AllocateScratchForTest((pointer), (bytes))
#define ncclLoadParam LoadCommonParameterForTest
#include "../common.cc"  // NOLINT(build/include)
#undef ncclLoadParam
#undef ncclIbMalloc

namespace {

/**
 * @brief Creates a zero-initialized communicator with sequential logical QPs.
 * @param[in] devices Number of local device entries exposed by the communicator.
 * @param[in] qps Number of logical and direct-mode physical QPs.
 * @return Owning pointer to the initialized communicator base.
 */
std::unique_ptr<ncclIbNetCommBase> MakeBaseWithActiveQps(int devices, int qps) {
  auto base = std::make_unique<ncclIbNetCommBase>();
  base->vProps.ndevs = devices;
  base->nqps = qps;
  base->nPhysicalQps = qps;
  base->hasStandbyPaths = false;
  for (int i = 0; i < qps; i++) {
    base->qps[i].logicalLane = i;
    base->activeQps[i] = &base->qps[i];
  }
  return base;
}

/**
 * @brief Reports whether every byte in a memory region is zero.
 * @param[in] data Readable byte range when @p size is nonzero.
 * @param[in] size Number of bytes to inspect.
 * @return true only when every inspected byte is zero.
 */
bool BytesAreZero(const void* data, size_t size) {
  const unsigned char* bytes = static_cast<const unsigned char*>(data);
  for (size_t index = 0; index < size; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

int commonPostReceiveResult = 0;
int commonPostReceiveCalls = 0;
ibv_qp* commonPostedQueuePair = nullptr;
ibv_recv_wr* commonPostedWorkRequest = nullptr;
ibv_recv_wr** commonBadWorkOutput = nullptr;

/**
 * @brief Records a receive post and supplies a controlled provider result.
 * @param[in] queuePair Optional queue pair recorded without modification.
 * @param[in] workRequest Optional receive work request recorded without modification.
 * @param[out] badWorkRequest Valid storage receiving NULL or @p workRequest.
 * @return The configured provider result.
 * @note Side effects: Records all arguments and increments the call count.
 */
int PostReceiveForCommonTest(ibv_qp* queuePair, ibv_recv_wr* workRequest,
    ibv_recv_wr** badWorkRequest) {
  ++commonPostReceiveCalls;
  commonPostedQueuePair = queuePair;
  commonPostedWorkRequest = workRequest;
  commonBadWorkOutput = badWorkRequest;
  *badWorkRequest = commonPostReceiveResult == 0 ? nullptr : workRequest;
  return commonPostReceiveResult;
}

}  // namespace

/**
 * @brief Verifies common parameter accessors preserve their keys, defaults, and configured values.
 */
TEST(NetIbCommonDirectTest, LoadsReceiveAndAsyncParameters) {
  SCOPED_TRACE("Parameter accessors must preserve production metadata and expose seam values.");
  commonParameterLoads.clear();
  prepostReceiveWorkRequestsParameter = 0;
  asyncEventsParameter = 1;
  EXPECT_EQ(ncclParamIbPrepostReceiveWorkRequests(), 0);
  EXPECT_EQ(ncclParamIbAsyncEvents(), 1);

  prepostReceiveWorkRequestsParameter = 1;
  asyncEventsParameter = 0;
  EXPECT_EQ(ncclParamIbPrepostReceiveWorkRequests(), 1);
  EXPECT_EQ(ncclParamIbAsyncEvents(), 0);

  ASSERT_EQ(commonParameterLoads.size(), 4u);
  EXPECT_EQ(commonParameterLoads[0].environment, "NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS");
  EXPECT_EQ(commonParameterLoads[0].defaultValue, 0);
  EXPECT_EQ(commonParameterLoads[1].environment, "NCCL_IB_RETURN_ASYNC_EVENTS");
  EXPECT_EQ(commonParameterLoads[1].defaultValue, 1);
  prepostReceiveWorkRequestsParameter = 0;
  asyncEventsParameter = 1;
}

/** @brief Verifies cold-field lookup returns the indexed embedded record without mutation. */
TEST(NetIbCommonDirectTest, SelectsQueuePairColdFields) {
  SCOPED_TRACE("Cold-field lookup must use the QP cold index exactly.");
  auto base = std::make_unique<ncclIbNetCommBase>();
  base->qpCold[0].remQpn = 17;
  base->qpCold[NCCL_IB_MAX_QPS - 1].remQpn = 29;
  ncclIbQp queuePair = {};

  queuePair.coldIdx = 0;
  const ncclIbQp firstSnapshot = queuePair;
  EXPECT_EQ(ncclIbCommBaseGetQpCold(base.get(), &queuePair), &base->qpCold[0]);
  EXPECT_EQ(std::memcmp(&queuePair, &firstSnapshot, sizeof(queuePair)), 0);
  EXPECT_EQ(base->qpCold[0].remQpn, 17u);

  queuePair.coldIdx = NCCL_IB_MAX_QPS - 1;
  const ncclIbQp lastSnapshot = queuePair;
  EXPECT_EQ(ncclIbCommBaseGetQpCold(base.get(), &queuePair),
            &base->qpCold[NCCL_IB_MAX_QPS - 1]);
  EXPECT_EQ(std::memcmp(&queuePair, &lastSnapshot, sizeof(queuePair)), 0);
  EXPECT_EQ(base->qpCold[NCCL_IB_MAX_QPS - 1].remQpn, 29u);
}

/** @brief Verifies receive posting forwards pointers and translates provider status. */
TEST(NetIbCommonDirectTest, PostsReceiveWorkRequest) {
  SCOPED_TRACE("Receive posting must forward the QP and WR and translate only provider status.");
  ibv_context context = {};
  context.ops.post_recv = PostReceiveForCommonTest;
  ibv_qp queuePair = {};
  queuePair.context = &context;
  ibv_recv_wr workRequest = {};
  workRequest.wr_id = 0x1234u;
  const ibv_qp queuePairSnapshot = queuePair;
  const ibv_recv_wr workRequestSnapshot = workRequest;
  commonPostReceiveResult = 0;
  commonPostReceiveCalls = 0;

  EXPECT_EQ(ncclIbPostRecvWorkRequest(&queuePair, &workRequest), ncclSuccess);
  EXPECT_EQ(commonPostReceiveCalls, 1);
  EXPECT_EQ(commonPostedQueuePair, &queuePair);
  EXPECT_EQ(commonPostedWorkRequest, &workRequest);
  EXPECT_NE(commonBadWorkOutput, nullptr);

  commonPostReceiveResult = EIO;
  EXPECT_EQ(ncclIbPostRecvWorkRequest(&queuePair, &workRequest), ncclSystemError);
  EXPECT_EQ(commonPostReceiveCalls, 2);
  EXPECT_EQ(std::memcmp(&queuePair, &queuePairSnapshot, sizeof(queuePair)), 0);
  EXPECT_EQ(std::memcmp(&workRequest, &workRequestSnapshot, sizeof(workRequest)), 0);
  commonPostReceiveResult = 0;
}

/** @brief Verifies the direct QP hash is deterministic and incorporates both input halves. */
TEST(NetIbCommonDirectTest, HashesQueuePairNumbers) {
  SCOPED_TRACE("QP hashing must be stable and mix both halves into the consumer-visible bucket.");
  constexpr uint32_t ordinary = 0x12345678u;
  const uint32_t first = qpHash32(ordinary);
  const uint32_t firstBucket = first & QP_MAP_MASK;
  EXPECT_EQ(qpHash32(ordinary), first);
  EXPECT_NE(qpHash32(ordinary ^ 0x00000001u) & QP_MAP_MASK, firstBucket);
  EXPECT_NE(qpHash32(ordinary ^ 0x00010000u) & QP_MAP_MASK, firstBucket);
}

/** @brief Verifies insertion advances a colliding QP number to the next slot. */
TEST(NetIbCommonDirectTest, InsertsCollidingQueuePairInNextSlot) {
  SCOPED_TRACE("QP insertion must preserve the first entry and place its collision next to it.");
  const QpCollisionNumbers qpNumbers = FindTwoCollidingQpNumbers();
  ASSERT_NE(0u, qpNumbers.first);
  ASSERT_NE(0u, qpNumbers.second);

  ncclIbQpNumHash hashMap = {};
  qpMapInsert(&hashMap, qpNumbers.first, 11u);
  qpMapInsert(&hashMap, qpNumbers.second, 22u);

  const uint32_t firstSlot = qpHash32(qpNumbers.first) & QP_MAP_MASK;
  const uint32_t secondSlot = (firstSlot + 1u) & QP_MAP_MASK;
  EXPECT_EQ(qpNumbers.first, hashMap.keys[firstSlot]);
  EXPECT_EQ(11u, hashMap.vals[firstSlot]);
  EXPECT_EQ(qpNumbers.second, hashMap.keys[secondSlot]);
  EXPECT_EQ(22u, hashMap.vals[secondSlot]);
}

/** @brief Verifies lookup finds a QP stored after a colliding entry. */
TEST(NetIbCommonDirectTest, FindsQueuePairPastHashCollision) {
  SCOPED_TRACE("QP lookup must probe past an occupied home slot and return the next entry.");
  const QpCollisionNumbers qpNumbers = FindTwoCollidingQpNumbers();
  ASSERT_NE(0u, qpNumbers.first);
  ASSERT_NE(0u, qpNumbers.second);

  ncclIbQpNumHash hashMap = {};
  const uint32_t firstSlot = qpHash32(qpNumbers.first) & QP_MAP_MASK;
  const uint32_t secondSlot = (firstSlot + 1u) & QP_MAP_MASK;
  hashMap.keys[firstSlot] = qpNumbers.first;
  hashMap.vals[firstSlot] = 11u;
  hashMap.keys[secondSlot] = qpNumbers.second;
  hashMap.vals[secondSlot] = 22u;

  EXPECT_EQ(22, qpMapFind(&hashMap, qpNumbers.second));
}

/** @brief Verifies a colliding lookup stops at the first empty slot. */
TEST(NetIbCommonDirectTest, StopsMissingQueuePairAtEmptySlot) {
  SCOPED_TRACE("A missing colliding QP must return -1 as soon as probing reaches an empty slot.");
  const QpCollisionNumbers qpNumbers = FindTwoCollidingQpNumbers();
  ASSERT_NE(0u, qpNumbers.first);
  ASSERT_NE(0u, qpNumbers.second);

  ncclIbQpNumHash hashMap = {};
  const uint32_t firstSlot = qpHash32(qpNumbers.first) & QP_MAP_MASK;
  const uint32_t emptySlot = (firstSlot + 1u) & QP_MAP_MASK;
  const uint32_t sentinelSlot = (emptySlot + 1u) & QP_MAP_MASK;
  hashMap.keys[firstSlot] = qpNumbers.first;
  hashMap.vals[firstSlot] = 11u;
  // A valid open-addressed map cannot contain this key beyond an empty slot. Keeping it there as
  // an observation sentinel makes the test fail if lookup incorrectly continues past the gap.
  hashMap.keys[sentinelSlot] = qpNumbers.second;
  hashMap.vals[sentinelSlot] = 22u;

  EXPECT_EQ(-1, qpMapFind(&hashMap, qpNumbers.second));
}

/** @brief Verifies a missing lookup terminates after probing a completely occupied map. */
TEST(NetIbCommonDirectTest, ReportsMissingQueuePairInFullHashMap) {
  SCOPED_TRACE("A full map miss must examine every slot before returning -1.");
  ncclIbQpNumHash hashMap = {};
  FillFullQpHashMapForLookup(&hashMap);
  for (uint32_t slot = 0; slot < QP_MAP_SIZE; ++slot) {
    ASSERT_NE(0u, hashMap.keys[slot]);
  }

  EXPECT_EQ(-1, qpMapFind(&hashMap, QP_MAP_SIZE + 1u));
}

/** @brief Verifies direction-specific device-base lookup at both valid index boundaries. */
TEST(NetIbCommonTest, ReturnsSendAndRecvDeviceBase) {
  SCOPED_TRACE("Device-base lookup must select direction-specific storage without mutation.");
  auto sendComm = std::make_unique<ncclIbSendComm>();
  sendComm->base.isSend = true;
  sendComm->base.vProps.ndevs = NCCL_IB_MAX_DEVS_PER_NIC;
  auto recvComm = std::make_unique<ncclIbRecvComm>();
  recvComm->base.isSend = false;
  recvComm->base.vProps.ndevs = NCCL_IB_MAX_DEVS_PER_NIC;
  constexpr int lastDevice = NCCL_IB_MAX_DEVS_PER_NIC - 1;

  EXPECT_EQ(ncclIbGetNetCommDevBase(&sendComm->base, 0), &sendComm->devs[0].base);
  EXPECT_EQ(ncclIbGetNetCommDevBase(&sendComm->base, lastDevice),
            &sendComm->devs[lastDevice].base);
  EXPECT_EQ(ncclIbGetNetCommDevBase(&recvComm->base, 0), &recvComm->devs[0].base);
  EXPECT_EQ(ncclIbGetNetCommDevBase(&recvComm->base, lastDevice),
            &recvComm->devs[lastDevice].base);
}

/** @brief Verifies byte-count rounding for aligned, unaligned, and boundary values. */
TEST(NetIbCommonTest, RoundsByteCounts) {
  SCOPED_TRACE("Byte rounding must return the least aligned representable value.");
  constexpr size_t maximumSize = SIZE_MAX;
  EXPECT_EQ(ibFrmRoundUp(0, 1), 0u);
  EXPECT_EQ(ibFrmRoundUp(0, 64), 0u);
  EXPECT_EQ(ibFrmRoundUp(37, 1), 37u);
  EXPECT_EQ(ibFrmRoundUp(64, 64), 64u);
  EXPECT_EQ(ibFrmRoundUp(1, 64), 64u);
  EXPECT_EQ(ibFrmRoundUp(63, 64), 64u);
  EXPECT_EQ(ibFrmRoundUp(65, 64), 128u);
  EXPECT_EQ(ibFrmRoundUp(7, 3), 9u);
  EXPECT_EQ(ibFrmRoundUp(maximumSize, 1), maximumSize);
  EXPECT_EQ(ibFrmRoundUp(1, maximumSize), maximumSize);
}

/**
 * @brief Verifies scratch allocation, layout, replay sidecars, cleanup, and failure propagation.
 */
TEST(NetIbCommonTest, InitializesMultiSendScratch) {
  SCOPED_TRACE("Scratch initialization must build a bounded layout and preserve safe ownership.");
  EXPECT_EQ(ncclIbSendCommInitMultiSendScratch(nullptr), ncclSuccess);
  auto comm = std::make_unique<ncclIbSendComm>();
  ncclIbMultiSendScratch* scratch = &comm->multiSendScratch;

  comm->base.nqps = -1;
  ASSERT_EQ(ncclIbSendCommInitMultiSendScratch(comm.get()), ncclSuccess);
  EXPECT_EQ(scratch->storage, nullptr);
  EXPECT_EQ(scratch->nqps, 0);
  comm->base.nqps = 0;
  ASSERT_EQ(ncclIbSendCommInitMultiSendScratch(comm.get()), ncclSuccess);
  EXPECT_EQ(scratch->storage, nullptr);

  comm->base.nqps = 1;
  ASSERT_EQ(ncclIbSendCommInitMultiSendScratch(comm.get()), ncclSuccess);
  ASSERT_NE(scratch->storage, nullptr);
  EXPECT_EQ(scratch->nqps, 1);
  EXPECT_EQ(scratch->replayPlanBytes, nullptr);
  EXPECT_EQ(scratch->replayPlanFrozen, nullptr);

  comm->base.nqps = NCCL_IB_MAX_QPS;
  comm->base.resiliency = reinterpret_cast<ncclIbResiliency*>(0x1);
  ASSERT_EQ(ncclIbSendCommInitMultiSendScratch(comm.get()), ncclSuccess);
  ASSERT_NE(scratch->storage, nullptr);
  EXPECT_EQ(scratch->nqps, NCCL_IB_MAX_QPS);
  ASSERT_NE(scratch->lwPerQp, nullptr);
  ASSERT_NE(scratch->equalChunkSize, nullptr);
  ASSERT_NE(scratch->reqOffsetShadow, nullptr);
  ASSERT_NE(scratch->sgeAddrShadow, nullptr);
  ASSERT_NE(scratch->remoteAddrShadow, nullptr);
  ASSERT_NE(scratch->chunkSizePerReq, nullptr);
  ASSERT_NE(scratch->qpPerPost, nullptr);
  ASSERT_NE(scratch->qpIndexPerPost, nullptr);
  ASSERT_NE(scratch->chainBytesPerPost, nullptr);
  ASSERT_NE(scratch->wrsPerPost, nullptr);
  ASSERT_NE(scratch->sgesPerPost, nullptr);
  ASSERT_NE(scratch->replayPlanBytes, nullptr);
  ASSERT_NE(scratch->replayPlanFrozen, nullptr);
  EXPECT_EQ(scratch->chunkRowBytes % 128u, 0u);
  EXPECT_EQ(scratch->wrRowBytes % 128u, 0u);
  EXPECT_EQ(scratch->sgeRowBytes % 128u, 0u);
  EXPECT_EQ(scratch->replayPlanFrozen,
            reinterpret_cast<uint8_t*>(scratch->replayPlanBytes +
                                       NET_IB_MAX_REQUESTS * scratch->nqps));

  for (int qpSlot = 0; qpSlot < NCCL_IB_MAX_QPS; ++qpSlot) {
    SCOPED_TRACE(::testing::Message() << "qpSlot=" << qpSlot);
    ibv_send_wr* wrRow = ncclIbMultiSendScratchWrRow(scratch, qpSlot);
    ibv_sge* sgeRow = ncclIbMultiSendScratchSgeRow(scratch, qpSlot);
    EXPECT_FLOAT_EQ(scratch->lwPerQp[qpSlot], 0.0f);
    EXPECT_EQ(scratch->qpPerPost[qpSlot], nullptr);
    EXPECT_EQ(scratch->qpIndexPerPost[qpSlot], 0);
    EXPECT_EQ(scratch->chainBytesPerPost[qpSlot], 0u);
    for (int requestIndex = 0; requestIndex < NCCL_NET_IB_MAX_RECVS; ++requestIndex) {
      EXPECT_EQ(wrRow[requestIndex].sg_list, &sgeRow[requestIndex]);
      EXPECT_EQ(wrRow[requestIndex].num_sge, 1);
      EXPECT_EQ(wrRow[requestIndex].opcode, IBV_WR_RDMA_WRITE);
      EXPECT_EQ(wrRow[requestIndex].send_flags, 0u);
      EXPECT_EQ(wrRow[requestIndex].next, &wrRow[requestIndex + 1]);
      EXPECT_EQ(sgeRow[requestIndex].addr, 0u);
      EXPECT_EQ(sgeRow[requestIndex].length, 0u);
      EXPECT_EQ(sgeRow[requestIndex].lkey, 0u);
    }
    EXPECT_EQ(wrRow[NCCL_NET_IB_MAX_RECVS].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
    EXPECT_EQ(wrRow[NCCL_NET_IB_MAX_RECVS].send_flags, IBV_SEND_SIGNALED);
    EXPECT_EQ(wrRow[NCCL_NET_IB_MAX_RECVS].next, nullptr);
  }

  for (int requestIndex = 0; requestIndex < NCCL_NET_IB_MAX_RECVS; ++requestIndex) {
    SCOPED_TRACE(::testing::Message() << "requestIndex=" << requestIndex);
    EXPECT_EQ(scratch->equalChunkSize[requestIndex], 0);
    EXPECT_EQ(scratch->reqOffsetShadow[requestIndex], 0);
    EXPECT_EQ(scratch->sgeAddrShadow[requestIndex], 0u);
    EXPECT_EQ(scratch->remoteAddrShadow[requestIndex], 0u);
    uint32_t* chunkRow = ncclIbMultiSendScratchChunkRow(scratch, requestIndex);
    for (int qpSlot = 0; qpSlot < NCCL_IB_MAX_QPS; ++qpSlot) {
      EXPECT_EQ(chunkRow[qpSlot], 0u);
    }
  }
  EXPECT_EQ(occlIbMultiSendScratchReplayPlanRow(scratch, 3),
            scratch->replayPlanBytes + 3 * scratch->nqps);
  EXPECT_EQ(occlIbMultiSendScratchReplayPlanRow(scratch, 3)[2], 0u);
  EXPECT_EQ(scratch->replayPlanFrozen[3], 0u);

  comm->base.nqps = 1;
  failScratchAllocation = true;
  const ncclResult_t allocationResult = ncclIbSendCommInitMultiSendScratch(comm.get());
  failScratchAllocation = false;
  EXPECT_EQ(allocationResult, ncclSystemError);
  EXPECT_EQ(scratch->storage, nullptr);
  EXPECT_EQ(scratch->nqps, 0);
  EXPECT_EQ(scratch->chunkRowBytes, 0u);
  EXPECT_EQ(scratch->wrRowBytes, 0u);
  EXPECT_EQ(scratch->sgeRowBytes, 0u);
  EXPECT_EQ(scratch->lwPerQp, nullptr);
  EXPECT_EQ(scratch->equalChunkSize, nullptr);
  EXPECT_EQ(scratch->reqOffsetShadow, nullptr);
  EXPECT_EQ(scratch->sgeAddrShadow, nullptr);
  EXPECT_EQ(scratch->remoteAddrShadow, nullptr);
  EXPECT_EQ(scratch->chunkSizePerReq, nullptr);
  EXPECT_EQ(scratch->qpPerPost, nullptr);
  EXPECT_EQ(scratch->qpIndexPerPost, nullptr);
  EXPECT_EQ(scratch->chainBytesPerPost, nullptr);
  EXPECT_EQ(scratch->wrsPerPost, nullptr);
  EXPECT_EQ(scratch->sgesPerPost, nullptr);
  EXPECT_EQ(scratch->replayPlanBytes, nullptr);
  EXPECT_EQ(scratch->replayPlanFrozen, nullptr);
  ncclIbSendCommFreeMultiSendScratch(comm.get());
}

/** @brief Verifies base initialization resets old and forced-geometry state for both directions. */
TEST(NetIbCommonTest, InitializesBaseComm) {
  SCOPED_TRACE("Base initialization must reset owned state, preserve QP handles, and propagate errors.");
  const bool directions[] = {false, true};
  for (bool isSend : directions) {
    SCOPED_TRACE(isSend ? "send" : "receive");
    auto base = std::make_unique<ncclIbNetCommBase>();
    auto qpHandles = std::make_unique<ibv_qp[]>(NCCL_IB_MAX_QPS);
    std::memset(&base->frmFields, 0xff, sizeof(base->frmFields));
    std::memset(&base->frmCold, 0xff, sizeof(base->frmCold));
    std::memset(&base->qpCold, 0xff, sizeof(base->qpCold));
    for (int index = 0; index < NCCL_IB_MAX_QPS; ++index) {
      base->qps[index].qp = &qpHandles[index];
      base->qps[index].devIndex = index;
      base->qps[index].remDevIdx = index;
      base->qps[index].logicalLane = static_cast<uint8_t>(index);
      base->qps[index].sig_head = 1;
      base->qps[index].sig_tail = 1;
      std::memset(base->qps[index].sig_tsc, 0xff, sizeof(base->qps[index].sig_tsc));
      std::memset(base->qps[index].sig_bytes, 0xff, sizeof(base->qps[index].sig_bytes));
      base->qps[index].coldIdx = NCCL_IB_QP_COLD_INVALID;
      base->activeQps[index] = nullptr;
    }
    base->nqps = 1;
    base->nPhysicalQps = 2;
    base->hasStandbyPaths = true;
    base->splitDataOnQps = -1;
    base->nDataQps = 1;
    base->connTc = 1;
    base->connSl = 1;
    base->isSend = !isSend;
    base->ready = 1;
    auto context = std::make_unique<ncclIbResiliency>();
    resiliencyInitResult = ncclSuccess;
    resiliencyInitContext = context.get();

    const ncclResult_t result = ncclIbBaseCommInit(base.get(), isSend);
    resiliencyInitContext = nullptr;
    ASSERT_EQ(result, ncclSuccess);
    EXPECT_EQ(base->nqps, -1);
    EXPECT_EQ(base->nPhysicalQps, -1);
    EXPECT_FALSE(base->hasStandbyPaths);
    EXPECT_EQ(base->splitDataOnQps, ncclParamIbSplitDataOnQps());
    EXPECT_EQ(base->nDataQps, -1);
    EXPECT_EQ(base->connTc, 0);
    EXPECT_EQ(base->connSl, 0);
    EXPECT_EQ(base->isSend, isSend);
    EXPECT_EQ(base->ready, 0);
    EXPECT_EQ(base->resiliency, context.get());
    EXPECT_TRUE(BytesAreZero(&base->frmFields, sizeof(base->frmFields)));
    EXPECT_TRUE(BytesAreZero(&base->frmCold, sizeof(base->frmCold)));
    for (int index = 0; index < NCCL_IB_MAX_QPS; ++index) {
      SCOPED_TRACE(index);
      EXPECT_EQ(base->qps[index].qp, &qpHandles[index]);
      EXPECT_EQ(base->qps[index].devIndex, -1);
      EXPECT_EQ(base->qps[index].remDevIdx, -1);
      EXPECT_EQ(base->qps[index].logicalLane, NCCL_IB_QP_COLD_INVALID);
      EXPECT_EQ(base->qps[index].sig_head, 0u);
      EXPECT_EQ(base->qps[index].sig_tail, 0u);
      EXPECT_TRUE(BytesAreZero(base->qps[index].sig_tsc,
                              sizeof(base->qps[index].sig_tsc)));
      EXPECT_TRUE(BytesAreZero(base->qps[index].sig_bytes,
                              sizeof(base->qps[index].sig_bytes)));
      EXPECT_EQ(base->qps[index].coldIdx, static_cast<uint8_t>(index));
      EXPECT_TRUE(BytesAreZero(&base->qpCold[index], sizeof(base->qpCold[index])));
      EXPECT_EQ(base->activeQps[index], &base->qps[index]);
    }
  }

  resiliencyInitResult = ncclSystemError;
  auto failedBase = std::make_unique<ncclIbNetCommBase>();
  const ncclResult_t result = ncclIbBaseCommInit(failedBase.get(), true);
  resiliencyInitResult = ncclSuccess;
  EXPECT_EQ(result, ncclSystemError);
}

/** @brief Verifies completion-record capacity spans logical receives and physical QPs. */
TEST(NetIbCommonTest, CompletionRecordLayoutCoversSizesAndPhysicalCompletions) {
  SCOPED_TRACE("Completion records must expose declared capacities and start zero initialized.");
  ncclIbRequestCompletionRecord record{};
  constexpr int expectedRecvs = 8;
  constexpr int expectedQps = 128;

  EXPECT_EQ(sizeof(record.sizes) / sizeof(record.sizes[0]), expectedRecvs);
  EXPECT_EQ(sizeof(record.completions) / sizeof(record.completions[0]), expectedQps);

  for (int i = 0; i < expectedRecvs; i++) {
    EXPECT_EQ(record.sizes[i], 0);
  }
  for (int i = 0; i < expectedQps; i++) {
    EXPECT_EQ(record.completions[i], 0);
  }
}

/** @brief Verifies profiling arrays cover every accepted logical QP. */
TEST(NetIbCommonTest, ProfilerStorageCoversEveryAcceptedLogicalQp) {
  SCOPED_TRACE("Profiling storage must scale with the negotiated logical width.");
  ncclProfilerInfo info{};
  constexpr size_t eventCapacity =
      sizeof(info.qpEventHandles) / sizeof(info.qpEventHandles[0]);
  constexpr size_t indexCapacity = sizeof(info.qpIndex) / sizeof(info.qpIndex[0]);

  static_assert(eventCapacity == NCCL_IB_MAX_QPS,
                "profiler event storage must cover every accepted logical QP");
  static_assert(indexCapacity == NCCL_IB_MAX_QPS,
                "profiler QP-index storage must cover every accepted logical QP");
  EXPECT_EQ(eventCapacity, static_cast<size_t>(NCCL_IB_MAX_QPS));
  EXPECT_EQ(indexCapacity, static_cast<size_t>(NCCL_IB_MAX_QPS));
}

/** @brief Verifies device-local QP lookup follows the interleaved direct-mode layout. */
TEST(NetIbCommonTest, QpByDeviceIndexUsesDeviceMajorLayout) {
  SCOPED_TRACE("A device and local index must resolve to the exact active-QP entry.");
  auto base = MakeBaseWithActiveQps(2, 8);
  ncclIbQp* qp = nullptr;

  EXPECT_EQ(ncclIbCommBaseGetQpByIndex(base.get(), 1, 2, &qp), ncclSuccess);

  EXPECT_EQ(qp, &base->qps[5]);
}

/** @brief Verifies request selection returns its logical lane and current physical QP. */
TEST(NetIbCommonTest, SelectsQpForRequestByRequestIdAndOffset) {
  SCOPED_TRACE("Direct request selection must keep pointer and logical-lane outputs consistent.");
  auto base = MakeBaseWithActiveQps(2, 8);
  ncclIbQp* qp = nullptr;
  int selected = -1;

  EXPECT_EQ(ncclIbCommBaseGetQpForRequest(base.get(), 6, 3, &qp, &selected), ncclSuccess);

  EXPECT_EQ(selected, 1);
  EXPECT_EQ(qp, &base->qps[1]);
}

/** @brief Verifies standby paths never inflate the logical request width. */
TEST(NetIbCommonTest, QpsPerRequestKeepsSplitZeroSingleLaneAndSplitOneLogicalWidth) {
  SCOPED_TRACE("Split policy must count logical lanes, not private physical standby QPs.");
  auto base = MakeBaseWithActiveQps(4, 8);
  base->nPhysicalQps = 32;
  base->nDataQps = 1;

  base->splitDataOnQps = 0;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(base.get()), 1);

  base->splitDataOnQps = 1;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(base.get()), 8);
}

/** @brief Verifies direct mode retains its configured data-QP width when striping is disabled. */
TEST(NetIbCommonTest, QpsPerRequestHonorsDirectSplitPolicy) {
  SCOPED_TRACE("Direct mode must choose nDataQps without striping and all logical QPs with it.");
  auto base = MakeBaseWithActiveQps(2, 8);
  base->nDataQps = 2;

  base->splitDataOnQps = 0;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(base.get()), 2);

  base->splitDataOnQps = 1;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(base.get()), 8);
}

/** @brief Verifies logical request selection follows an active-QP standby remap. */
TEST(NetIbCommonTest, ActiveLookupReturnsLogicalLaneBackedByPhysicalStandby) {
  SCOPED_TRACE("Failover may replace a physical QP without changing the selected logical lane.");
  auto base = MakeBaseWithActiveQps(3, 2);
  base->nPhysicalQps = 6;
  base->activeQps[0] = &base->qps[4];
  base->activeQps[1] = &base->qps[5];
  ncclIbQp* qp = nullptr;
  int logicalLane = -1;

  ASSERT_EQ(ncclIbCommBaseGetQpForRequest(base.get(), 2, 1, &qp, &logicalLane),
            ncclSuccess);

  EXPECT_EQ(logicalLane, 1);
  EXPECT_EQ(qp, &base->qps[5]);
  EXPECT_EQ(qp - base->qps, 5);
}

/** @brief Verifies physical QPs use path-major indices and device-scoped hash lookup. */
TEST(NetIbCommonTest, PhysicalQpGeometryAndHashUsePathMajorIndices) {
  SCOPED_TRACE("Physical lookup must retain path, logical-lane, and device identity.");
  auto base = MakeBaseWithActiveQps(3, 2);
  base->nPhysicalQps = 6;
  ibv_qp verbsQps[6] = {};
  for (int path = 0; path < 3; path++) {
    for (int logicalLane = 0; logicalLane < base->nqps; logicalLane++) {
      const int physicalIndex = occlIbCommPhysicalQpIndex(base.get(), logicalLane, path);
      EXPECT_EQ(physicalIndex, path * base->nqps + logicalLane);
      verbsQps[physicalIndex].qp_num = 200 + physicalIndex;
      base->qps[physicalIndex].qp = &verbsQps[physicalIndex];
      base->qps[physicalIndex].devIndex = path;
      base->qps[physicalIndex].logicalLane = logicalLane;
    }
  }
  qpMapBuild(base.get());

  EXPECT_EQ(occlIbCommPhysicalQpCount(base.get()), 6);
  EXPECT_EQ(qpMapFind(&base->frmFields.completion.qpHash, 205), 5);

  ncclIbQp* qp = nullptr;
  int physicalIndex = -1;
  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(base.get(), 2, 205, &qp, &physicalIndex),
            ncclSuccess);
  EXPECT_EQ(qp, &base->qps[5]);
  EXPECT_EQ(physicalIndex, 5);
  EXPECT_EQ(qp->logicalLane, 1);

  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(base.get(), 0, 205, &qp, nullptr),
            ncclInternalError);
  EXPECT_EQ(qp, nullptr);
}

/** @brief Verifies QP-number lookup rejects a matching number on the wrong device. */
TEST(NetIbCommonTest, QpLookupByNumberFindsDeviceScopedQp) {
  SCOPED_TRACE("QP-number lookup must set both outputs only for the requested device.");
  auto base = MakeBaseWithActiveQps(2, 4);
  ibv_qp verbsQps[4] = {};
  for (int i = 0; i < 4; i++) {
    verbsQps[i].qp_num = 100 + i;
    base->qps[i].qp = &verbsQps[i];
    base->qps[i].devIndex = i % 2;
  }

  ncclIbQp* qp = nullptr;
  int qpIndex = -1;
  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(base.get(), 1, 103, &qp, &qpIndex), ncclSuccess);
  EXPECT_EQ(qp, &base->qps[3]);
  EXPECT_EQ(qpIndex, 3);

  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(base.get(), 0, 103, &qp, nullptr), ncclInternalError);
  EXPECT_EQ(qp, nullptr);
}

/** @brief Verifies all-ones request tracking decodes as the no-attempt sentinel. */
TEST(NetIbCommonTest, SendRequestTrackingCanUseAllOnesAsNoAttemptSentinel) {
  SCOPED_TRACE("Every physical-QP tracking slot must preserve the -1 attempt sentinel.");
  ncclIbRequest request{};
  constexpr int expectedQps = 128;
  std::memset(request.send.pendingData, 0, sizeof(request.send.pendingData));
  std::memset(request.send.lastAttemptQpIndex, 0xff, sizeof(request.send.lastAttemptQpIndex));
  std::memset(request.send.lastAttemptOrder, 0, sizeof(request.send.lastAttemptOrder));
  std::memset(request.send.ackedCompletions, 0, sizeof(request.send.ackedCompletions));

  for (int i = 0; i < expectedQps; i++) {
    EXPECT_FALSE(request.send.pendingData[i]);
    EXPECT_EQ(request.send.lastAttemptQpIndex[i], -1);
    EXPECT_EQ(request.send.lastAttemptOrder[i], 0);
    EXPECT_EQ(request.send.ackedCompletions[i], 0);
  }
}

/** @brief Verifies replay plans are sidecars for resilient direct and standby geometries. */
TEST(NetIbCommonTest, ReplayPlanSidecarIsAllocatedForEveryResilientGeometry) {
  SCOPED_TRACE("Resiliency must allocate replay state without moving existing scratch regions.");
  ncclIbSendComm comm = {};
  comm.base.nqps = 4;
  comm.base.resiliency = reinterpret_cast<ncclIbResiliency*>(0x1);

  ASSERT_EQ(ncclIbSendCommInitMultiSendScratch(&comm), ncclSuccess);
  ASSERT_NE(comm.multiSendScratch.storage, nullptr);
  ASSERT_NE(comm.multiSendScratch.replayPlanBytes, nullptr);
  ASSERT_NE(comm.multiSendScratch.replayPlanFrozen, nullptr);
  const std::ptrdiff_t directChunkOffset = reinterpret_cast<char*>(comm.multiSendScratch.chunkSizePerReq) -
                                           reinterpret_cast<char*>(comm.multiSendScratch.storage);
  const std::ptrdiff_t directWrOffset = reinterpret_cast<char*>(comm.multiSendScratch.wrsPerPost) -
                                        reinterpret_cast<char*>(comm.multiSendScratch.storage);
  const std::ptrdiff_t directSgeOffset = reinterpret_cast<char*>(comm.multiSendScratch.sgesPerPost) -
                                         reinterpret_cast<char*>(comm.multiSendScratch.storage);
  EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen,
            reinterpret_cast<uint8_t*>(comm.multiSendScratch.replayPlanBytes +
                                       NET_IB_MAX_REQUESTS * comm.base.nqps));
  EXPECT_EQ(occlIbMultiSendScratchReplayPlanRow(&comm.multiSendScratch, 3),
            comm.multiSendScratch.replayPlanBytes + 3 * comm.base.nqps);
  EXPECT_EQ(occlIbMultiSendScratchReplayPlanRow(&comm.multiSendScratch, 3)[2], 0u);
  EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen[3], 0u);
  ncclIbSendCommFreeMultiSendScratch(&comm);

  comm.base.hasStandbyPaths = true;
  ASSERT_EQ(ncclIbSendCommInitMultiSendScratch(&comm), ncclSuccess);
  ASSERT_NE(comm.multiSendScratch.replayPlanBytes, nullptr);
  ASSERT_NE(comm.multiSendScratch.replayPlanFrozen, nullptr);
  EXPECT_EQ(reinterpret_cast<char*>(comm.multiSendScratch.chunkSizePerReq) -
                reinterpret_cast<char*>(comm.multiSendScratch.storage),
            directChunkOffset);
  EXPECT_EQ(reinterpret_cast<char*>(comm.multiSendScratch.wrsPerPost) -
                reinterpret_cast<char*>(comm.multiSendScratch.storage),
            directWrOffset);
  EXPECT_EQ(reinterpret_cast<char*>(comm.multiSendScratch.sgesPerPost) -
                reinterpret_cast<char*>(comm.multiSendScratch.storage),
            directSgeOffset);
  EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen,
            reinterpret_cast<uint8_t*>(comm.multiSendScratch.replayPlanBytes +
                                       NET_IB_MAX_REQUESTS * comm.base.nqps));
  EXPECT_EQ(occlIbMultiSendScratchReplayPlanRow(&comm.multiSendScratch, 3),
            comm.multiSendScratch.replayPlanBytes + 3 * comm.base.nqps);
  EXPECT_EQ(occlIbMultiSendScratchReplayPlanRow(&comm.multiSendScratch, 3)[2], 0u);
  EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen[3], 0u);
  ncclIbSendCommFreeMultiSendScratch(&comm);
  EXPECT_EQ(comm.multiSendScratch.replayPlanBytes, nullptr);
  EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen, nullptr);

  comm.base.resiliency = nullptr;
  ASSERT_EQ(ncclIbSendCommInitMultiSendScratch(&comm), ncclSuccess);
  EXPECT_EQ(comm.multiSendScratch.replayPlanBytes, nullptr);
  EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen, nullptr);
  ncclIbSendCommFreeMultiSendScratch(&comm);
}

#ifndef NCCL_ENABLE_NET_PROFILING
/** @brief Verifies replay planning does not enlarge each request object. */
TEST(NetIbCommonTest, ReplayPlanSidecarKeepsPlanBytesOutOfEveryRequest) {
  SCOPED_TRACE("Replay state must remain in communicator-owned sidecar storage.");
  EXPECT_EQ(sizeof(ncclIbRequest), 1216u);
}
#endif

/** @brief Verifies send and receive hot fields retain their 32-byte alignment contracts. */
TEST(NetIbCommonTest, SendAndRecvCommKeepThirtyTwoByteAlignmentContracts) {
  SCOPED_TRACE("Cache-sensitive communicator fields and records must stay 32-byte aligned.");
  EXPECT_EQ(sizeof(ncclIbSendFifo), 64u);
  EXPECT_EQ(sizeof(ncclIbNetCommBase) % 32, 0u);
  EXPECT_EQ(offsetof(ncclIbSendComm, ctsFifo) % 32, 0u);
  EXPECT_EQ(sizeof(ncclIbSendFifo) % 32, 0u);
  EXPECT_EQ(offsetof(ncclIbSendComm, sges) % 32, 0u);
  EXPECT_EQ(offsetof(ncclIbSendComm, wrs) % 32, 0u);
  EXPECT_EQ(offsetof(ncclIbRecvComm, remCtsFifo) % 32, 0u);
}

/** @brief Verifies send initialization replaces stale state and propagates base failures. */
TEST(NetIbCommonTest, SendCommInitPopulatesBaseDefaults) {
  SCOPED_TRACE("Send initialization must reset legacy and forced-geometry state deterministically.");
  auto comm = std::make_unique<ncclIbSendComm>();
  CIMState state = {};
  comm->base.isSend = false;
  comm->base.nqps = 4;
  comm->base.nPhysicalQps = 8;
  comm->base.hasStandbyPaths = true;
  comm->cim = &state;
  std::memset(&comm->multiSendScratch, 0xff, sizeof(comm->multiSendScratch));
  comm->frmGateOpen = true;
  comm->sendWrIdCounter = UINT64_MAX;
  resiliencyInitResult = ncclSuccess;
  resiliencyInitContext = nullptr;

  ASSERT_EQ(ncclIbSendCommInit(comm.get()), ncclSuccess);
  EXPECT_TRUE(comm->base.isSend);
  EXPECT_EQ(comm->base.ready, 0);
  EXPECT_EQ(comm->base.nqps, -1);
  EXPECT_EQ(comm->base.nPhysicalQps, -1);
  EXPECT_FALSE(comm->base.hasStandbyPaths);
  EXPECT_EQ(occlIbCommPhysicalQpCount(&comm->base), 0);
  EXPECT_EQ(comm->base.nDataQps, -1);
  EXPECT_EQ(comm->cim, nullptr);
  EXPECT_TRUE(BytesAreZero(&comm->multiSendScratch, sizeof(comm->multiSendScratch)));
  EXPECT_FALSE(comm->frmGateOpen);
  EXPECT_EQ(comm->sendWrIdCounter, 0u);

  auto failedComm = std::make_unique<ncclIbSendComm>();
  resiliencyInitResult = ncclSystemError;
  const ncclResult_t result = ncclIbSendCommInit(failedComm.get());
  resiliencyInitResult = ncclSuccess;
  EXPECT_EQ(result, ncclSystemError);
}

/** @brief Verifies receive initialization resolves preposting and forced-geometry defaults. */
TEST(NetIbCommonTest, RecvCommInitPopulatesReceiveDefaults) {
  SCOPED_TRACE("Receive initialization must combine configured and resiliency preposting policies.");
  /** @brief Describes one receive preposting and resiliency configuration. */
  struct Scenario {
    int64_t configuredPrepost;
    bool resiliencyEnabled;
    bool expectedPrepost;
  };
  const Scenario scenarios[] = {
      {0, false, false}, {1, false, true}, {0, true, true}, {1, true, true}};

  for (const Scenario& scenario : scenarios) {
    SCOPED_TRACE(::testing::Message()
                 << "configuredPrepost=" << scenario.configuredPrepost
                 << ", resiliencyEnabled=" << scenario.resiliencyEnabled);
    auto comm = std::make_unique<ncclIbRecvComm>();
    std::memset(&comm->ibRecvWorkRequest, 0xff, sizeof(comm->ibRecvWorkRequest));
    comm->base.nqps = 4;
    comm->base.nPhysicalQps = 8;
    comm->base.hasStandbyPaths = true;
    comm->prepostReceiveWorkRequests = !scenario.expectedPrepost;
    auto context = std::make_unique<ncclIbResiliency>();
    resiliencyInitContext = scenario.resiliencyEnabled ? context.get() : nullptr;
    prepostReceiveWorkRequestsParameter = scenario.configuredPrepost;

    const ncclResult_t result = ncclIbRecvCommInit(comm.get());
    resiliencyInitContext = nullptr;
    prepostReceiveWorkRequestsParameter = 0;
    ASSERT_EQ(result, ncclSuccess);
    EXPECT_FALSE(comm->base.isSend);
    EXPECT_EQ(comm->base.nqps, -1);
    EXPECT_EQ(comm->base.nPhysicalQps, -1);
    EXPECT_FALSE(comm->base.hasStandbyPaths);
    EXPECT_EQ(occlIbCommPhysicalQpCount(&comm->base), 0);
    EXPECT_EQ(comm->base.resiliency, scenario.resiliencyEnabled ? context.get() : nullptr);
    EXPECT_EQ(comm->prepostReceiveWorkRequests, scenario.expectedPrepost);
    EXPECT_EQ(comm->ibRecvWorkRequest.wr_id, NCCL_IB_RECV_WR_ID_DUMMY);
    EXPECT_EQ(comm->ibRecvWorkRequest.next, nullptr);
    EXPECT_EQ(comm->ibRecvWorkRequest.sg_list, nullptr);
    EXPECT_EQ(comm->ibRecvWorkRequest.num_sge, 0);
  }

  auto failedComm = std::make_unique<ncclIbRecvComm>();
  resiliencyInitResult = ncclSystemError;
  const ncclResult_t result = ncclIbRecvCommInit(failedComm.get());
  resiliencyInitResult = ncclSuccess;
  EXPECT_EQ(result, ncclSystemError);
}

/**
 * @brief Verifies asynchronous-event attribution, acknowledgment, and stop-on-error behavior.
 */
TEST(NetIbCommonTest, AsyncThreadMainBehavior) {
  SCOPED_TRACE("The async loop must process events in order and stop at each dependency error.");
  ncclIbDev device = {};
  ibv_context context = {};
  ncclIbStats cqStats = {};
  ncclIbStats qpStats = {};
  ibv_cq cq = {};
  ibv_qp qp = {};
  ibv_srq srq = {};
  device.context = &context;
  cq.cq_context = &cqStats;
  qp.qp_context = &qpStats;
  const enum ibv_event_type eventTypes[] = {
      IBV_EVENT_DEVICE_FATAL,
      IBV_EVENT_CQ_ERR,
      IBV_EVENT_QP_FATAL,
      IBV_EVENT_QP_REQ_ERR,
      IBV_EVENT_QP_ACCESS_ERR,
      IBV_EVENT_SRQ_ERR,
      IBV_EVENT_GID_CHANGE,
      IBV_EVENT_PATH_MIG_ERR,
      IBV_EVENT_PORT_ERR,
      IBV_EVENT_PATH_MIG,
      IBV_EVENT_PORT_ACTIVE,
      IBV_EVENT_SQ_DRAINED,
      IBV_EVENT_LID_CHANGE,
      IBV_EVENT_PKEY_CHANGE,
      IBV_EVENT_SM_CHANGE,
      IBV_EVENT_QP_LAST_WQE_REACHED,
      IBV_EVENT_CLIENT_REREGISTER,
      IBV_EVENT_SRQ_LIMIT_REACHED,
      IBV_EVENT_COMM_EST,
      static_cast<enum ibv_event_type>(-1)};

  ResetAsyncEventTestState();
  for (enum ibv_event_type eventType : eventTypes) {
    ibv_async_event event = {};
    event.event_type = eventType;
    if (eventType == IBV_EVENT_CQ_ERR) event.element.cq = &cq;
    if (eventType == IBV_EVENT_QP_FATAL || eventType == IBV_EVENT_QP_REQ_ERR ||
        eventType == IBV_EVENT_QP_ACCESS_ERR) {
      event.element.qp = &qp;
    }
    if (eventType == IBV_EVENT_SRQ_ERR) event.element.srq = &srq;
    queuedAsyncEvents.push_back(event);
  }

  EXPECT_EQ(ncclIbAsyncThreadMain(&device), nullptr);
  EXPECT_EQ(observedAsyncEventContext, &context);
  EXPECT_EQ(asyncEventGetCalls, static_cast<int>(queuedAsyncEvents.size()) + 1);
  EXPECT_EQ(asyncEventDecodeCalls, static_cast<int>(queuedAsyncEvents.size()));
  EXPECT_EQ(asyncEventAckCalls, static_cast<int>(queuedAsyncEvents.size()));
  EXPECT_EQ(decodedAsyncEventTypes.size(), queuedAsyncEvents.size());
  EXPECT_EQ(acknowledgedAsyncEventTypes.size(), queuedAsyncEvents.size());
  for (size_t index = 0; index < acknowledgedAsyncEventTypes.size(); ++index) {
    SCOPED_TRACE(::testing::Message() << "eventIndex=" << index);
    EXPECT_EQ(decodedAsyncEventTypes[index], eventTypes[index]);
    EXPECT_EQ(acknowledgedAsyncEventTypes[index], eventTypes[index]);
  }
  EXPECT_EQ(device.stats.fatalErrorCount, 1);
  EXPECT_EQ(cqStats.fatalErrorCount, 1);
  EXPECT_EQ(qpStats.fatalErrorCount, 3);

  ResetAsyncEventTestState();
  ibv_async_event decodeFailureEvent = {};
  decodeFailureEvent.event_type = IBV_EVENT_COMM_EST;
  queuedAsyncEvents.push_back(decodeFailureEvent);
  failAsyncEventDecode = true;
  EXPECT_EQ(ncclIbAsyncThreadMain(&device), nullptr);
  EXPECT_EQ(asyncEventGetCalls, 1);
  EXPECT_EQ(asyncEventDecodeCalls, 1);
  EXPECT_EQ(asyncEventAckCalls, 0);

  ResetAsyncEventTestState();
  ibv_async_event ackFailureEvent = {};
  ackFailureEvent.event_type = IBV_EVENT_COMM_EST;
  queuedAsyncEvents.push_back(ackFailureEvent);
  failAsyncEventAck = true;
  EXPECT_EQ(ncclIbAsyncThreadMain(&device), nullptr);
  EXPECT_EQ(asyncEventGetCalls, 1);
  EXPECT_EQ(asyncEventDecodeCalls, 1);
  EXPECT_EQ(asyncEventAckCalls, 1);
  ResetAsyncEventTestState();
}

/** @brief Verifies fatal-count checking distinguishes clean and failed statistics. */
TEST(NetIbCommonTest, FatalCountCheckHonorsAsyncEventParam) {
  SCOPED_TRACE("Fatal-count checking must report a system error only after a fatal event.");
  ncclIbStats stats = {};

  EXPECT_EQ(ncclIbStatsCheckFatalCount(&stats, "test"), ncclSuccess);

  stats.fatalErrorCount = 1;
  EXPECT_EQ(ncclIbStatsCheckFatalCount(&stats, "test"), ncclSystemError);
}

/** @brief Verifies fatal-error helpers increment their associated counters. */
TEST(NetIbCommonTest, FatalErrorHelpersIncrementSharedCounters) {
  SCOPED_TRACE("Each fatal helper must update only the counter associated with its argument.");
  ncclIbStats stats = {};
  ibv_qp qp = {};
  ibv_cq cq = {};
  ncclIbDev dev = {};
  qp.qp_context = &stats;
  cq.cq_context = &stats;

  ASSERT_EQ(ncclIbStatsInit(&stats), ncclSuccess);
  ncclIbStatsFatalError(&stats);
  EXPECT_EQ(stats.fatalErrorCount, 1);

  ncclIbQpFatalError(&qp);
  EXPECT_EQ(stats.fatalErrorCount, 2);

  ncclIbCqFatalError(&cq);
  EXPECT_EQ(stats.fatalErrorCount, 3);

  ncclIbDevFatalError(&dev);
  EXPECT_EQ(dev.stats.fatalErrorCount, 1);
}
