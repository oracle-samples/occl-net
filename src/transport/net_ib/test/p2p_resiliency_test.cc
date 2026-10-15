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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Holds provider outcomes and observations for allocation, QP rejoin, replay, and
 * peer-read recovery.
 * @note Fields configure injected failures and capture provider interactions under test.
 */
struct ResiliencyFakes {
  ncclResult_t allocationResult = ncclSuccess;
  int allocationCalls = 0;
  size_t allocatedBytes = 0;
  ncclResult_t queryPortResult = ncclSuccess;
  enum ibv_port_state portState = IBV_PORT_ACTIVE;
  enum ibv_mtu portMtu = IBV_MTU_1024;
  int queryPortCalls = 0;
  struct ibv_context* queriedContext = nullptr;
  uint8_t queriedPort = 0;
  ncclResult_t modifyQpResult = ncclSuccess;
  struct ibv_qp* modifiedQp = nullptr;
  enum ibv_qp_state modifiedQpState = IBV_QPS_RESET;
  int modifyQpMask = 0;
  int modifyQpCalls = 0;
  ncclResult_t registerMrResult = ncclSuccess;
  int registerMrFailAt = -1;
  int registerMrCalls = 0;
  ncclResult_t deregisterMrResult = ncclSuccess;
  int deregisterMrFailAt = -1;
  int deregisterMrCalls = 0;
  struct ibv_mr* deregisteredMrs[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  ncclResult_t createCqResult = ncclSuccess;
  int createCqCalls = 0;
  int createdCqSize = 0;
  void* createdCqContext = nullptr;
  ncclResult_t destroyCqResult = ncclSuccess;
  int destroyCqCalls = 0;
  ncclResult_t createQpResult = ncclSuccess;
  int createQpCalls = 0;
  enum ibv_qp_type createdQpType = IBV_QPT_RC;
  unsigned int createdQpAccess = 0;
  uint32_t createdQpMaxRecv = 0;
  uint32_t createdQpMaxSend = 0;
  ncclResult_t initQpResult = ncclSuccess;
  int initQpCalls = 0;
  struct ibv_qp* initializedQp = nullptr;
  uint8_t initializedPort = 0;
  unsigned int initializedAccess = 0;
  ncclResult_t setEceResult = ncclSuccess;
  int setEceCalls = 0;
  struct ibv_qp* eceQp = nullptr;
  int eceSupported = 1;
  ncclResult_t rtrQpResult = ncclSuccess;
  int rtrQpCalls = 0;
  struct ibv_qp* rtrQp = nullptr;
  uint32_t rtrRemoteQpn = 0;
  enum ibv_mtu rtrLocalMtu = IBV_MTU_256;
  enum ibv_mtu rtrRemoteMtu = IBV_MTU_256;
  bool rtrFifoTc = false;
  ncclResult_t rtsQpResult = ncclSuccess;
  int rtsQpCalls = 0;
  struct ibv_qp* rtsQp = nullptr;
  ncclResult_t multiSendResult = ncclSuccess;
  int multiSendCalls = 0;
  struct ncclIbSendComm* multiSendComm = nullptr;
  int multiSendSlot = -1;
  ncclResult_t postFifoResult = ncclSuccess;
  int postFifoCalls = 0;
  struct ncclIbRecvComm* postFifoComm = nullptr;
  struct ncclIbRequest* postFifoRequest = nullptr;
  int postFifoSlot = -1;
  int refreshTokenCalls = 0;
  struct ncclIbSendComm* refreshTokenComm = nullptr;
  int refreshTokenSlot = -1;
  int refreshTokenCount = 0;
  int postSendError = 0;
  int postSendCalls = 0;
  struct ibv_qp* postedSendQp = nullptr;
  struct ibv_send_wr postedSend = {};
  struct ibv_sge postedSendSge = {};
  int postRecvError = 0;
  int postRecvCalls = 0;
  struct ibv_qp* postedRecvQp = nullptr;
  struct ibv_recv_wr* postedRecvWr = nullptr;
  struct ibv_wc polledCompletions[4] = {};
  int polledCompletionCount = 0;
  int polledCompletionIndex = 0;
  bool pollFails = false;
  ncclResult_t destroyQpResult = ncclSuccess;
  int destroyQpCalls = 0;
  char rejoinStages[8] = {};
  int rejoinStageCount = 0;
};

ResiliencyFakes fakes;
struct ibv_context fakeContext = {};
struct ibv_cq fakeCq = {};
struct ibv_pd fakePds[NCCL_IB_MAX_DEVS_PER_NIC] = {};
struct ibv_mr fakeMrs[8] = {};
struct ibv_qp fakeCreatedQps[NCCL_IB_MAX_DEVS_PER_NIC] = {};
struct ibv_qp probingVerbsQps[NCCL_IB_MAX_DEVS_PER_NIC] = {};
struct ibv_context probingVerbsContext = {};
int probingQpCreateCalls = 0;
int capturedMaxSendWorkRequest = 0;
struct ibv_qp* capturedProbeQp = nullptr;
uint32_t capturedProbeLkey = 0;
uint32_t capturedProbeRkey = 0;
int capturedProbePosts = 0;
int64_t testPortFailover = 1;
int64_t testPortFailoverMaxAttempts = 1;
int64_t testPortFailoverProbeDelay = 1000;
/** @brief Records one generated resiliency-parameter lookup. */
struct ResiliencyParameterLoadCall {
  std::string environment;
  int64_t defaultValue;
};
std::vector<ResiliencyParameterLoadCall> resiliencyParameterLoads;

/**
 * @brief Appends one provider-stage marker to the bounded QP-rejoin trace.
 * @param[in] stage Marker for RESET, INIT, ECE, RTR, RTS, or receive-queue refill.
 * @note Side effects: Appends stage to fakes.rejoinStages and advances fakes.rejoinStageCount by
 * one.
 */
void RecordRejoinStage(char stage) {
  ASSERT_NE(stage, '\0') << "A QP-rejoin provider stage must have a printable non-NUL marker";
  ASSERT_LT(fakes.rejoinStageCount, static_cast<int>(sizeof(fakes.rejoinStages) - 1))
      << "The QP-rejoin trace must fit in its fixed test buffer";
  fakes.rejoinStages[fakes.rejoinStageCount++] = stage;
  fakes.rejoinStages[fakes.rejoinStageCount] = '\0';
}

/**
 * @brief Records one fake send post and returns the configured provider result.
 * @param[in] qp NULL or a valid fake QP; recorded without dereferencing.
 * @param[in] wr NULL or a valid work request whose first SGE is copied when present.
 * @param[out] badWr NULL or valid storage receiving wr when the configured result is nonzero.
 * @return The configured provider result in [INT_MIN, INT_MAX].
 * @note Side effects: Records the call, QP, work request, first SGE, and optional failure output.
 */
int FakePostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr) {
  fakes.postSendCalls++;
  fakes.postedSendQp = qp;
  if (wr != nullptr) {
    fakes.postedSend = *wr;
    if (wr->sg_list != nullptr) fakes.postedSendSge = *wr->sg_list;
  }
  if (fakes.postSendError != 0 && badWr != nullptr) *badWr = wr;
  return fakes.postSendError;
}

/**
 * @brief Records one fake receive post and returns the configured provider result.
 * @param[in] qp NULL or a valid fake QP; recorded without dereferencing.
 * @param[in] wr NULL or a valid receive work request; recorded without dereferencing.
 * @param[out] badWr NULL or valid storage receiving wr when the configured result is nonzero.
 * @return The configured provider result in [INT_MIN, INT_MAX].
 * @note Side effects: Records the call and arguments, writes @p badWr on failure, and appends
 * the first receive-post marker to the rejoin trace.
 */
int FakePostRecv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr) {
  if (fakes.postRecvCalls == 0) RecordRejoinStage('Q');
  fakes.postRecvCalls++;
  fakes.postedRecvQp = qp;
  fakes.postedRecvWr = wr;
  if (fakes.postRecvError != 0 && badWr != nullptr) *badWr = wr;
  return fakes.postRecvError;
}

/**
 * @brief Copies configured fake completions into one CQ polling result.
 * @param[in] cq NULL or a valid fake CQ; ignored by this test double.
 * @param[in] count Maximum completion count in [0, INT_MAX].
 * @param[out] completions Valid storage for at least count entries when count is positive.
 * @return -1 for a configured poll failure or the copied count in [0, count].
 * @note Side effects: Copies queued completions to @p completions and advances the fake CQ
 * cursor unless polling is configured to fail.
 */
int FakePollCq(struct ibv_cq* cq, int count, struct ibv_wc* completions) {
  (void)cq;
  if (fakes.pollFails) return -1;
  int produced = 0;
  while (produced < count && fakes.polledCompletionIndex < fakes.polledCompletionCount) {
    completions[produced++] = fakes.polledCompletions[fakes.polledCompletionIndex++];
  }
  return produced;
}

/**
 * @brief Restores provider fakes, parameter controls, and backing verbs objects to deterministic
 * defaults.
 * @note Side effects: Replaces fake and production device state, resets parameter controls, and
 * initializes backing verbs objects.
 */
void ResetFakes() {
  fakes = ResiliencyFakes{};
  testPortFailover = 1;
  testPortFailoverMaxAttempts = 1;
  testPortFailoverProbeDelay = 1000;
  resiliencyParameterLoads.clear();
  std::memset(&fakeContext, 0, sizeof(fakeContext));
  std::memset(&fakeCq, 0, sizeof(fakeCq));
  std::memset(fakePds, 0, sizeof(fakePds));
  std::memset(fakeMrs, 0, sizeof(fakeMrs));
  std::memset(fakeCreatedQps, 0, sizeof(fakeCreatedQps));
  std::memset(probingVerbsQps, 0, sizeof(probingVerbsQps));
  std::memset(&probingVerbsContext, 0, sizeof(probingVerbsContext));
  probingQpCreateCalls = 0;
  capturedMaxSendWorkRequest = 0;
  capturedProbeQp = nullptr;
  capturedProbeLkey = 0;
  capturedProbeRkey = 0;
  capturedProbePosts = 0;
  fakeContext.ops.post_send = FakePostSend;
  fakeContext.ops.post_recv = FakePostRecv;
  fakeContext.ops.poll_cq = FakePollCq;
  fakeCq.context = &fakeContext;
  for (int index = 0; index < 8; index++) {
    fakeMrs[index].lkey = 0x100u + static_cast<uint32_t>(index);
    fakeMrs[index].rkey = 0x200u + static_cast<uint32_t>(index);
  }
  for (int index = 0; index < NCCL_IB_MAX_DEVS_PER_NIC; index++) {
    fakeCreatedQps[index].context = &fakeContext;
    fakeCreatedQps[index].qp_num = 500u + static_cast<uint32_t>(index);
  }
  for (int index = 0; index < MAX_IB_DEVS; index++) {
    ncclIbDevs[index].context = nullptr;
    ncclIbDevs[index].portNum = 0;
    std::memset(&ncclIbDevs[index].portAttr, 0, sizeof(ncclIbDevs[index].portAttr));
  }
}

/**
 * @brief Captures one auxiliary probe post and its local and remote keys.
 * @param[in] qp Valid non-NULL probing QP.
 * @param[in] wr Valid non-NULL RDMA work request with one SGE.
 * @param[out] badWr NULL or output storage left unchanged on success.
 * @return Zero.
 * @note Side effects: Records the QP and keys and increments the probe-post count.
 */
int CaptureProbePostSend(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr) {
  (void)badWr;
  capturedProbeQp = qp;
  capturedProbeLkey = wr->sg_list->lkey;
  capturedProbeRkey = wr->wr.rdma.rkey;
  capturedProbePosts++;
  return 0;
}

/**
 * @brief Resets the independent auxiliary-probe capture context.
 * @note Side effects: Clears probe observations and installs the capture callback.
 */
void ResetProbePostCapture() {
  std::memset(&probingVerbsContext, 0, sizeof(probingVerbsContext));
  probingVerbsContext.ops.post_send = CaptureProbePostSend;
  capturedProbeQp = nullptr;
  capturedProbeLkey = 0;
  capturedProbeRkey = 0;
  capturedProbePosts = 0;
}

/**
 * @brief Resolves resiliency parameters through the production-generated accessors.
 * @param[in] environment Exact environment-variable name requested by the accessor.
 * @param[in] defaultValue Production default for the requested parameter.
 * @param[in] uninitialized Production cache sentinel, accepted without modification.
 * @param[in] cache Production cache storage left unmodified to keep scenarios independent.
 * @param[out] noCache Receives one so every scenario reaches the dependency seam.
 * @return The configured value selected by environment, or defaultValue for an unknown key.
 * @note Side effects: Records the parameter name and default value.
 */
int64_t LoadResiliencyParameterForTest(const char* environment, int64_t defaultValue,
    int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  (void)uninitialized;
  (void)cache;
  resiliencyParameterLoads.push_back({environment, defaultValue});
  *noCache = 1;
  if (std::strcmp(environment, "NCCL_IB_RESILIENCY_PORT_FAILOVER") == 0) return testPortFailover;
  if (std::strcmp(environment, "NCCL_IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS") == 0) {
    return testPortFailoverMaxAttempts;
  }
  if (std::strcmp(environment, "NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY") == 0) {
    return testPortFailoverProbeDelay;
  }
  return defaultValue;
}

/**
 * @brief Allocates zeroed resiliency-test storage unless allocation failure is configured.
 * @param[out] pointer Valid non-NULL storage receiving a heap allocation on success and remaining
 * unchanged on error.
 * @param[in] bytes Requested allocation size in [1, SIZE_MAX].
 * @return ncclSuccess, the configured allocation error, or ncclSystemError when calloc fails.
 * @note Side effects: Records the call and size and allocates memory freed by the production
 * destroy path.
 */
ncclResult_t AllocateResiliencyForTest(void** pointer, size_t bytes) {
  fakes.allocationCalls++;
  fakes.allocatedBytes = bytes;
  if (fakes.allocationResult != ncclSuccess) return fakes.allocationResult;
  void* allocation = std::calloc(1, bytes);
  if (allocation == nullptr) return ncclSystemError;
  *pointer = allocation;
  return ncclSuccess;
}

}  // namespace

const char* ncclIbReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut", "IGet"};

/**
 * @brief Supplies the default partition key expected by the resiliency unit under test.
 * @return 0.
 */
int64_t ncclParamIbPkey() {
  return 0;
}

/**
 * @brief Supplies the default QP retry count expected by the resiliency unit under test.
 * @return 7.
 */
int64_t ncclParamIbRetryCnt() {
  return 7;
}

/**
 * @brief Supplies the default QP timeout expected by the resiliency unit under test.
 * @return 20.
 */
int64_t ncclParamIbTimeout() {
  return 20;
}

/**
 * @brief Disables a dedicated FIFO traffic class in the resiliency unit under test.
 * @return -1.
 */
int64_t ncclParamIbFifoTc() {
  return -1;
}

/**
 * @brief Models successful port-recovery context initialization.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoveryInit(struct ncclIbResiliency* context) {
  (void)context;
  return ncclSuccess;
}

/**
 * @brief Models successful port-recovery context closure.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoveryClose(struct ncclIbResiliency* context) {
  (void)context;
  return ncclSuccess;
}

/**
 * @brief Models successful port-recovery initialization for one device.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @param[in] devIndex Device index in [INT_MIN, INT_MAX], ignored by this test double.
 * @param[in] device NULL or a valid device; ignored by this test double.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoveryDevInit(struct ncclIbResiliency* context, int devIndex, ncclIbDev* device) {
  (void)context;
  (void)devIndex;
  (void)device;
  return ncclSuccess;
}

/**
 * @brief Models successful port-recovery destruction for one device.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @param[in] devIndex Device index in [INT_MIN, INT_MAX], ignored by this test double.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoveryDevDestroy(struct ncclIbResiliency* context, int devIndex) {
  (void)context;
  (void)devIndex;
  return ncclSuccess;
}

/**
 * @brief Models successful creation of sender port-recovery QPs.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @param[in] qpInfo NULL or valid QP information storage; left unchanged.
 * @param[in] count QP count in [INT_MIN, INT_MAX], ignored by this test double.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoverySenderQpsCreate(struct ncclIbResiliency* context, struct ncclIbQpInfo* qpInfo, int count) {
  (void)context;
  (void)qpInfo;
  (void)count;
  return ncclSuccess;
}

/**
 * @brief Models successful sender port-recovery QP transitions to RTS.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @param[in] metadata NULL or valid remote connection metadata; ignored by this test double.
 * @param[in] count QP count in [INT_MIN, INT_MAX], ignored by this test double.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoverySenderQpsToRts(struct ncclIbResiliency* context, struct ncclIbConnectionMetadata* metadata, int count) {
  (void)context;
  (void)metadata;
  (void)count;
  return ncclSuccess;
}

/**
 * @brief Models successful receiver port-recovery QP creation and RTS transitions.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @param[in] metadata NULL or valid remote connection metadata; ignored by this test double.
 * @param[in] qpInfo NULL or valid QP information storage; left unchanged.
 * @param[in] count QP count in [INT_MIN, INT_MAX], ignored by this test double.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoveryReceiverQpsCreateToRts(struct ncclIbResiliency* context, struct ncclIbConnectionMetadata* metadata, struct ncclIbQpInfo* qpInfo, int count) {
  (void)context;
  (void)metadata;
  (void)qpInfo;
  (void)count;
  return ncclSuccess;
}

/**
 * @brief Models successful destruction of port-recovery QPs.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @param[in] count QP count in [INT_MIN, INT_MAX], ignored by this test double.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoveryQpsDestroy(struct ncclIbResiliency* context, int count) {
  (void)context;
  (void)count;
  return ncclSuccess;
}

/**
 * @brief Models successful handling of one port-recovery failure.
 * @param[in] context NULL or a valid resiliency context; left unchanged.
 * @param[in] devIndex Device index in [INT_MIN, INT_MAX], ignored by this test double.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPortRecoveryHandleFailure(struct ncclIbResiliency* context, int devIndex) {
  (void)context;
  (void)devIndex;
  return ncclSuccess;
}

/**
 * @brief Models successful batch posting of receive work requests.
 * @param[in] comm NULL or a valid receive communicator; left unchanged.
 * @param[in] qp NULL or a valid QP; left unchanged.
 * @return ncclSuccess.
 */
ncclResult_t ncclIbPostReceiveWorkRequestsOnQp(struct ncclIbRecvComm* comm, ncclIbQp* qp) {
  (void)comm;
  (void)qp;
  return ncclSuccess;
}

/**
 * @brief Records one fake multi-send replay attempt.
 * @param[in] comm NULL or a valid send communicator; recorded without dereferencing.
 * @param[in] slot Request slot in [INT_MIN, INT_MAX], recorded without validation.
 * @return The configured replay result.
 * @note Side effects: Increments the replay count and records @p comm and @p slot.
 */
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  fakes.multiSendCalls++;
  fakes.multiSendComm = comm;
  fakes.multiSendSlot = slot;
  return fakes.multiSendResult;
}

/**
 * @brief Records one fake receive FIFO repost attempt.
 * @param[in] comm NULL or a valid receive communicator; recorded without dereferencing.
 * @param[in] request NULL or a valid request; recorded without dereferencing.
 * @param[in] slot Request slot in [INT_MIN, INT_MAX], recorded without validation.
 * @return The configured FIFO-post result.
 * @note Side effects: Increments the FIFO-post count and records all arguments.
 */
ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* request, int slot) {
  fakes.postFifoCalls++;
  fakes.postFifoComm = comm;
  fakes.postFifoRequest = request;
  fakes.postFifoSlot = slot;
  return fakes.postFifoResult;
}

/**
 * @brief Records one fake refresh of send-request WR-ID tokens.
 * @param[in] comm NULL or a valid send communicator; recorded without dereferencing.
 * @param[in] slot Request slot in [INT_MIN, INT_MAX], recorded without validation.
 * @param[in] count Request count in [INT_MIN, INT_MAX], recorded without validation.
 * @note Side effects: Increments the refresh count and records all arguments.
 */
void ncclIbSendReqsRefreshWrIdToken(struct ncclIbSendComm* comm, int slot, int count) {
  fakes.refreshTokenCalls++;
  fakes.refreshTokenComm = comm;
  fakes.refreshTokenSlot = slot;
  fakes.refreshTokenCount = count;
}

/**
 * @brief Increments one request event counter for fake replay accounting.
 * @param[in,out] req Valid non-NULL request whose selected event is representable after increment.
 * @param[in] devIndex Device index in [0, NCCL_IB_MAX_DEVS_PER_NIC).
 * @note Side effects: Increments the selected request event counter.
 */
void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex) {
  req->events[devIndex]++;
}

/**
 * @brief Selects the send-side or receive-side device base for a communicator.
 * @param[in] base Valid non-NULL base embedded in the matching send or receive communicator.
 * @param[in] devIndex Device index in [0, base->vProps.ndevs).
 * @return The selected communicator-device base.
 */
ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) return &reinterpret_cast<ncclIbSendComm*>(base)->devs[devIndex].base;
  return &reinterpret_cast<ncclIbRecvComm*>(base)->devs[devIndex].base;
}

/**
 * @brief Returns the configured fake port state and MTU.
 * @param[in] context NULL or a valid verbs context; recorded without dereferencing.
 * @param[in] port Port number in [0, UINT8_MAX], recorded without validation.
 * @param[out] portAttr Valid non-NULL storage receiving state and MTU on success.
 * @return The configured query result.
 * @note Side effects: Records the call and inputs and writes @p portAttr on success.
 */
ncclResult_t wrap_ibv_query_port(struct ibv_context* context, uint8_t port, struct ibv_port_attr* portAttr) {
  fakes.queryPortCalls++;
  fakes.queriedContext = context;
  fakes.queriedPort = port;
  if (fakes.queryPortResult != ncclSuccess) return fakes.queryPortResult;
  portAttr->state = fakes.portState;
  portAttr->active_mtu = fakes.portMtu;
  return fakes.queryPortResult;
}

/**
 * @brief Returns one indexed fake memory registration unless failure is configured.
 * @param[out] result Valid non-NULL storage receiving a fake MR on success.
 * @param[in] pd NULL or a valid protection domain; ignored by this test double.
 * @param[in] address NULL or a valid region start; ignored by this test double.
 * @param[in] size Region size in [0, SIZE_MAX], ignored by this test double.
 * @param[in] access Access flags in [INT_MIN, INT_MAX], ignored by this test double.
 * @return ncclSuccess or ncclSystemError.
 * @note Side effects: Increments the registration count and writes @p result on success.
 */
ncclResult_t wrap_ibv_reg_mr(struct ibv_mr** result, struct ibv_pd* pd, void* address, size_t size, int access) {
  (void)pd;
  (void)address;
  (void)size;
  (void)access;
  int call = fakes.registerMrCalls++;
  if (call == fakes.registerMrFailAt || fakes.registerMrResult != ncclSuccess) return ncclSystemError;
  *result = &fakeMrs[call];
  return ncclSuccess;
}

/**
 * @brief Records one fake memory deregistration.
 * @param[in] mr NULL or a fake MR returned by wrap_ibv_reg_mr.
 * @return ncclSystemError on the selected failing call, otherwise the configured result.
 * @note Side effects: Increments the deregistration count and records up to four MR pointers.
 */
ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr* mr) {
  int call = fakes.deregisterMrCalls++;
  if (call < 4) fakes.deregisteredMrs[call] = mr;
  if (call == fakes.deregisterMrFailAt) return ncclSystemError;
  return fakes.deregisterMrResult;
}

/**
 * @brief Records CQ creation arguments and returns the shared fake CQ.
 * @param[out] result Valid non-NULL storage receiving the fake CQ on success.
 * @param[in] context NULL or a valid fake verbs context stored in the CQ.
 * @param[in] size Requested CQ size in [INT_MIN, INT_MAX].
 * @param[in] cqContext NULL or an opaque caller context recorded for verification.
 * @param[in] channel NULL or a valid completion channel; ignored by this test double.
 * @param[in] vector Completion vector in [INT_MIN, INT_MAX], ignored by this test double.
 * @return The configured CQ-creation result.
 * @note Side effects: Records the call and selected inputs and, on success, updates the fake CQ
 * and writes @p result.
 */
ncclResult_t wrap_ibv_create_cq(struct ibv_cq** result, struct ibv_context* context, int size, void* cqContext, struct ibv_comp_channel* channel, int vector) {
  (void)channel;
  (void)vector;
  fakes.createCqCalls++;
  fakes.createdCqSize = size;
  fakes.createdCqContext = cqContext;
  if (fakes.createCqResult != ncclSuccess) return fakes.createCqResult;
  fakeCq.context = context;
  *result = &fakeCq;
  return ncclSuccess;
}

/**
 * @brief Records one fake CQ destruction.
 * @param[in] cq NULL or the shared fake CQ; ignored by this test double.
 * @return The configured CQ-destruction result.
 * @note Side effects: Increments the CQ-destruction count.
 */
ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq* cq) {
  (void)cq;
  fakes.destroyCqCalls++;
  return fakes.destroyCqResult;
}

/**
 * @brief Provides a successful no-op for the unused direct verbs QP creation path.
 * @param[in] result NULL or valid QP-pointer storage; left unchanged.
 * @param[in] pd NULL or a valid protection domain; ignored by this test double.
 * @param[in] attributes NULL or valid QP attributes; ignored by this test double.
 * @return ncclSuccess.
 */
ncclResult_t wrap_ibv_create_qp(struct ibv_qp** result, struct ibv_pd* pd, struct ibv_qp_init_attr* attributes) {
  (void)result;
  (void)pd;
  (void)attributes;
  return ncclSuccess;
}

/**
 * @brief Records one fake QP destruction.
 * @param[in] qp NULL or a fake QP; ignored by this test double.
 * @return The configured QP-destruction result.
 * @note Side effects: Increments the QP-destruction count.
 */
ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp* qp) {
  (void)qp;
  fakes.destroyQpCalls++;
  return fakes.destroyQpResult;
}

/**
 * @brief Records one fake ECE application.
 * @param[in] qp NULL or a fake QP; recorded without dereferencing.
 * @param[in] ece NULL or valid ECE data; ignored by this test double.
 * @param[out] supported NULL or valid storage receiving the configured support result.
 * @return The configured ECE result.
 * @note Side effects: Appends an ECE-stage marker, records the call and QP, and may write
 * @p supported.
 */
ncclResult_t wrap_ibv_set_ece(struct ibv_qp* qp, struct ibv_ece* ece, int* supported) {
  (void)ece;
  RecordRejoinStage('E');
  fakes.setEceCalls++;
  fakes.eceQp = qp;
  if (supported != nullptr) *supported = fakes.eceSupported;
  return fakes.setEceResult;
}

/**
 * @brief Records the requested fake QP state transition.
 * @param[in] qp NULL or a fake QP; recorded without dereferencing.
 * @param[in] attr Valid non-NULL attributes whose QP state is recorded.
 * @param[in] mask Attribute mask in [INT_MIN, INT_MAX].
 * @return The configured modify-QP result.
 * @note Side effects: Appends a RESET-stage marker and records the call and arguments.
 */
ncclResult_t wrap_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int mask) {
  RecordRejoinStage('R');
  fakes.modifyQpCalls++;
  fakes.modifiedQp = qp;
  fakes.modifiedQpState = attr->qp_state;
  fakes.modifyQpMask = mask;
  return fakes.modifyQpResult;
}

/**
 * @brief Records one fake transition of a QP to INIT.
 * @param[in] qp NULL or a fake QP; recorded without dereferencing.
 * @param[in] port Port number in [0, UINT8_MAX], recorded without validation.
 * @param[in] access Access flags in [0, UINT_MAX], recorded without validation.
 * @return The configured INIT result.
 * @note Side effects: Appends an INIT-stage marker and records the call and arguments.
 */
ncclResult_t ibQpToInit(struct ibv_qp* qp, uint8_t port, unsigned int access) {
  RecordRejoinStage('I');
  fakes.initQpCalls++;
  fakes.initializedQp = qp;
  fakes.initializedPort = port;
  fakes.initializedAccess = access;
  return fakes.initQpResult;
}

/**
 * @brief Records one fake transition of a QP to RTR.
 * @param[in] qp NULL or a fake QP; recorded without dereferencing.
 * @param[in] gid NULL or valid local GID information; ignored by this test double.
 * @param[in] remoteQpn Remote QP number in [0, UINT32_MAX].
 * @param[in] remoteDevice Valid non-NULL remote device information whose MTU is recorded.
 * @param[in] localMtu Active MTU selected from the local physical port.
 * @param[in] fifoTc Whether FIFO traffic-class handling is requested.
 * @param[in] trafficClass Traffic class in [INT_MIN, INT_MAX], ignored by this test double.
 * @param[in] serviceLevel Service level in [INT_MIN, INT_MAX], ignored by this test double.
 * @return The configured RTR result.
 * @note Side effects: Appends an RTR-stage marker and records selected arguments.
 */
ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, struct ncclIbGidInfo* gid,
    uint32_t remoteQpn, struct ncclIbDevInfo* remoteDevice,
    enum ibv_mtu localMtu, bool fifoTc, int trafficClass, int serviceLevel) {
  (void)gid;
  (void)trafficClass;
  (void)serviceLevel;
  RecordRejoinStage('T');
  fakes.rtrQpCalls++;
  fakes.rtrQp = qp;
  fakes.rtrRemoteQpn = remoteQpn;
  fakes.rtrLocalMtu = localMtu;
  fakes.rtrRemoteMtu = remoteDevice->mtu;
  fakes.rtrFifoTc = fifoTc;
  return fakes.rtrQpResult;
}

/**
 * @brief Records one fake transition of a QP to RTS.
 * @param[in] qp NULL or a fake QP; recorded without dereferencing.
 * @return The configured RTS result.
 * @note Side effects: Appends an RTS-stage marker and records the call and QP.
 */
ncclResult_t ncclIbRtsQp(struct ibv_qp* qp) {
  RecordRejoinStage('S');
  fakes.rtsQpCalls++;
  fakes.rtsQp = qp;
  return fakes.rtsQpResult;
}

/**
 * @brief Compiles p2p_resiliency.cc with parameter loading and allocation redirected to fake
 * provider state.
 */
#define ncclLoadParam LoadResiliencyParameterForTest
#undef ncclIbMalloc
#define ncclIbMalloc(pointer, bytes) AllocateResiliencyForTest((pointer), (bytes))
#include "../p2p_resiliency.cc"  // NOLINT(build/include)
#undef ncclIbMalloc
#undef ncclLoadParam

/**
 * @brief Records QP creation attributes and returns one indexed fake QP.
 * @param[in] attributes Valid non-NULL QP creation attributes.
 * @param[in] context NULL or an opaque caller context; ignored by this test double.
 * @param[out] qp Valid non-NULL QP storage receiving the fake verbs QP on success.
 * @return The configured QP-creation result.
 * @note Side effects: Records the call and attributes and writes @p qp on success.
 */
ncclResult_t ncclIbCreateQp(struct ncclIbQpCreateAttr* attributes, void* context, struct ncclIbQp* qp) {
  (void)context;
  int call = fakes.createQpCalls++;
  probingQpCreateCalls++;
  capturedMaxSendWorkRequest = attributes->maxSendWorkRequest;
  fakes.createdQpType = attributes->type;
  fakes.createdQpAccess = attributes->accessFlags;
  fakes.createdQpMaxRecv = attributes->maxRecvWorkRequest;
  fakes.createdQpMaxSend = attributes->maxSendWorkRequest;
  if (fakes.createQpResult != ncclSuccess) return fakes.createQpResult;
  qp->qp = &fakeCreatedQps[call];
  return ncclSuccess;
}

namespace {

constexpr uint64_t kRemoteCompletionAddress = 0x10000;
constexpr uint32_t kRemoteCompletionKey = 0x44;

/**
 * @brief Verifies two resiliency accessors retain their keys, defaults, and configured values.
 */
TEST(P2pResiliencyDirectTest, LoadsFailoverParameters) {
  SCOPED_TRACE("The two generated accessors must preserve metadata and expose seam results.");
  ResetFakes();
  EXPECT_EQ(ncclParamIbResiliencyPortFailover(), 1)
      << "Port failover must retain its enabled production default";
  EXPECT_EQ(ncclParamIbResiliencyPortFailoverProbeDelay(), 1000)
      << "Probe delay must retain its one-second production default";
  testPortFailover = 0;
  testPortFailoverProbeDelay = 25;
  EXPECT_EQ(ncclParamIbResiliencyPortFailover(), 0)
      << "Port failover must expose a configured disabled value";
  EXPECT_EQ(ncclParamIbResiliencyPortFailoverProbeDelay(), 25)
      << "Probe delay must expose a configured positive value";

  ASSERT_EQ(resiliencyParameterLoads.size(), 4u)
      << "No-cache scenarios must invoke the loader for every direct call";
  EXPECT_EQ(resiliencyParameterLoads[0].environment, "NCCL_IB_RESILIENCY_PORT_FAILOVER")
      << "The failover accessor must request its exact production environment key";
  EXPECT_EQ(resiliencyParameterLoads[0].defaultValue, 1)
      << "The failover accessor must pass default one to the loader";
  EXPECT_EQ(resiliencyParameterLoads[1].environment,
      "NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY")
      << "The probe-delay accessor must request its exact production environment key";
  EXPECT_EQ(resiliencyParameterLoads[1].defaultValue, 1000)
      << "The probe-delay accessor must pass default one thousand to the loader";
  testPortFailover = 1;
  testPortFailoverProbeDelay = 1000;
}

/**
 * @brief Creates a zeroed sender or receiver communicator and its attached resiliency context.
 * @tparam IsSend true for sender types or false for receiver types.
 * @param[in] nqps QP count in [0, NCCL_IB_MAX_QPS].
 * @param[in] nDataQps data-QP count in [0, nqps].
 * @param[in] ndevs local device count in [0, NCCL_IB_MAX_DEVS_PER_NIC].
 * @return Owning communicator and matching context with healthy state and cross-links initialized.
 * @note Side effects: Allocates both returned objects on the heap.
 */
template <bool IsSend>
auto MakeState(int nqps, int nDataQps, int ndevs) {
  assert(nqps >= 0 && nqps <= NCCL_IB_MAX_QPS);
  assert(nDataQps >= 0 && nDataQps <= nqps);
  assert(ndevs >= 0 && ndevs <= NCCL_IB_MAX_DEVS_PER_NIC);
  using Comm = std::conditional_t<IsSend, ncclIbSendComm, ncclIbRecvComm>;
  using Context = std::conditional_t<IsSend, ncclIbResiliencySend, ncclIbResiliency>;
  auto comm = std::make_unique<Comm>();
  auto context = std::make_unique<Context>();
  ncclIbResiliency* baseContext;
  if constexpr (IsSend) baseContext = &context->base;
  else baseContext = context.get();
  comm->base.isSend = IsSend;
  comm->base.nqps = nqps;
  comm->base.nDataQps = nDataQps;
  comm->base.nPhysicalQps = nqps;
  comm->base.hasStandbyPaths = false;
  comm->base.vProps.ndevs = ndevs;
  comm->base.resiliency = baseContext;
  baseContext->baseComm = &comm->base;
  baseContext->ndevs = ndevs;
  baseContext->frmAllQpsHealthy = true;
  return std::make_pair(std::move(comm), std::move(context));
}

/**
 * @brief Reinitializes one base resiliency context while preserving its communicator.
 * @param[in,out] base Valid non-NULL communicator linked to the reset context.
 * @param[out] context Valid non-NULL context storage replaced with zeroes and supplied dimensions.
 * @param[in] ndevs Local device count in [0, NCCL_IB_MAX_DEVS_PER_NIC].
 * @note Side effects: Replaces the complete context and base->resiliency link.
 */
void ResetContext(ncclIbNetCommBase* base, ncclIbResiliency* context, int ndevs) {
  assert(base != nullptr);
  assert(context != nullptr);
  assert(ndevs >= 0 && ndevs <= NCCL_IB_MAX_DEVS_PER_NIC);
  *context = {};
  context->baseComm = base;
  context->ndevs = ndevs;
  base->resiliency = context;
}

/**
 * @brief Connects fake verbs QPs to logical QPs using one explicit device index per logical QP.
 * @param[in,out] base Valid non-NULL communicator with nqps equal to deviceIndices.size().
 * @param[in,out] context Valid non-NULL attached context whose configured QPs become healthy.
 * @param[in] deviceIndices Device indices in [0, context->ndevs), ordered by logical QP index.
 * @note Side effects: Replaces configured QP handles, device mappings, active-QP entries, and QP
 * states.
 */
void ConfigureActiveQps(
    ncclIbNetCommBase* base, ncclIbResiliency* context, std::initializer_list<int> deviceIndices) {
  assert(base != nullptr);
  assert(context != nullptr);
  assert(static_cast<int>(deviceIndices.size()) == base->nqps);
  int qpIndex = 0;
  for (int devIndex : deviceIndices) {
    assert(devIndex >= 0 && devIndex < context->ndevs);
    base->qps[qpIndex].qp = &fakeCreatedQps[qpIndex];
    base->qps[qpIndex].devIndex = devIndex;
    base->qps[qpIndex].remDevIdx = devIndex;
    base->qps[qpIndex].coldIdx = qpIndex;
    base->qps[qpIndex].logicalLane = qpIndex;
    base->activeQps[qpIndex] = &base->qps[qpIndex];
    context->qpInfo[qpIndex].state = ncclIbResiliencyQpStateOk;
    qpIndex++;
  }
  base->nPhysicalQps = base->nqps;
  base->hasStandbyPaths = false;
}

/**
 * @brief Restores every configured logical QP to its original physical QP.
 * @param[in,out] base Valid non-NULL communicator with nqps in [0, NCCL_IB_MAX_QPS].
 * @note Side effects: Replaces activeQps entries in [0, base->nqps).
 */
void ResetActiveQps(ncclIbNetCommBase* base) {
  assert(base != nullptr);
  for (int qpIndex = 0; qpIndex < base->nqps; qpIndex++) base->activeQps[qpIndex] = &base->qps[qpIndex];
}

/**
 * @brief Sets both local and remote readiness for selected QPs.
 * @param[in,out] context Valid non-NULL context whose ready masks are updated.
 * @param[in] qpIndices QP indices in [0, NCCL_IB_MAX_QPS).
 * @param[in] ready true to set both bits or false to clear both bits.
 * @note Side effects: Updates localReady.mask and remoteReady.mask for every selected QP.
 */
void SetQpsReady(
    ncclIbResiliency* context, std::initializer_list<int> qpIndices, bool ready) {
  assert(context != nullptr);
  for (int qpIndex : qpIndices) {
    ncclIbResiliencyReadyQpSetBit(context->localReady.mask, qpIndex, ready);
    ncclIbResiliencyReadyQpSetBit(context->remoteReady.mask, qpIndex, ready);
  }
}

/**
 * @brief Configures one sender device as a working probe path to fake remote completion records.
 * @param[in,out] context Valid non-NULL sender context with devIndex in [0, context->base.ndevs).
 * @param[in] devIndex Local probe device index in [0, context->base.ndevs).
 * @param[in] remDevIndex Remote probe device index in [0, NCCL_IB_MAX_DEVS_PER_NIC).
 * @note Side effects: Installs fake MR, CQ, QP, path counts, remote record metadata, and the
 * initial probe WR-ID token.
 */
void ConfigureProbePath(
    ncclIbResiliencySend* context, int devIndex, int remDevIndex) {
  assert(context != nullptr);
  assert(devIndex >= 0 && devIndex < context->base.ndevs);
  assert(remDevIndex >= 0 && remDevIndex < NCCL_IB_MAX_DEVS_PER_NIC);
  context->base.devs[devIndex].state = ncclIbResiliencyDevStateOk;
  context->base.devs[devIndex].probingResultMr = &fakeMrs[0];
  context->base.devs[devIndex].probingCq = &fakeCq;
  context->base.probingQps[devIndex].qp = &fakeCreatedQps[devIndex];
  context->base.probingQps[devIndex].devIndex = devIndex;
  context->base.probingQps[devIndex].remDevIdx = remDevIndex;
  context->base.nProbingQps = std::max(context->base.nProbingQps, devIndex + 1);
  context->base.baseComm->nRemDevs =
      std::max(context->base.baseComm->nRemDevs, remDevIndex + 1);
  context->remCmplRecordsInfo[remDevIndex].addr = kRemoteCompletionAddress;
  context->remCmplRecordsInfo[remDevIndex].rkey = kRemoteCompletionKey;
  context->nextProbeWrId = IB_RESILIENCY_PROBE_WR_ID_BASE;
}

/**
 * @brief Creates one zeroed request with its common identity fields initialized.
 * @param[in] base Valid non-NULL communicator stored in the request.
 * @param[in] id Request generation ID in [0, INT64_MAX].
 * @param[in] type Request type in [NCCL_NET_IB_REQ_UNUSED, NCCL_NET_IB_REQ_GIN_IPUT].
 * @param[in] nreqs Group size in [0, NCCL_NET_IB_MAX_RECVS].
 * @return Initialized request value with every scenario-specific field left zero.
 */
ncclIbRequest MakeRequest(ncclIbNetCommBase* base, int64_t id, int type, int nreqs) {
  assert(base != nullptr);
  assert(id >= 0);
  assert(nreqs >= 0 && nreqs <= NCCL_NET_IB_MAX_RECVS);
  ncclIbRequest request = {};
  request.base = base;
  request.id = id;
  request.type = type;
  request.nreqs = nreqs;
  return request;
}

/**
 * @brief Binds one request to its sender slot and selected group entry.
 * @param[in,out] comm Valid non-NULL sender communicator.
 * @param[in] request Valid non-NULL request whose nonnegative ID selects the slot.
 * @param[in] groupIndex Group index in [0, NCCL_NET_IB_MAX_RECVS).
 * @note Side effects: Replaces one sendReqs table entry.
 */
void BindSendRequest(ncclIbSendComm* comm, ncclIbRequest* request, int groupIndex) {
  assert(comm != nullptr);
  assert(request != nullptr);
  assert(groupIndex >= 0 && groupIndex < NCCL_NET_IB_MAX_RECVS);
  comm->sendReqs[request->id % NET_IB_MAX_REQUESTS][groupIndex] = request;
}

/**
 * @brief Attaches one request to its failed-send tracking slot in the requested state.
 * @param[in,out] context Valid non-NULL sender resiliency context.
 * @param[in] request Valid non-NULL request whose nonnegative ID selects the tracking slot.
 * @param[in] state Tracking state in [ncclIbResiliencyRequestStatePending,
 *   ncclIbResiliencyRequestStateReplayPosted].
 * @return Pointer to the initialized tracking entry.
 * @note Side effects: Replaces the entry's state, request pointer, generation guard, and probe
 * token.
 */
ncclIbResiliencyRequestSend* TrackFailedRequest(
    ncclIbResiliencySend* context, ncclIbRequest* request, ncclIbResiliencyRequestSendState state) {
  assert(context != nullptr);
  assert(request != nullptr);
  auto* failed = &context->failedRequests[request->id % NET_IB_MAX_REQUESTS];
  failed->state = state;
  failed->request = request;
  failed->id = request->id + 1;
  failed->probeWrId = 0;
  return failed;
}

/**
 * @brief Checks recorded refresh and multi-send observations for one grouped sender replay.
 * @param[in] comm Valid non-NULL sender communicator expected by both replay test doubles.
 * @param[in] slot Request slot in [0, NET_IB_MAX_REQUESTS).
 * @param[in] groupSize Grouped request count in [1, NCCL_NET_IB_MAX_RECVS].
 * @param[in] callCount Expected refresh and multi-send call count in [1, INT_MAX].
 * @note Side effects: Records nonfatal GoogleTest failures when replay observations differ.
 */
void ExpectReplayPosts(
    ncclIbSendComm* comm, int slot, int groupSize, int callCount) {
  SCOPED_TRACE(testing::Message()
      << "Replay observations for slot=" << slot << ", groupSize=" << groupSize
      << ", expectedCalls=" << callCount);
  EXPECT_EQ(fakes.refreshTokenCalls, callCount);
  EXPECT_EQ(fakes.refreshTokenComm, comm);
  EXPECT_EQ(fakes.refreshTokenSlot, slot);
  EXPECT_EQ(fakes.refreshTokenCount, groupSize);
  EXPECT_EQ(fakes.multiSendCalls, callCount);
  EXPECT_EQ(fakes.multiSendComm, comm);
  EXPECT_EQ(fakes.multiSendSlot, slot);
}

/**
 * @brief Resets all fakes and prepares one receiver QP for an immediate rejoin attempt.
 * @param[out] comm Valid non-NULL receive communicator storage replaced with a one-QP
 * configuration.
 * @param[out] context Valid non-NULL resiliency storage attached to comm and marked in progress.
 * @param[in] prepostReceives true to refill the receive queue after RTS or false to skip refill.
 * @note Side effects: Resets all test fakes, removes the probe delay, and replaces comm and
 * context.
 */
void ResetRejoinScenario(
    ncclIbRecvComm* comm, ncclIbResiliency* context, bool prepostReceives) {
  ResetFakes();
  testPortFailoverProbeDelay = 0;
  *comm = {};
  *context = {};
  comm->base.isSend = false;
  comm->base.nqps = 1;
  comm->base.nDataQps = 1;
  comm->base.nPhysicalQps = 1;
  comm->base.hasStandbyPaths = false;
  comm->base.vProps.ndevs = 1;
  comm->base.nRemDevs = 1;
  comm->base.qps[0].qp = &fakeCreatedQps[0];
  comm->base.qps[0].devIndex = 0;
  comm->base.qps[0].remDevIdx = 0;
  comm->base.qps[0].coldIdx = 0;
  comm->base.qps[0].logicalLane = 0;
  comm->base.qpCold[0].remQpn = 900;
  comm->base.qpCold[0].remEceSupported = 1;
  comm->base.qpCold[0].ibPort = 1;
  comm->base.activeQps[0] = &comm->base.qps[0];
  comm->base.remDevs[0].mtu = IBV_MTU_4096;
  comm->devs[0].base.ibDevN = 0;
  comm->prepostReceiveWorkRequests = prepostReceives;
  ncclIbDevs[0].context = &fakeContext;
  ncclIbDevs[0].portNum = 1;
  context->baseComm = &comm->base;
  context->ndevs = 1;
  context->devs[0].state = ncclIbResiliencyDevStateError;
  context->qpInfo[0].state = ncclIbResiliencyQpStateFailed;
  context->inProgress = true;
}

/**
 * @brief Replaces the fake CQ input with one completion ready for the next poll.
 * @param[in] status Completion status in the ibv_wc_status domain.
 * @param[in] wrId Work-request identifier in [0, UINT64_MAX].
 * @param[in] qpNum Completing QP number in [0, UINT32_MAX].
 * @note Side effects: Replaces completion slot zero and resets the fake CQ count and cursor.
 */
void QueueCompletion(enum ibv_wc_status status, uint64_t wrId, uint32_t qpNum = 0) {
  fakes.polledCompletions[0] = {};
  fakes.polledCompletions[0].status = status;
  fakes.polledCompletions[0].wr_id = wrId;
  fakes.polledCompletions[0].qp_num = qpNum;
  fakes.polledCompletionCount = 1;
  fakes.polledCompletionIndex = 0;
}

/** @brief Provides clean process-local fakes for standalone resiliency tests. */
struct NetIbResiliencyTest : public ::testing::Test {
  /**
   * @brief Restores every fake before each standalone resiliency test.
   * @note Side effects: Resets global fake and parameter state.
   */
  void SetUp() override { ResetFakes(); }
};

/** @brief Provides a two-device, four-QP direct resiliency geometry. */
struct ResiliencyFixture : public NetIbResiliencyTest {
  /**
   * @brief Restores fake state and prepares four QPs, including two data QPs, across two devices.
   * @note Side effects: Resets global fakes and replaces fixture communicator and context state.
   */
  void SetUp() override {
    NetIbResiliencyTest::SetUp();
    std::memset(&base, 0, sizeof(base));
    std::memset(&ctx, 0, sizeof(ctx));
    for (int i = 0; i < 4; i++) {
      verbsQps[i].qp_num = 100 + i;
      base.qps[i].qp = &verbsQps[i];
      base.qps[i].devIndex = i % 2;
      base.qps[i].remDevIdx = i % 2;
      base.qps[i].coldIdx = i;
      base.qps[i].logicalLane = i;
      base.qpCold[i].remQpn = 200 + i;
      base.activeQps[i] = &base.qps[i];
      ctx.qpInfo[i].state = ncclIbResiliencyQpStateOk;
    }
    base.nqps = 4;
    base.nDataQps = 2;
    base.nPhysicalQps = 4;
    base.hasStandbyPaths = false;
    base.vProps.ndevs = 2;
    base.nRemDevs = 2;
    base.resiliency = &ctx;
    ctx.baseComm = &base;
    ctx.ndevs = 2;
    ctx.nProbingQps = 2;
    ctx.frmAllQpsHealthy = true;
    ctx.devs[0].state = ncclIbResiliencyDevStateOk;
    ctx.devs[1].state = ncclIbResiliencyDevStateOk;
  }

  ncclIbNetCommBase base = {};
  ncclIbResiliency ctx = {};
  ibv_qp verbsQps[4] = {};
};

}  // namespace

/**
 * @brief Verifies disabled, receive, and send initialization plus idempotent destruction.
 */
TEST_F(NetIbResiliencyTest, InitAndDestroyContext) {
  SCOPED_TRACE("Resiliency context ownership must be initialized and released for send and receive roles.");
  // Prepare a receive communicator with two data QPs.
  ncclIbNetCommBase base = {};
  base.nqps = 2;
  ncclIbResiliency* ctx = nullptr;

  // Disabled failover must clear the caller's output without allocating a context.
  testPortFailover = 0;
  ctx = reinterpret_cast<ncclIbResiliency*>(0x1);
  ASSERT_EQ(ncclIbResiliencyInit(&base, &ctx), ncclSuccess);
  EXPECT_EQ(ctx, nullptr);
  testPortFailover = 1;

  // An allocator failure must be returned without publishing a partial receive context.
  fakes.allocationResult = ncclSystemError;
  ASSERT_EQ(ncclIbResiliencyInit(&base, &ctx), ncclSystemError);
  EXPECT_EQ(fakes.allocationCalls, 1);
  EXPECT_EQ(fakes.allocatedBytes, sizeof(ncclIbResiliency));
  EXPECT_EQ(ctx, nullptr);
  fakes.allocationResult = ncclSuccess;

  // Receive initialization must create an idle and healthy context.
  ASSERT_EQ(ncclIbResiliencyInit(&base, &ctx), ncclSuccess);
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->baseComm, &base);
  EXPECT_FALSE(ctx->inProgress);
  EXPECT_EQ(ctx->outstandingRequests, 0);
  EXPECT_TRUE(ctx->frmAllQpsHealthy);
  EXPECT_EQ(ctx->peerReadyReadQpIndex, -1);
  EXPECT_EQ(ctx->peerReadyReadDevIndex, -1);
  // Representative QP entries, including the last slot, must start healthy.
  for (int index : {0, 1, NCCL_IB_MAX_QPS - 1}) {
    EXPECT_EQ(ctx->qpInfo[index].state, ncclIbResiliencyQpStateOk);
    EXPECT_EQ(ctx->qpInfo[index].failCount, 0);
  }

  // Destruction must clear the pointer and accept repeated or null calls.
  ASSERT_EQ(ncclIbResiliencyDestroy(&ctx), ncclSuccess);
  EXPECT_EQ(ctx, nullptr);
  EXPECT_EQ(ncclIbResiliencyDestroy(&ctx), ncclSuccess);
  EXPECT_EQ(ncclIbResiliencyDestroy(nullptr), ncclSuccess);

  // Send initialization must also clear all send-only recovery storage.
  base.isSend = true;
  ASSERT_EQ(ncclIbResiliencyInit(&base, &ctx), ncclSuccess);
  ASSERT_NE(ctx, nullptr);
  auto* sendCtx = reinterpret_cast<ncclIbResiliencySend*>(ctx);
  EXPECT_EQ(sendCtx->nextProbeWrId, IB_RESILIENCY_PROBE_WR_ID_BASE);
  EXPECT_EQ(sendCtx->failedRequests[0].request, nullptr);
  EXPECT_EQ(sendCtx->failedRequests[NET_IB_MAX_REQUESTS - 1].request, nullptr);
  EXPECT_EQ(sendCtx->probingResults[0][0], 0u);
  EXPECT_EQ(sendCtx->remCmplRecordsInfo[NCCL_IB_MAX_DEVS_PER_NIC - 1].addr, 0u);
  EXPECT_EQ(ncclIbResiliencyDestroy(&ctx), ncclSuccess);
  EXPECT_EQ(ctx, nullptr);
}

/**
 * @brief Verifies local device and probing-QP counts for receive and send contexts.
 */
TEST_F(NetIbResiliencyTest, DeviceCounts) {
  SCOPED_TRACE(
      "Device-count helpers must report state and store configured device and probing counts.");
  // Prepare an unconfigured receive-side context.
  ncclIbNetCommBase base = {};
  ncclIbResiliency ctx = {};
  ctx.baseComm = &base;

  // A receiver needs one probing QP for every remote sender device.
  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 2, 3), ncclSuccess);

  EXPECT_EQ(ctx.ndevs, 2);
  EXPECT_EQ(ctx.nProbingQps, 3);

  // A sender needs one probing QP for every local device.
  base.isSend = true;
  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 1, 3), ncclSuccess);
  EXPECT_EQ(ctx.ndevs, 1);
  EXPECT_EQ(ctx.nProbingQps, 1);
}

/**
 * @brief Verifies send and receive device-resource creation, destruction, and primary failure
 * paths.
 */
TEST_F(NetIbResiliencyTest, DeviceResourcesLifecycle) {
  // Prepare one send device with fake registration and CQ resources.
  auto [sendComm, sendCtx] = MakeState<true>(0, 0, 1);
  sendComm->devs[0].base.pd = &fakePds[0];
  ncclIbDev device = {};
  device.context = &fakeContext;

  // Send initialization reserves one CQ entry beyond the request pool for peer-ready progress.
  ASSERT_EQ(ncclIbResiliencyDevInit(&sendCtx->base, 0, &device), ncclSuccess);
  EXPECT_EQ(sendCtx->base.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendCtx->base.devs[0].readyStateMr, &fakeMrs[0]);
  EXPECT_EQ(sendCtx->base.devs[0].peerReadyStateMr, &fakeMrs[1]);
  EXPECT_EQ(sendCtx->base.devs[0].probingResultMr, &fakeMrs[2]);
  EXPECT_EQ(sendCtx->base.devs[0].probingCq, &fakeCq);
  EXPECT_EQ(fakes.registerMrCalls, 3);
  EXPECT_EQ(fakes.createdCqSize, NET_IB_MAX_REQUESTS + 1);
  EXPECT_EQ(fakes.createdCqContext, &sendComm->base.stats);

  // Send destruction must release the CQ first and then all memory registrations.
  ASSERT_EQ(ncclIbResiliencyDevDestroy(&sendCtx->base, 0), ncclSuccess);
  EXPECT_EQ(fakes.destroyCqCalls, 1);
  ASSERT_EQ(fakes.deregisterMrCalls, 3);
  EXPECT_EQ(fakes.deregisteredMrs[0], &fakeMrs[2]);
  EXPECT_EQ(fakes.deregisteredMrs[1], &fakeMrs[1]);
  EXPECT_EQ(fakes.deregisteredMrs[2], &fakeMrs[0]);

  // A receiver needs two memory regions and only a one-entry probe CQ.
  ResetFakes();
  auto [recvComm, recvContext] = MakeState<false>(0, 0, 1);
  auto& recvCtx = *recvContext;
  recvComm->devs[0].base.pd = &fakePds[0];
  ASSERT_EQ(ncclIbResiliencyDevInit(&recvCtx, 0, &device), ncclSuccess);
  EXPECT_EQ(fakes.registerMrCalls, 2);
  EXPECT_EQ(fakes.createdCqSize, 1);
  EXPECT_EQ(recvCtx.devs[0].probingResultMr, nullptr);
  ASSERT_EQ(ncclIbResiliencyDevDestroy(&recvCtx, 0), ncclSuccess);
  EXPECT_EQ(fakes.deregisterMrCalls, 2);

  // A compact failpoint sweep must stop sender initialization at each memory-registration stage.
  for (int failAt = 0; failAt < 3; failAt++) {
    SCOPED_TRACE(::testing::Message() << "registerMrFailAt=" << failAt);
    ResetFakes();
    ResetContext(&sendComm->base, &sendCtx->base, 1);
    fakes.registerMrFailAt = failAt;
    EXPECT_EQ(ncclIbResiliencyDevInit(&sendCtx->base, 0, &device), ncclSystemError);
    EXPECT_EQ(fakes.registerMrCalls, failAt + 1);
    EXPECT_EQ(fakes.createCqCalls, 0);
  }

  // CQ creation failure must be returned after both receiver registrations.
  ResetFakes();
  ResetContext(&recvComm->base, &recvCtx, 1);
  fakes.createCqResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencyDevInit(&recvCtx, 0, &device), ncclSystemError);
  EXPECT_EQ(fakes.registerMrCalls, 2);
  EXPECT_EQ(fakes.createCqCalls, 1);

  // CQ destruction failure must stop cleanup before deregistration.
  ResetFakes();
  ResetContext(&recvComm->base, &recvCtx, 1);
  recvCtx.devs[0].probingCq = &fakeCq;
  recvCtx.devs[0].peerReadyStateMr = &fakeMrs[0];
  recvCtx.devs[0].readyStateMr = &fakeMrs[1];
  fakes.destroyCqResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencyDevDestroy(&recvCtx, 0), ncclSystemError);
  EXPECT_EQ(fakes.destroyCqCalls, 1);
  EXPECT_EQ(fakes.deregisterMrCalls, 0);

  // A compact failpoint sweep must stop sender cleanup at each memory-deregistration stage.
  for (int failAt = 0; failAt < 3; failAt++) {
    SCOPED_TRACE(::testing::Message() << "deregisterMrFailAt=" << failAt);
    ResetFakes();
    ResetContext(&sendComm->base, &sendCtx->base, 1);
    sendCtx->base.devs[0].probingCq = &fakeCq;
    sendCtx->base.devs[0].probingResultMr = &fakeMrs[2];
    sendCtx->base.devs[0].peerReadyStateMr = &fakeMrs[1];
    sendCtx->base.devs[0].readyStateMr = &fakeMrs[0];
    fakes.deregisterMrFailAt = failAt;
    EXPECT_EQ(ncclIbResiliencyDevDestroy(&sendCtx->base, 0), ncclSystemError);
    EXPECT_EQ(fakes.destroyCqCalls, 1);
    EXPECT_EQ(fakes.deregisterMrCalls, failAt + 1);
  }
}

/**
 * @brief Verifies sender and receiver probing-QP creation, connection metadata, transitions,
 * and cleanup.
 */
TEST_F(NetIbResiliencyTest, ProbingQpSetup) {
  // Prepare two sender devices with distinct ports, MTUs, and memory keys.
  auto [sendComm, sendStorage] = MakeState<true>(0, 0, 2);
  auto& sendCtx = sendStorage->base;
  sendComm->base.nRemDevs = 2;
  sendCtx.nProbingQps = 2;
  for (int index = 0; index < 2; index++) {
    sendComm->devs[index].base.ibDevN = index;
    sendComm->devs[index].base.pd = &fakePds[index];
    ncclIbDevs[index].portNum = static_cast<uint8_t>(index + 1);
    ncclIbDevs[index].portAttr.active_mtu = index == 0 ? IBV_MTU_1024 : IBV_MTU_2048;
    sendCtx.devs[index].probingCq = &fakeCq;
    sendCtx.devs[index].readyStateMr = &fakeMrs[index];
  }
  ncclIbResiliencyInfo localInfo = {};

  // Sender QP creation must use the probe-specific attributes and export local metadata.
  ASSERT_EQ(ncclIbResiliencySenderCreateQps(&sendCtx, &localInfo), ncclSuccess);
  EXPECT_EQ(fakes.createQpCalls, 2);
  EXPECT_EQ(fakes.createdQpType, IBV_QPT_RC);
  EXPECT_EQ(fakes.createdQpAccess, static_cast<unsigned int>(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
  EXPECT_EQ(fakes.createdQpMaxRecv, 0u);
  EXPECT_EQ(fakes.createdQpMaxSend, static_cast<uint32_t>(NET_IB_MAX_REQUESTS + 1));
  EXPECT_EQ(sendCtx.probingQps[0].devIndex, 0);
  EXPECT_EQ(sendCtx.probingQps[1].devIndex, 1);
  EXPECT_EQ(localInfo.probingQpsInfo[0].qpn, 500u);
  EXPECT_EQ(localInfo.probingQpsInfo[1].qpn, 501u);
  EXPECT_EQ(localInfo.readyStateAddr, reinterpret_cast<uint64_t>(&sendCtx.localReady));
  EXPECT_EQ(localInfo.readyStateRkeys[0], fakeMrs[0].rkey);
  EXPECT_EQ(localInfo.readyStateRkeys[1], fakeMrs[1].rkey);

  // Prepare asymmetric remote metadata to verify QP mapping and independent local MTU input.
  ncclIbConnectionMetadata remoteInfo = {};
  remoteInfo.resiliencyInfo.probingQpsInfo[0].qpn = 700;
  remoteInfo.resiliencyInfo.probingQpsInfo[0].devIndex = 1;
  remoteInfo.resiliencyInfo.probingQpsInfo[1].qpn = 701;
  remoteInfo.resiliencyInfo.probingQpsInfo[1].devIndex = 0;
  remoteInfo.resiliencyInfo.readyStateAddr = 0x12340000;
  remoteInfo.resiliencyInfo.readyStateRkeys[0] = 0x31;
  remoteInfo.resiliencyInfo.readyStateRkeys[1] = 0x32;
  remoteInfo.devs[0].mtu = IBV_MTU_4096;
  remoteInfo.devs[1].mtu = IBV_MTU_4096;

  // One two-row failpoint table must prove that sender connection stops at RTR or RTS.
  const ncclIbResiliency sendRtsBaseline = sendCtx;
  const ncclIbConnectionMetadata sendRemoteBaseline = remoteInfo;
  for (int failStage = 0; failStage < 2; failStage++) {
    SCOPED_TRACE(::testing::Message() << "senderFailStage=" << (failStage == 0 ? "RTR" : "RTS"));
    fakes = ResiliencyFakes{};
    sendCtx = sendRtsBaseline;
    remoteInfo = sendRemoteBaseline;
    if (failStage == 0) fakes.rtrQpResult = ncclSystemError;
    if (failStage == 1) fakes.rtsQpResult = ncclSystemError;
    EXPECT_EQ(ncclIbResiliencySenderQpsToRts(&sendCtx, &remoteInfo), ncclSystemError);
    EXPECT_EQ(fakes.rtrQpCalls, 1);
    EXPECT_EQ(fakes.rtsQpCalls, failStage);
    EXPECT_EQ(sendCtx.remoteReadyStateAddr, 0u);
  }

  // Sender connection must transition every probe QP and cache remote ready-state access.
  fakes = ResiliencyFakes{};
  sendCtx = sendRtsBaseline;
  remoteInfo = sendRemoteBaseline;
  ASSERT_EQ(ncclIbResiliencySenderQpsToRts(&sendCtx, &remoteInfo), ncclSuccess);
  EXPECT_EQ(fakes.rtrQpCalls, 2);
  EXPECT_EQ(fakes.rtsQpCalls, 2);
  EXPECT_FALSE(fakes.rtrFifoTc);
  EXPECT_EQ(sendCtx.probingQps[0].remDevIdx, 1);
  EXPECT_EQ(sendCtx.probingQps[1].remDevIdx, 0);
  EXPECT_EQ(remoteInfo.devs[0].mtu, IBV_MTU_4096);
  EXPECT_EQ(remoteInfo.devs[1].mtu, IBV_MTU_4096);
  EXPECT_EQ(fakes.rtrLocalMtu, IBV_MTU_2048);
  EXPECT_EQ(fakes.rtrRemoteMtu, IBV_MTU_4096);
  EXPECT_EQ(sendCtx.remoteReadyStateAddr, 0x12340000u);
  EXPECT_EQ(sendCtx.remoteReadyStateRkeys[0], 0x31u);
  EXPECT_EQ(sendCtx.remoteReadyStateRkeys[1], 0x32u);
  // Closing must accept null and destroy every configured sender probe QP.
  EXPECT_EQ(ncclIbResiliencyClose(nullptr), ncclSuccess);
  ASSERT_EQ(ncclIbResiliencyClose(&sendCtx), ncclSuccess);
  EXPECT_EQ(fakes.destroyQpCalls, 2);

  // Prepare three receiver probing QPs striped across two local devices.
  ResetFakes();
  auto [recvComm, recvStorage] = MakeState<false>(0, 0, 2);
  auto& recvCtx = *recvStorage;
  recvComm->base.nRemDevs = 2;
  recvCtx.nProbingQps = 3;
  for (int index = 0; index < 2; index++) {
    recvComm->devs[index].base.ibDevN = index;
    recvComm->devs[index].base.pd = &fakePds[index];
    recvCtx.devs[index].probingCq = &fakeCq;
    recvCtx.devs[index].readyStateMr = &fakeMrs[index];
  }
  remoteInfo = {};
  remoteInfo.resiliencyInfo.readyStateAddr = 0x56780000;
  for (int index = 0; index < 3; index++) {
    remoteInfo.resiliencyInfo.probingQpsInfo[index].qpn = 800u + static_cast<uint32_t>(index);
    remoteInfo.resiliencyInfo.probingQpsInfo[index].devIndex = index % 2;
  }
  remoteInfo.resiliencyInfo.readyStateRkeys[0] = 0x41;
  remoteInfo.resiliencyInfo.readyStateRkeys[1] = 0x42;
  remoteInfo.devs[0].mtu = IBV_MTU_1024;
  remoteInfo.devs[1].mtu = IBV_MTU_1024;
  localInfo = {};

  // One three-row failpoint table must prove receiver setup stops at create, RTR, or RTS.
  const ncclIbResiliency recvRtsBaseline = recvCtx;
  const ncclIbConnectionMetadata recvRemoteBaseline = remoteInfo;
  for (int failStage = 0; failStage < 3; failStage++) {
    SCOPED_TRACE(::testing::Message() << "receiverFailStage=" << failStage);
    fakes = ResiliencyFakes{};
    recvCtx = recvRtsBaseline;
    remoteInfo = recvRemoteBaseline;
    localInfo = {};
    if (failStage == 0) fakes.createQpResult = ncclSystemError;
    if (failStage == 1) fakes.rtrQpResult = ncclSystemError;
    if (failStage == 2) fakes.rtsQpResult = ncclSystemError;
    EXPECT_EQ(ncclIbResiliencyReceiverQpsCreateToRts(&recvCtx, &remoteInfo, &localInfo), ncclSystemError);
    EXPECT_EQ(fakes.createQpCalls, 1);
    EXPECT_EQ(fakes.rtrQpCalls, failStage == 0 ? 0 : 1);
    EXPECT_EQ(fakes.rtsQpCalls, failStage == 2 ? 1 : 0);
  }

  // Receiver creation must stripe QPs, connect them, and exchange ready-state metadata.
  fakes = ResiliencyFakes{};
  recvCtx = recvRtsBaseline;
  remoteInfo = recvRemoteBaseline;
  localInfo = {};
  ASSERT_EQ(ncclIbResiliencyReceiverQpsCreateToRts(&recvCtx, &remoteInfo, &localInfo), ncclSuccess);
  EXPECT_EQ(fakes.createQpCalls, 3);
  EXPECT_EQ(fakes.rtrQpCalls, 3);
  EXPECT_EQ(fakes.rtsQpCalls, 3);
  EXPECT_TRUE(fakes.rtrFifoTc);
  EXPECT_EQ(recvCtx.probingQps[0].devIndex, 0);
  EXPECT_EQ(recvCtx.probingQps[1].devIndex, 1);
  EXPECT_EQ(recvCtx.probingQps[2].devIndex, 0);
  EXPECT_EQ(localInfo.readyStateAddr, reinterpret_cast<uint64_t>(&recvCtx.localReady));
  EXPECT_EQ(recvCtx.remoteReadyStateAddr, 0x56780000u);
  ASSERT_EQ(ncclIbResiliencyClose(&recvCtx), ncclSuccess);
  EXPECT_EQ(fakes.destroyQpCalls, 3);

  // A QP creation error must be returned immediately.
  ResetFakes();
  sendCtx.nProbingQps = 1;
  fakes.createQpResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencySenderCreateQps(&sendCtx, &localInfo), ncclSystemError);
  EXPECT_EQ(fakes.createQpCalls, 1);

  // A QP destruction error must stop close at the failing entry.
  ResetFakes();
  recvCtx.nProbingQps = 2;
  recvCtx.probingQps[0].qp = &fakeCreatedQps[0];
  recvCtx.probingQps[1].qp = &fakeCreatedQps[1];
  fakes.destroyQpResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencyClose(&recvCtx), ncclSystemError);
  EXPECT_EQ(fakes.destroyQpCalls, 1);
}

/**
 * @brief Verifies request completion accounting for null, zero, positive, and failover-compensated
 * events.
 */
TEST_F(NetIbResiliencyTest, RequestCompletion) {
  // Prepare a request whose completion state spans two devices.
  auto [comm, context] = MakeState<false>(0, 0, 2);
  ncclIbRequest request = MakeRequest(&comm->base, 0, NCCL_NET_IB_REQ_UNUSED, 0);
  bool complete = false;

  // A null request is never reported as complete.
  ASSERT_EQ(ncclIbResiliencyRequestIsComplete(nullptr, &complete), ncclSuccess);
  EXPECT_FALSE(complete);

  /** @brief Describes one pair of device-event counts and its completion result. */
  struct EventCase { int first; int second; bool expected; };
  const EventCase eventCases[] = {{0, 0, true}, {0, 1, false}, {-1, 1, true}, {-2, 1, false}};
  // Empty, outstanding, compensated, and unbalanced event totals must report their exact
  // completion state.
  for (const EventCase& eventCase : eventCases) {
    SCOPED_TRACE(::testing::Message() << eventCase.first << ',' << eventCase.second);
    request.events[0] = eventCase.first;
    request.events[1] = eventCase.second;
    ASSERT_EQ(ncclIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
    EXPECT_EQ(complete, eventCase.expected);
  }
}

/**
 * @brief Verifies QP failure validation, recovery state, ready-mask clearing, and repeated calls.
 */
TEST_F(NetIbResiliencyTest, MarkFailedQp) {
  SCOPED_TRACE("Marking a failed queue pair must update only its owning device and recovery state.");
  // Prepare a healthy QP advertised as ready by both peers.
  auto [comm, context] = MakeState<false>(2, 2, 2);
  auto& base = comm->base;
  auto& ctx = *context;
  ConfigureActiveQps(&base, &ctx, {0, 1});
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, 1, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 1, true);

  // Indices outside the configured QP range must be rejected.
  EXPECT_EQ(ncclIbResiliencyMarkQpFailed(&ctx, -1), ncclInternalError);
  EXPECT_EQ(ncclIbResiliencyMarkQpFailed(&ctx, 2), ncclInternalError);

  // A new failure must enter recovery and withdraw both ready indications.
  ASSERT_EQ(ncclIbResiliencyMarkQpFailed(&ctx, 1), ncclSuccess);

  EXPECT_TRUE(ctx.inProgress);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 1));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 1));
  EXPECT_FALSE(ctx.frmAllQpsHealthy);
  // Reporting the same failure again must not consume another retry.
  ASSERT_EQ(ncclIbResiliencyMarkQpFailed(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);

  // A permanent failure must remain permanent and unchanged.
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateFailedPermanent;
  ASSERT_EQ(ncclIbResiliencyMarkQpFailed(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailedPermanent);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);
}

/**
 * @brief Verifies WR-ID encoding, QP-state predicates, ready masks, health state, and backoff
 * boundaries.
 */
TEST_F(ResiliencyFixture, StateHelpers) {
  SCOPED_TRACE("State helpers must classify valid recovery states and preserve unrelated fixture data.");
  // Probe WR IDs must preserve the first and last valid request slots.
  EXPECT_EQ(ibResiliencyProbeWrIdEncode(IB_RESILIENCY_PROBE_WR_ID_BASE, 0), IB_RESILIENCY_PROBE_WR_ID_BASE);
  const uint64_t lastSlot = NET_IB_MAX_REQUESTS - 1;
  const uint64_t encoded = ibResiliencyProbeWrIdEncode(IB_RESILIENCY_PROBE_WR_ID_BASE, lastSlot);
  EXPECT_EQ(ibResiliencyProbeWrIdDecodeSlot(encoded), lastSlot);
  EXPECT_EQ(ibResiliencyProbeWrIdDecodeSlot(UINT64_MAX), IB_SEND_REQ_WR_ID_SLOT_MASK);

  // Only Failed and LocalReady QPs require recovery, while only Ok permits traffic.
  EXPECT_TRUE(ncclIbResiliencyQpIsPendingRecovery(ncclIbResiliencyQpStateFailed));
  EXPECT_TRUE(ncclIbResiliencyQpIsPendingRecovery(ncclIbResiliencyQpStateLocalReady));
  EXPECT_FALSE(ncclIbResiliencyQpIsPendingRecovery(ncclIbResiliencyQpStateOk));
  EXPECT_FALSE(ncclIbResiliencyQpIsPendingRecovery(ncclIbResiliencyQpStateFailedPermanent));
  EXPECT_TRUE(ncclIbResiliencyQpIsTrafficReady(ncclIbResiliencyQpStateOk));
  EXPECT_FALSE(ncclIbResiliencyQpIsTrafficReady(ncclIbResiliencyQpStateFailed));
  EXPECT_FALSE(ncclIbResiliencyQpIsTrafficReady(ncclIbResiliencyQpStateLocalReady));
  EXPECT_FALSE(ncclIbResiliencyQpIsTrafficReady(ncclIbResiliencyQpStateFailedPermanent));

  // Ready bits must work across both sides of the 64-bit word boundary.
  uint64_t mask[IB_RESILIENCY_READY_QP_WORDS] = {};
  for (int index : {0, 63, 64, NCCL_IB_MAX_QPS - 1}) {
    ncclIbResiliencyReadyQpSetBit(mask, index, true);
    EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(mask, index)) << "index=" << index;
  }
  ncclIbResiliencyReadyQpSetBit(mask, 63, false);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(mask, 63));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(mask, 64));

  // Reset must clear every field that identifies an in-flight peer-ready read.
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 1;
  ctx.peerReadyReadDevIndex = 1;
  ncclIbResiliencyResetPeerReadyReadState(&ctx);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, -1);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, -1);

  // An empty context is healthy, while any failed configured QP makes it unhealthy.
  ctx.baseComm = nullptr;
  ctx.frmAllQpsHealthy = false;
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);
  ctx.baseComm = &base;
  const int configuredQps = base.nqps;
  base.nqps = 0;
  ctx.frmAllQpsHealthy = false;
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);
  base.nqps = configuredQps;
  ctx.qpInfo[2].state = ncclIbResiliencyQpStateFailed;
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_FALSE(ctx.frmAllQpsHealthy);
  ctx.qpInfo[2].state = ncclIbResiliencyQpStateOk;
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);

  // Delay and backoff helpers must clamp both integer boundaries.
  EXPECT_EQ(ncclIbResiliencyQpRejoinCooldownNs(), 1000ULL * MSEC_TO_NSEC);
  EXPECT_EQ(ncclIbResiliencyPeerReadyBackoffNs(std::numeric_limits<int>::min()), 1000ULL * MSEC_TO_NSEC);
  EXPECT_EQ(ncclIbResiliencyPeerReadyBackoffNs(3), 3000ULL * MSEC_TO_NSEC);
  EXPECT_EQ(ncclIbResiliencyPeerReadyBackoffNs(std::numeric_limits<int>::max()), 16000ULL * MSEC_TO_NSEC);
}

/**
 * @brief Verifies probe-QP usability accepts only configured, healthy, unquarantined paths.
 * @note Side effects: Mutates only local fixture setup between read-only helper calls.
 */
TEST_F(ResiliencyFixture, EvaluatesProbeQpUsabilityDirectly) {
  SCOPED_TRACE(
      "Probe-QP usability must validate index, quarantine, local device, and remote mapping.");
  ibv_qp probingVerbs[2] = {};
  probingVerbs[0].qp_num = 700;
  probingVerbs[1].qp_num = 701;
  ctx.nProbingQps = 2;
  for (int probingQpIndex = 0; probingQpIndex < ctx.nProbingQps; probingQpIndex++) {
    ctx.probingQps[probingQpIndex].qp = &probingVerbs[probingQpIndex];
    ctx.probingQps[probingQpIndex].devIndex = probingQpIndex;
    ctx.probingQps[probingQpIndex].remDevIdx = probingQpIndex;
  }

  EXPECT_FALSE(ibResiliencyProbeQpUsable(&ctx, -1))
      << "A negative probing-QP index must be rejected before array access";
  EXPECT_FALSE(ibResiliencyProbeQpUsable(&ctx, ctx.nProbingQps))
      << "An index at the configured probing-QP count must be rejected";
  EXPECT_TRUE(ibResiliencyProbeQpUsable(&ctx, 0))
      << "The first fully configured healthy probing path must be usable";
  EXPECT_TRUE(ibResiliencyProbeQpUsable(&ctx, ctx.nProbingQps - 1))
      << "The final configured healthy probing path must be usable";

  ctx.failedProbeQpMask = 1ULL;
  EXPECT_FALSE(ibResiliencyProbeQpUsable(&ctx, 0))
      << "A quarantined probing QP must not be selected again";
  EXPECT_TRUE(ibResiliencyProbeQpUsable(&ctx, 1))
      << "Quarantining one path must not disable a distinct healthy path";
  ctx.failedProbeQpMask = 0;

  ctx.probingQps[0].qp = nullptr;
  EXPECT_FALSE(ibResiliencyProbeQpUsable(&ctx, 0))
      << "A path without a created verbs QP must be unusable";
  ctx.probingQps[0].qp = &probingVerbs[0];

  for (int invalidDevIndex : {-1, ctx.ndevs}) {
    SCOPED_TRACE(testing::Message() << "invalid local devIndex=" << invalidDevIndex);
    ctx.probingQps[0].devIndex = invalidDevIndex;
    EXPECT_FALSE(ibResiliencyProbeQpUsable(&ctx, 0))
        << "A probing path must reject a local device outside the configured range";
  }
  ctx.probingQps[0].devIndex = 0;

  for (ncclIbResiliencyDevState state :
      {ncclIbResiliencyDevStateError, ncclIbResiliencyDevStateErrorPermanent}) {
    SCOPED_TRACE(testing::Message() << "unhealthy device state=" << static_cast<int>(state));
    ctx.devs[0].state = state;
    EXPECT_FALSE(ibResiliencyProbeQpUsable(&ctx, 0))
        << "A probing path must reject every non-operational local-device state";
  }
  ctx.devs[0].state = ncclIbResiliencyDevStateOk;

  for (int invalidRemoteIndex : {-1, base.nRemDevs}) {
    SCOPED_TRACE(testing::Message() << "invalid remote devIndex=" << invalidRemoteIndex);
    ctx.probingQps[0].remDevIdx = invalidRemoteIndex;
    EXPECT_FALSE(ibResiliencyProbeQpUsable(&ctx, 0))
        << "A probing path must reject a remote mapping outside the peer-device range";
  }
  ctx.probingQps[0].remDevIdx = base.nRemDevs - 1;

  std::vector<unsigned char> contextBefore(sizeof(ctx));
  std::vector<unsigned char> baseBefore(sizeof(base));
  std::vector<unsigned char> verbsBefore(sizeof(probingVerbs));
  std::memcpy(contextBefore.data(), &ctx, sizeof(ctx));
  std::memcpy(baseBefore.data(), &base, sizeof(base));
  std::memcpy(verbsBefore.data(), probingVerbs, sizeof(probingVerbs));
  EXPECT_TRUE(ibResiliencyProbeQpUsable(&ctx, 0))
      << "The last valid remote-device boundary must remain usable";
  EXPECT_EQ(std::memcmp(contextBefore.data(), &ctx, sizeof(ctx)), 0)
      << "Usability inspection must preserve the complete resiliency context";
  EXPECT_EQ(std::memcmp(baseBefore.data(), &base, sizeof(base)), 0)
      << "Usability inspection must preserve the complete base communicator";
  EXPECT_EQ(std::memcmp(verbsBefore.data(), probingVerbs, sizeof(probingVerbs)), 0)
      << "Usability inspection must preserve the backing verbs QPs";
}

/**
 * @brief Verifies probe completions resolve by the exact local device and verbs QP number.
 * @note Side effects: Mutates only local fixture setup between read-only helper calls.
 */
TEST_F(ResiliencyFixture, FindsCompletingProbeQpDirectly) {
  SCOPED_TRACE(
      "Completion lookup must identify physical probe paths without applying usability filters.");
  ibv_qp probingVerbs[3] = {};
  probingVerbs[0].qp_num = 800;
  probingVerbs[1].qp_num = 801;
  probingVerbs[2].qp_num = 800;
  ctx.nProbingQps = 3;
  ctx.probingQps[0].qp = &probingVerbs[0];
  ctx.probingQps[0].devIndex = 0;
  ctx.probingQps[1].qp = &probingVerbs[1];
  ctx.probingQps[1].devIndex = 1;
  ctx.probingQps[2].qp = &probingVerbs[2];
  ctx.probingQps[2].devIndex = 1;

  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, 0, 800), 0)
      << "The first probing path must resolve from its exact device and QP number";
  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, 1, 801), 1)
      << "An interior probing path must resolve from its exact device and QP number";
  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, 1, 800), 2)
      << "The final probing path must resolve even when another device uses the same QP number";
  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, -1, 800), -1)
      << "A negative local-device index must not match a probing path";
  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, ctx.ndevs, 800), -1)
      << "An index at the local-device count must not match a probing path";
  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, 0, 801), -1)
      << "A QP number observed on the wrong local device must not match";
  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, 1, 999), -1)
      << "An unknown QP number must report no matching probing path";

  ctx.probingQps[1].qp = nullptr;
  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, 1, 801), -1)
      << "An uncreated probing path must not match a completion";
  ctx.probingQps[1].qp = &probingVerbs[1];

  ctx.failedProbeQpMask = 1ULL << 2;
  ctx.devs[1].state = ncclIbResiliencyDevStateError;
  std::vector<unsigned char> contextBefore(sizeof(ctx));
  std::vector<unsigned char> baseBefore(sizeof(base));
  std::vector<unsigned char> verbsBefore(sizeof(probingVerbs));
  std::memcpy(contextBefore.data(), &ctx, sizeof(ctx));
  std::memcpy(baseBefore.data(), &base, sizeof(base));
  std::memcpy(verbsBefore.data(), probingVerbs, sizeof(probingVerbs));
  EXPECT_EQ(ibResiliencyProbeQpIndexFromCompletion(&ctx, 1, 800), 2)
      << "A failed completion must still identify its quarantined or unhealthy physical path";
  EXPECT_EQ(std::memcmp(contextBefore.data(), &ctx, sizeof(ctx)), 0)
      << "Completion lookup must preserve the complete resiliency context";
  EXPECT_EQ(std::memcmp(baseBefore.data(), &base, sizeof(base)), 0)
      << "Completion lookup must preserve the complete base communicator";
  EXPECT_EQ(std::memcmp(verbsBefore.data(), probingVerbs, sizeof(probingVerbs)), 0)
      << "Completion lookup must preserve the backing verbs QPs";
}

/**
 * @brief Verifies local-port query results and QP reset success and failure.
 */
TEST_F(NetIbResiliencyTest, PortQueryAndQpReset) {
  // Prepare one communicator device backed by a controllable fake port.
  auto sendComm = std::make_unique<ncclIbSendComm>();
  sendComm->base.isSend = true;
  sendComm->base.vProps.ndevs = 1;
  sendComm->devs[0].base.ibDevN = 0;
  ncclIbDevs[0].context = &fakeContext;
  ncclIbDevs[0].portNum = 7;

  bool active = true;
  enum ibv_mtu mtu = IBV_MTU_4096;
  // Both invalid device-index boundaries must be rejected without querying the provider.
  for (int invalidIndex : {-1, sendComm->base.vProps.ndevs}) {
    SCOPED_TRACE(::testing::Message() << "devIndex=" << invalidIndex);
    active = true;
    mtu = IBV_MTU_4096;
    EXPECT_EQ(ncclIbResiliencyIsLocalPortActive(&sendComm->base, invalidIndex, &active, &mtu), ncclInternalError);
    EXPECT_FALSE(active);
    EXPECT_EQ(mtu, IBV_MTU_256);
  }
  EXPECT_EQ(fakes.queryPortCalls, 0);

  // An active port must return its current MTU.
  EXPECT_EQ(ncclIbResiliencyIsLocalPortActive(&sendComm->base, 0, &active, &mtu), ncclSuccess);
  EXPECT_TRUE(active);
  EXPECT_EQ(mtu, IBV_MTU_1024);
  EXPECT_EQ(fakes.queryPortCalls, 1);
  EXPECT_EQ(fakes.queriedContext, &fakeContext);
  EXPECT_EQ(fakes.queriedPort, 7);

  // A provider-reported down port remains a successful query with active=false.
  fakes.portState = IBV_PORT_DOWN;
  fakes.portMtu = IBV_MTU_2048;
  EXPECT_EQ(ncclIbResiliencyIsLocalPortActive(&sendComm->base, 0, &active, &mtu), ncclSuccess);
  EXPECT_FALSE(active);
  EXPECT_EQ(mtu, IBV_MTU_2048);

  // A query failure must fall back to inactive and the minimum MTU.
  fakes.queryPortResult = ncclSystemError;
  active = true;
  mtu = IBV_MTU_4096;
  EXPECT_EQ(ncclIbResiliencyIsLocalPortActive(&sendComm->base, 0, &active, &mtu), ncclSuccess);
  EXPECT_FALSE(active);
  EXPECT_EQ(mtu, IBV_MTU_256);

  // QP reset must request only IBV_QP_STATE and propagate provider errors.
  EXPECT_EQ(ncclIbResiliencyQpToReset(&fakeCreatedQps[0]), ncclSuccess);
  EXPECT_EQ(fakes.modifyQpCalls, 1);
  EXPECT_EQ(fakes.modifiedQp, &fakeCreatedQps[0]);
  EXPECT_EQ(fakes.modifiedQpState, IBV_QPS_RESET);
  EXPECT_EQ(fakes.modifyQpMask, IBV_QP_STATE);
  fakes.modifyQpResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencyQpToReset(&fakeCreatedQps[0]), ncclSystemError);
}

/**
 * @brief Verifies disabled, successful, and failed receive-queue refill behavior.
 */
TEST_F(NetIbResiliencyTest, ReceiveQueueRefill) {
  SCOPED_TRACE("Receive-queue refill must post the required work requests and propagate provider failures.");
  // Prepare a receive QP with optional pre-posting disabled.
  auto [recvComm, recvContext] = MakeState<false>(2, 1, 1);
  auto& ctx = *recvContext;
  recvComm->prepostReceiveWorkRequests = false;
  ncclIbQp qp = {};
  qp.qp = &fakeCreatedQps[0];
  qp.devIndex = 0;

  // Send communicators must skip receive-queue work before interpreting the communicator layout.
  recvComm->base.isSend = true;
  EXPECT_EQ(ncclIbResiliencyRefillRecvQueueForQp(&ctx, &qp), ncclSuccess);
  EXPECT_EQ(fakes.postRecvCalls, 0);
  recvComm->base.isSend = false;

  // Disabled pre-posting must leave the receive queue untouched.
  EXPECT_EQ(ncclIbResiliencyRefillRecvQueueForQp(&ctx, &qp), ncclSuccess);
  EXPECT_EQ(fakes.postRecvCalls, 0);

  // Enabled pre-posting must restore the complete required queue depth.
  recvComm->prepostReceiveWorkRequests = true;
  EXPECT_EQ(ncclIbResiliencyRefillRecvQueueForQp(&ctx, &qp), ncclSuccess);
  EXPECT_EQ(fakes.postRecvCalls, NET_IB_MAX_REQUESTS);
  EXPECT_EQ(fakes.postedRecvQp, qp.qp);
  EXPECT_EQ(fakes.postedRecvWr, &recvComm->ibRecvWorkRequest);
  EXPECT_STREQ(fakes.rejoinStages, "Q");

  // The first provider post failure must be returned immediately.
  fakes.postRecvCalls = 0;
  fakes.rejoinStages[0] = '\0';
  fakes.rejoinStageCount = 0;
  fakes.postRecvError = EIO;
  EXPECT_EQ(ncclIbResiliencyRefillRecvQueueForQp(&ctx, &qp), ncclSystemError);
  EXPECT_EQ(fakes.postRecvCalls, 1);
  EXPECT_EQ(fakes.postedRecvQp, qp.qp);
  EXPECT_EQ(fakes.postedRecvWr, &recvComm->ibRecvWorkRequest);
  EXPECT_STREQ(fakes.rejoinStages, "Q");
}

/**
 * @brief Verifies completion fatality classification, QP replacement, and device failover.
 */
TEST_F(ResiliencyFixture, DeviceFailover) {
  SCOPED_TRACE("Device failover must classify errors and remap affected QPs to a healthy device.");
  // Prepare a recoverable completion error with one alternate device.
  ibv_wc wc = {};
  wc.status = IBV_WC_WR_FLUSH_ERR;
  ctx.devs[1].state = ncclIbResiliencyDevStateError;

  // Flush and retry errors are recoverable while another device is healthy.
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&ctx, &wc, 1), ncclSuccess);
  wc.status = IBV_WC_RETRY_EXC_ERR;
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&ctx, &wc, 1), ncclSuccess);
  // Unsupported completion statuses must remain fatal.
  wc.status = IBV_WC_LOC_PROT_ERR;
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&ctx, &wc, 1), ncclRemoteError);

  // A supported status is still fatal when no functional device remains.
  wc.status = IBV_WC_WR_FLUSH_ERR;
  ctx.devs[0].state = ncclIbResiliencyDevStateError;
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&ctx, &wc, 1), ncclRemoteError);

  // Replacing device-one QPs must preserve QPs already using device zero.
  ctx.devs[0].state = ncclIbResiliencyDevStateOk;
  ctx.devs[1].state = ncclIbResiliencyDevStateError;
  ASSERT_EQ(ncclIbResiliencyReplaceQps(&ctx, 1), ncclSuccess);
  EXPECT_EQ(base.activeQps[0], &base.qps[0]);
  EXPECT_EQ(base.activeQps[1], &base.qps[2]);
  EXPECT_EQ(base.activeQps[2], &base.qps[2]);
  EXPECT_EQ(base.activeQps[3], &base.qps[0]);

  // Replacement must skip a non-traffic-ready QP even when its device itself is healthy.
  ResetActiveQps(&base);
  ctx.qpInfo[2].state = ncclIbResiliencyQpStateLocalReady;
  ASSERT_EQ(ncclIbResiliencyReplaceQps(&ctx, 1), ncclSuccess);
  EXPECT_EQ(base.activeQps[1], &base.qps[0]);
  EXPECT_EQ(base.activeQps[3], &base.qps[0]);
  ctx.qpInfo[2].state = ncclIbResiliencyQpStateOk;

  // Failover handling must preserve a permanent device state while redirecting and marking its QPs.
  ResetActiveQps(&base);
  ctx.devs[1].state = ncclIbResiliencyDevStateErrorPermanent;
  ASSERT_EQ(ncclIbResiliencyHandleDeviceFailure(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].state, ncclIbResiliencyDevStateErrorPermanent);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[3].state, ncclIbResiliencyQpStateFailed);

  // Full failure handling must mark every affected QP and clear its ready state.
  ResetActiveQps(&base);
  ctx.devs[1].state = ncclIbResiliencyDevStateOk;
  for (int index : {1, 3}) {
    ctx.qpInfo[index].state = ncclIbResiliencyQpStateOk;
    ctx.qpInfo[index].failCount = 0;
  }
  SetQpsReady(&ctx, {1, 3}, true);
  ASSERT_EQ(ncclIbResiliencyHandleDeviceFailure(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);
  EXPECT_EQ(ctx.qpInfo[3].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[3].failCount, 1);
  EXPECT_EQ(base.activeQps[1], &base.qps[2]);
  EXPECT_EQ(base.activeQps[3], &base.qps[0]);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 1));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 1));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 3));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 3));
  EXPECT_TRUE(ctx.inProgress);
  EXPECT_FALSE(ctx.frmAllQpsHealthy);

  // Failure must be reported when no traffic-ready replacement exists.
  ResetActiveQps(&base);
  ctx.devs[0].state = ncclIbResiliencyDevStateError;
  ctx.devs[1].state = ncclIbResiliencyDevStateOk;
  for (int index : {1, 3}) {
    ctx.qpInfo[index].state = ncclIbResiliencyQpStateOk;
    ctx.qpInfo[index].failCount = 0;
  }
  SetQpsReady(&ctx, {1, 3}, true);
  EXPECT_EQ(ncclIbResiliencyHandleDeviceFailure(&ctx, 1), ncclInternalError);
  EXPECT_EQ(ctx.devs[1].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 0);
  EXPECT_EQ(ctx.qpInfo[3].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(ctx.qpInfo[3].failCount, 0);
  EXPECT_EQ(base.activeQps[1], &base.qps[1]);
  EXPECT_EQ(base.activeQps[3], &base.qps[3]);
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 1));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 1));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 3));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 3));
}

/**
 * @brief Verifies pending-recovery detection and derived device-state reconciliation.
 */
TEST_F(ResiliencyFixture, DeviceReconciliation) {
  SCOPED_TRACE("Device reconciliation must derive device state from its assigned QP states.");
  // A healthy context must report no pending QP recovery.
  EXPECT_FALSE(ncclIbResiliencyHasPendingQpRecovery(&ctx));
  // One failed QP must make recovery pending and place its device in Error.
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateFailed;
  EXPECT_TRUE(ncclIbResiliencyHasPendingQpRecovery(&ctx));

  ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].state, ncclIbResiliencyDevStateError);

  // A device with only permanently failed QPs must become permanently failed.
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateFailedPermanent;
  ctx.qpInfo[3].state = ncclIbResiliencyQpStateFailedPermanent;
  ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].state, ncclIbResiliencyDevStateErrorPermanent);

  // A configured device with no assigned QPs must resolve to the default healthy state.
  base.nqps = 0;
  ctx.ndevs = 1;
  ctx.devs[0].state = ncclIbResiliencyDevStateError;
  ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateOk);
}

/**
 * @brief Verifies rejoin retry exhaustion and peer-ready failure backoff.
 */
TEST_F(ResiliencyFixture, RejoinFailure) {
  // Start with both peers advertising every QP as ready.
  ctx.localReady.mask[0] = UINT64_MAX;
  ctx.remoteReady.mask[0] = UINT64_MAX;

  // A failed rejoin step must consume one retry and withdraw readiness.
  ASSERT_EQ(ncclIbResiliencyMarkQpRejoinFailure(&ctx, 1, "test"), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);
  EXPECT_TRUE(ctx.inProgress);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 1));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 1));

  // Reaching the retry limit must make the QP permanently failed and stable.
  ctx.qpInfo[1].failCount = IB_RESILIENCY_QP_REJOIN_MAX_FAILURES - 1;
  ASSERT_EQ(ncclIbResiliencyMarkQpRejoinFailure(&ctx, 1, "limit"), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailedPermanent);
  EXPECT_EQ(ctx.qpInfo[1].failCount, IB_RESILIENCY_QP_REJOIN_MAX_FAILURES);
  ASSERT_EQ(ncclIbResiliencyMarkQpRejoinFailure(&ctx, 1, "ignored"), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].failCount, IB_RESILIENCY_QP_REJOIN_MAX_FAILURES);

  // Peer-ready read failure must clear its snapshot and start device-local backoff.
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 2;
  ctx.peerReadyReadDevIndex = 1;
  ctx.remoteReady.mask[0] = UINT64_MAX;
  ASSERT_EQ(ncclIbResiliencyHandlePeerReadyReadFailure(&ctx, 1, "unit", nullptr), ncclSuccess);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.remoteReady.mask[0], 0u);
  EXPECT_EQ(ctx.devs[1].peerReadyFailureCount, 1);
  EXPECT_GT(ctx.devs[1].peerReadyRetryAfterNs, 0u);

  // A later success must clear backoff.
  ASSERT_EQ(ncclIbResiliencyHandlePeerReadyReadSuccess(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].peerReadyFailureCount, 0);
  EXPECT_EQ(ctx.devs[1].peerReadyRetryAfterNs, 0u);

  // Both invalid index boundaries must be rejected by the success and failure handlers.
  for (int invalidIndex : {-1, ctx.ndevs}) {
    SCOPED_TRACE(::testing::Message() << "devIndex=" << invalidIndex);
    EXPECT_EQ(ncclIbResiliencyHandlePeerReadyReadFailure(&ctx, invalidIndex, "unit", nullptr), ncclInternalError);
    EXPECT_EQ(ncclIbResiliencyHandlePeerReadyReadSuccess(&ctx, invalidIndex), ncclInternalError);
  }
}

/**
 * @brief Verifies successful QP rejoin, peer activation, inactive-port deferral, and RTR failure.
 */
TEST_F(NetIbResiliencyTest, QpRejoinLifecycle) {
  SCOPED_TRACE("Queue-pair rejoin must execute every provider stage in order and restore usable metadata.");
  // Prepare one failed receiver QP with complete local and remote connection data.
  auto recvComm = std::make_unique<ncclIbRecvComm>();
  ncclIbResiliency ctx = {};
  ResetRejoinScenario(recvComm.get(), &ctx, false);
  ctx.qpInfo[0].lastAttemptNs = 0;

  // Rejoin must perform RESET, INIT, ECE, RTR, and RTS before publishing local readiness.
  ASSERT_EQ(ncclIbResiliencyTryRejoinQps(&ctx), ncclSuccess);
  EXPECT_EQ(fakes.queryPortCalls, 1);
  EXPECT_EQ(fakes.queriedContext, &fakeContext);
  EXPECT_EQ(fakes.queriedPort, 1);
  EXPECT_EQ(fakes.modifyQpCalls, 1);
  EXPECT_EQ(fakes.modifiedQp, &fakeCreatedQps[0]);
  EXPECT_EQ(fakes.initQpCalls, 1);
  EXPECT_EQ(fakes.initializedQp, &fakeCreatedQps[0]);
  EXPECT_EQ(fakes.initializedPort, 1);
  EXPECT_EQ(fakes.initializedAccess, 0u);
  EXPECT_EQ(fakes.rtrQpCalls, 1);
  EXPECT_EQ(fakes.rtrQp, &fakeCreatedQps[0]);
  EXPECT_EQ(fakes.rtrRemoteQpn, 900u);
  EXPECT_EQ(fakes.rtrLocalMtu, IBV_MTU_1024);
  EXPECT_EQ(fakes.rtrRemoteMtu, IBV_MTU_4096);
  EXPECT_TRUE(fakes.rtrFifoTc);
  EXPECT_EQ(fakes.rtsQpCalls, 1);
  EXPECT_EQ(fakes.rtsQp, &fakeCreatedQps[0]);
  EXPECT_EQ(fakes.setEceCalls, 1);
  EXPECT_EQ(fakes.eceQp, &fakeCreatedQps[0]);
  EXPECT_STREQ(fakes.rejoinStages, "RIETS");
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateLocalReady);
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 0));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 0));
  EXPECT_EQ(ctx.localReady.epoch[0], 1u);
  EXPECT_FALSE(ctx.frmAllQpsHealthy);

  // A zero remote epoch must not activate a locally rebuilt QP.
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 0, true);
  ctx.remoteReady.epoch[0] = 0;
  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateLocalReady);
  // A fresh remote epoch must restore the original QP and device.
  ctx.remoteReady.epoch[0] = 1;
  ASSERT_EQ(ncclIbResiliencyTryRejoinQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(recvComm->base.activeQps[0], &recvComm->base.qps[0]);
  EXPECT_EQ(ctx.activatedRemoteReadyEpoch[0], 1u);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);

  // Receiver progress must stop after all QPs return to service.
  ASSERT_EQ(ncclIbResiliencyProgress(&ctx), ncclSuccess);
  EXPECT_FALSE(ctx.inProgress);

  // Receiver progress must remain active while a locally rebuilt QP still waits for peer readiness.
  ctx.qpInfo[0].state = ncclIbResiliencyQpStateLocalReady;
  ctx.inProgress = true;
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, 0, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 0, false);
  ASSERT_EQ(ncclIbResiliencyProgress(&ctx), ncclSuccess);
  EXPECT_TRUE(ctx.inProgress);

  // A recent rejoin attempt must be deferred without querying or changing the failed QP.
  ctx.qpInfo[0].state = ncclIbResiliencyQpStateFailed;
  ctx.qpInfo[0].failCount = 0;
  ctx.qpInfo[0].lastAttemptNs = clockNano();
  const int queriesBeforeCooldown = fakes.queryPortCalls;
  ASSERT_EQ(ncclIbResiliencyTryRejoinSingleQp(&ctx, 0), ncclSuccess);
  EXPECT_EQ(fakes.queryPortCalls, queriesBeforeCooldown);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[0].failCount, 0);

  // An inactive local port must defer rejoin without consuming a retry.
  fakes.portState = IBV_PORT_DOWN;
  ctx.qpInfo[0].state = ncclIbResiliencyQpStateFailed;
  ctx.qpInfo[0].lastAttemptNs = 0;
  ctx.qpInfo[0].failCount = 0;
  ASSERT_EQ(ncclIbResiliencyTryRejoinSingleQp(&ctx, 0), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[0].failCount, 0);
  EXPECT_TRUE(ctx.qpInfo[0].localPortRetryLogged);

  // Repeated inactive-port checks must preserve the one-time diagnostic guard and recovery state.
  ctx.qpInfo[0].lastAttemptNs = 0;
  const int queriesBeforeRepeat = fakes.queryPortCalls;
  ASSERT_EQ(ncclIbResiliencyTryRejoinSingleQp(&ctx, 0), ncclSuccess);
  EXPECT_EQ(fakes.queryPortCalls, queriesBeforeRepeat + 1);
  EXPECT_TRUE(ctx.qpInfo[0].localPortRetryLogged);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[0].failCount, 0);

  // A failed RTR transition must return the QP to Failed and consume one retry.
  fakes.portState = IBV_PORT_ACTIVE;
  fakes.rtrQpResult = ncclSystemError;
  fakes.rejoinStages[0] = '\0';
  fakes.rejoinStageCount = 0;
  ctx.qpInfo[0].lastAttemptNs = 0;
  ASSERT_EQ(ncclIbResiliencyTryRejoinSingleQp(&ctx, 0), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[0].failCount, 1);
  EXPECT_TRUE(ctx.inProgress);
  EXPECT_STREQ(fakes.rejoinStages, "RIET");
}

/**
 * @brief Verifies that RESET, INIT, ECE, RTS, and receive-refill failures stop at their stage.
 */
TEST_F(NetIbResiliencyTest, QpRejoinStageFailures) {
  auto recvComm = std::make_unique<ncclIbRecvComm>();
  ncclIbResiliency context = {};

  // The rejoin sweep must skip both invalid local-device-index boundaries before provider access.
  for (int invalidIndex : {-1, 1}) {
    SCOPED_TRACE(::testing::Message() << "devIndex=" << invalidIndex);
    ResetRejoinScenario(recvComm.get(), &context, false);
    recvComm->base.qps[0].devIndex = invalidIndex;
    ASSERT_EQ(ncclIbResiliencyTryRejoinQps(&context), ncclSuccess);
    EXPECT_EQ(fakes.queryPortCalls, 0);
    EXPECT_EQ(fakes.modifyQpCalls, 0);
    EXPECT_EQ(context.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  }

  // A permanently failed local device must also be skipped before any provider transition.
  ResetRejoinScenario(recvComm.get(), &context, false);
  context.devs[0].state = ncclIbResiliencyDevStateErrorPermanent;
  ASSERT_EQ(ncclIbResiliencyTryRejoinQps(&context), ncclSuccess);
  EXPECT_EQ(fakes.queryPortCalls, 0);
  EXPECT_EQ(fakes.modifyQpCalls, 0);
  EXPECT_EQ(context.qpInfo[0].state, ncclIbResiliencyQpStateFailed);

  // Both invalid remote-device-index boundaries must fail before provider access.
  for (int invalidIndex : {-1, 1}) {
    SCOPED_TRACE(::testing::Message() << "remDevIdx=" << invalidIndex);
    ResetRejoinScenario(recvComm.get(), &context, false);
    recvComm->base.qps[0].remDevIdx = invalidIndex;
    ASSERT_EQ(ncclIbResiliencyTryRejoinSingleQp(&context, 0), ncclSuccess);
    EXPECT_EQ(fakes.queryPortCalls, 0);
    EXPECT_EQ(fakes.modifyQpCalls, 0);
    EXPECT_STREQ(fakes.rejoinStages, "");
    EXPECT_EQ(context.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
    EXPECT_EQ(context.qpInfo[0].failCount, 1);
  }

  /**
   * @brief Encodes the first failing rejoin stage and the provider-call prefix expected
   * before it stops.
   * @note The stage character selects the injected failure; trace is the ordered stage sequence.
   */
  struct FailureCase {
    const char* name;
    char stage;
    bool prepostReceives;
    int initCalls;
    int eceCalls;
    int rtrCalls;
    int rtsCalls;
    int recvCalls;
    const char* trace;
  };
  const FailureCase failureCases[] = {
      {"RESET", 'R', false, 0, 0, 0, 0, 0, "R"},
      {"INIT", 'I', false, 1, 0, 0, 0, 0, "RI"},
      {"ECE", 'E', false, 1, 1, 0, 0, 0, "RIE"},
      {"RTS", 'S', false, 1, 1, 1, 1, 0, "RIETS"},
      {"receive refill", 'Q', true, 1, 1, 1, 1, 1, "RIETSQ"},
  };

  // Configured RESET, INIT, ECE, RTS, and refill failures stop at that stage and consume one retry.
  for (const FailureCase& failure : failureCases) {
    SCOPED_TRACE(failure.name);
    ResetRejoinScenario(recvComm.get(), &context, failure.prepostReceives);
    if (failure.stage == 'R') fakes.modifyQpResult = ncclSystemError;
    if (failure.stage == 'I') fakes.initQpResult = ncclSystemError;
    if (failure.stage == 'E') fakes.setEceResult = ncclSystemError;
    if (failure.stage == 'S') fakes.rtsQpResult = ncclSystemError;
    if (failure.stage == 'Q') fakes.postRecvError = EIO;
    ASSERT_EQ(ncclIbResiliencyTryRejoinSingleQp(&context, 0), ncclSuccess);
    EXPECT_EQ(fakes.modifyQpCalls, 1);
    EXPECT_EQ(fakes.initQpCalls, failure.initCalls);
    EXPECT_EQ(fakes.setEceCalls, failure.eceCalls);
    EXPECT_EQ(fakes.rtrQpCalls, failure.rtrCalls);
    EXPECT_EQ(fakes.rtsQpCalls, failure.rtsCalls);
    EXPECT_EQ(fakes.postRecvCalls, failure.recvCalls);
    EXPECT_STREQ(fakes.rejoinStages, failure.trace);
    EXPECT_EQ(context.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
    EXPECT_EQ(context.qpInfo[0].failCount, 1);
    EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(context.localReady.mask, 0));
    EXPECT_TRUE(context.inProgress);
    if (failure.prepostReceives) {
      EXPECT_EQ(fakes.postedRecvQp, recvComm->base.qps[0].qp);
      EXPECT_EQ(fakes.postedRecvWr, &recvComm->ibRecvWorkRequest);
    }
  }

  // A provider report that ECE is unsupported must not block later transitions or local readiness.
  ResetRejoinScenario(recvComm.get(), &context, false);
  fakes.eceSupported = 0;
  ASSERT_EQ(ncclIbResiliencyTryRejoinSingleQp(&context, 0), ncclSuccess);
  EXPECT_EQ(fakes.setEceCalls, 1);
  EXPECT_EQ(fakes.rtrQpCalls, 1);
  EXPECT_EQ(fakes.rtsQpCalls, 1);
  EXPECT_STREQ(fakes.rejoinStages, "RIETS");
  EXPECT_EQ(context.qpInfo[0].state, ncclIbResiliencyQpStateLocalReady);
  EXPECT_EQ(context.qpInfo[0].failCount, 0);
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(context.localReady.mask, 0));

  // Rejoin without cached ECE must omit only that optional provider stage.
  ResetRejoinScenario(recvComm.get(), &context, false);
  recvComm->base.qpCold[0].remEceSupported = 0;
  context.localReady.epoch[0] = UINT32_MAX;
  ASSERT_EQ(ncclIbResiliencyTryRejoinSingleQp(&context, 0), ncclSuccess);
  EXPECT_EQ(fakes.setEceCalls, 0);
  EXPECT_EQ(fakes.rtrQpCalls, 1);
  EXPECT_EQ(fakes.rtsQpCalls, 1);
  EXPECT_STREQ(fakes.rejoinStages, "RITS");
  EXPECT_EQ(context.qpInfo[0].state, ncclIbResiliencyQpStateLocalReady);
  EXPECT_EQ(context.qpInfo[0].failCount, 0);
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(context.localReady.mask, 0));
  EXPECT_EQ(context.localReady.epoch[0], 1u);
}

/**
 * @brief Verifies send-recovery tracking initialization, replay, rearming, completion,
 * and rejection.
 */
TEST_F(ResiliencyFixture, SendRequestTracking) {
  // Prepare one live send request that can be tracked by its request slot.
  ncclIbResiliencySend sendCtx = {};
  base.isSend = true;
  sendCtx.base.baseComm = &base;
  sendCtx.base.ndevs = ctx.ndevs;
  ncclIbRequest request = MakeRequest(&base, 7, NCCL_NET_IB_REQ_SEND, 0);

  // Initial tracking must attach the request and increment outstanding recovery work.
  ASSERT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &request, 1), ncclSuccess);
  auto* failed = &sendCtx.failedRequests[request.id % NET_IB_MAX_REQUESTS];
  EXPECT_EQ(failed->request, &request);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->id, request.id + 1);
  EXPECT_EQ(failed->errorInfo.devIndex, 1);
  EXPECT_GT(failed->errorInfo.time, 0u);
  EXPECT_EQ(failed->failedAttempts, 0);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 1);
  EXPECT_TRUE(sendCtx.base.inProgress);

  // Only pre-replay states count as active recovery.
  EXPECT_TRUE(ncclIbResiliencySendRequestRecoveryActive(ncclIbResiliencyRequestStatePending));
  EXPECT_TRUE(ncclIbResiliencySendRequestRecoveryActive(ncclIbResiliencyRequestStateProbePosted));
  EXPECT_TRUE(ncclIbResiliencySendRequestRecoveryActive(ncclIbResiliencyRequestStateProbeCompleted));
  EXPECT_FALSE(ncclIbResiliencySendRequestRecoveryActive(ncclIbResiliencyRequestStateReplayPosted));

  // Posting a replay must keep tracking attached but remove outstanding recovery work.
  ASSERT_EQ(ncclIbResiliencySendRequestReplayPosted(&sendCtx, failed), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateReplayPosted);
  EXPECT_EQ(failed->request, &request);
  EXPECT_EQ(failed->errorInfo.devIndex, 0);
  EXPECT_EQ(failed->errorInfo.time, 0u);
  EXPECT_EQ(failed->failedAttempts, 0);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);

  // A new failure of the replayed request must rearm the same tracking entry.
  failed->failedAttempts = 4;
  failed->probeWrId = 0x1234;
  ASSERT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &request, 1), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->request, &request);
  EXPECT_EQ(failed->errorInfo.devIndex, 1);
  EXPECT_GT(failed->errorInfo.time, 0u);
  EXPECT_EQ(failed->failedAttempts, 0);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 1);

  // Free must clear an active entry once and reject a repeated free.
  const uint64_t releasedGeneration = failed->id;
  ASSERT_EQ(ncclIbResiliencySendRequestFree(&sendCtx, failed), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->request, nullptr);
  EXPECT_EQ(failed->errorInfo.devIndex, 0);
  EXPECT_EQ(failed->errorInfo.time, 0u);
  EXPECT_EQ(failed->failedAttempts, 0);
  EXPECT_EQ(failed->id, releasedGeneration);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);
  EXPECT_EQ(ncclIbResiliencySendRequestFree(&sendCtx, failed), ncclInternalError);

  // A late error for the request generation just freed must be ignored without rearming recovery.
  ASSERT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &request, 0), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->request, nullptr);
  EXPECT_EQ(failed->id, releasedGeneration);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);

  // A newer request generation in the same slot must be independently completable.
  request.id += NET_IB_MAX_REQUESTS;
  ASSERT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &request, 0), ncclSuccess);
  ASSERT_EQ(sendCtx.base.outstandingRequests, 1);
  ASSERT_EQ(ncclIbResiliencySendRequestComplete(&sendCtx.base, &request), ncclSuccess);
  EXPECT_EQ(failed->request, nullptr);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);

  // An older generation reusing the same slot must be rejected and leave the newer generation
  // marker intact.
  ncclIbRequest oldRequest = MakeRequest(
      &base, request.id - NET_IB_MAX_REQUESTS, NCCL_NET_IB_REQ_SEND, 0);
  const uint64_t trackedGeneration = failed->id;
  EXPECT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &oldRequest, 0), ncclInternalError);
  EXPECT_EQ(failed->id, trackedGeneration);
  EXPECT_EQ(failed->request, nullptr);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);

  // Completion after replay must clear tracking without decrementing the count twice.
  request.id += NET_IB_MAX_REQUESTS;
  ASSERT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &request, 0), ncclSuccess);
  ASSERT_EQ(ncclIbResiliencySendRequestReplayPosted(&sendCtx, failed), ncclSuccess);
  ASSERT_EQ(sendCtx.base.outstandingRequests, 0);
  ASSERT_EQ(ncclIbResiliencySendRequestComplete(&sendCtx.base, &request), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->request, nullptr);
  EXPECT_EQ(failed->errorInfo.devIndex, 0);
  EXPECT_EQ(failed->errorInfo.time, 0u);
  EXPECT_EQ(failed->failedAttempts, 0);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);

  // Null completion inputs must remain harmless no-ops.
  EXPECT_EQ(ncclIbResiliencySendRequestComplete(nullptr, &request), ncclSuccess);
  EXPECT_EQ(ncclIbResiliencySendRequestComplete(&sendCtx.base, nullptr), ncclSuccess);

  // An untracked generation must not alter its empty recovery slot or the outstanding count.
  ncclIbRequest untrackedRequest = MakeRequest(&base, 10, NCCL_NET_IB_REQ_SEND, 0);
  auto* untracked = &sendCtx.failedRequests[untrackedRequest.id % NET_IB_MAX_REQUESTS];
  ASSERT_EQ(ncclIbResiliencySendRequestComplete(&sendCtx.base, &untrackedRequest), ncclSuccess);
  EXPECT_EQ(untracked->request, nullptr);
  EXPECT_EQ(untracked->id, 0u);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);

  // A matching generation owned by another request pointer must remain attached to its real
  // request.
  ncclIbRequest pointerMismatch = untrackedRequest;
  untracked = TrackFailedRequest(
      &sendCtx, &untrackedRequest, ncclIbResiliencyRequestStatePending);
  untracked->errorInfo.devIndex = 1;
  untracked->errorInfo.time = 8;
  untracked->failedAttempts = 3;
  untracked->probeWrId = 0x5678;
  sendCtx.base.outstandingRequests = 1;
  ASSERT_EQ(ncclIbResiliencySendRequestComplete(&sendCtx.base, &pointerMismatch), ncclSuccess);
  EXPECT_EQ(untracked->request, &untrackedRequest);
  EXPECT_EQ(untracked->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(untracked->errorInfo.devIndex, 1);
  EXPECT_EQ(untracked->errorInfo.time, 8u);
  EXPECT_EQ(untracked->failedAttempts, 3);
  EXPECT_EQ(untracked->probeWrId, 0x5678u);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 1);
  untracked->request = nullptr;
  sendCtx.base.outstandingRequests = 0;

  // Receive communicators and non-SEND requests must not release a matching active tracking entry.
  ncclIbRequest guardedRequest = MakeRequest(&base, 9, NCCL_NET_IB_REQ_SEND, 0);
  auto* guarded = TrackFailedRequest(
      &sendCtx, &guardedRequest, ncclIbResiliencyRequestStatePending);
  guarded->errorInfo.devIndex = 1;
  guarded->errorInfo.time = 7;
  guarded->failedAttempts = 2;
  guarded->probeWrId = 0x1234;
  sendCtx.base.outstandingRequests = 1;
  /** @brief Describes one communicator direction and guarded request type. */
  struct CompletionGuardCase { bool isSend; int requestType; };
  const CompletionGuardCase guardCases[] = {
      {false, NCCL_NET_IB_REQ_SEND}, {true, NCCL_NET_IB_REQ_RECV}};
  for (const CompletionGuardCase& guardCase : guardCases) {
    SCOPED_TRACE(::testing::Message() << "isSend=" << guardCase.isSend << ", type=" << guardCase.requestType);
    base.isSend = guardCase.isSend;
    guardedRequest.type = guardCase.requestType;
    ASSERT_EQ(ncclIbResiliencySendRequestComplete(&sendCtx.base, &guardedRequest), ncclSuccess);
    EXPECT_EQ(guarded->request, &guardedRequest);
    EXPECT_EQ(guarded->state, ncclIbResiliencyRequestStatePending);
    EXPECT_EQ(guarded->errorInfo.devIndex, 1);
    EXPECT_EQ(guarded->errorInfo.time, 7u);
    EXPECT_EQ(guarded->failedAttempts, 2);
    EXPECT_EQ(guarded->probeWrId, 0x1234u);
    EXPECT_EQ(sendCtx.base.outstandingRequests, 1);
  }

  // Recovery tracking must reject non-send requests.
  ncclIbRequest nonSend = MakeRequest(&base, 8, NCCL_NET_IB_REQ_RECV, 0);
  EXPECT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &nonSend, 0), ncclInternalError);
}

/**
 * @brief Verifies request-type rejection and receive/send replay behavior.
 */
TEST_F(NetIbResiliencyTest, RequestRepost) {
  SCOPED_TRACE("Request replay must repost the recorded operation once and propagate posting errors.");
  // Unsupported request types must be rejected before any provider operation.
  ncclIbRequest request = {};
  request.id = 5;
  request.type = NCCL_NET_IB_REQ_UNUSED;
  EXPECT_EQ(ncclIbResiliencyRepostRequest(&request), ncclInternalError);
  request.type = NCCL_NET_IB_REQ_FLUSH;
  EXPECT_EQ(ncclIbResiliencyRepostRequest(&request), ncclInternalError);

  // Receive replay must repost the CTS in the request's slot and propagate errors.
  auto recvComm = std::make_unique<ncclIbRecvComm>();
  request.type = NCCL_NET_IB_REQ_RECV;
  request.base = &recvComm->base;
  EXPECT_EQ(ncclIbResiliencyRepostRequest(&request), ncclSuccess);
  EXPECT_EQ(fakes.postFifoCalls, 1);
  EXPECT_EQ(fakes.postFifoComm, recvComm.get());
  EXPECT_EQ(fakes.postFifoRequest, &request);
  EXPECT_EQ(fakes.postFifoSlot, 5 % NET_IB_MAX_REQUESTS);
  fakes.postFifoResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencyRepostRequest(&request), ncclSystemError);
  EXPECT_EQ(fakes.postFifoCalls, 2);
  EXPECT_EQ(fakes.postFifoComm, recvComm.get());
  EXPECT_EQ(fakes.postFifoRequest, &request);
  EXPECT_EQ(fakes.postFifoSlot, 5 % NET_IB_MAX_REQUESTS);

  // Prepare a send replay with one delivered and one still-pending logical chunk.
  ResetFakes();
  auto [sendComm, unusedSendContext] = MakeState<true>(2, 2, 2);
  ConfigureActiveQps(&sendComm->base, &unusedSendContext->base, {0, 1});
  request = MakeRequest(&sendComm->base, 6, NCCL_NET_IB_REQ_SEND, 2);
  request.send.pendingData[0] = false;
  request.send.pendingData[1] = true;
  request.events[0] = 9;
  request.events[1] = 9;
  ncclIbRequest groupedRequest = MakeRequest(&sendComm->base, request.id, NCCL_NET_IB_REQ_SEND, request.nreqs);
  groupedRequest.events[0] = 7;
  groupedRequest.events[1] = 7;
  const int slot = request.id % NET_IB_MAX_REQUESTS;
  BindSendRequest(sendComm.get(), &request, 0);
  BindSendRequest(sendComm.get(), &groupedRequest, 1);

  // Replay must rebuild grouped requests only for pending data and request a shared WR-ID refresh.
  ASSERT_EQ(ncclIbResiliencyRepostRequest(&request), ncclSuccess);
  EXPECT_EQ(request.events[0], 0);
  EXPECT_EQ(request.events[1], 1);
  EXPECT_EQ(groupedRequest.events[0], 0);
  EXPECT_EQ(groupedRequest.events[1], 1);
  ExpectReplayPosts(sendComm.get(), slot, 2, 1);

  // A send-post failure must be propagated to the caller.
  fakes.multiSendResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencyRepostRequest(&request), ncclSystemError);
  ExpectReplayPosts(sendComm.get(), slot, 2, 2);
}

/**
 * @brief Verifies probe construction, completion, stale handling, failover, limits, and provider
 * errors.
 */
TEST_F(NetIbResiliencyTest, ProbePostingAndProgress) {
  SCOPED_TRACE("Probe posting and completion must advance recovery state with exact work-request metadata.");
  // Prepare one failed and one healthy device with a pending send request.
  auto [sendComm, sendCtx] = MakeState<true>(2, 2, 2);
  ConfigureActiveQps(&sendComm->base, &sendCtx->base, {0, 1});
  sendCtx->base.devs[0].state = ncclIbResiliencyDevStateError;
  ConfigureProbePath(sendCtx.get(), 1, 0);
  ncclIbRequest request = MakeRequest(&sendComm->base, 3, NCCL_NET_IB_REQ_SEND, 0);
  const int slot = request.id % NET_IB_MAX_REQUESTS;
  ncclIbResiliencyRequestSend* failed =
      TrackFailedRequest(sendCtx.get(), &request, ncclIbResiliencyRequestStatePending);
  failed->errorInfo.time = clockNano();

  // A pending request whose probe delay has not elapsed must remain pending without posting.
  testPortFailoverProbeDelay = 60000;
  ASSERT_EQ(ncclIbResiliencyProbePost(sendCtx.get(), failed), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 0);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(sendCtx->nextProbeWrId, IB_RESILIENCY_PROBE_WR_ID_BASE);
  testPortFailoverProbeDelay = 0;

  // Probe posting must build the expected RDMA read and attach its unique token.
  ASSERT_EQ(ncclIbResiliencyProbePost(sendCtx.get(), failed), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 1);
  EXPECT_EQ(fakes.postedSendQp, &fakeCreatedQps[1]);
  EXPECT_EQ(fakes.postedSend.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(fakes.postedSend.send_flags, IBV_SEND_SIGNALED);
  EXPECT_EQ(fakes.postedSend.num_sge, 1);
  EXPECT_EQ(fakes.postedSend.next, nullptr);
  EXPECT_EQ(ibResiliencyProbeWrIdDecodeSlot(fakes.postedSend.wr_id), static_cast<uint64_t>(slot));
  EXPECT_EQ(fakes.postedSendSge.addr, reinterpret_cast<uint64_t>(&sendCtx->probingResults[slot]));
  EXPECT_EQ(fakes.postedSendSge.length, sizeof(sendCtx->probingResults[slot]));
  EXPECT_EQ(fakes.postedSendSge.lkey, fakeMrs[0].lkey);
  const uint64_t expectedRemoteAddress = kRemoteCompletionAddress + sizeof(ncclIbRequestCompletionRecord) * slot +
      sizeof(int) * NCCL_NET_IB_MAX_RECVS;
  EXPECT_EQ(fakes.postedSend.wr.rdma.remote_addr, expectedRemoteAddress);
  EXPECT_EQ(fakes.postedSend.wr.rdma.rkey, kRemoteCompletionKey);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateProbePosted);
  EXPECT_EQ(failed->probeWrId, fakes.postedSend.wr_id);
  EXPECT_EQ(sendCtx->base.devs[1].nOutstandingProbes, 1);
  EXPECT_EQ(sendCtx->nextProbeWrId, IB_RESILIENCY_PROBE_WR_ID_BASE + IB_RESILIENCY_PROBE_WR_ID_STRIDE);

  // A matching successful completion must finish the probe and clear its token.
  QueueCompletion(IBV_WC_SUCCESS, failed->probeWrId);
  ASSERT_EQ(ncclIbResiliencyProbeProgress(sendCtx.get()), ncclSuccess);
  EXPECT_EQ(sendCtx->base.devs[1].nOutstandingProbes, 0);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateProbeCompleted);
  EXPECT_EQ(failed->probeWrId, 0u);

  // Probe progress must also consume a successful peer-ready CQE found on the shared probing CQ.
  sendCtx->base.peerReadyReadInFlight = true;
  sendCtx->base.peerReadyReadQpIndex = 1;
  sendCtx->base.peerReadyReadDevIndex = 1;
  sendCtx->base.devs[1].peerReadyFailureCount = 2;
  sendCtx->base.devs[1].peerReadyRetryAfterNs = UINT64_MAX;
  sendCtx->base.devs[1].nOutstandingProbes = 1;
  QueueCompletion(IBV_WC_SUCCESS, IB_RESILIENCY_PEER_READY_WR_ID);
  ASSERT_EQ(ncclIbResiliencyProbeProgress(sendCtx.get()), ncclSuccess);
  EXPECT_FALSE(sendCtx->base.peerReadyReadInFlight);
  EXPECT_EQ(sendCtx->base.peerReadyReadQpIndex, -1);
  EXPECT_EQ(sendCtx->base.peerReadyReadDevIndex, -1);
  EXPECT_EQ(sendCtx->base.devs[1].peerReadyFailureCount, 0);
  EXPECT_EQ(sendCtx->base.devs[1].peerReadyRetryAfterNs, 0u);
  EXPECT_EQ(sendCtx->base.devs[1].nOutstandingProbes, 0);

  // The same loop must convert a failed peer-ready CQE into backoff without failing the data
  // device.
  sendCtx->base.peerReadyReadInFlight = true;
  sendCtx->base.peerReadyReadQpIndex = 1;
  sendCtx->base.peerReadyReadDevIndex = 1;
  sendCtx->base.remoteReady.mask[0] = UINT64_MAX;
  sendCtx->base.devs[1].nOutstandingProbes = 1;
  QueueCompletion(IBV_WC_WR_FLUSH_ERR, IB_RESILIENCY_PEER_READY_WR_ID,
      fakeCreatedQps[1].qp_num);
  ASSERT_EQ(ncclIbResiliencyProbeProgress(sendCtx.get()), ncclSuccess);
  EXPECT_FALSE(sendCtx->base.peerReadyReadInFlight);
  EXPECT_EQ(sendCtx->base.remoteReady.mask[0], 0u);
  EXPECT_EQ(sendCtx->base.devs[1].peerReadyFailureCount, 1);
  EXPECT_GT(sendCtx->base.devs[1].peerReadyRetryAfterNs, 0u);
  EXPECT_EQ(sendCtx->base.devs[1].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendCtx->base.devs[1].nOutstandingProbes, 0);

  // A completion carrying an already-consumed token must be ignored as stale.
  ibv_wc stale = {};
  stale.status = IBV_WC_SUCCESS;
  stale.wr_id = fakes.postedSend.wr_id;
  EXPECT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(sendCtx.get(), &stale, 1), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateProbeCompleted);

  // A matching late probe CQE for an already-freed request must leave the empty tracking slot
  // untouched.
  failed->state = ncclIbResiliencyRequestStateProbePosted;
  failed->probeWrId = ibResiliencyProbeWrIdEncode(sendCtx->nextProbeWrId, slot);
  failed->request = nullptr;
  stale.wr_id = failed->probeWrId;
  ASSERT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(sendCtx.get(), &stale, 1), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateProbePosted);
  EXPECT_EQ(failed->probeWrId, stale.wr_id);
  EXPECT_EQ(sendCtx->base.devs[1].state, ncclIbResiliencyDevStateOk);
  failed->request = &request;

  // A recoverable probe error must quarantine only its auxiliary QP and return the request to
  // Pending without changing data-path state.
  sendCtx->base.devs[0].state = ncclIbResiliencyDevStateOk;
  failed->state = ncclIbResiliencyRequestStateProbePosted;
  failed->probeWrId = ibResiliencyProbeWrIdEncode(sendCtx->nextProbeWrId, slot);
  failed->failedAttempts = 0;
  ibv_wc failedProbe = {};
  failedProbe.status = IBV_WC_WR_FLUSH_ERR;
  failedProbe.wr_id = failed->probeWrId;
  failedProbe.qp_num = fakeCreatedQps[1].qp_num;
  ASSERT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(sendCtx.get(), &failedProbe, 1), ncclSuccess);
  EXPECT_EQ(sendCtx->base.failedProbeQpMask, 1ULL << 1);
  EXPECT_EQ(sendCtx->base.devs[1].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendCtx->base.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(sendComm->base.activeQps[1], &sendComm->base.qps[1]);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->failedAttempts, 1);
  EXPECT_EQ(failed->probeWrId, 0u);

  // A repeated recoverable failure from that QP must preserve data geometry while advancing the
  // request's retry state.
  sendCtx->base.devs[0].state = ncclIbResiliencyDevStateOk;
  sendCtx->base.devs[1].state = ncclIbResiliencyDevStateOk;
  sendCtx->base.qpInfo[0].state = ncclIbResiliencyQpStateLocalReady;
  sendCtx->base.qpInfo[1].state = ncclIbResiliencyQpStateOk;
  sendComm->base.activeQps[1] = &sendComm->base.qps[1];
  failed->state = ncclIbResiliencyRequestStateProbePosted;
  failed->probeWrId = ibResiliencyProbeWrIdEncode(sendCtx->nextProbeWrId, slot);
  failed->failedAttempts = 0;
  failedProbe.wr_id = failed->probeWrId;
  EXPECT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(sendCtx.get(), &failedProbe, 1), ncclSuccess);
  EXPECT_EQ(sendCtx->base.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendCtx->base.devs[1].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendCtx->base.qpInfo[0].state, ncclIbResiliencyQpStateLocalReady);
  EXPECT_EQ(sendCtx->base.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(sendComm->base.activeQps[1], &sendComm->base.qps[1]);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(failed->failedAttempts, 1);

  // Attempts beyond the configured limit must stop before posting another probe.
  failed->state = ncclIbResiliencyRequestStatePending;
  failed->failedAttempts = static_cast<int>(ncclParamIbResiliencyPortFailoverMaxAttempts()) + 1;
  EXPECT_EQ(ncclIbResiliencyProbePost(sendCtx.get(), failed), ncclRemoteError);
  EXPECT_EQ(fakes.postSendCalls, 1);

  // Probe posting must fail when no usable auxiliary QP remains.
  failed->failedAttempts = 0;
  sendCtx->base.devs[0].state = ncclIbResiliencyDevStateError;
  sendCtx->base.devs[1].state = ncclIbResiliencyDevStateError;
  EXPECT_EQ(ncclIbResiliencyProbePost(sendCtx.get(), failed), ncclRemoteError);

  // A provider post error advances the next token but preserves request and outstanding-probe
  // state.
  sendCtx->base.devs[1].state = ncclIbResiliencyDevStateOk;
  sendCtx->base.failedProbeQpMask = 0;
  fakes.postSendError = EIO;
  EXPECT_EQ(ncclIbResiliencyProbePost(sendCtx.get(), failed), ncclSystemError);
  EXPECT_EQ(fakes.postSendCalls, 2);
  EXPECT_EQ(fakes.postedSendQp, &fakeCreatedQps[1]);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(sendCtx->base.devs[1].nOutstandingProbes, 0);

  // A CQ polling error must be returned without consuming outstanding probes.
  fakes.postSendError = 0;
  fakes.pollFails = true;
  sendCtx->base.devs[1].nOutstandingProbes = 1;
  EXPECT_EQ(ncclIbResiliencyProbeProgress(sendCtx.get()), ncclSystemError);
  EXPECT_EQ(sendCtx->base.devs[1].nOutstandingProbes, 1);

  // The same CQ error must propagate through public sender progress without consuming probe
  // bookkeeping.
  for (int index = 0; index < 2; index++) sendCtx->base.qpInfo[index].state = ncclIbResiliencyQpStateOk;
  sendCtx->base.inProgress = true;
  EXPECT_EQ(ncclIbResiliencyProgress(&sendCtx->base), ncclSystemError);
  EXPECT_EQ(sendCtx->base.devs[1].nOutstandingProbes, 1);
  EXPECT_TRUE(sendCtx->base.inProgress);
}

/**
 * @brief Verifies that probe results complete delivered data and replay missing data.
 */
TEST_F(NetIbResiliencyTest, ProbeResultsCompleteOrReplay) {
  // Prepare a request whose two logical chunks were both observed remotely.
  auto [sendComm, sendCtx] = MakeState<true>(2, 2, 2);
  ConfigureActiveQps(&sendComm->base, &sendCtx->base, {0, 1});
  ncclIbRequest request = MakeRequest(&sendComm->base, 4, NCCL_NET_IB_REQ_SEND, 2);
  ncclIbRequest groupedRequest = MakeRequest(&sendComm->base, request.id, NCCL_NET_IB_REQ_SEND, request.nreqs);
  const int slot = request.id % NET_IB_MAX_REQUESTS;
  BindSendRequest(sendComm.get(), &request, 0);
  BindSendRequest(sendComm.get(), &groupedRequest, 1);
  ncclIbResiliencyRequestSend* failed =
      TrackFailedRequest(sendCtx.get(), &request, ncclIbResiliencyRequestStatePending);
  for (int index = 0; index < 2; index++) {
    request.events[index] = 1;
    request.send.pendingData[index] = true;
    request.send.lastAttemptQpIndex[index] = index;
    request.send.lastAttemptOrder[index] = 0;
    groupedRequest.events[index] = 2;
    groupedRequest.send.pendingData[index] = true;
    groupedRequest.send.lastAttemptQpIndex[index] = index;
    groupedRequest.send.lastAttemptOrder[index] = 0;
    sendCtx->probingResults[slot][index] = 1;
  }

  // Fully observed data must be credited and completed without replay.
  bool reposted = true;
  ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(sendCtx.get(), failed, &reposted), ncclSuccess);
  EXPECT_FALSE(reposted);
  for (ncclIbRequest* current : {&request, &groupedRequest}) {
    SCOPED_TRACE(current == &request ? "primary" : "grouped");
    EXPECT_FALSE(current->send.pendingData[0]);
    EXPECT_FALSE(current->send.pendingData[1]);
    EXPECT_EQ(current->send.ackedCompletions[0], 1u);
    EXPECT_EQ(current->send.ackedCompletions[1], 1u);
    EXPECT_EQ(current->events[0], 0);
    EXPECT_EQ(current->events[1], 0);
  }
  EXPECT_EQ(fakes.multiSendCalls, 0);

  // A missing logical chunk must preserve only its event and trigger one replay.
  std::memset(&request.send, 0, sizeof(request.send));
  std::memset(&groupedRequest.send, 0, sizeof(groupedRequest.send));
  std::memset(sendCtx->probingResults[slot], 0, sizeof(sendCtx->probingResults[slot]));
  for (ncclIbRequest* current : {&request, &groupedRequest}) {
    current->send.pendingData[1] = true;
    current->send.lastAttemptQpIndex[1] = 1;
    current->send.lastAttemptOrder[1] = 0;
  }
  reposted = false;
  ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(sendCtx.get(), failed, &reposted), ncclSuccess);
  EXPECT_TRUE(reposted);
  EXPECT_EQ(request.events[0], 0);
  EXPECT_EQ(request.events[1], 1);
  EXPECT_EQ(groupedRequest.events[0], 0);
  EXPECT_EQ(groupedRequest.events[1], 1);
  ExpectReplayPosts(sendComm.get(), slot, 2, 1);

  // An extra physical completion must credit the only pending chunk and stop without a false
  // replay.
  std::memset(&request.send, 0, sizeof(request.send));
  std::memset(&groupedRequest.send, 0, sizeof(groupedRequest.send));
  std::memset(sendCtx->probingResults[slot], 0, sizeof(sendCtx->probingResults[slot]));
  request.events[0] = 3;
  request.events[1] = 3;
  groupedRequest.events[0] = 4;
  groupedRequest.events[1] = 4;
  for (ncclIbRequest* current : {&request, &groupedRequest}) {
    current->send.pendingData[0] = true;
    current->send.lastAttemptQpIndex[0] = 0;
  }
  sendCtx->probingResults[slot][0] = 2;
  fakes.refreshTokenCalls = 0;
  fakes.multiSendCalls = 0;
  reposted = true;
  ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(sendCtx.get(), failed, &reposted), ncclSuccess);
  EXPECT_FALSE(reposted);
  for (ncclIbRequest* current : {&request, &groupedRequest}) {
    SCOPED_TRACE(current == &request ? "primary" : "grouped");
    EXPECT_FALSE(current->send.pendingData[0]);
    EXPECT_EQ(current->send.lastAttemptQpIndex[0], -1);
    EXPECT_EQ(current->send.ackedCompletions[0], 2u);
    EXPECT_EQ(current->events[0], 0);
    EXPECT_EQ(current->events[1], 0);
  }
  EXPECT_EQ(fakes.refreshTokenCalls, 0);
  EXPECT_EQ(fakes.multiSendCalls, 0);

  // Shared physical-QP credit must satisfy the earliest send order rather than the lowest logical
  // index.
  std::memset(&request.send, 0, sizeof(request.send));
  std::memset(&groupedRequest.send, 0, sizeof(groupedRequest.send));
  std::memset(sendCtx->probingResults[slot], 0, sizeof(sendCtx->probingResults[slot]));
  sendComm->base.nqps = 3;
  sendComm->base.nDataQps = 3;
  ConfigureActiveQps(&sendComm->base, &sendCtx->base, {0, 1, 1});
  request.send.pendingData[0] = true;
  request.send.pendingData[1] = true;
  request.send.pendingData[2] = true;
  request.send.lastAttemptQpIndex[0] = 0;
  request.send.lastAttemptQpIndex[1] = 0;
  request.send.lastAttemptQpIndex[2] = 0;
  request.send.lastAttemptOrder[0] = 2;
  request.send.lastAttemptOrder[1] = 0;
  request.send.lastAttemptOrder[2] = 1;
  groupedRequest.send = request.send;
  sendCtx->probingResults[slot][0] = 1;
  fakes.refreshTokenCalls = 0;
  fakes.multiSendCalls = 0;
  reposted = false;
  ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(sendCtx.get(), failed, &reposted), ncclSuccess);
  EXPECT_TRUE(reposted);
  for (ncclIbRequest* current : {&request, &groupedRequest}) {
    SCOPED_TRACE(current == &request ? "primary" : "grouped");
    EXPECT_TRUE(current->send.pendingData[0]);
    EXPECT_FALSE(current->send.pendingData[1]);
    EXPECT_TRUE(current->send.pendingData[2]);
    EXPECT_EQ(current->send.lastAttemptQpIndex[0], 0);
    EXPECT_EQ(current->send.lastAttemptQpIndex[1], -1);
    EXPECT_EQ(current->send.lastAttemptQpIndex[2], 0);
    EXPECT_EQ(current->send.ackedCompletions[0], 1u);
    EXPECT_EQ(current->events[0], 1);
    EXPECT_EQ(current->events[1], 1);
  }
  ExpectReplayPosts(sendComm.get(), slot, 2, 1);
}

/**
 * @brief Verifies sender progress completion and QP lookup-map rebuilding.
 */
TEST_F(NetIbResiliencyTest, SenderProgressCompletesProbe) {
  SCOPED_TRACE("Sender progress must consume a successful probe completion and finish the probe phase.");
  // Prepare one completed probe with no data left to replay.
  auto [sendComm, sendCtx] = MakeState<true>(1, 1, 1);
  ConfigureActiveQps(&sendComm->base, &sendCtx->base, {0});
  sendCtx->base.inProgress = true;
  sendCtx->base.outstandingRequests = 1;
  ncclIbRequest request = MakeRequest(&sendComm->base, 9, NCCL_NET_IB_REQ_SEND, 1);
  request.events[0] = 1;
  BindSendRequest(sendComm.get(), &request, 0);
  ncclIbResiliencyRequestSend* failed =
      TrackFailedRequest(sendCtx.get(), &request, ncclIbResiliencyRequestStateProbeCompleted);

  // Progress must complete the request, stop recovery, and rebuild the QP lookup map.
  ASSERT_EQ(ncclIbResiliencyProgress(&sendCtx->base), ncclSuccess);
  EXPECT_EQ(request.events[0], 0);
  EXPECT_EQ(failed->request, nullptr);
  EXPECT_EQ(sendCtx->base.outstandingRequests, 0);
  EXPECT_FALSE(sendCtx->base.inProgress);
  EXPECT_EQ(qpMapFind(&sendComm->base.frmFields.completion.qpHash, fakeCreatedQps[0].qp_num), 0);
}

/**
 * @brief Verifies that sender progress posts a pending probe and replays data after completion.
 */
TEST_F(NetIbResiliencyTest, SenderProgressTransitions) {
  SCOPED_TRACE("Sender progress must follow each valid recovery transition and return dependency failures.");
  // Prepare one healthy probe path and one pending failed send request.
  testPortFailoverProbeDelay = 0;
  auto [sendComm, sendContext] = MakeState<true>(1, 1, 1);
  ConfigureActiveQps(&sendComm->base, &sendContext->base, {0});
  sendContext->base.inProgress = true;
  sendContext->base.outstandingRequests = 1;
  ConfigureProbePath(sendContext.get(), 0, 0);
  ncclIbRequest request = MakeRequest(&sendComm->base, 10, NCCL_NET_IB_REQ_SEND, 1);
  request.send.pendingData[0] = true;
  request.send.lastAttemptQpIndex[0] = 0;
  request.send.lastAttemptOrder[0] = 0;
  const int slot = request.id % NET_IB_MAX_REQUESTS;
  BindSendRequest(sendComm.get(), &request, 0);
  ncclIbResiliencyRequestSend* failed =
      TrackFailedRequest(sendContext.get(), &request, ncclIbResiliencyRequestStatePending);

  // A provider post failure through public progress must leave the request pending and recovery
  // active.
  fakes.postSendError = EIO;
  EXPECT_EQ(ncclIbResiliencyProgress(&sendContext->base), ncclSystemError);
  EXPECT_EQ(fakes.postSendCalls, 1);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(sendContext->base.devs[0].nOutstandingProbes, 0);
  EXPECT_EQ(sendContext->base.outstandingRequests, 1);
  EXPECT_TRUE(sendContext->base.inProgress);
  EXPECT_EQ(sendContext->nextProbeWrId, IB_RESILIENCY_PROBE_WR_ID_BASE + IB_RESILIENCY_PROBE_WR_ID_STRIDE);
  fakes.postSendError = 0;
  fakes.postSendCalls = 0;

  // The first progress call must move Pending to ProbePosted and keep recovery active.
  ASSERT_EQ(ncclIbResiliencyProgress(&sendContext->base), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 1);
  EXPECT_EQ(fakes.postedSendQp, &fakeCreatedQps[0]);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateProbePosted);
  EXPECT_NE(failed->probeWrId, 0u);
  EXPECT_EQ(sendContext->base.devs[0].nOutstandingProbes, 1);
  EXPECT_EQ(sendContext->base.outstandingRequests, 1);
  EXPECT_TRUE(sendContext->base.inProgress);

  // A completed probe with missing data must trigger replay and leave its tracking armed for stale
  // CQEs.
  failed->state = ncclIbResiliencyRequestStateProbeCompleted;
  failed->probeWrId = 0;
  sendContext->base.devs[0].nOutstandingProbes = 0;
  sendContext->probingResults[slot][0] = 0;

  // A replay-post failure must propagate through progress without advancing the tracked request
  // state.
  fakes.multiSendResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencyProgress(&sendContext->base), ncclSystemError);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateProbeCompleted);
  EXPECT_EQ(failed->request, &request);
  EXPECT_EQ(sendContext->base.outstandingRequests, 1);
  EXPECT_TRUE(sendContext->base.inProgress);
  fakes.multiSendResult = ncclSuccess;
  fakes.refreshTokenCalls = 0;
  fakes.multiSendCalls = 0;

  ASSERT_EQ(ncclIbResiliencyProgress(&sendContext->base), ncclSuccess);
  ExpectReplayPosts(sendComm.get(), slot, 1, 1);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateReplayPosted);
  EXPECT_EQ(failed->request, &request);
  EXPECT_TRUE(request.send.pendingData[0]);
  EXPECT_EQ(request.events[0], 1);
  EXPECT_EQ(sendContext->base.outstandingRequests, 0);
  EXPECT_FALSE(sendContext->base.inProgress);

  // ReplayPosted tracking must be left alone while a LocalReady QP keeps sender recovery active.
  sendContext->base.qpInfo[0].state = ncclIbResiliencyQpStateLocalReady;
  ncclIbResiliencyReadyQpSetBit(sendContext->base.localReady.mask, 0, true);
  ncclIbResiliencyReadyQpSetBit(sendContext->base.remoteReady.mask, 0, false);
  sendContext->base.inProgress = true;
  const int probePostsBeforeReplayWait = fakes.postSendCalls;
  const int replayPostsBeforeReplayWait = fakes.multiSendCalls;
  ASSERT_EQ(ncclIbResiliencyProgress(&sendContext->base), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateReplayPosted);
  EXPECT_EQ(failed->request, &request);
  EXPECT_EQ(fakes.postSendCalls, probePostsBeforeReplayWait);
  EXPECT_EQ(fakes.multiSendCalls, replayPostsBeforeReplayWait);
  EXPECT_TRUE(sendContext->base.inProgress);
}

/**
 * @brief Verifies peer-ready read posting, polling, backoff, and QP activation.
 */
TEST_F(ResiliencyFixture, PeerReadyHandshake) {
  // Polling without an in-flight read is a harmless no-op.
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclSuccess);
  // Both invalid stored device-index boundaries must be rejected before CQ access.
  for (int invalidIndex : {-1, ctx.ndevs}) {
    SCOPED_TRACE(::testing::Message() << "devIndex=" << invalidIndex);
    ctx.peerReadyReadInFlight = true;
    ctx.peerReadyReadDevIndex = invalidIndex;
    EXPECT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclInternalError);
  }
  ctx.peerReadyReadInFlight = false;

  // A peer-ready CQ polling failure must propagate through rejoin and public progress without
  // consuming state.
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadDevIndex = 0;
  ctx.devs[0].probingCq = &fakeCq;
  ctx.devs[0].nOutstandingProbes = 1;
  ctx.inProgress = true;
  fakes.pollFails = true;
  EXPECT_EQ(ncclIbResiliencyProgress(&ctx), ncclSystemError);
  EXPECT_TRUE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.devs[0].nOutstandingProbes, 1);
  EXPECT_TRUE(ctx.inProgress);
  fakes.pollFails = false;
  ctx.peerReadyReadInFlight = false;
  ctx.devs[0].nOutstandingProbes = 0;

  // Posting must be skipped when remote access data or a LocalReady QP is absent.
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  ctx.remoteReadyStateAddr = 0x1000;
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateOk;
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 0);

  // A LocalReady QP without any configured probing QP must remain pending without a provider call.
  const int configuredProbingQps = ctx.nProbingQps;
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateLocalReady;
  ctx.nProbingQps = 0;
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 0);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  ctx.nProbingQps = configuredProbingQps;

  // A LocalReady data QP without any usable probing QP must remain pending without a provider call.
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 0);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);

  // A valid LocalReady QP must post one peer-ready RDMA read.
  ctx.probingQps[0].qp = &fakeCreatedQps[0];
  ctx.probingQps[0].devIndex = 0;
  ctx.probingQps[0].remDevIdx = 0;
  ctx.devs[0].peerReadyStateMr = &fakeMrs[0];
  ctx.devs[0].probingCq = &fakeCq;
  ctx.remoteReadyStateRkeys[0] = 0x55;
  const ncclIbResiliency validPeerReadyContext = ctx;
  /** @brief Identifies which peer-read precondition a blocked-probe row invalidates. */
  enum class ProbeBlock { LocalIndex, DeviceState, RetryDelay, RemoteIndex, RemoteKey, PollInterval };
  /**
   * @brief Encodes one invalid peer-read precondition.
   * @note @p value is interpreted according to @p field; RetryDelay ignores it.
   */
  struct BlockedProbeCase {
    const char* name;
    ProbeBlock field;
    int64_t value;
  };
  const BlockedProbeCase blockedProbeCases[] = {
      {"negative local device", ProbeBlock::LocalIndex, -1},
      {"upper local device", ProbeBlock::LocalIndex, ctx.ndevs},
      {"failed local device", ProbeBlock::DeviceState, ncclIbResiliencyDevStateError},
      {"device retry backoff", ProbeBlock::RetryDelay, -1},
      {"negative remote device", ProbeBlock::RemoteIndex, -1},
      {"upper remote device", ProbeBlock::RemoteIndex, base.nRemDevs},
      {"missing remote key", ProbeBlock::RemoteKey, 0},
      {"minimum poll interval", ProbeBlock::PollInterval, 60000},
  };

  // Every invalid probing candidate must leave the LocalReady QP pending without a provider call.
  for (const BlockedProbeCase& blocked : blockedProbeCases) {
    SCOPED_TRACE(blocked.name);
    ctx = validPeerReadyContext;
    fakes.postSendCalls = 0;
    if (blocked.field == ProbeBlock::LocalIndex) ctx.probingQps[0].devIndex = blocked.value;
    if (blocked.field == ProbeBlock::DeviceState) {
      ctx.devs[0].state = static_cast<enum ncclIbResiliencyDevState>(blocked.value);
    }
    if (blocked.field == ProbeBlock::RetryDelay) ctx.devs[0].peerReadyRetryAfterNs = UINT64_MAX;
    if (blocked.field == ProbeBlock::RemoteIndex) ctx.probingQps[0].remDevIdx = blocked.value;
    if (blocked.field == ProbeBlock::RemoteKey) ctx.remoteReadyStateRkeys[0] = blocked.value;
    if (blocked.field == ProbeBlock::PollInterval) {
      testPortFailoverProbeDelay = blocked.value;
      ctx.lastPeerReadyReadNs = clockNano();
    }
    ASSERT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
    EXPECT_EQ(fakes.postSendCalls, 0);
    EXPECT_FALSE(ctx.peerReadyReadInFlight);
    testPortFailoverProbeDelay = 1000;
  }
  ctx = validPeerReadyContext;
  testPortFailoverProbeDelay = 1000;
  ctx.lastPeerReadyReadNs = 0;

  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  EXPECT_TRUE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, 0);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, 0);
  EXPECT_EQ(ctx.nextPeerReadyQpIndex, 1);
  EXPECT_EQ(ctx.devs[0].nOutstandingProbes, 1);
  EXPECT_EQ(fakes.postSendCalls, 1);
  EXPECT_EQ(fakes.postedSendQp, &fakeCreatedQps[0]);
  EXPECT_EQ(fakes.postedSend.wr_id, IB_RESILIENCY_PEER_READY_WR_ID);
  EXPECT_EQ(fakes.postedSend.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(fakes.postedSend.send_flags, IBV_SEND_SIGNALED);
  EXPECT_EQ(fakes.postedSend.num_sge, 1);
  EXPECT_EQ(fakes.postedSendSge.addr, reinterpret_cast<uint64_t>(&ctx.remoteReady));
  EXPECT_EQ(fakes.postedSendSge.length, sizeof(ctx.remoteReady));
  EXPECT_EQ(fakes.postedSendSge.lkey, fakeMrs[0].lkey);
  EXPECT_EQ(fakes.postedSend.wr.rdma.remote_addr, 0x1000u);
  EXPECT_EQ(fakes.postedSend.wr.rdma.rkey, 0x55u);

  // Repeated post and empty poll calls must preserve the single in-flight operation.
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 1);
  EXPECT_EQ(ctx.devs[0].nOutstandingProbes, 1);
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclSuccess);
  EXPECT_TRUE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, 0);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, 0);
  EXPECT_EQ(ctx.devs[0].nOutstandingProbes, 1);

  // Successful completion must preserve the fetched snapshot while clearing only in-flight
  // bookkeeping.
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 1, true);
  ctx.remoteReady.epoch[1] = 7;
  QueueCompletion(IBV_WC_SUCCESS, IB_RESILIENCY_PEER_READY_WR_ID);
  ctx.devs[0].nOutstandingProbes = 0;
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclSuccess);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, -1);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, -1);
  EXPECT_EQ(ctx.devs[0].nOutstandingProbes, 0);
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 1));
  EXPECT_EQ(ctx.remoteReady.epoch[1], 7u);

  // Round-robin selection must skip an empty next candidate and wrap to the usable probing QP.
  ctx.lastPeerReadyReadNs = 0;
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 2);
  EXPECT_EQ(fakes.postedSendQp, &fakeCreatedQps[0]);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, 0);
  QueueCompletion(IBV_WC_SUCCESS, IB_RESILIENCY_PEER_READY_WR_ID);
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclSuccess);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);

  // Sender polling must route a foreign regular-probe CQE without consuming its peer-ready read.
  auto [foreignSendComm, foreignSendContext] = MakeState<true>(0, 0, 1);
  foreignSendContext->base.peerReadyReadInFlight = true;
  foreignSendContext->base.peerReadyReadQpIndex = 0;
  foreignSendContext->base.peerReadyReadDevIndex = 0;
  foreignSendContext->base.devs[0].probingCq = &fakeCq;
  foreignSendContext->base.devs[0].nOutstandingProbes = 2;
  ncclIbRequest failedRequest = MakeRequest(&foreignSendComm->base, 3, NCCL_NET_IB_REQ_SEND, 0);
  const int failedSlot = failedRequest.id % NET_IB_MAX_REQUESTS;
  auto* failedTracking = TrackFailedRequest(
      foreignSendContext.get(), &failedRequest, ncclIbResiliencyRequestStateProbePosted);
  failedTracking->probeWrId = ibResiliencyProbeWrIdEncode(IB_RESILIENCY_PROBE_WR_ID_BASE, failedSlot);
  QueueCompletion(IBV_WC_SUCCESS, failedTracking->probeWrId);
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPoll(&foreignSendContext->base), ncclSuccess);
  EXPECT_EQ(failedTracking->state, ncclIbResiliencyRequestStateProbeCompleted);
  EXPECT_EQ(failedTracking->probeWrId, 0u);
  EXPECT_TRUE(foreignSendContext->base.peerReadyReadInFlight);
  EXPECT_EQ(foreignSendContext->base.devs[0].nOutstandingProbes, 1);

  // Receiver polling must ignore an unexpected WR while keeping the peer-ready read in flight.
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 0;
  ctx.peerReadyReadDevIndex = 0;
  ctx.devs[0].nOutstandingProbes = 2;
  QueueCompletion(IBV_WC_SUCCESS, 17);
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclSuccess);
  EXPECT_TRUE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, 0);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, 0);
  EXPECT_EQ(ctx.devs[0].nOutstandingProbes, 1);

  // A failed peer-ready completion must clear the snapshot and start backoff without failing the
  // data device.
  ctx.remoteReady.mask[0] = UINT64_MAX;
  ctx.remoteReady.mask[IB_RESILIENCY_READY_QP_WORDS - 1] = UINT64_MAX;
  ctx.remoteReady.epoch[NCCL_IB_MAX_QPS - 1] = 9;
  ctx.devs[0].nOutstandingProbes = 1;
  QueueCompletion(IBV_WC_WR_FLUSH_ERR, IB_RESILIENCY_PEER_READY_WR_ID,
      fakeCreatedQps[0].qp_num);
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclSuccess);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, -1);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, -1);
  const ibReadyStateSnapshot emptyRemoteReady = {};
  EXPECT_EQ(std::memcmp(&ctx.remoteReady, &emptyRemoteReady, sizeof(ctx.remoteReady)), 0);
  EXPECT_EQ(ctx.devs[0].peerReadyFailureCount, 1);
  EXPECT_GT(ctx.devs[0].peerReadyRetryAfterNs, 0u);
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateOk);

  // A provider post failure must remain local to peer-ready backoff.
  ctx.probingQps[1].qp = &fakeCreatedQps[1];
  ctx.probingQps[1].devIndex = 1;
  ctx.probingQps[1].remDevIdx = 1;
  ctx.devs[1].peerReadyStateMr = &fakeMrs[1];
  ctx.remoteReadyStateRkeys[1] = 0x66;
  ctx.lastPeerReadyReadNs = 0;
  fakes.postSendCalls = 0;
  fakes.postSendError = EIO;
  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  EXPECT_EQ(fakes.postSendCalls, 1);
  EXPECT_EQ(fakes.postedSendQp, &fakeCreatedQps[1]);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.devs[1].peerReadyFailureCount, 1);
  EXPECT_GT(ctx.devs[1].peerReadyRetryAfterNs, 0u);

  // Fresh peer readiness must reactivate the original data QP.
  fakes.postSendError = 0;
  ctx.localReady.epoch[1] = 2;
  ctx.remoteReady.epoch[1] = 3;
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 1, true);
  base.activeQps[1] = &base.qps[0];

  // Peer readiness alone must not activate a QP whose local-ready bit is absent.
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, 1, false);
  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateLocalReady);
  EXPECT_EQ(base.activeQps[1], &base.qps[0]);

  // Matching local and peer readiness must reactivate the original data QP.
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, 1, true);

  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(base.activeQps[1], &base.qps[1]);
  EXPECT_EQ(ctx.activatedRemoteReadyEpoch[1], 3u);

  // A stale peer epoch must not reactivate a QP from a later local recovery cycle.
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateLocalReady;
  ctx.frmAllQpsHealthy = false;
  ctx.localReady.epoch[1] = 4;
  base.activeQps[1] = &base.qps[0];
  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateLocalReady);
  EXPECT_EQ(base.activeQps[1], &base.qps[0]);
  EXPECT_EQ(ctx.activatedRemoteReadyEpoch[1], 3u);
  EXPECT_FALSE(ctx.frmAllQpsHealthy);

  // A newer peer epoch must then reactivate the original QP and refresh health.
  ctx.remoteReady.epoch[1] = 4;
  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(base.activeQps[1], &base.qps[1]);
  EXPECT_EQ(ctx.activatedRemoteReadyEpoch[1], 4u);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);
}

/**
 * @brief Verifies stale completion filtering and receiver flush-error handling.
 */
TEST_F(ResiliencyFixture, CompletionErrors) {
  SCOPED_TRACE(
      "Completion handling must distinguish stale events from actionable device failures.");
  // A non-request sender completion must be treated as stale.
  bool ignore = false;
  ncclIbResiliencySend sendCtx = {};
  sendCtx.base.baseComm = &base;
  sendCtx.base.ndevs = ctx.ndevs;
  ibv_wc wc = {};
  wc.wr_id = 1;
  wc.status = IBV_WC_WR_FLUSH_ERR;

  EXPECT_EQ(ncclIbResiliencyShouldIgnoreSenderCompletionError(&sendCtx.base, &wc, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorSender(&sendCtx.base, &wc, 0), ncclSuccess);

  // A dummy receive flush from an already failed device must be ignored.
  ignore = false;
  base.isSend = false;
  ctx.devs[0].state = ncclIbResiliencyDevStateError;
  wc.wr_id = NCCL_IB_RECV_WR_ID_DUMMY;
  EXPECT_EQ(ncclIbResiliencyShouldIgnoreReceiverCompletionError(&ctx, &wc, 0, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorReceiver(&ctx, &wc, 0), ncclSuccess);

  // Public dispatch must also ignore the drain CQE without marking another QP or changing failover
  // routing.
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&ctx, &wc, 0), ncclSuccess);
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(ctx.qpInfo[0].failCount, 0);
  EXPECT_EQ(base.activeQps[0], &base.qps[0]);

  // The same dummy drain on a healthy device is actionable and must fail over every QP on that
  // device.
  ctx.devs[0].state = ncclIbResiliencyDevStateOk;
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, 0, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 0, true);
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, 2, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 2, true);
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&ctx, &wc, 0), ncclSuccess);
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[0].failCount, 1);
  EXPECT_EQ(ctx.qpInfo[2].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[2].failCount, 1);
  EXPECT_EQ(base.activeQps[0], &base.qps[1]);
  EXPECT_EQ(base.activeQps[2], &base.qps[3]);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 0));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 0));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 2));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 2));

  // Public sender dispatch must leave an ignored stale completion without new effects.
  base.isSend = true;
  wc.wr_id = 1;
  EXPECT_EQ(ncclIbResiliencyHandleCompletionError(&ctx, &wc, 0), ncclSuccess);
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateError);

  // An actionable receiver flush must fail over the device and clear its event.
  base.isSend = false;
  ctx.devs[0].state = ncclIbResiliencyDevStateOk;
  ctx.devs[1].state = ncclIbResiliencyDevStateOk;
  ResetActiveQps(&base);
  for (int index = 0; index < base.nqps; index++) {
    ctx.qpInfo[index].state = ncclIbResiliencyQpStateOk;
  }
  ncclIbRequest* flushRequest = &base.reqs[2];
  flushRequest->base = &base;
  flushRequest->type = NCCL_NET_IB_REQ_FLUSH;
  flushRequest->events[0] = 1;
  wc.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + 2;
  wc.status = IBV_WC_WR_FLUSH_ERR;
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&ctx, &wc, 0), ncclSuccess);
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[2].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(flushRequest->events[0], 0);
}

/**
 * @brief Verifies stale sender filtering and receiver request lookup and repost error paths.
 */
TEST_F(NetIbResiliencyTest, CompletionErrorEdges) {
  SCOPED_TRACE("Completion edge cases must preserve ownership and choose the documented recovery outcome.");
  // Prepare a healthy two-device sender so ignored CQEs can be distinguished from failover.
  auto [sendComm, sendContext] = MakeState<true>(2, 2, 2);
  ConfigureActiveQps(&sendComm->base, &sendContext->base, {0, 1});
  SetQpsReady(&sendContext->base, {0, 1}, true);
  ibv_wc completion = {};
  completion.status = IBV_WC_WR_FLUSH_ERR;
  completion.wr_id = ibReqWrIdEncode(7);

  // An encoded sender CQE with no matching live request must be ignored without failing its device.
  bool ignore = false;
  ASSERT_EQ(
      ncclIbResiliencyShouldIgnoreSenderCompletionError(&sendContext->base, &completion, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorSender(&sendContext->base, &completion, 0), ncclSuccess);
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&sendContext->base, &completion, 0), ncclSuccess);
  EXPECT_EQ(sendContext->base.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendContext->base.qpInfo[0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(sendComm->base.activeQps[0], &sendComm->base.qps[0]);
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(sendContext->base.localReady.mask, 0));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(sendContext->base.remoteReady.mask, 0));
  EXPECT_EQ(sendContext->base.outstandingRequests, 0);
  EXPECT_TRUE(sendContext->base.frmAllQpsHealthy);

  // A live table entry whose generation was already released must also be ignored as stale.
  ncclIbRequest sendRequest = MakeRequest(&sendComm->base, 7, NCCL_NET_IB_REQ_SEND, 0);
  const int sendSlot = sendRequest.id % NET_IB_MAX_REQUESTS;
  BindSendRequest(sendComm.get(), &sendRequest, 0);
  sendContext->failedRequests[sendSlot].id = sendRequest.id + 1;
  sendContext->failedRequests[sendSlot].request = nullptr;
  ignore = false;
  ASSERT_EQ(
      ncclIbResiliencyShouldIgnoreSenderCompletionError(&sendContext->base, &completion, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&sendContext->base, &completion, 0), ncclSuccess);
  EXPECT_EQ(sendContext->base.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendContext->base.qpInfo[0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(sendComm->base.activeQps[0], &sendComm->base.qps[0]);

  // If live lookup succeeds but tracking rejects the request type, public handling must propagate
  // that failure.
  sendContext->failedRequests[sendSlot] = {};
  sendRequest.type = NCCL_NET_IB_REQ_RECV;
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&sendContext->base, &completion, 0), ncclInternalError);
  EXPECT_EQ(sendContext->failedRequests[sendSlot].request, nullptr);
  EXPECT_EQ(sendContext->base.outstandingRequests, 0);
  EXPECT_EQ(sendContext->base.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(sendContext->base.devs[1].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendContext->base.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(sendContext->base.qpInfo[0].failCount, 1);
  EXPECT_EQ(sendContext->base.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(sendComm->base.activeQps[0], &sendComm->base.qps[1]);
  EXPECT_EQ(sendComm->base.activeQps[1], &sendComm->base.qps[1]);
  EXPECT_FALSE(sendContext->base.frmAllQpsHealthy);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(sendContext->base.localReady.mask, 0));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(sendContext->base.remoteReady.mask, 0));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(sendContext->base.localReady.mask, 1));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(sendContext->base.remoteReady.mask, 1));

  // Prepare a healthy two-device receiver for invalid, stale, live, completed, and wrong-type CQEs.
  ResetFakes();
  auto [recvComm, recvStorage] = MakeState<false>(2, 2, 2);
  auto& recvContext = *recvStorage;
  ConfigureActiveQps(&recvComm->base, &recvContext, {0, 1});
  SetQpsReady(&recvContext, {0, 1}, true);
  completion = {};
  completion.status = IBV_WC_WR_FLUSH_ERR;
  completion.wr_id = 1;

  // A receiver WR ID outside request, flush, and dummy ranges must be rejected before request
  // lookup.
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorReceiver(&recvContext, &completion, 0), ncclInternalError);
  EXPECT_EQ(fakes.postFifoCalls, 0);

  // The ignore classifier must leave the same below-range WR ID actionable for the main error
  // handler.
  ignore = true;
  ASSERT_EQ(
      ncclIbResiliencyShouldIgnoreReceiverCompletionError(&recvContext, &completion, 0, &ignore), ncclSuccess);
  EXPECT_FALSE(ignore);

  // The first WR ID above the flush range must evaluate the upper bound and remain actionable.
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + NET_IB_MAX_REQUESTS;
  ignore = true;
  ASSERT_EQ(
      ncclIbResiliencyShouldIgnoreReceiverCompletionError(&recvContext, &completion, 0, &ignore), ncclSuccess);
  EXPECT_FALSE(ignore);

  // A valid encoded ID with no current table entry is stale and must remain a no-op.
  completion.wr_id = ibReqWrIdEncode(5);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorReceiver(&recvContext, &completion, 0), ncclSuccess);
  ignore = false;
  ASSERT_EQ(
      ncclIbResiliencyShouldIgnoreReceiverCompletionError(&recvContext, &completion, 0, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionError(&recvContext, &completion, 0), ncclSuccess);
  EXPECT_EQ(recvContext.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(fakes.postFifoCalls, 0);

  // Bind one live request so completed, actionable, and wrong-type receiver paths can use the same
  // table entry.
  ncclIbRequest recvRequest = MakeRequest(&recvComm->base, 5, NCCL_NET_IB_REQ_UNUSED, 0);
  const int recvSlot = recvRequest.id % NET_IB_MAX_REQUESTS;
  recvComm->recvReqs[recvSlot] = &recvRequest;

  // A completed request must be ignored by both dispatch paths without failing a device or
  // reposting it.
  recvRequest.type = NCCL_NET_IB_REQ_UNUSED;
  ignore = false;
  ASSERT_EQ(
      ncclIbResiliencyShouldIgnoreReceiverCompletionError(&recvContext, &completion, 0, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionError(&recvContext, &completion, 0), ncclSuccess);
  EXPECT_EQ(recvContext.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorReceiver(&recvContext, &completion, 0), ncclSuccess);
  EXPECT_EQ(fakes.postFifoCalls, 0);

  // A live receive CTS error through the public path must fail over its QP and repost exactly the
  // same slot.
  recvRequest.type = NCCL_NET_IB_REQ_RECV;
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&recvContext, &completion, 0), ncclSuccess);
  EXPECT_EQ(recvContext.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(recvContext.devs[1].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(recvContext.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(recvContext.qpInfo[0].failCount, 1);
  EXPECT_EQ(recvContext.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(recvComm->base.activeQps[0], &recvComm->base.qps[1]);
  EXPECT_EQ(recvComm->base.activeQps[1], &recvComm->base.qps[1]);
  EXPECT_TRUE(recvContext.inProgress);
  EXPECT_FALSE(recvContext.frmAllQpsHealthy);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(recvContext.localReady.mask, 0));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(recvContext.remoteReady.mask, 0));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(recvContext.localReady.mask, 1));
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(recvContext.remoteReady.mask, 1));
  EXPECT_EQ(fakes.postFifoCalls, 1);
  EXPECT_EQ(fakes.postFifoComm, recvComm.get());
  EXPECT_EQ(fakes.postFifoRequest, &recvRequest);
  EXPECT_EQ(fakes.postFifoSlot, recvSlot);

  // A second live CTS must propagate a FIFO-post failure without repeating the already-recorded
  // failover.
  ncclIbRequest failingRecvRequest = MakeRequest(&recvComm->base, 6, NCCL_NET_IB_REQ_RECV, 0);
  const int failingRecvSlot = failingRecvRequest.id % NET_IB_MAX_REQUESTS;
  recvComm->recvReqs[failingRecvSlot] = &failingRecvRequest;
  completion.wr_id = ibReqWrIdEncode(failingRecvRequest.id);
  fakes.postFifoResult = ncclSystemError;
  EXPECT_EQ(ncclIbResiliencyHandleCompletionError(&recvContext, &completion, 0), ncclSystemError);
  EXPECT_EQ(recvContext.qpInfo[0].failCount, 1);
  EXPECT_EQ(fakes.postFifoCalls, 2);
  EXPECT_EQ(fakes.postFifoComm, recvComm.get());
  EXPECT_EQ(fakes.postFifoRequest, &failingRecvRequest);
  EXPECT_EQ(fakes.postFifoSlot, failingRecvSlot);

  // A wrong live request type must propagate an error without another failover count or CTS repost.
  failingRecvRequest.type = NCCL_NET_IB_REQ_SEND;
  EXPECT_EQ(ncclIbResiliencyHandleCompletionError(&recvContext, &completion, 0), ncclInternalError);
  EXPECT_EQ(recvContext.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(recvContext.devs[1].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(recvContext.qpInfo[0].failCount, 1);
  EXPECT_EQ(fakes.postFifoCalls, 2);
}

/**
 * @brief Verifies sender error failover, single recovery tracking, and completion cleanup.
 */
TEST_F(NetIbResiliencyTest, SenderCompletionStartsRecovery) {
  SCOPED_TRACE("A qualifying sender completion failure must start recovery exactly once for its device.");
  // Prepare one live sender request on the device that will fail.
  auto [sendComm, sendCtx] = MakeState<true>(2, 2, 2);
  ConfigureActiveQps(&sendComm->base, &sendCtx->base, {0, 1});
  ncclIbRequest request = MakeRequest(&sendComm->base, 7, NCCL_NET_IB_REQ_SEND, 0);
  const int slot = request.id % NET_IB_MAX_REQUESTS;
  BindSendRequest(sendComm.get(), &request, 0);
  ibv_wc completion = {};
  completion.wr_id = ibReqWrIdEncode(request.id);
  completion.status = IBV_WC_WR_FLUSH_ERR;

  // A fatal completion status must return before changing device, QP, routing, or request-tracking
  // state.
  completion.status = IBV_WC_LOC_PROT_ERR;
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&sendCtx->base, &completion, 0), ncclRemoteError);
  EXPECT_EQ(sendCtx->base.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendCtx->base.qpInfo[0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(sendCtx->base.qpInfo[0].failCount, 0);
  EXPECT_EQ(sendComm->base.activeQps[0], &sendComm->base.qps[0]);
  EXPECT_EQ(sendCtx->failedRequests[slot].request, nullptr);
  EXPECT_EQ(sendCtx->base.outstandingRequests, 0);
  EXPECT_FALSE(sendCtx->base.inProgress);

  // The first error must fail over the QP and create one tracked recovery request.
  completion.status = IBV_WC_WR_FLUSH_ERR;
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&sendCtx->base, &completion, 0), ncclSuccess);
  EXPECT_EQ(sendCtx->base.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(sendCtx->base.qpInfo[0].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(sendCtx->base.qpInfo[0].failCount, 1);
  EXPECT_EQ(sendCtx->base.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(sendComm->base.activeQps[0], &sendComm->base.qps[1]);
  EXPECT_EQ(sendComm->base.activeQps[1], &sendComm->base.qps[1]);
  EXPECT_FALSE(sendCtx->base.frmAllQpsHealthy);
  EXPECT_EQ(sendCtx->failedRequests[slot].request, &request);
  EXPECT_EQ(sendCtx->failedRequests[slot].state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(sendCtx->failedRequests[slot].errorInfo.devIndex, 0);
  EXPECT_EQ(sendCtx->base.outstandingRequests, 1);
  EXPECT_TRUE(sendCtx->base.inProgress);

  // A duplicate completion for the same request must not double-count recovery.
  ASSERT_EQ(ncclIbResiliencyHandleCompletionError(&sendCtx->base, &completion, 0), ncclSuccess);
  EXPECT_EQ(sendCtx->base.outstandingRequests, 1);
  EXPECT_EQ(sendCtx->failedRequests[slot].request, &request);
  EXPECT_EQ(sendCtx->base.qpInfo[0].failCount, 1);
  EXPECT_EQ(sendComm->base.activeQps[0], &sendComm->base.qps[1]);

  // Normal request completion must release the tracked recovery entry once.
  ASSERT_EQ(ncclIbResiliencySendRequestComplete(&sendCtx->base, &request), ncclSuccess);
  EXPECT_EQ(sendCtx->failedRequests[slot].request, nullptr);
  EXPECT_EQ(sendCtx->base.outstandingRequests, 0);

  // Public handling must propagate replacement failure before marking or tracking the affected
  // request.
  request.id += NET_IB_MAX_REQUESTS;
  BindSendRequest(sendComm.get(), &request, 0);
  completion.wr_id = ibReqWrIdEncode(request.id);
  sendCtx->base.devs[0].state = ncclIbResiliencyDevStateOk;
  sendCtx->base.devs[1].state = ncclIbResiliencyDevStateOk;
  sendCtx->base.qpInfo[0].state = ncclIbResiliencyQpStateOk;
  sendCtx->base.qpInfo[0].failCount = 0;
  sendCtx->base.qpInfo[1].state = ncclIbResiliencyQpStateLocalReady;
  sendComm->base.activeQps[0] = &sendComm->base.qps[0];
  EXPECT_EQ(ncclIbResiliencyHandleCompletionError(&sendCtx->base, &completion, 0), ncclInternalError);
  EXPECT_EQ(sendCtx->base.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_EQ(sendCtx->base.qpInfo[0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(sendCtx->base.qpInfo[0].failCount, 0);
  EXPECT_EQ(sendCtx->failedRequests[slot].request, nullptr);
  EXPECT_EQ(sendCtx->base.outstandingRequests, 0);
}

/**
 * @brief Verifies queue sizing, remote record storage, incomplete metadata, and idle progress.
 */
TEST_F(ResiliencyFixture, DataSizesRecordsAndIdleProgress) {
  SCOPED_TRACE(
      "Role-specific queue formulas and recovery metadata must remain internally consistent.");
  // Receiver CQ and RQ sizes must include the complete failover workload.
  int cqSize = 0;
  uint32_t rqSize = 0;
  ASSERT_EQ(ncclIbResiliencyDataCqSizeGet(&ctx, 0, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * 2 * base.nDataQps * ctx.ndevs);
  ASSERT_EQ(ncclIbResiliencyDataRqSizeGet(&ctx, 0, &rqSize), ncclSuccess);
  EXPECT_EQ(rqSize, static_cast<uint32_t>(NET_IB_MAX_REQUESTS * base.nDataQps));

  // Sender CQ sizing must omit the receiver's second completion factor.
  base.isSend = true;
  ASSERT_EQ(ncclIbResiliencyDataCqSizeGet(&ctx, 0, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * base.nDataQps * ctx.ndevs);

  // Remote completion-record metadata must be stored in the selected device slot.
  ncclIbResiliencySend sendCtx = {};
  sendCtx.base.baseComm = &base;
  sendCtx.base.nProbingQps = 2;
  EXPECT_EQ(ncclIbResiliencyRemoteCompletionRecordsSet(&sendCtx.base, 0x77, 0x1234, 1), ncclSuccess);
  EXPECT_EQ(sendCtx.remCmplRecordsInfo[1].rkey, 0x77u);
  EXPECT_EQ(sendCtx.remCmplRecordsInfo[1].addr, 0x1234u);

  // Incomplete remote QP metadata must consume one rejoin attempt without crashing.
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateFailed;
  base.qpCold[base.qps[1].coldIdx].remQpn = 0;
  EXPECT_EQ(ncclIbResiliencyTryRejoinSingleQp(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);

  // Progress on an idle context must remain a no-op.
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateOk;
  ctx.inProgress = false;
  EXPECT_EQ(ncclIbResiliencyProgress(&ctx), ncclSuccess);
}

namespace {

constexpr int kStripedTestPhysicalQps = 64;

/**
 * @brief Builds path-major forced geometry with two active ports.
 * @param[out] base Valid non-NULL communicator storage replaced by the helper.
 * @param[out] ctx Valid non-NULL resiliency storage attached to @p base.
 * @param[in,out] dataQps Storage for the computed physical QP count.
 * @param[in] qpsPerPort Positive logical QP count for each active port.
 * @param[in] localDevices Positive local device count within the transport limit.
 * @param[in] remoteDevices Positive remote device count within the transport limit.
 * @note Side effects: Resets fakes, replaces @p base and @p ctx, and initializes used QPs.
 */
void ConfigureTwoPortGeometry(ncclIbNetCommBase* base, ncclIbResiliency* ctx,
                              ibv_qp* dataQps, int qpsPerPort,
                              int localDevices, int remoteDevices) {
  ResetFakes();
  std::memset(base, 0, sizeof(*base));
  std::memset(ctx, 0, sizeof(*ctx));
  base->nqps = 2 * qpsPerPort;
  base->nDataQps = 2;
  base->nPhysicalQps = base->nqps * std::max(localDevices, remoteDevices);
  ASSERT_LE(base->nPhysicalQps, kStripedTestPhysicalQps);
  base->hasStandbyPaths = true;
  base->vProps.ndevs = localDevices;
  base->nRemDevs = remoteDevices;
  base->resiliency = ctx;
  ctx->baseComm = base;
  ctx->ndevs = localDevices;
  ctx->nProbingQps = std::max(localDevices, remoteDevices);
  ctx->frmAllQpsHealthy = true;
  for (int path = 0; path < ctx->nProbingQps; path++) {
    for (int lane = 0; lane < base->nqps; lane++) {
      const int physical = occlIbCommPhysicalQpIndex(base, lane, path);
      dataQps[physical].qp_num = 300 + physical;
      auto* qp = &base->qps[physical];
      qp->qp = &dataQps[physical];
      qp->devIndex = (path + lane % 2) % localDevices;
      qp->remDevIdx = (path + lane % 2) % remoteDevices;
      qp->logicalLane = lane;
      qp->coldIdx = physical;
      base->qpCold[physical].remQpn = 500 + physical;
      ctx->qpInfo[physical].state = ncclIbResiliencyQpStateOk;
      if (path == 0) base->activeQps[lane] = qp;
    }
  }
}

/** @brief Provides configurable forced-path geometry with two active ports. */
struct TwoPortForcedPathResiliencyFixture : public ::testing::Test {
  /**
   * @brief Rebuilds the fixture with one forced two-port geometry.
   * @param[in] qpsPerPort Logical QPs assigned to each active port.
   * @param[in] localDevices Number of local devices in the geometry.
   * @param[in] remoteDevices Number of remote devices in the geometry.
   * @note Side effects: Replaces base and ctx and initializes used dataQps entries.
   */
  void Configure(int qpsPerPort, int localDevices, int remoteDevices) {
    ConfigureTwoPortGeometry(&base, &ctx, dataQps, qpsPerPort,
                             localDevices, remoteDevices);
  }

  ncclIbNetCommBase base = {};
  ncclIbResiliency ctx = {};
  ibv_qp dataQps[kStripedTestPhysicalQps] = {};
};

/** @brief Provides two logical lanes with one primary and one standby path each. */
struct ForcedPathResiliencyFixture : public ::testing::Test {
  /**
   * @brief Resolves one logical lane and path to its physical QP index.
   * @param[in] logicalLane Logical lane in [0, base.nqps).
   * @param[in] path Path index in [0, ctx.nProbingQps).
   * @return The path-major physical QP index.
   * @note This function has no externally visible side effects.
   */
  int Physical(int logicalLane, int path) const {
    return occlIbCommPhysicalQpIndex(&base, logicalLane, path);
  }

  /**
   * @brief Restores fakes and builds the fixture's primary and standby paths.
   * @note Side effects: Resets process-local fakes and replaces fixture state.
   */
  void SetUp() override {
    ResetFakes();
    std::memset(&base, 0, sizeof(base));
    std::memset(&ctx, 0, sizeof(ctx));
    base.nqps = 2;
    base.nPhysicalQps = 4;
    base.hasStandbyPaths = true;
    for (int path = 0; path < 2; path++) {
      for (int logicalLane = 0; logicalLane < base.nqps; logicalLane++) {
        int physicalQpIndex = Physical(logicalLane, path);
        verbsQps[physicalQpIndex].qp_num = 100 + physicalQpIndex;
        base.qps[physicalQpIndex].qp = &verbsQps[physicalQpIndex];
        base.qps[physicalQpIndex].devIndex = path;
        base.qps[physicalQpIndex].remDevIdx = path;
        base.qps[physicalQpIndex].coldIdx = physicalQpIndex;
        base.qps[physicalQpIndex].logicalLane = logicalLane;
        base.qpCold[physicalQpIndex].remQpn = 200 + physicalQpIndex;
        ctx.qpInfo[physicalQpIndex].state = ncclIbResiliencyQpStateOk;
      }
    }
    for (int logicalLane = 0; logicalLane < base.nqps; logicalLane++) {
      base.activeQps[logicalLane] = &base.qps[Physical(logicalLane, 0)];
    }
    base.nDataQps = 1;
    base.vProps.ndevs = 2;
    base.nRemDevs = 2;
    ctx.baseComm = &base;
    ctx.ndevs = 2;
    ctx.nProbingQps = 2;
    ctx.devs[0].state = ncclIbResiliencyDevStateOk;
    ctx.devs[1].state = ncclIbResiliencyDevStateOk;
  }

  ncclIbNetCommBase base = {};
  ncclIbResiliency ctx = {};
  ibv_qp verbsQps[8] = {};
};

}  // namespace

/**
 * @brief Verifies direct device setup uses the peer-matched probing width.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(NetIbResiliencyTest, DirectDeviceNumSetRetainsPeerMatchedProbeCounts) {
  ncclIbNetCommBase base = {};
  ncclIbResiliency ctx = {};
  ctx.baseComm = &base;

  base.isSend = true;
  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 2, 3), ncclSuccess);
  EXPECT_EQ(ctx.ndevs, 2);
  EXPECT_EQ(ctx.nProbingQps, 2);

  base.isSend = false;
  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 3, 2), ncclSuccess);
  EXPECT_EQ(ctx.ndevs, 3);
  EXPECT_EQ(ctx.nProbingQps, 2);
}

/**
 * @brief Verifies standby device setup probes the larger endpoint path count.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(NetIbResiliencyTest, StandbyDeviceNumSetUsesMaximumPathCount) {
  ncclIbNetCommBase base = {};
  ncclIbResiliency ctx = {};
  base.hasStandbyPaths = true;
  ctx.baseComm = &base;

  base.isSend = true;
  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 2, 3), ncclSuccess);
  EXPECT_EQ(ctx.ndevs, 2);
  EXPECT_EQ(ctx.nProbingQps, 3);

  base.isSend = false;
  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 3, 2), ncclSuccess);
  EXPECT_EQ(ctx.ndevs, 3);
  EXPECT_EQ(ctx.nProbingQps, 3);
}

/**
 * @brief Verifies standby sender probes map peer paths onto local devices by modulo.
 * @note Side effects: Mutates process-local provider fakes.
 */
TEST_F(NetIbResiliencyTest, StandbySenderProbingPathsUseModuloLocalPlacement) {
  ncclIbSendComm sendComm = {};
  ibv_mr readyStateMrs[2] = {};
  sendComm.base.isSend = true;
  sendComm.base.hasStandbyPaths = true;
  sendComm.base.vProps.ndevs = 2;
  sendComm.devs[0].base.ibDevN = 0;
  sendComm.devs[1].base.ibDevN = 1;

  ncclIbResiliency ctx = {};
  ctx.baseComm = &sendComm.base;
  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 2, 3), ncclSuccess);
  for (int device = 0; device < ctx.ndevs; device++) {
    readyStateMrs[device].rkey = 0x40 + device;
    ctx.devs[device].readyStateMr = &readyStateMrs[device];
  }

  probingQpCreateCalls = 0;
  fakes.createQpCalls = 0;
  fakes.createQpResult = ncclSuccess;
  std::memset(probingVerbsQps, 0, sizeof(probingVerbsQps));
  ncclIbResiliencyInfo localInfo = {};
  ASSERT_EQ(ncclIbResiliencySenderCreateQps(&ctx, &localInfo), ncclSuccess);
  ASSERT_EQ(probingQpCreateCalls, 3);
  EXPECT_EQ(capturedMaxSendWorkRequest, NET_IB_MAX_REQUESTS + 1);
  for (int path = 0; path < ctx.nProbingQps; path++) {
    EXPECT_EQ(ctx.probingQps[path].devIndex, path % 2);
    EXPECT_EQ(localInfo.probingQpsInfo[path].devIndex, path % 2);
  }
}

/**
 * @brief Verifies direct sender probing retains one path per local device.
 * @note Side effects: Mutates process-local provider fakes.
 */
TEST_F(NetIbResiliencyTest, DirectSenderProbingPathsRetainOnePerLocalDevice) {
  ncclIbSendComm sendComm = {};
  ibv_mr readyStateMrs[2] = {};
  sendComm.base.isSend = true;
  sendComm.base.vProps.ndevs = 2;
  sendComm.devs[0].base.ibDevN = 0;
  sendComm.devs[1].base.ibDevN = 1;

  ncclIbResiliency ctx = {};
  ctx.baseComm = &sendComm.base;
  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 2, 3), ncclSuccess);
  for (int device = 0; device < ctx.ndevs; device++) {
    readyStateMrs[device].rkey = 0x40 + device;
    ctx.devs[device].readyStateMr = &readyStateMrs[device];
  }

  probingQpCreateCalls = 0;
  fakes.createQpCalls = 0;
  fakes.createQpResult = ncclSuccess;
  std::memset(probingVerbsQps, 0, sizeof(probingVerbsQps));
  ncclIbResiliencyInfo localInfo = {};
  ASSERT_EQ(ncclIbResiliencySenderCreateQps(&ctx, &localInfo), ncclSuccess);
  ASSERT_EQ(probingQpCreateCalls, 2);
  EXPECT_EQ(ctx.probingQps[0].devIndex, 0);
  EXPECT_EQ(ctx.probingQps[1].devIndex, 1);
  EXPECT_EQ(localInfo.probingQpsInfo[0].devIndex, 0);
  EXPECT_EQ(localInfo.probingQpsInfo[1].devIndex, 1);
}

/**
 * @brief Verifies request completion accounts for events on every communicator device.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(NetIbResiliencyTest, RequestCompletionUsesAllDeviceEvents) {
  ncclIbResiliency ctx = {};
  ctx.ndevs = 2;
  ncclIbNetCommBase base = {};
  base.resiliency = &ctx;
  ncclIbRequest request = {};
  request.base = &base;
  request.events[0] = 0;
  request.events[1] = 0;
  bool complete = false;

  ASSERT_EQ(ncclIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
  EXPECT_TRUE(complete);

  request.events[0] = 0;
  request.events[1] = 1;
  ASSERT_EQ(ncclIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
  EXPECT_FALSE(complete);
}

/**
 * @brief Verifies replay completion balances event accounting across both active ports.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(TwoPortForcedPathResiliencyFixture, ReplayCompletionBalancesBothActivePorts) {
  for (int qpsPerPort : {1, 2, 4}) {
    for (int split : {0, 1}) {
      for (int failedDevice : {0, 1}) {
        SCOPED_TRACE(::testing::Message() << "qps=" << qpsPerPort
            << " split=" << split << " failed=" << failedDevice);
        Configure(qpsPerPort, 4, 4);
        base.splitDataOnQps = split;
        const int eventsPerPort = ncclIbCommBaseGetNqpsPerRequest(&base) / 2;
        ncclIbRequest request = {};
        request.base = &base;
        request.type = NCCL_NET_IB_REQ_RECV;
        request.events[0] = eventsPerPort;
        request.events[1] = eventsPerPort;
        bool complete = true;
        ASSERT_EQ(ncclIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
        EXPECT_FALSE(complete);

        request.events[1 - failedDevice] = 0;
        request.events[2] = -(eventsPerPort - 1);
        ASSERT_EQ(ncclIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
        EXPECT_FALSE(complete);
        request.events[2]--;
        ASSERT_EQ(ncclIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
        EXPECT_TRUE(complete);
        EXPECT_EQ(request.events[failedDevice], eventsPerPort);
        EXPECT_EQ(request.events[2], -eventsPerPort);

        // Both primary ports can be replayed on the same surviving member.
        request.events[0] = eventsPerPort;
        request.events[1] = eventsPerPort;
        request.events[2] = -(2 * eventsPerPort - 1);
        ASSERT_EQ(ncclIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
        EXPECT_FALSE(complete);
        request.events[2]--;
        ASSERT_EQ(ncclIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
        EXPECT_TRUE(complete);
      }
    }
  }
}

/**
 * @brief Verifies queue capacities cover two-port request width and failover traffic.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(TwoPortForcedPathResiliencyFixture, QueueCapacitiesCoverTwoPortRequestWidthAndFailover) {
  for (int devices : {2, 4, 8}) {
    for (int qpsPerPort : {1, 2, 4}) {
      for (int split : {0, 1}) {
        SCOPED_TRACE(::testing::Message() << "devices=" << devices
            << " qps=" << qpsPerPort << " split=" << split);
        Configure(qpsPerPort, devices, devices);
        base.splitDataOnQps = split;
        const int requestLanes = split ? 2 * qpsPerPort : 2;
        for (int device = 0; device < devices; device++) {
          int cqSize = 0;
          uint32_t rqSize = 0;
          base.isSend = true;
          ASSERT_EQ(ncclIbResiliencyDataCqSizeGet(&ctx, device, &cqSize), ncclSuccess);
          EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * requestLanes * devices);
          base.isSend = false;
          ASSERT_EQ(ncclIbResiliencyDataCqSizeGet(&ctx, device, &cqSize), ncclSuccess);
          EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * 2 * requestLanes * devices);
          ASSERT_EQ(ncclIbResiliencyDataRqSizeGet(&ctx, device, &rqSize), ncclSuccess);
          EXPECT_EQ(rqSize, (uint32_t)(NET_IB_MAX_REQUESTS * requestLanes));
        }
      }
    }
  }
}

/**
 * @brief Verifies partial probe completion never credits another lane or a repeated record.
 * @note Side effects: Mutates process-local provider fakes.
 */
TEST_F(NetIbResiliencyTest, TwoPortPartialProbeNeverCreditsOtherLanesOrRepeatedRecords) {
  for (int qpsPerPort : {1, 2, 4}) {
    for (int split : {0, 1}) {
      for (int failedDevice : {0, 1}) {
        SCOPED_TRACE(::testing::Message() << "qps=" << qpsPerPort
            << " split=" << split << " failed=" << failedDevice);
        ncclIbSendComm sendComm = {};
        ncclIbResiliencySend sendCtx = {};
        ibv_qp dataQps[kStripedTestPhysicalQps] = {};
        ConfigureTwoPortGeometry(&sendComm.base, &sendCtx.base, dataQps,
                                 qpsPerPort, 4, 4);
        auto* base = &sendComm.base;
        base->isSend = true;
        base->splitDataOnQps = split;
        constexpr int requestId = 1;
        auto* request = &base->reqs[requestId];
        request->base = base;
        request->id = requestId;
        request->type = NCCL_NET_IB_REQ_SEND;
        request->nreqs = 1;
        sendComm.sendReqs[requestId][0] = request;
        const int requestLanes = ncclIbCommBaseGetNqpsPerRequest(base);
        for (int i = 0; i < requestLanes; i++) {
          const int lane = (requestId + i) % base->nqps;
          request->send.pendingData[lane] = true;
          request->send.lastAttemptQpIndex[lane] = lane;
          request->events[lane % 2]++;
          if (lane % 2 != failedDevice) {
            sendCtx.probingResults[requestId][lane] = 1;
          }
        }
        ASSERT_EQ(ncclIbResiliencyHandleDeviceFailure(&sendCtx.base, failedDevice), ncclSuccess);
        ncclIbResiliencyRequestSend failed = {};
        failed.request = request;
        bool reposted = false;
        ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(&sendCtx, &failed, &reposted), ncclSuccess);
        EXPECT_TRUE(reposted);
        for (int i = 0; i < requestLanes; i++) {
          const int lane = (requestId + i) % base->nqps;
          const bool missing = lane % 2 == failedDevice;
          EXPECT_EQ(request->send.pendingData[lane], missing);
          EXPECT_EQ(request->send.ackedCompletions[lane], missing ? 0 : 1);
          if (missing) {
            // MultiSend is stubbed in this unit: model its recorded new attempt.
            request->send.lastAttemptQpIndex[lane] = base->activeQps[lane] - base->qps;
            request->send.lastAttemptOrder[lane] = 0;
          }
        }

        // Re-reading the same record, including extra completions for an
        // already-delivered lane, cannot satisfy another port's missing lane.
        for (int i = 0; i < requestLanes; i++) {
          const int lane = (requestId + i) % base->nqps;
          if (lane % 2 != failedDevice) sendCtx.probingResults[requestId][lane] = 2;
        }
        for (int repeat = 0; repeat < 2; repeat++) {
          ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(&sendCtx, &failed, &reposted), ncclSuccess);
          EXPECT_TRUE(reposted);
          int remaining = 0;
          for (int i = 0; i < requestLanes; i++) {
            const int lane = (requestId + i) % base->nqps;
            EXPECT_EQ(request->send.pendingData[lane], lane % 2 == failedDevice);
            remaining += request->send.pendingData[lane];
          }
          EXPECT_EQ(remaining, requestLanes / 2);
        }

        for (int i = 0; i < requestLanes; i++) {
          const int lane = (requestId + i) % base->nqps;
          if (!request->send.pendingData[lane]) continue;
          const int physical = request->send.lastAttemptQpIndex[lane];
          ASSERT_EQ(base->qps[physical].logicalLane, lane);
          sendCtx.probingResults[requestId][physical] = 1;
        }
        ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(&sendCtx, &failed, &reposted), ncclSuccess);
        EXPECT_FALSE(reposted);
        for (int lane = 0; lane < base->nqps; lane++) {
          EXPECT_FALSE(request->send.pendingData[lane]);
        }
        for (int device = 0; device < sendCtx.base.ndevs; device++) {
          EXPECT_EQ(request->events[device], 0);
        }
        // A duplicate final snapshot remains complete and cannot restart replay.
        ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(&sendCtx, &failed, &reposted), ncclSuccess);
        EXPECT_FALSE(reposted);
      }
    }
  }
}

/**
 * @brief Verifies marking a failed QP updates its state and recovery-progress flag.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(NetIbResiliencyTest, MarkFailedQpUpdatesStateAndProgressFlag) {
  ncclIbResiliency ctx = {};
  ncclIbNetCommBase base = {};
  ibv_qp verbsQp = {};
  verbsQp.qp_num = 101;
  base.nqps = 2;
  base.nPhysicalQps = 2;
  base.qps[1].qp = &verbsQp;
  base.qps[1].logicalLane = 1;
  base.activeQps[1] = &base.qps[1];
  ctx.baseComm = &base;
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateOk;

  ASSERT_EQ(ncclIbResiliencyMarkQpFailed(&ctx, 1), ncclSuccess);

  EXPECT_TRUE(ctx.inProgress);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailed);
}

/**
 * @brief Verifies state predicates, masks, and retry backoff at stable boundaries.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, StatePredicatesMasksAndBackoffHelpersAreStable) {
  EXPECT_TRUE(ncclIbResiliencyQpIsPendingRecovery(ncclIbResiliencyQpStateFailed));
  EXPECT_TRUE(ncclIbResiliencyQpIsPendingRecovery(ncclIbResiliencyQpStateLocalReady));
  EXPECT_FALSE(ncclIbResiliencyQpIsPendingRecovery(ncclIbResiliencyQpStateOk));
  EXPECT_TRUE(ncclIbResiliencyQpIsTrafficReady(ncclIbResiliencyQpStateOk));
  EXPECT_FALSE(ncclIbResiliencyQpIsTrafficReady(ncclIbResiliencyQpStateFailedPermanent));

  uint64_t mask[IB_RESILIENCY_READY_QP_WORDS] = {};
  ncclIbResiliencyReadyQpSetBit(mask, 3, true);
  EXPECT_TRUE(ncclIbResiliencyReadyQpGetBit(mask, 3));
  ncclIbResiliencyReadyQpSetBit(mask, 3, false);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(mask, 3));

  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 1;
  ctx.peerReadyReadDevIndex = 1;
  ncclIbResiliencyResetPeerReadyReadState(&ctx);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, -1);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, -1);

  EXPECT_EQ(ncclIbResiliencyQpRejoinCooldownNs(), 1000ULL * MSEC_TO_NSEC);
  EXPECT_EQ(ncclIbResiliencyPeerReadyBackoffNs(0), 1000ULL * MSEC_TO_NSEC);
  EXPECT_EQ(ncclIbResiliencyPeerReadyBackoffNs(3), 3000ULL * MSEC_TO_NSEC);
}

/**
 * @brief Verifies fatality and replacement decisions respect one failed device.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, ErrorFatalityAndReplacementRespectOneFailedDevice) {
  ibv_wc wc = {};
  wc.status = IBV_WC_WR_FLUSH_ERR;
  ctx.devs[0].state = ncclIbResiliencyDevStateError;

  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&ctx, &wc, 0), ncclSuccess);

  ctx.devs[1].state = ncclIbResiliencyDevStateError;
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&ctx, &wc, 0), ncclRemoteError);

  ctx.devs[0].state = ncclIbResiliencyDevStateError;
  ctx.devs[1].state = ncclIbResiliencyDevStateOk;
  ASSERT_EQ(ncclIbResiliencyReplaceQps(&ctx, 0), ncclSuccess);
  EXPECT_EQ(base.activeQps[0], &base.qps[Physical(0, 1)]);
  EXPECT_EQ(base.activeQps[1], &base.qps[Physical(1, 1)]);
  EXPECT_EQ(base.activeQps[0]->logicalLane, 0);
  EXPECT_EQ(base.activeQps[1]->logicalLane, 1);
}

/**
 * @brief Verifies remapped QPs from a failed device remain eligible for recovery.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, FailedDeviceQpsRemainRecoverableAfterRemap) {
  ctx.localReady.mask[0] = UINT64_MAX;
  ctx.remoteReady.mask[0] = UINT64_MAX;

  ASSERT_EQ(ncclIbResiliencyHandleDeviceFailure(&ctx, 0), ncclSuccess);

  for (int logicalLane = 0; logicalLane < base.nqps; logicalLane++) {
    int failedQpIndex = Physical(logicalLane, 0);
    EXPECT_EQ(base.activeQps[logicalLane], &base.qps[Physical(logicalLane, 1)]);
    EXPECT_EQ(ctx.qpInfo[failedQpIndex].state, ncclIbResiliencyQpStateFailed);
    EXPECT_EQ(ctx.qpInfo[failedQpIndex].failCount, 1);
    EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, failedQpIndex));
    EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, failedQpIndex));
  }
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateError);
  EXPECT_TRUE(ncclIbResiliencyHasPendingQpRecovery(&ctx));
  EXPECT_TRUE(ctx.inProgress);
}

/**
 * @brief Verifies standby replacement never borrows a different logical lane.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, ReplacementNeverBorrowsAnotherLogicalLane) {
  ctx.devs[0].state = ncclIbResiliencyDevStateError;
  ctx.qpInfo[Physical(0, 1)].state = ncclIbResiliencyQpStateFailed;

  EXPECT_EQ(ncclIbResiliencyReplaceQps(&ctx, 0), ncclInternalError);
  EXPECT_EQ(base.activeQps[0], &base.qps[Physical(0, 0)]);
  EXPECT_EQ(ctx.qpInfo[Physical(1, 1)].state, ncclIbResiliencyQpStateOk);
}

/**
 * @brief Verifies legacy striped geometry retains cross-QP replacement behavior.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, LegacyStripedGeometryRetainsCrossQpReplacement) {
  base.nqps = 6;
  base.nPhysicalQps = 6;
  base.hasStandbyPaths = false;
  for (int qpIndex = 0; qpIndex < base.nqps; qpIndex++) {
    verbsQps[qpIndex].qp_num = 100 + qpIndex;
    base.qps[qpIndex].qp = &verbsQps[qpIndex];
    base.qps[qpIndex].devIndex = qpIndex % 3;
    base.qps[qpIndex].logicalLane = qpIndex;
    base.activeQps[qpIndex] = &base.qps[qpIndex];
  }
  base.vProps.ndevs = 3;
  ctx.ndevs = 3;
  ctx.devs[2].state = ncclIbResiliencyDevStateOk;
  ctx.devs[0].state = ncclIbResiliencyDevStateError;

  ASSERT_EQ(ncclIbResiliencyReplaceQps(&ctx, 0), ncclSuccess);
  EXPECT_EQ(base.activeQps[0], &base.qps[1]);
  EXPECT_EQ(base.activeQps[3], &base.qps[4]);

  ctx.devs[1].state = ncclIbResiliencyDevStateError;
  ASSERT_EQ(ncclIbResiliencyReplaceQps(&ctx, 1), ncclSuccess);
  for (int logicalLane = 0; logicalLane < base.nqps; logicalLane++) {
    EXPECT_EQ(base.activeQps[logicalLane]->devIndex, 2);
  }
}

/**
 * @brief Verifies sequential failures consume every standby path on the same lane.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, SequentialFailuresUseEverySameLaneStandbyPath) {
  base.nPhysicalQps = 8;
  base.vProps.ndevs = 4;
  base.nRemDevs = 4;
  ctx.ndevs = 4;
  for (int path = 2; path < 4; path++) {
    ctx.devs[path].state = ncclIbResiliencyDevStateOk;
    for (int logicalLane = 0; logicalLane < base.nqps; logicalLane++) {
      int physicalQpIndex = Physical(logicalLane, path);
      verbsQps[physicalQpIndex].qp_num = 100 + physicalQpIndex;
      base.qps[physicalQpIndex].qp = &verbsQps[physicalQpIndex];
      base.qps[physicalQpIndex].devIndex = path;
      base.qps[physicalQpIndex].remDevIdx = path;
      base.qps[physicalQpIndex].coldIdx = physicalQpIndex;
      base.qps[physicalQpIndex].logicalLane = logicalLane;
      base.qpCold[physicalQpIndex].remQpn = 200 + physicalQpIndex;
      ctx.qpInfo[physicalQpIndex].state = ncclIbResiliencyQpStateOk;
    }
  }

  for (int failedPath = 0; failedPath < 3; failedPath++) {
    ASSERT_EQ(ncclIbResiliencyHandleDeviceFailure(&ctx, failedPath), ncclSuccess);
    for (int logicalLane = 0; logicalLane < base.nqps; logicalLane++) {
      EXPECT_EQ(base.activeQps[logicalLane], &base.qps[Physical(logicalLane, failedPath + 1)]);
      EXPECT_EQ(base.activeQps[logicalLane]->logicalLane, logicalLane);
    }
  }
}

/**
 * @brief Verifies either active port fails over without moving healthy lanes.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(TwoPortForcedPathResiliencyFixture, EitherActivePortFailsOverWithoutMovingHealthyLanes) {
  const int devicePairs[][2] = {{2, 2}, {4, 4}, {8, 8}, {2, 4}, {4, 2}};
  for (const auto& devices : devicePairs) {
    for (int qpsPerPort : {1, 2, 4}) {
      for (int failedDevice : {0, 1}) {
        SCOPED_TRACE(::testing::Message() << "local=" << devices[0]
            << " remote=" << devices[1] << " qps=" << qpsPerPort
            << " failed=" << failedDevice);
        Configure(qpsPerPort, devices[0], devices[1]);
        ASSERT_EQ(ncclIbResiliencyHandleDeviceFailure(&ctx, failedDevice), ncclSuccess);
        for (int lane = 0; lane < base.nqps; lane++) {
          const auto* active = base.activeQps[lane];
          ASSERT_NE(active, nullptr);
          EXPECT_EQ(active->logicalLane, lane);
          EXPECT_NE(active->devIndex, failedDevice);
          EXPECT_EQ((active - base.qps) % base.nqps, lane);
          if (lane % 2 != failedDevice) {
            EXPECT_EQ(active, &base.qps[lane]);
          } else {
            EXPECT_EQ(ctx.qpInfo[lane].state, ncclIbResiliencyQpStateFailed);
            EXPECT_EQ(ctx.qpInfo[lane].failCount, 1);
          }
        }
        for (int physical = 0; physical < base.nPhysicalQps; physical++) {
          EXPECT_EQ(ctx.qpInfo[physical].state,
                    base.qps[physical].devIndex == failedDevice
                        ? ncclIbResiliencyQpStateFailed : ncclIbResiliencyQpStateOk);
        }
      }
    }
  }
}

/**
 * @brief Verifies sequential failures select each lane's sole surviving path.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(TwoPortForcedPathResiliencyFixture, SequentialFailuresReachEverySoleSurvivingMember) {
  const int devicePairs[][2] = {{2, 2}, {4, 4}, {8, 8}, {2, 4}, {4, 2}};
  for (const auto& devices : devicePairs) {
    for (int qpsPerPort : {1, 2, 4}) {
      for (int survivor = 0; survivor < devices[0]; survivor++) {
        SCOPED_TRACE(::testing::Message() << "local=" << devices[0]
            << " remote=" << devices[1] << " qps=" << qpsPerPort
            << " survivor=" << survivor);
        Configure(qpsPerPort, devices[0], devices[1]);
        for (int failedDevice = 0; failedDevice < devices[0]; failedDevice++) {
          if (failedDevice == survivor) continue;
          ASSERT_EQ(ncclIbResiliencyHandleDeviceFailure(&ctx, failedDevice), ncclSuccess);
          for (int lane = 0; lane < base.nqps; lane++) {
            const auto* active = base.activeQps[lane];
            ASSERT_NE(active, nullptr);
            EXPECT_EQ(active->logicalLane, lane);
            EXPECT_EQ((active - base.qps) % base.nqps, lane);
            EXPECT_EQ(ctx.devs[active->devIndex].state, ncclIbResiliencyDevStateOk);
            EXPECT_EQ(ctx.qpInfo[active - base.qps].state, ncclIbResiliencyQpStateOk);
          }
        }
        for (int lane = 0; lane < base.nqps; lane++) {
          EXPECT_EQ(base.activeQps[lane]->devIndex, survivor);
        }
      }
    }
  }
}

/**
 * @brief Verifies an exhausted lane cannot borrow another active port's lane.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(TwoPortForcedPathResiliencyFixture, ExhaustedLaneCannotBorrowOtherActivePortsLanes) {
  for (int devices : {2, 4, 8}) {
    for (int qpsPerPort : {1, 2, 4}) {
      for (int failedDevice : {0, 1}) {
        SCOPED_TRACE(::testing::Message() << "devices=" << devices
            << " qps=" << qpsPerPort << " failed=" << failedDevice);
        Configure(qpsPerPort, devices, devices);
        const int strandedLane = failedDevice;
        ctx.devs[failedDevice].state = ncclIbResiliencyDevStateError;
        for (int path = 1; path < devices; path++) {
          const int physical = occlIbCommPhysicalQpIndex(&base, strandedLane, path);
          ctx.qpInfo[physical].state = ncclIbResiliencyQpStateFailed;
        }
        EXPECT_EQ(ncclIbResiliencyReplaceQps(&ctx, failedDevice), ncclInternalError);
        EXPECT_EQ(base.activeQps[strandedLane], &base.qps[strandedLane]);
        const int healthyLane = 1 - failedDevice;
        EXPECT_EQ(base.activeQps[healthyLane], &base.qps[healthyLane]);
        EXPECT_EQ(ctx.qpInfo[healthyLane].state, ncclIbResiliencyQpStateOk);
      }
    }
  }
}

/**
 * @brief Verifies fresh peer readiness restores each primary port instead of a standby.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(TwoPortForcedPathResiliencyFixture, FreshPeerReadyRestoresEachPrimaryPortNotStandbys) {
  for (int qpsPerPort : {1, 2, 4}) {
    SCOPED_TRACE(qpsPerPort);
    Configure(qpsPerPort, 4, 4);
    for (int failedDevice : {3, 0, 1}) {
      ASSERT_EQ(ncclIbResiliencyHandleDeviceFailure(&ctx, failedDevice), ncclSuccess);
    }
    for (int lane = 0; lane < base.nqps; lane++) {
      ASSERT_EQ(base.activeQps[lane]->devIndex, 2);
    }

    // Device 3 has only standby QPs; its recovery cannot replace a live lane.
    for (int physical = 0; physical < base.nPhysicalQps; physical++) {
      if (base.qps[physical].devIndex != 3) continue;
      ctx.qpInfo[physical].state = ncclIbResiliencyQpStateLocalReady;
      ctx.localReady.epoch[physical] = 2;
      ctx.remoteReady.epoch[physical] = 3;
      ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, physical, true);
      ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, physical, true);
    }
    ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
    ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
    EXPECT_EQ(ctx.devs[3].state, ncclIbResiliencyDevStateOk);
    for (int lane = 0; lane < base.nqps; lane++) {
      EXPECT_EQ(base.activeQps[lane]->devIndex, 2);
    }

    // An already consumed peer epoch must not fail either primary back.
    for (int physical = 0; physical < base.nPhysicalQps; physical++) {
      if (base.qps[physical].devIndex > 1) continue;
      ctx.qpInfo[physical].state = ncclIbResiliencyQpStateLocalReady;
      ctx.localReady.epoch[physical] = 4;
      ctx.remoteReady.epoch[physical] = 5;
      ctx.activatedRemoteReadyEpoch[physical] = 5;
      ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, physical, true);
      ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, physical, true);
    }
    ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
    for (int lane = 0; lane < base.nqps; lane++) {
      EXPECT_EQ(base.activeQps[lane]->devIndex, 2);
      EXPECT_EQ(ctx.qpInfo[lane].state, ncclIbResiliencyQpStateLocalReady);
    }

    for (int recoveredDevice : {0, 1}) {
      for (int physical = 0; physical < base.nPhysicalQps; physical++) {
        if (base.qps[physical].devIndex == recoveredDevice) {
          ctx.remoteReady.epoch[physical] = 6;
        }
      }
      ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
      ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
      EXPECT_EQ(ctx.devs[recoveredDevice].state, ncclIbResiliencyDevStateOk);
      for (int lane = 0; lane < base.nqps; lane++) {
        if (lane % 2 <= recoveredDevice) {
          EXPECT_EQ(base.activeQps[lane], &base.qps[lane]);
          EXPECT_EQ(base.activeQps[lane]->devIndex, lane % 2);
          EXPECT_EQ(ctx.activatedRemoteReadyEpoch[lane], 6u);
        } else {
          EXPECT_EQ(base.activeQps[lane]->devIndex, 2);
        }
      }
    }
  }
}

/**
 * @brief Verifies FRM health tracks only active logical lanes.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, FrmHealthTracksOnlyActiveLogicalLanes) {
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);

  ctx.qpInfo[Physical(0, 0)].state = ncclIbResiliencyQpStateFailed;
  base.activeQps[0] = &base.qps[Physical(0, 1)];
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);

  ctx.qpInfo[Physical(1, 1)].state = ncclIbResiliencyQpStateFailed;
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);

  ctx.qpInfo[Physical(0, 1)].state = ncclIbResiliencyQpStateFailed;
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_FALSE(ctx.frmAllQpsHealthy);
}

/**
 * @brief Verifies standby-path health rejects a logical lane without an active QP.
 * @note Side effects: Clears one fixture active-QP pointer.
 */
TEST_F(ForcedPathResiliencyFixture, FrmHealthRejectsMissingActiveQp) {
  base.activeQps[1] = nullptr;

  ibResiliencyRefreshFrmQpHealth(&ctx);

  EXPECT_FALSE(ctx.frmAllQpsHealthy);
  EXPECT_EQ(base.activeQps[0], &base.qps[Physical(0, 0)]);
  EXPECT_EQ(ctx.qpInfo[Physical(0, 0)].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(ctx.qpInfo[Physical(1, 0)].state, ncclIbResiliencyQpStateOk);
}

/**
 * @brief Verifies direct FRM health retains original QP slots after cross-lane remapping.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, DirectFrmHealthRetainsOriginalQpSlotsAfterCrossLaneRemap) {
  base.nPhysicalQps = base.nqps;
  base.hasStandbyPaths = false;
  base.activeQps[0] = &base.qps[1];

  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);

  ctx.qpInfo[0].state = ncclIbResiliencyQpStateFailed;
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_FALSE(ctx.frmAllQpsHealthy);

  ctx.qpInfo[0].state = ncclIbResiliencyQpStateOk;
  ibResiliencyRefreshFrmQpHealth(&ctx);
  EXPECT_TRUE(ctx.frmAllQpsHealthy);
}

/**
 * @brief Verifies pending recovery and device reconciliation track current QP states.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, PendingRecoveryAndDeviceStateReconcileTrackQpStates) {
  EXPECT_FALSE(ncclIbResiliencyHasPendingQpRecovery(&ctx));
  ctx.qpInfo[Physical(0, 1)].state = ncclIbResiliencyQpStateFailed;
  EXPECT_TRUE(ncclIbResiliencyHasPendingQpRecovery(&ctx));

  ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].state, ncclIbResiliencyDevStateError);

  ctx.qpInfo[Physical(0, 1)].state = ncclIbResiliencyQpStateFailedPermanent;
  ctx.qpInfo[Physical(1, 1)].state = ncclIbResiliencyQpStateFailedPermanent;
  ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].state, ncclIbResiliencyDevStateErrorPermanent);
}

/**
 * @brief Verifies rejoin failure and peer-ready handlers reset recovery bookkeeping.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, RejoinFailureAndPeerReadyHandlersResetBookkeeping) {
  ctx.localReady.mask[0] = UINT64_MAX;
  ctx.remoteReady.mask[0] = UINT64_MAX;

  ASSERT_EQ(ncclIbResiliencyMarkQpRejoinFailure(&ctx, 1, "test"), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);
  EXPECT_TRUE(ctx.inProgress);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 1));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 1));

  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 1;
  ctx.peerReadyReadDevIndex = 1;
  ctx.remoteReady.mask[0] = UINT64_MAX;
  ASSERT_EQ(ncclIbResiliencyHandlePeerReadyReadFailure(&ctx, 1, "unit", nullptr), ncclSuccess);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.remoteReady.mask[0], 0u);
  EXPECT_EQ(ctx.devs[1].peerReadyFailureCount, 1);
  EXPECT_GT(ctx.devs[1].peerReadyRetryAfterNs, 0u);
  EXPECT_EQ(ctx.failedProbeQpMask, 0u);

  ASSERT_EQ(ncclIbResiliencyHandlePeerReadyReadSuccess(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].peerReadyFailureCount, 0);
  EXPECT_EQ(ctx.devs[1].peerReadyRetryAfterNs, 0u);

  probingVerbsQps[0].qp_num = 900;
  ctx.probingQps[0].qp = &probingVerbsQps[0];
  ctx.probingQps[0].devIndex = 0;
  ctx.probingQps[0].remDevIdx = 0;
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 0;
  ctx.peerReadyReadDevIndex = 0;
  ibv_wc failedReady = {};
  failedReady.status = IBV_WC_WR_FLUSH_ERR;
  failedReady.qp_num = probingVerbsQps[0].qp_num;
  ASSERT_EQ(ncclIbResiliencyHandlePeerReadyReadFailure(&ctx, 0, "unit", &failedReady), ncclSuccess);
  EXPECT_EQ(ctx.failedProbeQpMask, 1u);
  EXPECT_EQ(ctx.devs[0].state, ncclIbResiliencyDevStateOk);
}

/**
 * @brief Verifies stale peer-ready failure quarantines only its completing QP.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, StalePeerReadyFailureOnlyQuarantinesItsCompletingQp) {
  for (int probingQpIndex = 0; probingQpIndex < ctx.nProbingQps; probingQpIndex++) {
    probingVerbsQps[probingQpIndex].qp_num = 900 + probingQpIndex;
    ctx.probingQps[probingQpIndex].qp = &probingVerbsQps[probingQpIndex];
    ctx.probingQps[probingQpIndex].devIndex = probingQpIndex;
    ctx.probingQps[probingQpIndex].remDevIdx = probingQpIndex;
  }
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 1;
  ctx.peerReadyReadDevIndex = 1;
  ctx.remoteReady.mask[0] = 0x55;

  ibv_wc staleFailure = {};
  staleFailure.status = IBV_WC_WR_FLUSH_ERR;
  staleFailure.qp_num = probingVerbsQps[0].qp_num;
  ASSERT_EQ(ncclIbResiliencyHandlePeerReadyReadFailure(&ctx, 0, "unit", &staleFailure), ncclSuccess);

  EXPECT_EQ(ctx.failedProbeQpMask, 1u);
  EXPECT_TRUE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, 1);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, 1);
  EXPECT_EQ(ctx.remoteReady.mask[0], 0x55u);
  EXPECT_EQ(ctx.devs[0].peerReadyFailureCount, 0);
  EXPECT_EQ(ctx.devs[0].peerReadyRetryAfterNs, 0u);
}

/**
 * @brief Verifies a failed peer-ready completion from an unknown QP is rejected unchanged.
 * @note Side effects: Configures one fixture probing QP and peer-read identity.
 */
TEST_F(ForcedPathResiliencyFixture, UnknownPeerReadyProbeCompletionIsRejected) {
  probingVerbsQps[0].qp_num = 900;
  ctx.probingQps[0].qp = &probingVerbsQps[0];
  ctx.probingQps[0].devIndex = 0;
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 0;
  ctx.peerReadyReadDevIndex = 0;
  ctx.remoteReady.mask[0] = 0x55;

  ibv_wc completion = {};
  completion.status = IBV_WC_WR_FLUSH_ERR;
  completion.qp_num = 999;
  EXPECT_EQ(ncclIbResiliencyHandlePeerReadyReadFailure(
                &ctx, 0, "unit", &completion),
      ncclInternalError);

  EXPECT_EQ(ctx.failedProbeQpMask, 0u);
  EXPECT_TRUE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, 0);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, 0);
  EXPECT_EQ(ctx.remoteReady.mask[0], 0x55u);
  EXPECT_EQ(ctx.devs[0].peerReadyFailureCount, 0);
}

/**
 * @brief Verifies send-request state transitions through init, free, and replay-posted.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, SendRequestStateMachineCoversInitFreeAndReplayPosted) {
  ncclIbResiliencySend sendCtx = {};
  sendCtx.base.baseComm = &base;
  sendCtx.base.ndevs = ctx.ndevs;
  ncclIbRequest request = {};
  request.id = 7;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.base = &base;

  ASSERT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &request, 1), ncclSuccess);
  auto* failed = &sendCtx.failedRequests[request.id % NET_IB_MAX_REQUESTS];
  EXPECT_EQ(failed->request, &request);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 1);

  EXPECT_TRUE(ncclIbResiliencySendRequestRecoveryActive(ncclIbResiliencyRequestStatePending));
  EXPECT_TRUE(ncclIbResiliencySendRequestRecoveryActive(ncclIbResiliencyRequestStateProbePosted));
  EXPECT_TRUE(ncclIbResiliencySendRequestRecoveryActive(ncclIbResiliencyRequestStateProbeCompleted));
  EXPECT_FALSE(ncclIbResiliencySendRequestRecoveryActive(ncclIbResiliencyRequestStateReplayPosted));

  ASSERT_EQ(ncclIbResiliencySendRequestReplayPosted(&sendCtx, failed), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateReplayPosted);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);

  ASSERT_EQ(ncclIbResiliencySendRequestInit(&sendCtx, &request, 0), ncclSuccess);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 1);

  ASSERT_EQ(ncclIbResiliencySendRequestFree(&sendCtx, failed), ncclSuccess);
  EXPECT_EQ(failed->request, nullptr);
  EXPECT_EQ(sendCtx.base.outstandingRequests, 0);
  EXPECT_EQ(ncclIbResiliencySendRequestFree(&sendCtx, failed), ncclInternalError);
}

/**
 * @brief Verifies request-probe failure quarantines only the completing auxiliary QP.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, RequestProbeFailureQuarantinesOnlyCompletingAuxiliaryQp) {
  ncclIbResiliencySend sendCtx = {};
  sendCtx.base = ctx;
  sendCtx.base.nProbingQps = 3;
  for (int probingQpIndex = 0; probingQpIndex < sendCtx.base.nProbingQps; probingQpIndex++) {
    probingVerbsQps[probingQpIndex].qp_num = 900 + probingQpIndex;
    sendCtx.base.probingQps[probingQpIndex].qp = &probingVerbsQps[probingQpIndex];
    sendCtx.base.probingQps[probingQpIndex].devIndex = probingQpIndex % 2;
    sendCtx.base.probingQps[probingQpIndex].remDevIdx = probingQpIndex % 2;
  }
  probingVerbsQps[2].qp_num = probingVerbsQps[1].qp_num;

  ncclIbRequest request = {};
  request.id = 7;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.base = &base;
  int slot = request.id % NET_IB_MAX_REQUESTS;
  auto* failed = &sendCtx.failedRequests[slot];
  failed->request = &request;
  failed->state = ncclIbResiliencyRequestStateProbePosted;
  failed->probeWrId = ibResiliencyProbeWrIdEncode(IB_RESILIENCY_PROBE_WR_ID_BASE, slot);

  ibv_wc wc = {};
  wc.status = IBV_WC_WR_FLUSH_ERR;
  wc.wr_id = failed->probeWrId;
  wc.qp_num = probingVerbsQps[2].qp_num;
  ASSERT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(&sendCtx, &wc, 0), ncclSuccess);

  EXPECT_EQ(sendCtx.base.failedProbeQpMask, 1ULL << 2);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed->probeWrId, 0u);
  EXPECT_EQ(failed->failedAttempts, 1);
  EXPECT_EQ(sendCtx.base.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendCtx.base.devs[1].state, ncclIbResiliencyDevStateOk);
  for (int qpIndex = 0; qpIndex < base.nPhysicalQps; qpIndex++) {
    EXPECT_EQ(sendCtx.base.qpInfo[qpIndex].state, ncclIbResiliencyQpStateOk);
  }
  EXPECT_EQ(base.activeQps[0], &base.qps[Physical(0, 0)]);
  EXPECT_EQ(base.activeQps[1], &base.qps[Physical(1, 0)]);
}

/**
 * @brief Verifies a failed request probe from an unknown QP is rejected unchanged.
 * @note Side effects: Configures one auxiliary QP in a local sender context.
 */
TEST_F(ForcedPathResiliencyFixture, UnknownRequestProbeCompletionIsRejected) {
  ncclIbResiliencySend sendContext = {};
  sendContext.base = ctx;
  sendContext.base.nProbingQps = 1;
  probingVerbsQps[0].qp_num = 900;
  sendContext.base.probingQps[0].qp = &probingVerbsQps[0];
  sendContext.base.probingQps[0].devIndex = 0;

  ibv_wc completion = {};
  completion.status = IBV_WC_WR_FLUSH_ERR;
  completion.wr_id = ibResiliencyProbeWrIdEncode(IB_RESILIENCY_PROBE_WR_ID_BASE, 0);
  completion.qp_num = 999;
  EXPECT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(&sendContext, &completion, 0),
      ncclInternalError);

  EXPECT_EQ(sendContext.base.failedProbeQpMask, 0u);
  EXPECT_EQ(sendContext.base.devs[0].state, ncclIbResiliencyDevStateOk);
  EXPECT_EQ(sendContext.base.devs[1].state, ncclIbResiliencyDevStateOk);
}

/**
 * @brief Verifies a fatal request-probe status returns an error after identifying its QP.
 * @note Side effects: Quarantines one auxiliary QP without changing data-device health.
 */
TEST_F(ForcedPathResiliencyFixture, FatalRequestProbeCompletionIsRejected) {
  ncclIbResiliencySend sendContext = {};
  sendContext.base = ctx;
  sendContext.base.nProbingQps = 1;
  probingVerbsQps[0].qp_num = 900;
  sendContext.base.probingQps[0].qp = &probingVerbsQps[0];
  sendContext.base.probingQps[0].devIndex = 0;

  ncclIbRequest request = MakeRequest(&base, 7, NCCL_NET_IB_REQ_SEND, 0);
  const int slot = request.id % NET_IB_MAX_REQUESTS;
  ncclIbResiliencyRequestSend* failed = TrackFailedRequest(
      &sendContext, &request, ncclIbResiliencyRequestStateProbePosted);
  failed->probeWrId = ibResiliencyProbeWrIdEncode(IB_RESILIENCY_PROBE_WR_ID_BASE, slot);

  ibv_wc completion = {};
  completion.status = IBV_WC_GENERAL_ERR;
  completion.wr_id = failed->probeWrId;
  completion.qp_num = probingVerbsQps[0].qp_num;
  EXPECT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(&sendContext, &completion, 0),
      ncclRemoteError);

  // Request bookkeeping after a terminal error is intentionally not part of this contract.
  EXPECT_EQ(sendContext.base.failedProbeQpMask, 1u);
  EXPECT_EQ(sendContext.base.devs[0].state, ncclIbResiliencyDevStateOk);
}

/**
 * @brief Verifies request-probe posting skips a quarantined auxiliary QP.
 * @note Side effects: Mutates process-local provider fakes.
 */
TEST_F(ForcedPathResiliencyFixture, RequestProbePostSkipsQuarantinedAuxiliaryQp) {
  ResetProbePostCapture();
  ncclIbResiliencySend sendCtx = {};
  sendCtx.base = ctx;
  sendCtx.base.nProbingQps = 2;
  sendCtx.base.failedProbeQpMask = 1ULL;
  ibv_mr probingResultMrs[2] = {};
  for (int probingQpIndex = 0; probingQpIndex < sendCtx.base.nProbingQps; probingQpIndex++) {
    probingVerbsQps[probingQpIndex].context = &probingVerbsContext;
    probingVerbsQps[probingQpIndex].qp_num = 900 + probingQpIndex;
    sendCtx.base.probingQps[probingQpIndex].qp = &probingVerbsQps[probingQpIndex];
    sendCtx.base.probingQps[probingQpIndex].devIndex = probingQpIndex;
    sendCtx.base.probingQps[probingQpIndex].remDevIdx = probingQpIndex;
    probingResultMrs[probingQpIndex].lkey = 0x10 + probingQpIndex;
    sendCtx.base.devs[probingQpIndex].probingResultMr = &probingResultMrs[probingQpIndex];
    sendCtx.remCmplRecordsInfo[probingQpIndex].addr = 0x1000 + 0x100 * probingQpIndex;
    sendCtx.remCmplRecordsInfo[probingQpIndex].rkey = 0x20 + probingQpIndex;
  }
  sendCtx.nextProbeWrId = IB_RESILIENCY_PROBE_WR_ID_BASE;

  ncclIbRequest request = {};
  request.id = 9;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.base = &base;
  auto* failed = &sendCtx.failedRequests[request.id % NET_IB_MAX_REQUESTS];
  failed->request = &request;
  failed->state = ncclIbResiliencyRequestStatePending;
  failed->errorInfo.time = 0;

  ASSERT_EQ(ncclIbResiliencyProbePost(&sendCtx, failed), ncclSuccess);
  EXPECT_EQ(capturedProbePosts, 1);
  EXPECT_EQ(capturedProbeQp, &probingVerbsQps[1]);
  EXPECT_EQ(capturedProbeLkey, probingResultMrs[1].lkey);
  EXPECT_EQ(capturedProbeRkey, sendCtx.remCmplRecordsInfo[1].rkey);
  EXPECT_EQ(failed->state, ncclIbResiliencyRequestStateProbePosted);
  EXPECT_NE(failed->probeWrId, 0u);
  EXPECT_EQ(sendCtx.base.devs[0].nOutstandingProbes, 0);
  EXPECT_EQ(sendCtx.base.devs[1].nOutstandingProbes, 1);
}

/**
 * @brief Verifies probe completion credits the standby physical QP that completed it.
 * @note Side effects: Mutates process-local provider fakes.
 */
TEST_F(NetIbResiliencyTest, ProbeCompletionCreditsAStandbyPhysicalQp) {
  ncclIbSendComm sendComm = {};
  sendComm.base.nqps = 2;
  sendComm.base.nPhysicalQps = 4;
  sendComm.base.hasStandbyPaths = true;

  ncclIbRequest* request = &sendComm.base.reqs[0];
  request->base = &sendComm.base;
  request->id = 3;
  request->type = NCCL_NET_IB_REQ_SEND;
  request->nreqs = 1;
  int slot = request->id % NET_IB_MAX_REQUESTS;
  int standbyPhysicalQp = occlIbCommPhysicalQpIndex(&sendComm.base, 0, 1);
  request->send.pendingData[0] = true;
  request->send.lastAttemptQpIndex[0] = standbyPhysicalQp;
  sendComm.sendReqs[slot][0] = request;

  ncclIbResiliencySend sendCtx = {};
  sendCtx.base.baseComm = &sendComm.base;
  sendCtx.failedRequests[slot].request = request;
  sendCtx.failedRequests[slot].state = ncclIbResiliencyRequestStateProbeCompleted;
  sendCtx.probingResults[slot][standbyPhysicalQp] = 1;
  bool reposted = false;

  ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(&sendCtx, &sendCtx.failedRequests[slot], &reposted), ncclSuccess);
  EXPECT_FALSE(reposted);
  EXPECT_FALSE(request->send.pendingData[0]);
  EXPECT_EQ(request->send.ackedCompletions[standbyPhysicalQp], 1);
}

/**
 * @brief Verifies peer-ready post and poll helpers accept their no-op paths.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, PeerReadyPostAndPollHandleNoopPaths) {
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclSuccess);
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadDevIndex = 9;
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclInternalError);
  ctx.peerReadyReadInFlight = false;

  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  ctx.remoteReadyStateAddr = 0x1000;
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateOk;
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
}

/**
 * @brief Verifies peer-ready posting skips a quarantined auxiliary QP.
 * @note Side effects: Mutates process-local provider fakes.
 */
TEST_F(ForcedPathResiliencyFixture, PeerReadyPostSkipsQuarantinedAuxiliaryQp) {
  ResetProbePostCapture();
  ibv_mr peerReadyMrs[2] = {};
  for (int probingQpIndex = 0; probingQpIndex < ctx.nProbingQps; probingQpIndex++) {
    probingVerbsQps[probingQpIndex].context = &probingVerbsContext;
    probingVerbsQps[probingQpIndex].qp_num = 900 + probingQpIndex;
    ctx.probingQps[probingQpIndex].qp = &probingVerbsQps[probingQpIndex];
    ctx.probingQps[probingQpIndex].devIndex = probingQpIndex;
    ctx.probingQps[probingQpIndex].remDevIdx = probingQpIndex;
    peerReadyMrs[probingQpIndex].lkey = 0x30 + probingQpIndex;
    ctx.devs[probingQpIndex].peerReadyStateMr = &peerReadyMrs[probingQpIndex];
    ctx.remoteReadyStateRkeys[probingQpIndex] = 0x40 + probingQpIndex;
  }
  ctx.failedProbeQpMask = 1ULL;
  ctx.remoteReadyStateAddr = 0x1000;
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateLocalReady;

  ASSERT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  EXPECT_EQ(capturedProbePosts, 1);
  EXPECT_EQ(capturedProbeQp, &probingVerbsQps[1]);
  EXPECT_EQ(capturedProbeLkey, peerReadyMrs[1].lkey);
  EXPECT_EQ(capturedProbeRkey, ctx.remoteReadyStateRkeys[1]);
  EXPECT_EQ(ctx.peerReadyReadQpIndex, 1);
  EXPECT_EQ(ctx.peerReadyReadDevIndex, 1);
  EXPECT_TRUE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.devs[0].nOutstandingProbes, 0);
  EXPECT_EQ(ctx.devs[1].nOutstandingProbes, 1);
}

/**
 * @brief Verifies peer-ready recovery fails back only primary paths.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, PeerReadyRecoveryFailsBackOnlyPrimaryPaths) {
  base.activeQps[0] = &base.qps[Physical(0, 1)];
  base.activeQps[1] = &base.qps[Physical(1, 1)];

  int primaryLane0 = Physical(0, 0);
  ctx.qpInfo[primaryLane0].state = ncclIbResiliencyQpStateLocalReady;
  ctx.localReady.epoch[primaryLane0] = 2;
  ctx.remoteReady.epoch[primaryLane0] = 3;
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, primaryLane0, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, primaryLane0, true);

  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[primaryLane0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(base.activeQps[0], &base.qps[primaryLane0]);
  EXPECT_EQ(base.activeQps[1], &base.qps[Physical(1, 1)]);
  EXPECT_EQ(ctx.activatedRemoteReadyEpoch[primaryLane0], 3u);

  int standbyLane0 = Physical(0, 1);
  ctx.qpInfo[standbyLane0].state = ncclIbResiliencyQpStateLocalReady;
  ctx.localReady.epoch[standbyLane0] = 4;
  ctx.remoteReady.epoch[standbyLane0] = 5;
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, standbyLane0, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, standbyLane0, true);

  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[standbyLane0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(base.activeQps[0], &base.qps[primaryLane0]);

  int primaryLane1 = Physical(1, 0);
  ctx.qpInfo[primaryLane1].state = ncclIbResiliencyQpStateLocalReady;
  ctx.localReady.epoch[primaryLane1] = 6;
  ctx.remoteReady.epoch[primaryLane1] = 7;
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, primaryLane1, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, primaryLane1, true);

  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(base.activeQps[1], &base.qps[primaryLane1]);
}

/**
 * @brief Verifies peer-ready activation rejects both invalid logical-lane boundaries.
 * @note Side effects: Temporarily marks one fixture QP locally and remotely ready.
 */
TEST_F(ForcedPathResiliencyFixture, PeerReadyActivationRejectsInvalidLogicalLane) {
  const int physicalQpIndex = Physical(0, 0);
  ctx.qpInfo[physicalQpIndex].state = ncclIbResiliencyQpStateLocalReady;
  ctx.remoteReady.epoch[physicalQpIndex] = 1;
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, physicalQpIndex, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, physicalQpIndex, true);

  for (int invalidLogicalLane : {-1, base.nqps}) {
    SCOPED_TRACE(testing::Message() << "logicalLane=" << invalidLogicalLane);
    base.qps[physicalQpIndex].logicalLane = invalidLogicalLane;

    EXPECT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclInternalError);
    EXPECT_EQ(ctx.qpInfo[physicalQpIndex].state, ncclIbResiliencyQpStateLocalReady);
    EXPECT_EQ(ctx.activatedRemoteReadyEpoch[physicalQpIndex], 0u);
    EXPECT_EQ(base.activeQps[0], &base.qps[physicalQpIndex]);
  }
}

/**
 * @brief Verifies direct peer-ready recovery restores the original QP slot.
 * @note Side effects: Resets process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, DirectPeerReadyRecoveryRestoresOriginalQpSlot) {
  base.nPhysicalQps = base.nqps;
  base.hasStandbyPaths = false;
  base.activeQps[0] = &base.qps[1];

  ctx.qpInfo[0].state = ncclIbResiliencyQpStateLocalReady;
  ctx.localReady.epoch[0] = 2;
  ctx.remoteReady.epoch[0] = 3;
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, 0, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 0, true);

  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[0].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(base.activeQps[0], &base.qps[0]);
  EXPECT_EQ(ctx.activatedRemoteReadyEpoch[0], 3u);
}

/**
 * @brief Verifies completion ignore and handle paths safely accept stale entries.
 * @note Side effects: Mutates process-local resiliency fakes.
 */
TEST_F(ForcedPathResiliencyFixture, CompletionIgnoreAndHandlePathsTreatStaleEntriesAsSafe) {
  bool ignore = false;
  ncclIbResiliencySend sendCtx = {};
  sendCtx.base.baseComm = &base;
  sendCtx.base.ndevs = ctx.ndevs;
  ibv_wc wc = {};
  wc.wr_id = 1;
  wc.status = IBV_WC_WR_FLUSH_ERR;

  EXPECT_EQ(ncclIbResiliencyShouldIgnoreSenderCompletionError(&sendCtx.base, &wc, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorSender(&sendCtx.base, &wc, 0), ncclSuccess);

  ignore = false;
  base.isSend = false;
  ctx.devs[0].state = ncclIbResiliencyDevStateError;
  wc.wr_id = NCCL_IB_RECV_WR_ID_DUMMY;
  EXPECT_EQ(ncclIbResiliencyShouldIgnoreReceiverCompletionError(&ctx, &wc, 0, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorReceiver(&ctx, &wc, 0), ncclSuccess);
}

/**
 * @brief Verifies public helpers for data sizes, remote records, close, and progress.
 * @note Side effects: Mutates process-local resiliency and provider fakes.
 */
TEST_F(ForcedPathResiliencyFixture, PublicHelpersCoverDataSizesRemoteRecordsCloseAndProgress) {
  int cqSize = 0;
  uint32_t rqSize = 0;

  ASSERT_EQ(ncclIbResiliencyDataCqSizeGet(&ctx, 0, &cqSize), ncclSuccess);
  EXPECT_GT(cqSize, 0);
  ASSERT_EQ(ncclIbResiliencyDataRqSizeGet(&ctx, 0, &rqSize), ncclSuccess);
  EXPECT_GT(rqSize, 0u);

  ncclIbResiliencySend sendCtx = {};
  sendCtx.base.baseComm = &base;
  sendCtx.base.nProbingQps = 1;
  base.isSend = true;
  base.nRemDevs = 3;
  EXPECT_EQ(ncclIbResiliencyRemoteCompletionRecordsSet(&sendCtx.base, 0x77, 0x1234, 2), ncclSuccess);
  EXPECT_EQ(sendCtx.remCmplRecordsInfo[2].rkey, 0x77u);
  EXPECT_EQ(sendCtx.remCmplRecordsInfo[2].addr, 0x1234u);
  EXPECT_EQ(ncclIbResiliencyRemoteCompletionRecordsSet(&sendCtx.base, 0x88, 0x5678, 3), ncclInvalidArgument);

  int standbyPhysicalQp = Physical(0, 1);
  ctx.qpInfo[standbyPhysicalQp].state = ncclIbResiliencyQpStateFailed;
  base.qpCold[standbyPhysicalQp].remQpn = 0;
  EXPECT_EQ(ncclIbResiliencyTryRejoinSingleQp(&ctx, standbyPhysicalQp), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[standbyPhysicalQp].failCount, 1);

  ctx.nProbingQps = 0;
  EXPECT_EQ(ncclIbResiliencyClose(&ctx), ncclSuccess);

  ctx.qpInfo[standbyPhysicalQp].state = ncclIbResiliencyQpStateOk;
  ctx.inProgress = false;
  EXPECT_EQ(ncclIbResiliencyProgress(&ctx), ncclSuccess);
}
