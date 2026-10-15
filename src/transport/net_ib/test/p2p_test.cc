/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

/** @brief Records one send work request observed by the provider seam. */
struct PostedSend {
  uint32_t qpNumber;
  uint64_t wrId;
  ibv_wr_opcode opcode;
  int sendFlags;
  uint32_t immData;
  uint64_t remoteAddress;
  uint32_t remoteKey;
  int numSge;
  ibv_sge sge;
};

/**
 * @brief Captures provider calls, completions, resiliency callbacks, and diagnostic formatting
 * from p2p.cc.
 * @note Fields provide injected results and retain observations for the current scenario.
 */
struct P2pFakeState {
  int postSendCalls;
  int postRecvCalls;
  int postSendError;
  int postRecvError;
  int pollError;
  int pollCalls;
  int resiliencyProgressCalls;
  int resiliencyCompletionCalls;
  int resiliencySendCompleteCalls;
  int resiliencyErrorCalls;
  int statsCalls;
  ncclResult_t statsResult;
  bool resiliencyComplete;
  int debugWeightCalls;
  int debugLatencyCalls;
  int debugByteCalls;
  int debugViewCalls;
  int debugSequenceCalls;
  std::array<PostedSend, 64> postedSends;
  int postedSendCount;
  uint64_t lastRecvWrId;
  std::array<ibv_wc, 16> completions;
  int completionCount;
  int completionIndex;
  int warningCalls;
  std::string lastWarning;
  int socketAddressCalls;
  ncclSocket* addressedSocket;
  int socketStringCalls;
  int socketNumericForm;
  int statusStringCalls;
  ibv_wc_status formattedStatus;
  int opcodeStringCalls;
  ibv_wc_opcode formattedOpcode;
  int gidStringCalls;
  std::array<ibv_gid*, 2> formattedGids;
  std::array<size_t, 2> gidStringCapacities;
};

/** @brief Records one generated P2P-parameter lookup. */
struct P2pParameterLoadCall {
  std::string environment;
  int64_t defaultValue;
};

P2pFakeState g_fake;
int64_t testArThreshold = 8192;
int64_t testReceiverMatchingScheme = 1;
int64_t testFrmSpreadEpsUs = 4;
int64_t testFrmZeroStampHoldoff = 2;
bool returnUnnormalizedActiveWeights = false;
std::vector<P2pParameterLoadCall> p2pParameterLoads;

/**
 * @brief Calls the active-window weight helper and optionally supplies a controlled unnormalized
 * result.
 * @param[in] latency Valid non-NULL latency array with totalQps entries.
 * @param[in] totalQps Total QP count in [1, CIM_MAX_QPS].
 * @param[in] baseQp Active-window base index in [0, totalQps).
 * @param[in] activeQps Active-window count in [1, totalQps].
 * @param[out] weights Valid non-NULL array with activeQps entries.
 * @note Side effects: Replaces weights; test control may replace its first two entries with 0.2f
 * and 0.3f.
 */
void ComputeActiveWindowWeightsForTest(
    const uint64_t* latency, int totalQps, int baseQp, int activeQps, float* weights) {
  CIM_computeActiveWindowWeights(latency, totalQps, baseQp, activeQps, weights);
  if (returnUnnormalizedActiveWeights && activeQps >= 2) {
    weights[0] = 0.2f;
    weights[1] = 0.3f;
  }
}

/**
 * @brief Resets every fake-provider record and process-local p2p test control.
 * @note Side effects: Clears fake state and restores injected parameter and weight controls to
 * defaults.
 */
void ResetFakes() {
  g_fake = {};
  testArThreshold = 8192;
  testReceiverMatchingScheme = 1;
  testFrmSpreadEpsUs = 4;
  testFrmZeroStampHoldoff = 2;
  returnUnnormalizedActiveWeights = false;
  p2pParameterLoads.clear();
}

/**
 * @brief Resolves p2p parameters through the production-generated accessors.
 * @param[in] environment Exact environment-variable name requested by the accessor.
 * @param[in] defaultValue Production default for the requested parameter.
 * @param[in] uninitialized Production cache sentinel, accepted without modification.
 * @param[in] cache Production cache storage left unmodified to keep scenarios independent.
 * @param[out] noCache Receives one so every direct scenario reaches the dependency seam.
 * @return The configured value selected by environment, or defaultValue for an unknown key.
 * @note Side effects: Records the parameter name and default value.
 */
int64_t LoadP2pParameterForTest(const char* environment, int64_t defaultValue,
    int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  (void)uninitialized;
  (void)cache;
  p2pParameterLoads.push_back({environment, defaultValue});
  *noCache = 1;
  if (std::strcmp(environment, "NCCL_IB_AR_THRESHOLD") == 0) return testArThreshold;
  if (std::strcmp(environment, "NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME") == 0) {
    return testReceiverMatchingScheme;
  }
  if (std::strcmp(environment, "NCCL_FRM_SPREAD_EPS_US") == 0) return testFrmSpreadEpsUs;
  if (std::strcmp(environment, "NCCL_FRM_ZERO_STAMP_HOLDOFF") == 0) return testFrmZeroStampHoldoff;
  return defaultValue;
}

/**
 * @brief Captures one production warning after formatting all variadic arguments.
 * @param[in] format Valid printf-style format string.
 * @note Side effects: Replaces g_fake.lastWarning and increments warningCalls.
 */
void CaptureP2pWarningForTest(const char* format, ...) {
  char message[2048] = {};
  va_list arguments;
  va_start(arguments, format);
  std::vsnprintf(message, sizeof(message), format, arguments);
  va_end(arguments);
  ++g_fake.warningCalls;
  g_fake.lastWarning = message;
}

/**
 * @brief Records the completion status passed to diagnostic formatting.
 * @param[in] status Provider completion status.
 * @return Stable diagnostic text.
 * @note Side effects: Records status and increments statusStringCalls.
 */
const char* CompletionStatusForP2pTest(ibv_wc_status status) {
  ++g_fake.statusStringCalls;
  g_fake.formattedStatus = status;
  return "test-wc-status";
}

/**
 * @brief Records the completion opcode passed to diagnostic formatting.
 * @param[in] opcode Provider completion opcode.
 * @return Stable diagnostic text.
 * @note Side effects: Records opcode and increments opcodeStringCalls.
 */
const char* CompletionOpcodeForP2pTest(ibv_wc_opcode opcode) {
  ++g_fake.opcodeStringCalls;
  g_fake.formattedOpcode = opcode;
  return "test-wc-opcode";
}

/**
 * @brief Records GID formatting arguments and delegates to the production header helper.
 * @param[in] gid Exact binary GID selected for diagnostics.
 * @param[out] output Destination receiving canonical IPv6 text.
 * @param[in] capacity Destination capacity.
 * @return The production GID formatting result.
 * @note Side effects: Records gid and capacity and writes output on successful conversion.
 */
const char* FormatGidForP2pTest(union ibv_gid* gid, char* output, size_t capacity) {
  const int index = g_fake.gidStringCalls++;
  if (index < static_cast<int>(g_fake.formattedGids.size())) {
    g_fake.formattedGids[index] = gid;
    g_fake.gidStringCapacities[index] = capacity;
  }
  return ibvGetGidStr(gid, output, capacity);
}

/**
 * @brief Captures each posted send work request and returns the configured provider result.
 * @param[in] qp Valid non-NULL fake QP; its contents are not modified.
 * @param[in] wr NULL or the first work request in a finite linked chain.
 * @param[out] badWr Valid non-NULL storage receiving wr on configured failure or NULL on success.
 * @return The configured provider result in [INT_MIN, INT_MAX].
 * @note Side effects: Increments fake post counters, stores bounded chain copies, and writes
 * @p badWr.
 */
int FakePostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** badWr) {
  g_fake.postSendCalls++;
  for (ibv_send_wr* current = wr; current != nullptr; current = current->next) {
    if (g_fake.postedSendCount >= static_cast<int>(g_fake.postedSends.size())) break;
    PostedSend& captured = g_fake.postedSends[g_fake.postedSendCount++];
    captured = {};
    captured.qpNumber = qp->qp_num;
    captured.wrId = current->wr_id;
    captured.opcode = current->opcode;
    captured.sendFlags = current->send_flags;
    captured.immData = current->imm_data;
    captured.remoteAddress = current->wr.rdma.remote_addr;
    captured.remoteKey = current->wr.rdma.rkey;
    captured.numSge = current->num_sge;
    if (current->num_sge > 0 && current->sg_list != nullptr) captured.sge = current->sg_list[0];
  }
  *badWr = g_fake.postSendError == 0 ? nullptr : wr;
  return g_fake.postSendError;
}

/**
 * @brief Captures one posted receive work-request ID and returns the configured provider result.
 * @param[in] qp Valid non-NULL fake QP; its contents are not modified.
 * @param[in] wr Valid non-NULL receive work request.
 * @param[out] badWr Valid non-NULL storage receiving wr on configured failure or NULL on success.
 * @return The configured provider result in [INT_MIN, INT_MAX].
 * @note Side effects: Increments the receive-post count, records wr->wr_id, and writes
 * @p badWr.
 */
int FakePostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** badWr) {
  (void)qp;
  g_fake.postRecvCalls++;
  g_fake.lastRecvWrId = wr->wr_id;
  *badWr = g_fake.postRecvError == 0 ? nullptr : wr;
  return g_fake.postRecvError;
}

/**
 * @brief Returns queued fake completions up to the caller-provided capacity.
 * @param[in] cq Valid non-NULL fake completion queue; its contents are not modified.
 * @param[in] count Maximum output count in [0, INT_MAX].
 * @param[out] completions NULL for zero count or an array of at least count entries.
 * @return A configured negative provider error or the number of copied completions in [0, count].
 * @note Side effects: Increments the poll count, writes copied completions, and advances the
 * fake completion cursor by the returned nonnegative count.
 */
int FakePollCq(ibv_cq* cq, int count, ibv_wc* completions) {
  (void)cq;
  g_fake.pollCalls++;
  if (g_fake.pollError < 0) return g_fake.pollError;
  int available = g_fake.completionCount - g_fake.completionIndex;
  int copied = std::min(count, available);
  for (int i = 0; i < copied; i++) completions[i] = g_fake.completions[g_fake.completionIndex + i];
  g_fake.completionIndex += copied;
  return copied;
}

/**
 * @brief Initializes one fake verbs context, queue pair, and completion queue.
 * @param[out] context Valid non-NULL context storage receiving fake provider callbacks.
 * @param[out] qp Valid non-NULL QP storage receiving context and qpNumber.
 * @param[out] cq Valid non-NULL CQ storage receiving context.
 * @param[in] qpNumber QP number in [0, UINT32_MAX].
 * @note Side effects: Replaces all three supplied fake verbs objects.
 */
void InitFakeVerbs(ibv_context* context, ibv_qp* qp, ibv_cq* cq, uint32_t qpNumber) {
  *context = {};
  context->ops.post_send = FakePostSend;
  context->ops.post_recv = FakePostRecv;
  context->ops.poll_cq = FakePollCq;
  *qp = {};
  qp->context = context;
  qp->qp_num = qpNumber;
  *cq = {};
  cq->context = context;
}

/** @brief Captures one RDMA write in a posted send chain. */
struct PostedWrite {
  uint64_t localAddress;
  uint64_t remoteAddress;
  uint32_t bytes;
  uint32_t localKey;
  uint32_t remoteKey;
  int numSge;
};

/** @brief Summarizes one posted send chain and its captured writes. */
struct PostedRange {
  ibv_qp* qp;
  uint64_t localAddress;
  uint64_t remoteAddress;
  uint32_t bytes;
  uint32_t immediateRkey;
  uint32_t localKey;
  unsigned int sendFlags;
  unsigned int immediatePosts;
  unsigned int emptyImmediatePosts;
  unsigned int signaledPosts;
  std::vector<PostedWrite> writes;
};

ibv_context verbsContext = {};
constexpr int kTestVerbsQps = 16;
ibv_qp verbsQps[kTestVerbsQps] = {};
std::vector<PostedRange> postedRanges;
int postRecvCalls = 0;

int CapturePostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr**) {
  PostedRange range = {};
  range.qp = qp;
  for (ibv_send_wr* current = wr; current != nullptr; current = current->next) {
    range.sendFlags |= current->send_flags;
    if (current->send_flags & IBV_SEND_SIGNALED) range.signaledPosts++;
    if (current->opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
      range.immediateRkey = current->wr.rdma.rkey;
      range.immediatePosts++;
      if (current->num_sge == 0) range.emptyImmediatePosts++;
    }
    PostedWrite write = {0, current->wr.rdma.remote_addr, 0, 0,
                         current->wr.rdma.rkey, current->num_sge};
    if (current->num_sge > 0 && current->sg_list != nullptr) {
      write.localAddress = current->sg_list[0].addr;
      write.bytes = current->sg_list[0].length;
      write.localKey = current->sg_list[0].lkey;
    }
    range.writes.push_back(write);
    if (current->num_sge == 0 || current->sg_list == nullptr) continue;
    if (range.bytes == 0) {
      range.localAddress = current->sg_list[0].addr;
      range.remoteAddress = current->wr.rdma.remote_addr;
      range.localKey = current->sg_list[0].lkey;
    }
    range.bytes += current->sg_list[0].length;
  }
  postedRanges.push_back(range);
  return 0;
}

int CapturePostRecv(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**) {
  postRecvCalls++;
  return 0;
}

void ResetVerbsFakes() {
  std::memset(&verbsContext, 0, sizeof(verbsContext));
  std::memset(verbsQps, 0, sizeof(verbsQps));
  verbsContext.ops.post_send = CapturePostSend;
  verbsContext.ops.post_recv = CapturePostRecv;
  for (int i = 0; i < kTestVerbsQps; i++) {
    verbsQps[i].context = &verbsContext;
    verbsQps[i].qp_num = 700 + i;
  }
  postedRanges.clear();
  postRecvCalls = 0;
}

/** @brief Owns fixed backing storage for one communicator's multi-send scratch views. */
struct MultiSendBuffers {
  float weights[kTestVerbsQps] = {};
  int equalChunks[NCCL_NET_IB_MAX_RECVS] = {};
  int requestOffsets[NCCL_NET_IB_MAX_RECVS] = {};
  uintptr_t localAddresses[NCCL_NET_IB_MAX_RECVS] = {};
  uint64_t remoteAddresses[NCCL_NET_IB_MAX_RECVS] = {};
  uint32_t chunks[NCCL_NET_IB_MAX_RECVS][kTestVerbsQps] = {};
  ncclIbQp* qps[kTestVerbsQps] = {};
  int qpIndices[kTestVerbsQps] = {};
  uint32_t chainBytes[kTestVerbsQps] = {};
  ibv_send_wr wrs[kTestVerbsQps][NCCL_NET_IB_MAX_RECVS + 1] = {};
  ibv_sge sges[kTestVerbsQps][NCCL_NET_IB_MAX_RECVS] = {};
  uint32_t replayPlans[NET_IB_MAX_REQUESTS * kTestVerbsQps] = {};
  uint8_t replayPlanFrozen[NET_IB_MAX_REQUESTS] = {};
};

void AttachMultiSendBuffers(ncclIbSendComm* comm, MultiSendBuffers* buffers,
                            int nqps = 2, bool attachReplayPlan = false) {
  ncclIbMultiSendScratch& scratch = comm->multiSendScratch;
  scratch.storage = buffers;
  scratch.nqps = nqps;
  scratch.chunkRowBytes = sizeof(buffers->chunks[0]);
  scratch.wrRowBytes = sizeof(buffers->wrs[0]);
  scratch.sgeRowBytes = sizeof(buffers->sges[0]);
  scratch.lwPerQp = buffers->weights;
  scratch.equalChunkSize = buffers->equalChunks;
  scratch.reqOffsetShadow = buffers->requestOffsets;
  scratch.sgeAddrShadow = buffers->localAddresses;
  scratch.remoteAddrShadow = buffers->remoteAddresses;
  scratch.chunkSizePerReq = &buffers->chunks[0][0];
  scratch.qpPerPost = buffers->qps;
  scratch.qpIndexPerPost = buffers->qpIndices;
  scratch.chainBytesPerPost = buffers->chainBytes;
  scratch.wrsPerPost = &buffers->wrs[0][0];
  scratch.sgesPerPost = &buffers->sges[0][0];
  scratch.replayPlanBytes = attachReplayPlan ? buffers->replayPlans : nullptr;
  scratch.replayPlanFrozen = attachReplayPlan ? buffers->replayPlanFrozen : nullptr;
}

/**
 * @brief Creates a minimal one-device send communicator for request-only tests.
 * @return A zero-initialized send communicator with send mode and one local device selected.
 * @note This function has no externally visible side effects.
 */
ncclIbSendComm MakeSendComm() {
  ncclIbSendComm comm = {};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  return comm;
}

/**
 * @brief Connects a send communicator to one fake data QP and CQ.
 * @param[in,out] comm Valid non-NULL send communicator to initialize.
 * @param[in] qp Valid non-NULL fake verbs QP.
 * @param[in] cq Valid non-NULL fake verbs CQ.
 * @note Side effects: Replaces the communicator's one-device and one-QP routing fields.
 */
void InitSendPath(ncclIbSendComm* comm, ibv_qp* qp, ibv_cq* cq) {
  comm->base.isSend = true;
  comm->base.ready = 1;
  comm->base.vProps.ndevs = 1;
  comm->base.nqps = 1;
  comm->base.nDataQps = 1;
  comm->base.qps[0].qp = qp;
  comm->base.qps[0].devIndex = 0;
  comm->base.qps[0].remDevIdx = 0;
  comm->base.activeQps[0] = &comm->base.qps[0];
  comm->devs[0].base.cq = cq;
}

/**
 * @brief Extends a one-lane send path with one physical standby QP for that logical lane.
 * @param[in,out] comm Valid non-NULL send communicator initialized by InitSendPath.
 * @param[in] standbyQp Valid non-NULL fake verbs QP with a distinct QP number.
 * @note Side effects: Enables standby geometry and rebuilds the communicator's QP-number map.
 */
void AddStandbySendPathForTest(ncclIbSendComm* comm, ibv_qp* standbyQp) {
  comm->base.hasStandbyPaths = true;
  comm->base.nPhysicalQps = 2;
  comm->base.qps[0].logicalLane = 0;
  comm->base.qps[1].qp = standbyQp;
  comm->base.qps[1].devIndex = 0;
  comm->base.qps[1].remDevIdx = 0;
  comm->base.qps[1].logicalLane = 0;
  qpMapBuild(&comm->base);
}

/**
 * @brief Creates one live send request bound to device zero.
 * @param[in,out] comm Valid non-NULL one-device send communicator.
 * @param[in] size Submitted byte count reported when the request completes.
 * @return The communicator-owned request prepared in slot zero.
 * @note Side effects: Replaces request slot zero and its send lookup entry.
 */
ncclIbRequest* PrepareSingleSendRequestForTest(ncclIbSendComm* comm, int size) {
  ncclIbRequest* request = &comm->base.reqs[0];
  *request = {};
  request->base = &comm->base;
  request->type = NCCL_NET_IB_REQ_SEND;
  request->id = 0;
  request->nreqs = 1;
  request->send.size = size;
  request->events[0] = 1;
  request->devBases[0] = &comm->devs[0].base;
  comm->sendReqs[0][0] = request;
  comm->sendReqsCnt[0] = 1;
  return request;
}

/**
 * @brief Connects a receive communicator to one fake data QP and CQ.
 * @param[in,out] comm Valid non-NULL receive communicator to initialize.
 * @param[in] qp Valid non-NULL fake verbs QP.
 * @param[in] cq Valid non-NULL fake verbs CQ.
 * @note Side effects: Replaces the communicator's one-device and one-QP routing fields.
 */
void InitRecvPath(ncclIbRecvComm* comm, ibv_qp* qp, ibv_cq* cq) {
  comm->base.isSend = false;
  comm->base.ready = 1;
  comm->base.vProps.ndevs = 1;
  comm->base.nqps = 1;
  comm->base.nDataQps = 1;
  comm->base.qps[0].qp = qp;
  comm->base.qps[0].devIndex = 0;
  comm->base.qps[0].remDevIdx = 0;
  comm->base.activeQps[0] = &comm->base.qps[0];
  comm->devs[0].base.cq = cq;
  comm->devs[0].gpuFlush.qp.qp = qp;
}

/**
 * @brief Provides stable aligned backing for every two-QP multi-send scratch view.
 * @note AttachFrmScratch exposes these arrays through production pointer-and-stride fields.
 */
struct alignas(128) FrmScratchStorage {
  float weights[2];
  int equalChunks[NCCL_NET_IB_MAX_RECVS];
  int requestOffsets[NCCL_NET_IB_MAX_RECVS];
  uintptr_t localAddresses[NCCL_NET_IB_MAX_RECVS];
  uint64_t remoteAddresses[NCCL_NET_IB_MAX_RECVS];
  uint32_t chunks[NCCL_NET_IB_MAX_RECVS][2];
  ncclIbQp* qps[2];
  int qpIndices[2];
  uint32_t chainBytes[2];
  ibv_send_wr workRequests[2][NCCL_NET_IB_MAX_RECVS + 1];
  ibv_sge scatterGatherEntries[2][NCCL_NET_IB_MAX_RECVS];
  uint32_t replayPlans[NET_IB_MAX_REQUESTS * 2];
  uint8_t replayPlanFrozen[NET_IB_MAX_REQUESTS];
#if FRM_DEBUG
  uint32_t sentBytes[2][NCCL_NET_IB_MAX_RECVS];
#endif
};

/**
 * @brief Attaches bounded two-QP scratch storage to a fake send communicator.
 * @param[in,out] comm Valid non-NULL send communicator whose base.nqps is in [1, 2].
 * @param[out] storage Valid non-NULL storage cleared and retained for the communicator's full use.
 * @note Side effects: Replaces comm->multiSendScratch and clears every byte of storage.
 */
void AttachFrmScratch(ncclIbSendComm* comm, FrmScratchStorage* storage) {
  *storage = {};
  ncclIbMultiSendScratch& scratch = comm->multiSendScratch;
  scratch = {};
  scratch.storage = storage;
  scratch.nqps = comm->base.nqps;
  scratch.chunkRowBytes = sizeof(storage->chunks[0]);
  scratch.wrRowBytes = sizeof(storage->workRequests[0]);
  scratch.sgeRowBytes = sizeof(storage->scatterGatherEntries[0]);
  scratch.lwPerQp = storage->weights;
  scratch.equalChunkSize = storage->equalChunks;
  scratch.reqOffsetShadow = storage->requestOffsets;
  scratch.sgeAddrShadow = storage->localAddresses;
  scratch.remoteAddrShadow = storage->remoteAddresses;
  scratch.chunkSizePerReq = &storage->chunks[0][0];
  scratch.qpPerPost = storage->qps;
  scratch.qpIndexPerPost = storage->qpIndices;
  scratch.chainBytesPerPost = storage->chainBytes;
  scratch.wrsPerPost = &storage->workRequests[0][0];
  scratch.sgesPerPost = &storage->scatterGatherEntries[0][0];
  scratch.replayPlanBytes = storage->replayPlans;
  scratch.replayPlanFrozen = storage->replayPlanFrozen;
#if FRM_DEBUG
  scratch.sendSzRowBytes = sizeof(storage->sentBytes[0]);
  scratch.sendSzPerPost = &storage->sentBytes[0][0];
#endif
}

}  // namespace

double g_inv_cyc_per_us = 1.0;
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS] = {};

/**
 * @brief Returns the device base belonging to a fake send or receive communicator.
 * @param[in] base Valid non-NULL communicator base.
 * @param[in] devIndex Device index in [0, base->vProps.ndevs).
 * @return The selected embedded device base.
 * @note This function has no side effects.
 */
struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) return &reinterpret_cast<ncclIbSendComm*>(base)->devs[devIndex].base;
  return &reinterpret_cast<ncclIbRecvComm*>(base)->devs[devIndex].base;
}

/**
 * @brief Records one fake resiliency-progress call.
 * @param[in] resiliency NULL or a valid resiliency context; it is not inspected.
 * @return ncclSuccess.
 * @note Side effects: Increments the fake resiliency-progress counter.
 */
ncclResult_t ncclIbResiliencyProgress(struct ncclIbResiliency* resiliency) {
  (void)resiliency;
  g_fake.resiliencyProgressCalls++;
  return ncclSuccess;
}

/**
 * @brief Records one fake fatal-statistics check.
 * @param[in] stats NULL or valid statistics; they are not inspected.
 * @param[in] functionName NULL or a valid function name; it is not inspected.
 * @return The configured NCCL result.
 * @note Side effects: Increments the fake statistics-call counter.
 */
ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stats, const char* functionName) {
  (void)stats;
  (void)functionName;
  g_fake.statsCalls++;
  return g_fake.statsResult;
}

/**
 * @brief Returns the configured fake resiliency completion decision.
 * @param[in] request Valid non-NULL request; it is not modified.
 * @param[out] isComplete Valid non-NULL storage receiving the configured boolean result.
 * @return ncclSuccess.
 * @note Side effects: Writes isComplete and increments the fake completion-query counter.
 */
ncclResult_t ncclIbResiliencyRequestIsComplete(struct ncclIbRequest* request, bool* isComplete) {
  (void)request;
  g_fake.resiliencyCompletionCalls++;
  *isComplete = g_fake.resiliencyComplete;
  return ncclSuccess;
}

/**
 * @brief Records one fake resilient-send completion callback.
 * @param[in] resiliency NULL or a valid resiliency context; it is not inspected.
 * @param[in] request Valid non-NULL request; it is not modified.
 * @return ncclSuccess.
 * @note Side effects: Increments the fake resilient-send completion counter.
 */
ncclResult_t ncclIbResiliencySendRequestComplete(
    struct ncclIbResiliency* resiliency, struct ncclIbRequest* request) {
  (void)resiliency;
  (void)request;
  g_fake.resiliencySendCompleteCalls++;
  return ncclSuccess;
}

/**
 * @brief Records one fake resiliency completion-error callback.
 * @param[in] resiliency NULL or a valid resiliency context; it is not inspected.
 * @param[in] completion Valid non-NULL failed completion; it is not modified.
 * @param[in] devIndex Device index in [0, NCCL_IB_MAX_DEVS_PER_NIC).
 * @return ncclSuccess.
 * @note Side effects: Increments the fake completion-error counter.
 */
ncclResult_t ncclIbResiliencyHandleCompletionError(
    struct ncclIbResiliency* resiliency, struct ibv_wc* completion, int devIndex) {
  (void)resiliency;
  (void)completion;
  (void)devIndex;
  g_fake.resiliencyErrorCalls++;
  return ncclSuccess;
}

/**
 * @brief Records registration of a fake debug weight source.
 * @param[in] label NULL or a valid label; it is not inspected.
 * @param[in] weights NULL or a valid weight array; it is not inspected.
 * @param[in] count Element count in [0, CIM_MAX_QPS].
 * @note Side effects: Increments the fake weight-source counter.
 */
void CIM_DebugLatPrinterSetWeightSource(const char* label, volatile float* weights, int count) {
  (void)label;
  (void)weights;
  (void)count;
  g_fake.debugWeightCalls++;
}

/**
 * @brief Records registration of fake normalized and raw latency sources.
 * @param[in] label NULL or a valid label; it is not inspected.
 * @param[in] latency NULL or a valid latency array; it is not inspected.
 * @param[in] rawLatency NULL or a valid raw-latency array; it is not inspected.
 * @param[in] count Element count in [0, CIM_MAX_QPS].
 * @note Side effects: Increments the fake latency-source counter.
 */
void CIM_DebugLatPrinterSetSource(
    const char* label, volatile uint64_t* latency, volatile uint64_t* rawLatency, int count) {
  (void)label;
  (void)latency;
  (void)rawLatency;
  (void)count;
  g_fake.debugLatencyCalls++;
}

/**
 * @brief Records registration of a fake byte-count source.
 * @param[in] label NULL or a valid label; it is not inspected.
 * @param[in] bytes NULL or a valid byte-count array; it is not inspected.
 * @param[in] count Element count in [0, CIM_MAX_QPS].
 * @note Side effects: Increments the fake byte-source counter.
 */
void CIM_DebugLatPrinterSetByteSource(const char* label, volatile uint32_t* bytes, int count) {
  (void)label;
  (void)bytes;
  (void)count;
  g_fake.debugByteCalls++;
}

/**
 * @brief Records registration of fake debug-view sources.
 * @param[in] viewCount NULL or valid view-count storage; it is not inspected.
 * @param[in] viewQpIndex NULL or a valid view-index array; it is not inspected.
 * @note Side effects: Increments the fake view-source counter.
 */
void CIM_DebugLatPrinterSetViewSource(volatile uint32_t* viewCount, volatile uint32_t* viewQpIndex) {
  (void)viewCount;
  (void)viewQpIndex;
  g_fake.debugViewCalls++;
}

/**
 * @brief Records registration of a fake debug synchronization sequence.
 * @param[in] sequence NULL or valid sequence storage; it is not inspected.
 * @note Side effects: Increments the fake sequence-source counter.
 */
void CIM_DebugLatPrinterSetSyncSeq(volatile uint32_t* sequence) {
  (void)sequence;
  g_fake.debugSequenceCalls++;
}

/**
 * @brief Returns a zeroed fake peer socket address.
 * @param[in] socket NULL or a valid socket; it is not inspected.
 * @param[out] address Valid non-NULL address storage receiving zero bytes.
 * @return ncclSuccess.
 * @note Side effects: Records @p socket, increments the call count, and replaces @p address.
 */
ncclResult_t ncclSocketGetAddr(struct ncclSocket* socket, union ncclSocketAddress* address) {
  ++g_fake.socketAddressCalls;
  g_fake.addressedSocket = socket;
  *address = {};
  return ncclSuccess;
}

/**
 * @brief Writes a stable fake peer name for diagnostic paths.
 * @param[in] address NULL or a valid socket address; it is not inspected.
 * @param[out] buffer Valid non-NULL buffer of at least 10 bytes.
 * @param[in] numericHostForm Any integer; it is not inspected.
 * @return buffer.
 * @note Side effects: Records @p numericHostForm, increments the call count, and writes
 * test-peer to @p buffer.
 */
const char* ncclSocketToString(
    const union ncclSocketAddress* address, char* buffer, const int numericHostForm) {
  (void)address;
  ++g_fake.socketStringCalls;
  g_fake.socketNumericForm = numericHostForm;
  std::strcpy(buffer, "test-peer");
  return buffer;
}

#undef WARN
/**
 * @brief Compiles p2p.cc with diagnostics, parameters, and active-weight calculation
 * redirected to test seams.
 */
#define WARN(...) CaptureP2pWarningForTest(__VA_ARGS__)
#define ibvWcStatusStr CompletionStatusForP2pTest
#define ibvWcOpcodeStr CompletionOpcodeForP2pTest
#define ibvGetGidStr FormatGidForP2pTest
#define ncclLoadParam LoadP2pParameterForTest
#define CIM_computeActiveWindowWeights ComputeActiveWindowWeightsForTest
#ifdef NCCL_ENABLE_NET_PROFILING
namespace {

ncclResult_t occlTestProfilerCallback(void**, int, void*, int64_t, void*) {
  return ncclSuccess;
}

}  // namespace

ncclProfilerCallback_t ncclProfilerFunction = occlTestProfilerCallback;
#endif
#include "../p2p.cc"  // NOLINT(build/include)
#undef CIM_computeActiveWindowWeights
#undef ncclLoadParam
#undef ibvGetGidStr
#undef ibvWcOpcodeStr
#undef ibvWcStatusStr
#undef WARN

/**
 * @brief Verifies two p2p accessors retain their keys, defaults, and configured values.
 */
TEST(P2pDirectTest, LoadsMatchingAndHoldoffParameters) {
  SCOPED_TRACE("The two generated accessors must preserve metadata and expose seam results.");
  ResetFakes();
  EXPECT_EQ(ncclParamIbReceiverSideMatchingScheme(), 1)
      << "Receiver-side matching must retain BY_ID as its production default";
  EXPECT_EQ(ncclParamFrmZeroStampHoldoff(), 2)
      << "Zero-stamp holdoff must retain its production default";
  testReceiverMatchingScheme = 0;
  testFrmZeroStampHoldoff = 7;
  EXPECT_EQ(ncclParamIbReceiverSideMatchingScheme(), 0)
      << "Receiver-side matching must expose a configured BY_INDEX value";
  EXPECT_EQ(ncclParamFrmZeroStampHoldoff(), 7)
      << "Zero-stamp holdoff must expose a configured positive value";

  ASSERT_EQ(p2pParameterLoads.size(), 4u)
      << "No-cache scenarios must invoke the loader for every direct call";
  EXPECT_EQ(p2pParameterLoads[0].environment, "NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME")
      << "The matching accessor must request its exact production environment key";
  EXPECT_EQ(p2pParameterLoads[0].defaultValue, 1)
      << "The matching accessor must pass default one to the loader";
  EXPECT_EQ(p2pParameterLoads[1].environment, "NCCL_FRM_ZERO_STAMP_HOLDOFF")
      << "The holdoff accessor must request its exact production environment key";
  EXPECT_EQ(p2pParameterLoads[1].defaultValue, 2)
      << "The holdoff accessor must pass default two to the loader";
  testReceiverMatchingScheme = 1;
  testFrmZeroStampHoldoff = 2;
}

/**
 * @brief Verifies FRM gate decisions and telemetry reset on a healthy-to-unhealthy transition.
 * @note Side effects: Mutates only local communicator, CIM, and resiliency fixtures.
 */
TEST(P2pFrmTest, GateAndTelemetryReset) {
  // A missing communicator or resiliency context is healthy by definition.
  EXPECT_TRUE(ibFrmResiliencyHealthy(nullptr));
  ncclIbSendComm comm = {};
  EXPECT_TRUE(ibFrmResiliencyHealthy(&comm.base));
  EXPECT_NO_FATAL_FAILURE(ibFrmFlushTelemetry(nullptr));

  // Healthy resiliency opens the gate when CIM is present.
  CIMState cim = {};
  cim.nqps = 2;
  ncclIbResiliency resiliency = {};
  resiliency.frmAllQpsHealthy = true;
  comm.base.nqps = 2;
  comm.base.resiliency = &resiliency;
  comm.cim = &cim;
  EXPECT_TRUE(ibFrmGateUpdate(&comm));
  EXPECT_TRUE(comm.frmGateOpen);

  // Completion-side FRM rejects null, unbound, and receive-side requests.
  ncclIbRequest request = {};
  EXPECT_FALSE(ibFrmEnabledForTest(nullptr));
  EXPECT_FALSE(ibFrmEnabledForTest(&request));
  request.base = &comm.base;
  EXPECT_FALSE(ibFrmEnabledForTest(&request));

  // A send request without resiliency must use CIM and open its completion-side gate.
  comm.base.isSend = true;
  comm.base.resiliency = nullptr;
  EXPECT_TRUE(ibFrmEnabledForTest(&request));
  EXPECT_TRUE(comm.frmGateOpen);
  comm.base.resiliency = &resiliency;

  // Closing an open gate clears every FRM telemetry family and signature entry.
  comm.base.frmFields.send.qp_lat[0] = 11;
  comm.base.frmCold.qp_raw_lat[0] = 12;
  comm.base.frmCold.qp_lat_bytes[0] = 13;
  comm.base.frmFields.completion.qp_zero_stamp_holdoff[0] = 2;
  comm.base.qps[0].sig_head = 2;
  comm.base.qps[0].sig_tail = 0;
  comm.base.qps[0].sig_tsc[0] = 14;
  comm.base.qps[0].sig_bytes[0] = 15;
  cim.lw[0] = 0.5f;
  cim.latency[0] = 16;
  cim.raw_latency[0] = 17;
  cim.lat_bytes[0] = 18;
  cim.split0_mild_score[0] = 3;
  cim.split0_mild_slow_qp[0] = 1;
  resiliency.inProgress = true;
  EXPECT_FALSE(ibFrmEnabledForSend(&comm));
  EXPECT_FALSE(comm.frmGateOpen);
  EXPECT_EQ(comm.base.frmFields.send.qp_lat[0], 0u);
  EXPECT_EQ(comm.base.frmCold.qp_raw_lat[0], 0u);
  EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 0u);
  EXPECT_EQ(comm.base.frmFields.completion.qp_zero_stamp_holdoff[0], 0u);
  EXPECT_EQ(comm.base.qps[0].sig_tail, comm.base.qps[0].sig_head);
  EXPECT_EQ(comm.base.qps[0].sig_tsc[0], 0u);
  EXPECT_EQ(comm.base.qps[0].sig_bytes[0], 0u);
  EXPECT_FLOAT_EQ(cim.lw[0], 0.0f);
  EXPECT_EQ(cim.latency[0], 0u);
  EXPECT_EQ(cim.raw_latency[0], 0u);
  EXPECT_EQ(cim.lat_bytes[0], 0u);
  EXPECT_EQ(cim.split0_mild_score[0], 0u);
  EXPECT_EQ(cim.split0_mild_slow_qp[0], kIbFrmSplit0MildSlowInvalid);
  EXPECT_EQ(cim.dbg_view_count, 2u);
  EXPECT_EQ(cim.dbg_view_qp_index[0], 0u);
  EXPECT_EQ(cim.dbg_view_qp_index[1], 1u);
}

/**
 * @brief Verifies direct and active-window FRM weight generation without hardware state.
 * @note Side effects: Mutates local fixtures and temporarily changes process-local parameter and
 * weight controls.
 */
TEST(P2pFrmTest, WeightComputation) {
  ResetFakes();

  // Test-controlled getters must expose defaults and configured values without environment
  // caching.
  EXPECT_EQ(ncclParamIbArThreshold(), 8192);
  EXPECT_EQ(ncclParamFrmSpreadEpsUs(), 4);
  testArThreshold = 257;
  testFrmSpreadEpsUs = 9;
  EXPECT_EQ(ncclParamIbArThreshold(), 257);
  EXPECT_EQ(ncclParamFrmSpreadEpsUs(), 9);

  // A communicator without CIM must always produce uniform weights.
  ncclIbSendComm comm = {};
  comm.base.nqps = 4;
  comm.base.nDataQps = 2;
  float weights[4] = {-1.0f, -1.0f, 91.0f, 92.0f};
  ibFrmComputeLw(&comm, 2, weights, false, 0);
  EXPECT_FLOAT_EQ(weights[0], 0.5f);
  EXPECT_FLOAT_EQ(weights[1], 0.5f);
  EXPECT_FLOAT_EQ(weights[2], 91.0f);
  EXPECT_FLOAT_EQ(weights[3], 92.0f);

  // Equal measured latencies must remain a normalized uniform split.
  CIMState cim = {};
  comm.cim = &cim;
  comm.base.frmFields.send.qp_lat[0] = 100;
  comm.base.frmFields.send.qp_lat[1] = 100;
  ibFrmComputeLw(&comm, 2, weights, true, 0);
  EXPECT_FLOAT_EQ(weights[0], 0.5f);
  EXPECT_FLOAT_EQ(weights[1], 0.5f);

  // A wrapped active window must produce normalized weights only for requested entries.
  comm.base.frmFields.send.qp_lat[2] = 200;
  comm.base.frmFields.send.qp_lat[3] = 300;
  weights[2] = 91.0f;
  weights[3] = 92.0f;
  ibFrmComputeLw(&comm, 2, weights, false, 3);
  EXPECT_GE(weights[0], 0.0f);
  EXPECT_GE(weights[1], 0.0f);
  EXPECT_LT(weights[0], weights[1]);
  EXPECT_NEAR(weights[0] + weights[1], 1.0f, 1e-6f);
  EXPECT_FLOAT_EQ(weights[2], 91.0f);
  EXPECT_FLOAT_EQ(weights[3], 92.0f);

  // A dependency result whose positive weights sum to 0.5 must be defensively renormalized to 1.0.
  returnUnnormalizedActiveWeights = true;
  ibFrmComputeLw(&comm, 2, weights, false, 0);
  EXPECT_NEAR(weights[0], 0.4f, 1e-6f);
  EXPECT_NEAR(weights[1], 0.6f, 1e-6f);
  EXPECT_NEAR(weights[0] + weights[1], 1.0f, 1e-6f);
  EXPECT_FLOAT_EQ(weights[2], 91.0f);
  EXPECT_FLOAT_EQ(weights[3], 92.0f);

  // Restore temporary controls so this test remains isolated from later tests and shuffled
  // execution.
  returnUnnormalizedActiveWeights = false;
  testArThreshold = 8192;
  testFrmSpreadEpsUs = 4;
}

/**
 * @brief Verifies split-zero equal, mild, and full decisions plus mild-state decay.
 * @note Side effects: Mutates only local communicator, CIM, and output-weight fixtures.
 */
TEST(P2pFrmTest, SplitZeroDecisions) {
  // Missing inputs and a single active lane cannot request nonuniform weighting.
  float weights[2] = {0.25f, 0.75f};
  EXPECT_EQ(ibFrmSelectSplit0Decision(nullptr, 0, weights), IB_FRM_SPLIT0_EQUAL);
  EXPECT_FLOAT_EQ(weights[0], 0.25f);
  EXPECT_FLOAT_EQ(weights[1], 0.75f);
  ncclIbSendComm comm = {};
  CIMState cim = {};
  comm.cim = &cim;
  comm.base.nqps = 2;
  comm.base.nDataQps = 1;
  cim.nqps = 2;
  EXPECT_EQ(ibFrmSelectSplit0Decision(&comm, 0, weights), IB_FRM_SPLIT0_EQUAL);
  EXPECT_FLOAT_EQ(weights[0], 0.25f);
  EXPECT_FLOAT_EQ(weights[1], 0.75f);

  // Equal nonzero samples must remain equal and decay any previous mild classification.
  comm.base.nDataQps = 2;
  cim.spread_eps_us = 4;
  cim.split0_mild_score[0] = 1;
  cim.split0_mild_slow_qp[0] = 1;
  comm.base.frmFields.send.qp_lat[0] = 10;
  comm.base.frmFields.send.qp_lat[1] = 10;
  EXPECT_EQ(ibFrmSelectSplit0Decision(&comm, 0, weights), IB_FRM_SPLIT0_EQUAL);
  EXPECT_EQ(cim.split0_mild_score[0], 0u);
  EXPECT_EQ(cim.split0_mild_slow_qp[0], kIbFrmSplit0MildSlowInvalid);
  EXPECT_FLOAT_EQ(weights[0], 0.25f);
  EXPECT_FLOAT_EQ(weights[1], 0.75f);

  // A one-microsecond spread at the minimum threshold must remain equal and decay mild state.
  cim.spread_eps_us = kIbFrmSplit0MildEpsUs;
  cim.split0_mild_score[0] = 1;
  cim.split0_mild_slow_qp[0] = 1;
  comm.base.frmFields.send.qp_lat[1] = 11;
  EXPECT_EQ(ibFrmSelectSplit0Decision(&comm, 0, weights), IB_FRM_SPLIT0_EQUAL);
  EXPECT_EQ(cim.split0_mild_score[0], 0u);
  EXPECT_EQ(cim.split0_mild_slow_qp[0], kIbFrmSplit0MildSlowInvalid);
  EXPECT_FLOAT_EQ(weights[0], 0.25f);
  EXPECT_FLOAT_EQ(weights[1], 0.75f);

  // Three repeated weak spreads on the same slow QP must activate the mild bias.
  cim.spread_eps_us = 4;
  comm.base.frmFields.send.qp_lat[0] = 10;
  comm.base.frmFields.send.qp_lat[1] = 12;
  EXPECT_EQ(ibFrmSelectSplit0Decision(&comm, 0, weights), IB_FRM_SPLIT0_EQUAL);
  EXPECT_EQ(ibFrmSelectSplit0Decision(&comm, 0, weights), IB_FRM_SPLIT0_EQUAL);
  EXPECT_EQ(ibFrmSelectSplit0Decision(&comm, 0, weights), IB_FRM_SPLIT0_MILD);
  EXPECT_NEAR(weights[0], 0.6f, 1e-6f);
  EXPECT_NEAR(weights[1], 0.4f, 1e-6f);
  EXPECT_NEAR(weights[0] + weights[1], 1.0f, 1e-6f);

  // A strong spread must select full FRM and clear accumulated mild state.
  comm.base.frmFields.send.qp_lat[1] = 20;
  EXPECT_EQ(ibFrmSelectSplit0Decision(&comm, 0, weights), IB_FRM_SPLIT0_FULL);
  EXPECT_EQ(cim.split0_mild_score[0], 0u);
  EXPECT_EQ(cim.split0_mild_slow_qp[0], kIbFrmSplit0MildSlowInvalid);
  EXPECT_NEAR(weights[0], 0.6f, 1e-6f);
  EXPECT_NEAR(weights[1], 0.4f, 1e-6f);

  // Invalid samples decay an existing score and clear the marker at zero.
  weights[0] = 0.3f;
  weights[1] = 0.7f;
  cim.split0_mild_score[0] = 1;
  cim.split0_mild_slow_qp[0] = 1;
  comm.base.frmFields.send.qp_lat[1] = 0;
  EXPECT_EQ(ibFrmSelectSplit0Decision(&comm, 0, weights), IB_FRM_SPLIT0_EQUAL);
  EXPECT_EQ(cim.split0_mild_score[0], 0u);
  EXPECT_EQ(cim.split0_mild_slow_qp[0], kIbFrmSplit0MildSlowInvalid);
  EXPECT_FLOAT_EQ(weights[0], 0.3f);
  EXPECT_FLOAT_EQ(weights[1], 0.7f);

  // Direct decay must ignore invalid windows and clear a slow marker exactly when its score reaches
  // zero.
  cim.split0_mild_score[1] = 1;
  cim.split0_mild_slow_qp[1] = 0;
  ibFrmSplit0DecayMildState(&cim, -1);
  EXPECT_EQ(cim.split0_mild_score[1], 1u);
  ibFrmSplit0DecayMildState(&cim, 1);
  EXPECT_EQ(cim.split0_mild_score[1], 0u);
  EXPECT_EQ(cim.split0_mild_slow_qp[1], kIbFrmSplit0MildSlowInvalid);

  // A zero score with a stale slow-QP marker must clear the marker without underflowing the score.
  cim.split0_mild_score[1] = 0;
  cim.split0_mild_slow_qp[1] = 0;
  ibFrmSplit0DecayMildState(&cim, 1);
  EXPECT_EQ(cim.split0_mild_score[1], 0u);
  EXPECT_EQ(cim.split0_mild_slow_qp[1], kIbFrmSplit0MildSlowInvalid);

  // Direct mild weighting must support its defined single-lane and missing-slow-lane fallbacks.
  float fallbackWeight = 0.25f;
  ibFrmSplit0ApplyMildWeights(&comm, 0, 1, 0, &fallbackWeight);
  EXPECT_FLOAT_EQ(fallbackWeight, 1.0f);
  fallbackWeight = 0.25f;
  ibFrmSplit0ApplyMildWeights(&comm, 0, 2, -1, &fallbackWeight);
  EXPECT_FLOAT_EQ(fallbackWeight, 1.0f);
}

/**
 * @brief Verifies signature-ring ordering, empty behavior, and oldest-entry eviction at capacity.
 * @note Side effects: Mutates only a local QP signature ring and output variables.
 */
TEST(P2pFrmTest, SignatureRing) {
  // An empty ring must report no entry and preserve both outputs.
  ncclIbQp qp = {};
  uint64_t timestamp = 91;
  uint32_t bytes = 92;
  EXPECT_EQ(ibQpSigPop(&qp, &timestamp, &bytes), 0);
  EXPECT_EQ(timestamp, 91u);
  EXPECT_EQ(bytes, 92u);

  // More than one ring of pushes must retain exactly the newest entries in FIFO order.
  for (uint32_t i = 0; i < QP_SIG_RING + 2; i++) ibQpSigPush(&qp, 100 + i, 200 + i);
  EXPECT_EQ(qp.sig_head - qp.sig_tail, QP_SIG_RING);
  for (uint32_t i = 2; i < QP_SIG_RING + 2; i++) {
    ASSERT_EQ(ibQpSigPop(&qp, &timestamp, &bytes), 1);
    EXPECT_EQ(timestamp, 100u + i);
    EXPECT_EQ(bytes, 200u + i);
  }
  EXPECT_EQ(ibQpSigPop(&qp, &timestamp, &bytes), 0);
}

/**
 * @brief Verifies debug-view mapping, clamping, and one-time source registration.
 * @note Side effects: Mutates local debug state and process-local fake registration counters.
 */
TEST(P2pFrmTest, DebugPublishing) {
  ResetFakes();
  ncclIbSendComm comm = {};
  CIMState cim = {};
  comm.cim = &cim;
  comm.base.nqps = 4;
  cim.nqps = 4;

  // A split view maps entries directly, while an active view wraps physical QP indices.
  ibFrmPublishDebugView(&comm, 3, true, 2);
  EXPECT_EQ(cim.dbg_view_count, 3u);
  EXPECT_EQ(cim.dbg_view_qp_index[0], 0u);
  EXPECT_EQ(cim.dbg_view_qp_index[2], 2u);
  ibFrmPublishDebugView(&comm, 3, false, 3);
  EXPECT_EQ(cim.dbg_view_qp_index[0], 3u);
  EXPECT_EQ(cim.dbg_view_qp_index[1], 0u);
  EXPECT_EQ(cim.dbg_view_qp_index[2], 1u);

  // Out-of-range view counts must clamp to the documented empty and full bounds.
  ibFrmPublishDebugView(&comm, -1, true, 0);
  EXPECT_EQ(cim.dbg_view_count, 0u);
  ibFrmPublishDebugView(&comm, 99, true, 0);
  EXPECT_EQ(cim.dbg_view_count, 4u);

  // Disabled publication is a no-op, and enabled publication registers every source only once.
  int sourcesSet = 0;
  ibFrmFinalizeDebugPublishing(&comm, false, &sourcesSet);
  EXPECT_EQ(sourcesSet, 0);
  ibFrmFinalizeDebugPublishing(&comm, true, &sourcesSet);
  EXPECT_EQ(sourcesSet, 1);
  EXPECT_EQ(g_fake.debugWeightCalls, 1);
  EXPECT_EQ(g_fake.debugLatencyCalls, 1);
  EXPECT_EQ(g_fake.debugByteCalls, 1);
  EXPECT_EQ(g_fake.debugViewCalls, 1);
  EXPECT_EQ(g_fake.debugSequenceCalls, 1);
  uint32_t sequenceAfterRegistration = cim.dbg_seq;
  ibFrmFinalizeDebugPublishing(&comm, true, &sourcesSet);
  EXPECT_GT(cim.dbg_seq, sequenceAfterRegistration);
  EXPECT_EQ(g_fake.debugWeightCalls, 1);
}

/**
 * @brief Verifies aligned chunk construction, exact total preservation, and no-op boundaries.
 * @note Side effects: Mutates only local output arrays.
 */
TEST(P2pFrmTest, ChunkComputation) {
  // Rounded chunks that overshoot must be reduced from the tail to preserve the exact total.
  const float weights[3] = {0.5f, 0.3f, 0.2f};
  uint32_t chunks[3] = {91, 92, 93};
  ibFrmComputeChunksAll(weights, 3, 1000, 128, chunks);
  EXPECT_EQ(chunks[0], 512u);
  EXPECT_EQ(chunks[1], 384u);
  EXPECT_EQ(chunks[2], 104u);
  EXPECT_EQ(chunks[0] + chunks[1] + chunks[2], 1000u);

  // Missing weight mass must be assigned to the final chunk.
  const float zeroWeights[3] = {0.0f, 0.0f, 0.0f};
  ibFrmComputeChunksAll(zeroWeights, 3, 100, 128, chunks);
  EXPECT_EQ(chunks[0], 0u);
  EXPECT_EQ(chunks[1], 0u);
  EXPECT_EQ(chunks[2], 100u);

  // Positive sub-alignment shares must round up to the minimum aligned chunk before tail
  // correction.
  const float smallWeights[2] = {0.25f, 0.75f};
  uint32_t smallChunks[2] = {0, 0};
  ibFrmComputeChunksAll(smallWeights, 2, 256, 128, smallChunks);
  EXPECT_EQ(smallChunks[0], 128u);
  EXPECT_EQ(smallChunks[1], 128u);
  EXPECT_EQ(smallChunks[0] + smallChunks[1], 256u);

  // A nonpositive count must leave existing output bytes untouched.
  chunks[0] = 91;
  ibFrmComputeChunksAll(weights, 0, 1000, 128, chunks);
  EXPECT_EQ(chunks[0], 91u);
  EXPECT_NO_FATAL_FAILURE(ibFrmComputeChunksAll(weights, 3, 1000, 128, nullptr));
}

/**
 * @brief Verifies request allocation skips used slots and resets common bookkeeping.
 * @note Side effects: Mutates only a local communicator request pool.
 */
TEST(P2pRequestTest, AllocationSkipsUsedSlots) {
  // The allocator must skip both occupied entries before selecting the first unused slot.
  auto comm = MakeSendComm();
  comm.base.reqs[0].type = NCCL_NET_IB_REQ_SEND;
  comm.base.reqs[1].type = NCCL_NET_IB_REQ_RECV;
  comm.base.reqs[2].id = 9;
  comm.base.reqs[2].nreqs = 3;
  comm.base.reqs[2].events[0] = 4;
  comm.base.reqs[2].send.size = 77;
  ncclIbRequest* request = nullptr;

  // A successful allocation must return the selected slot and reset only its common fields.
  ASSERT_EQ(ncclIbGetRequest(&comm.base, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request, &comm.base.reqs[2]);
  EXPECT_EQ(request->base, &comm.base);
  EXPECT_EQ(request->sock, nullptr);
  EXPECT_EQ(request->id, 0u);
  EXPECT_EQ(request->nreqs, 0);
  EXPECT_EQ(request->events[0], 0);
  EXPECT_EQ(request->send.size, 77);
}

/**
 * @brief Verifies a full request pool reports failure and clears the output pointer.
 * @note Side effects: Mutates only a local output pointer.
 */
TEST(P2pRequestTest, AllocationFailsWhenPoolIsFull) {
  // Marking every request occupied creates the allocator's documented failure boundary.
  auto comm = MakeSendComm();
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    comm.base.reqs[i].type = NCCL_NET_IB_REQ_SEND;
  }
  ncclIbRequest* request = reinterpret_cast<ncclIbRequest*>(0x1);

  // Failure must not leak a stale request pointer back to the caller.
  EXPECT_EQ(ncclIbGetRequest(&comm.base, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
}

/**
 * @brief Verifies event accounting retains the selected communicator-device base.
 * @note Side effects: Mutates only a local request's event bookkeeping.
 */
TEST(P2pRequestTest, AddEventTracksDeviceBase) {
  // A request bound to a one-device communicator has exactly one valid event counter.
  auto comm = MakeSendComm();
  ncclIbRequest request = {};
  request.base = &comm.base;

  // Repeated events must accumulate on the same counter and preserve its device association.
  ncclIbAddEvent(&request, 0);
  ncclIbAddEvent(&request, 0);
  EXPECT_EQ(request.events[0], 2);
  EXPECT_EQ(request.devBases[0], &comm.devs[0].base);
}

/**
 * @brief Verifies completion accounting covers the expanded eight-device ABI.
 */
TEST(NetIbP2pTest, RequestCompletionTracksAllEightDevices) {
  ResetFakes();
  auto comm = MakeSendComm();
  comm.base.vProps.ndevs = NCCL_IB_MAX_DEVS_PER_NIC;
  ncclIbRequest request = {};
  request.base = &comm.base;

  request.events[6] = 1;
  EXPECT_FALSE(ncclIbRequestIsComplete(&request));
  request.events[6] = 0;
  EXPECT_TRUE(ncclIbRequestIsComplete(&request));

  ncclIbResiliency resiliency = {};
  resiliency.ndevs = NCCL_IB_MAX_DEVS_PER_NIC;
  comm.base.resiliency = &resiliency;
  request.events[0] = 1;
  request.events[6] = -1;
  g_fake.resiliencyComplete = true;
  EXPECT_TRUE(ncclIbRequestIsComplete(&request));
}

#ifdef NCCL_ENABLE_NET_PROFILING
/**
 * @brief Verifies profiling lookup covers all sixteen live QP event slots.
 */
TEST(NetIbP2pTest, ProfilingQpLookupSupportsSixteenLiveEvents) {
  ResetVerbsFakes();
  auto comm = MakeSendComm();
  comm.base.nqps = kTestVerbsQps;
  comm.base.nPhysicalQps = kTestVerbsQps;

  ncclIbRequest request = {};
  request.base = &comm.base;
  request.pInfo[0].nEventHandles = kTestVerbsQps;
  for (int event = 0; event < kTestVerbsQps; event++) {
    comm.base.qps[event].qp = &verbsQps[event];
    request.pInfo[0].qpIndex[event] = event;
  }

  EXPECT_EQ(getReqQpIndex(&request, 0, verbsQps[15].qp_num), 15);
  request.pInfo[0].nEventHandles = 15;
  EXPECT_EQ(getReqQpIndex(&request, 0, verbsQps[15].qp_num), 0);
}
#endif

/**
 * @brief Verifies periodic CTS signaling follows logical lanes across physical paths.
 */
TEST(NetIbP2pTest, CtsPeriodicSignalFollowsLogicalLane) {
  ResetVerbsFakes();
  ncclIbRecvComm receive = {};
  receive.base.vProps.ndevs = 2;
  receive.base.nqps = 2;
  receive.base.nPhysicalQps = 4;
  receive.base.hasStandbyPaths = true;
  receive.base.nRemDevs = 2;
  for (int logicalLane = 0; logicalLane < receive.base.nqps; logicalLane++) {
    receive.base.qps[logicalLane].qp = &verbsQps[logicalLane];
    receive.base.qps[logicalLane].devIndex = 0;
    receive.base.qps[logicalLane].remDevIdx = 0;
    receive.base.qps[logicalLane].logicalLane = logicalLane;
    receive.base.activeQps[logicalLane] = &receive.base.qps[logicalLane];
  }
  receive.base.remDevs[0].rkey = 0x51;
  receive.remCtsFifo.addr = 0x100000;

  ncclIbRequest request = {};
  request.base = &receive.base;
  request.id = 1;
  request.nreqs = 1;
  ASSERT_EQ(ncclIbPostFifo(&receive, &request, 1), ncclSuccess);
  ASSERT_EQ(postedRanges.size(), 1u);
  EXPECT_EQ(postedRanges[0].qp, &verbsQps[1]);
  EXPECT_NE(postedRanges[0].sendFlags & IBV_SEND_SIGNALED, 0u);

  postedRanges.clear();
  request.id = 3;
  ASSERT_EQ(ncclIbPostFifo(&receive, &request, 3), ncclSuccess);
  ASSERT_EQ(postedRanges.size(), 1u);
  EXPECT_EQ(postedRanges[0].sendFlags & IBV_SEND_SIGNALED, 0u);

  postedRanges.clear();
  ncclIbResiliency resiliency = {};
  receive.base.resiliency = &resiliency;
  ASSERT_EQ(ncclIbPostFifo(&receive, &request, 3), ncclSuccess);
  ASSERT_EQ(postedRanges.size(), 1u);
  EXPECT_NE(postedRanges[0].sendFlags & IBV_SEND_SIGNALED, 0u);
}

/**
 * @brief Verifies direct request-ID decoding and sender/receiver lookup of exact IDs and stale
 * aliases.
 */
TEST(P2pRequestTest, DecodesAndFindsRequestsDirectly) {
  SCOPED_TRACE("Request helpers must decode IDs and validate the full logical ID after slot selection.");
  for (uint64_t requestId : {uint64_t{0}, uint64_t{17}, uint64_t{NET_IB_MAX_REQUESTS + 5}}) {
    SCOPED_TRACE(testing::Message() << "requestId=" << requestId);
    EXPECT_EQ(ibReqWrIdDecode(IB_REQ_WR_ID_FLAG | requestId), requestId)
        << "Decoding a valid encoded request ID must recover every logical ID bit";
  }

  auto sendComm = std::make_unique<ncclIbSendComm>();
  ncclIbRequest sendRequest = {};
  constexpr uint64_t sendId = NET_IB_MAX_REQUESTS + 5;
  EXPECT_EQ(ibSendReqFind(sendComm.get(), sendId), nullptr)
      << "Sender lookup must return NULL when the selected slot has no live request";
  sendRequest.id = sendId;
  sendRequest.type = NCCL_NET_IB_REQ_SEND;
  sendComm->sendReqs[sendId % NET_IB_MAX_REQUESTS][0] = &sendRequest;
  EXPECT_EQ(ibSendReqFind(sendComm.get(), sendId), &sendRequest)
      << "Sender lookup must return the exact request stored in the selected slot";
  EXPECT_EQ(ibSendReqFind(sendComm.get(), sendId + NET_IB_MAX_REQUESTS), nullptr)
      << "A stale logical ID aliasing the same slot must be rejected";
  EXPECT_EQ(sendRequest.id, sendId) << "Sender lookup must preserve the request ID";
  EXPECT_EQ(sendRequest.type, NCCL_NET_IB_REQ_SEND) << "Sender lookup must preserve request state";

  auto recvComm = std::make_unique<ncclIbRecvComm>();
  ncclIbRequest receiveRequest = {};
  constexpr uint64_t receiveId = 2 * NET_IB_MAX_REQUESTS + 7;
  EXPECT_EQ(ibRecvReqFind(recvComm.get(), receiveId), nullptr)
      << "Receiver lookup must return NULL when the selected slot has no live request";
  receiveRequest.id = receiveId;
  receiveRequest.type = NCCL_NET_IB_REQ_RECV;
  recvComm->recvReqs[receiveId % NET_IB_MAX_REQUESTS] = &receiveRequest;
  EXPECT_EQ(ibRecvReqFind(recvComm.get(), receiveId), &receiveRequest)
      << "Receiver lookup must return the exact request stored in the selected slot";
  EXPECT_EQ(ibRecvReqFind(recvComm.get(), receiveId + NET_IB_MAX_REQUESTS), nullptr)
      << "A stale receiver ID aliasing the same slot must be rejected";
  EXPECT_EQ(receiveRequest.id, receiveId) << "Receiver lookup must preserve the request ID";
  EXPECT_EQ(receiveRequest.type, NCCL_NET_IB_REQ_RECV) << "Receiver lookup must preserve request state";
}

/**
 * @brief Verifies direct CTS-QP selection and indexed request retrieval output behavior.
 */
TEST(P2pRequestTest, SelectsCtsQueuePairAndRetrievesIndex) {
  SCOPED_TRACE("CTS routing must select device zero QPs, and indexed retrieval must validate its complete range.");
  auto recvComm = std::make_unique<ncclIbRecvComm>();
  recvComm->base.vProps.ndevs = 2;
  recvComm->base.activeQps[0] = &recvComm->base.qps[0];
  recvComm->base.activeQps[1] = &recvComm->base.qps[1];
  const int deviceCountBefore = recvComm->base.vProps.ndevs;
  ncclIbQp* const firstActiveBefore = recvComm->base.activeQps[0];
  ncclIbQp* const secondActiveBefore = recvComm->base.activeQps[1];
  ncclIbQp* selected = reinterpret_cast<ncclIbQp*>(uintptr_t{1});

  EXPECT_EQ(ncclIbRecvCommGetQpForCts(recvComm.get(), 1, &selected), ncclSuccess)
      << "A valid odd request ID must route CTS to the second device";
  EXPECT_EQ(selected, &recvComm->base.qps[1])
      << "CTS must use the first QP belonging to request ID modulo device count";
  EXPECT_EQ(ncclIbRecvCommGetQpForCts(recvComm.get(), 2, &selected), ncclSuccess)
      << "A valid even request ID must route CTS to the first device";
  EXPECT_EQ(selected, &recvComm->base.qps[0])
      << "CTS routing must wrap exactly at the device count";
  EXPECT_EQ(recvComm->base.vProps.ndevs, deviceCountBefore)
      << "CTS routing must preserve the communicator's device count";
  EXPECT_EQ(recvComm->base.activeQps[0], firstActiveBefore)
      << "CTS routing must preserve the first caller-owned active-QP binding";
  EXPECT_EQ(recvComm->base.activeQps[1], secondActiveBefore)
      << "CTS routing must preserve the second caller-owned active-QP binding";

  auto requests = std::make_unique<ncclIbRequest[]>(NET_IB_MAX_REQUESTS);
  requests[NET_IB_MAX_REQUESTS - 1].id = 91;
  ncclIbRequest* request = reinterpret_cast<ncclIbRequest*>(uintptr_t{1});
  EXPECT_EQ(ncclIbRequestRetrieveAsIndex(
      requests.get(), NET_IB_MAX_REQUESTS - 1, &request), ncclSuccess)
      << "The final valid request index must succeed";
  EXPECT_EQ(request, &requests[NET_IB_MAX_REQUESTS - 1])
      << "Successful indexed retrieval must publish the exact array element";
  EXPECT_EQ(requests[NET_IB_MAX_REQUESTS - 1].id, 91u)
      << "Indexed retrieval must not modify the selected request";

  request = reinterpret_cast<ncclIbRequest*>(uintptr_t{1});
  EXPECT_EQ(ncclIbRequestRetrieveAsIndex(requests.get(), NET_IB_MAX_REQUESTS, &request),
      ncclInternalError)
      << "An index equal to the request capacity must return ncclInternalError";
  EXPECT_EQ(request, reinterpret_cast<ncclIbRequest*>(uintptr_t{1}))
      << "Rejected indexed retrieval must preserve the caller's output pointer";
}

/**
 * @brief Verifies direct completion-error logging emits peer, verbs, GID, and HCA details without
 * changing inputs.
 */
TEST(P2pCompletionTest, LogsCompletionErrorDirectly) {
  SCOPED_TRACE("Completion error logging must include routable GIDs only for Ethernet links.");
  ResetFakes();
  auto comm = std::make_unique<ncclIbSendComm>();
  comm->base.isSend = true;
  comm->base.vProps.ndevs = 1;
  comm->base.nRemDevs = 1;
  ibv_device device = {};
  std::strcpy(device.name, "mlx5-test");
  ibv_context context = {};
  context.device = &device;
  ibv_pd protectionDomain = {};
  protectionDomain.context = &context;
  comm->devs[0].base.pd = &protectionDomain;
  comm->devs[0].base.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  ASSERT_EQ(inet_pton(AF_INET6, "2001:db8::1", comm->devs[0].base.gidInfo.localGid.raw), 1)
      << "The test fixture must create a valid local GID";
  ASSERT_EQ(inet_pton(AF_INET6, "2001:db8::2", comm->base.remDevs[0].remoteGid.raw), 1)
      << "The test fixture must create a valid remote GID";
  ibv_wc completion = {};
  completion.status = IBV_WC_REM_ACCESS_ERR;
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.vendor_err = 77;
  const ibv_wc completionSnapshot = completion;
  const union ibv_gid localGidSnapshot = comm->devs[0].base.gidInfo.localGid;
  const union ibv_gid remoteGidSnapshot = comm->base.remDevs[0].remoteGid;

  EXPECT_EQ(ncclIbLogCompletionWithError(&comm->base, &completion, 0), ncclSuccess)
      << "Logging a valid failed completion must succeed";
  EXPECT_EQ(g_fake.warningCalls, 1) << "One failed completion must emit exactly one warning";
  EXPECT_EQ(g_fake.socketAddressCalls, 1) << "Logging must read the peer address exactly once";
  EXPECT_EQ(g_fake.addressedSocket, &comm->base.sock)
      << "Peer-address lookup must receive the communicator's socket";
  EXPECT_EQ(g_fake.socketStringCalls, 1) << "Logging must format the peer address exactly once";
  EXPECT_EQ(g_fake.statusStringCalls, 1) << "Logging must format the completion status exactly once";
  EXPECT_EQ(g_fake.formattedStatus, completion.status)
      << "Status formatting must receive the failed completion's exact status";
  EXPECT_EQ(g_fake.opcodeStringCalls, 1) << "Logging must format the completion opcode exactly once";
  EXPECT_EQ(g_fake.formattedOpcode, completion.opcode)
      << "Opcode formatting must receive the failed completion's exact opcode";
  ASSERT_EQ(g_fake.gidStringCalls, 2) << "Ethernet logging must format local and remote GIDs";
  EXPECT_EQ(g_fake.formattedGids[0], &comm->devs[0].base.gidInfo.localGid)
      << "The first GID conversion must receive the local GID";
  EXPECT_EQ(g_fake.formattedGids[1], &comm->base.remDevs[0].remoteGid)
      << "The second GID conversion must receive the remote GID";
  EXPECT_EQ(g_fake.gidStringCapacities[0], INET6_ADDRSTRLEN)
      << "Local GID formatting must receive the complete destination capacity";
  EXPECT_EQ(g_fake.gidStringCapacities[1], INET6_ADDRSTRLEN)
      << "Remote GID formatting must receive the complete destination capacity";
  EXPECT_NE(g_fake.lastWarning.find("peer test-peer"), std::string::npos)
      << "The warning must identify the connected peer";
  EXPECT_NE(g_fake.lastWarning.find("status=test-wc-status"), std::string::npos)
      << "The warning must include the provider completion status";
  EXPECT_NE(g_fake.lastWarning.find("opcode=test-wc-opcode"), std::string::npos)
      << "The warning must include the provider completion opcode";
  EXPECT_NE(g_fake.lastWarning.find("vendor_err=77"), std::string::npos)
      << "The warning must include the vendor error code";
  EXPECT_NE(g_fake.lastWarning.find("localGid 2001:db8::1"), std::string::npos)
      << "Ethernet warnings must include the local GID";
#if 0 // ADDRESS_LATER
  EXPECT_NE(g_fake.lastWarning.find("remoteGid 2001:db8::2"), std::string::npos)
      << "Ethernet warnings must label and separate the remote GID clearly";
#endif 0 // ADDRESS_LATER
  EXPECT_NE(g_fake.lastWarning.find("hca mlx5-test"), std::string::npos)
      << "The warning must identify the local HCA";
  EXPECT_EQ(std::memcmp(&completion, &completionSnapshot, sizeof(completion)), 0)
      << "Logging must not modify the completion";
  EXPECT_EQ(std::memcmp(&comm->devs[0].base.gidInfo.localGid, &localGidSnapshot,
      sizeof(localGidSnapshot)), 0)
      << "Logging must not modify the local GID";
  EXPECT_EQ(std::memcmp(&comm->base.remDevs[0].remoteGid, &remoteGidSnapshot,
      sizeof(remoteGidSnapshot)), 0)
      << "Logging must not modify the remote GID";

  comm->devs[0].base.gidInfo.link_layer = IBV_LINK_LAYER_INFINIBAND;
  g_fake.warningCalls = 0;
  g_fake.lastWarning.clear();
  g_fake.gidStringCalls = 0;
  EXPECT_EQ(ncclIbLogCompletionWithError(&comm->base, &completion, 0), ncclSuccess)
      << "Logging an InfiniBand completion error must succeed";
  EXPECT_EQ(g_fake.warningCalls, 1) << "The InfiniBand scenario must emit one warning";
  EXPECT_EQ(g_fake.gidStringCalls, 0)
      << "InfiniBand logging must not attempt RoCE GID conversion";
  EXPECT_EQ(g_fake.lastWarning.find("localGid"), std::string::npos)
      << "InfiniBand warnings must omit RoCE GID fields";
  EXPECT_EQ(g_fake.lastWarning.find("remoteGids"), std::string::npos)
      << "InfiniBand warnings must omit remote RoCE GID fields";
}

/**
 * @brief Verifies sender completion IDs encode request identity and expose its request-pool slot.
 * @note Side effects: None.
 */
TEST(P2pRequestTest, EncodesSenderRequestIdentity) {
  SCOPED_TRACE(
      "A sender completion token must preserve the request ID and decode its modulo-pool slot.");
  constexpr uint64_t requestId = 72;
  constexpr uint64_t ignoredSlot = 3;

  const uint64_t token = ibSendReqWrIdEncode(requestId, ignoredSlot);

  EXPECT_EQ(token, ibReqWrIdEncode(requestId))
      << "Sender completion encoding must preserve the current request-ID contract";
  EXPECT_TRUE(ibReqWrIdIsRequest(token))
      << "A sender completion token must remain in the request WR-ID namespace";
  EXPECT_EQ(ibReqWrIdDecode(token), requestId)
      << "Decoding a sender completion token must recover the request ID";
  EXPECT_EQ(ibSendReqWrIdDecodeSlot(token), requestId % NET_IB_MAX_REQUESTS)
      << "Sender slot decoding must select the pool slot encoded by the request ID's low bits";
  EXPECT_EQ(ibSendReqWrIdEncode(requestId, ignoredSlot + 1), token)
      << "The retained slot parameter must not alter the request-ID token";

  for (uint64_t boundaryId : {uint64_t{0}, uint64_t{NET_IB_MAX_REQUESTS - 1},
           uint64_t{NET_IB_MAX_REQUESTS}, uint64_t{2 * NET_IB_MAX_REQUESTS + 19}}) {
    SCOPED_TRACE(testing::Message() << "boundary requestId=" << boundaryId);
    const uint64_t boundaryToken = ibSendReqWrIdEncode(boundaryId, ignoredSlot);
    EXPECT_EQ(ibSendReqWrIdDecodeSlot(boundaryToken), boundaryId % NET_IB_MAX_REQUESTS)
        << "Slot decoding must wrap exactly at the fixed request-pool capacity";
  }
}

/**
 * @brief Verifies grouped requests receive their shared request-ID completion token.
 * @note Side effects: Mutates the requests' WR-ID tokens without changing logical request state.
 */
TEST(P2pRequestTest, RefreshesGroupedSendTokens) {
  // Two grouped requests must share the representative request's encoded ID.
  auto comm = MakeSendComm();
  ncclIbRequest req0 = {};
  ncclIbRequest req1 = {};
  constexpr int slot = 3;
  req0.id = NET_IB_MAX_REQUESTS + slot;
  req1.id = req0.id;
  req0.type = NCCL_NET_IB_REQ_SEND;
  req1.type = NCCL_NET_IB_REQ_SEND;
  comm.sendReqs[slot][0] = &req0;
  comm.sendReqs[slot][1] = &req1;
  comm.sendWrIdCounter = 71;
  const uint64_t expectedToken = ibReqWrIdEncode(req0.id);

  // Each refresh must publish the same lookup token for the whole logical request group.
  ncclIbSendReqsRefreshWrIdToken(&comm, slot, 2);
  const uint64_t firstToken = req0.send.wrIdToken;
  EXPECT_EQ(firstToken, expectedToken)
      << "The representative request must receive its encoded request ID";
  EXPECT_EQ(req1.send.wrIdToken, firstToken)
      << "Every request in one grouped post must receive the same completion token";
  EXPECT_EQ(ibSendReqFindByWrId(&comm, firstToken), &req0)
      << "A current completion token must resolve to the group's representative request";
  EXPECT_EQ(comm.sendWrIdCounter, 71u)
      << "Request-ID token refresh must not consume the unused generation counter";

  // Repeating the refresh must preserve the stable token and lookup result.
  ncclIbSendReqsRefreshWrIdToken(&comm, slot, 2);
  const uint64_t secondToken = req0.send.wrIdToken;
  EXPECT_EQ(secondToken, expectedToken)
      << "Repeated refresh must preserve the encoded request ID";
  EXPECT_EQ(req1.send.wrIdToken, secondToken)
      << "Every grouped request must retain the shared request-ID token";
  EXPECT_EQ(secondToken, firstToken)
      << "Refreshing an unchanged logical request must preserve its token";
  EXPECT_EQ(ibSendReqFindByWrId(&comm, secondToken), &req0)
      << "The repeated token must resolve to the live representative request";
  EXPECT_EQ(comm.sendWrIdCounter, 71u)
      << "Repeated refresh must leave the unused generation counter unchanged";

  // Refreshing completion tokens must not alter the logical request ID or type.
  EXPECT_EQ(req0.id, static_cast<uint64_t>(NET_IB_MAX_REQUESTS + slot))
      << "Refreshing a WR-ID token must preserve the first logical request ID";
  EXPECT_EQ(req1.id, req0.id)
      << "Refreshing a WR-ID token must preserve grouped logical request IDs";
  EXPECT_EQ(req0.type, NCCL_NET_IB_REQ_SEND)
      << "Refreshing a WR-ID token must preserve the first request type";
  EXPECT_EQ(req1.type, NCCL_NET_IB_REQ_SEND)
      << "Refreshing a WR-ID token must preserve the second request type";
}

/**
 * @brief Verifies request release changes only the request type marker.
 * @note Side effects: Mutates only a local request's type.
 */
TEST(P2pRequestTest, FreeRequestMarksSlotUnused) {
  // A live flush request provides a non-unused type and unrelated state to preserve.
  ncclIbRequest request = {};
  request.type = NCCL_NET_IB_REQ_FLUSH;
  request.id = 123;

  // Release must make the slot reusable without rewriting other fields.
  EXPECT_EQ(ncclIbFreeRequest(&request), ncclSuccess);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(request.id, 123u);
}

/**
 * @brief Verifies completion lookup for send, receive-ID, flush, stale, and invalid records.
 * @note Side effects: Mutates local fixtures and temporarily changes the process-local matching
 * control.
 */
TEST(P2pRequestTest, CompletionLookup) {
  ResetFakes();

  // A send completion must resolve only an encoded ID that still names the live slot request.
  ncclIbSendComm sendComm = MakeSendComm();
  ncclIbRequest sendRequest = {};
  sendRequest.id = 7;
  sendComm.sendReqs[7][0] = &sendRequest;
  ibv_wc completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.wr_id = ibReqWrIdEncode(7);
  ncclIbRequest* result = nullptr;
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&sendComm.base, &completion, 0, &result), ncclSuccess);
  EXPECT_EQ(result, &sendRequest);
  completion.wr_id = ibReqWrIdEncode(8);
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&sendComm.base, &completion, 0, &result), ncclSuccess);
  EXPECT_EQ(result, nullptr);

  // A receive immediate value must resolve the matching full ID under the default ID scheme.
  ncclIbRecvComm recvComm = {};
  ncclIbRequest recvRequest = {};
  recvRequest.id = 11;
  recvComm.recvReqs[11] = &recvRequest;
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
  completion.imm_data = htobe32(11);
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&recvComm.base, &completion, 0, &result), ncclSuccess);
  EXPECT_EQ(result, &recvRequest);
  recvRequest.id = 12;
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&recvComm.base, &completion, 0, &result), ncclSuccess);
  EXPECT_EQ(result, nullptr);

  // BY_INDEX must ignore immediate-data IDs and retrieve the receive request from the encoded
  // WR-ID.
  testReceiverMatchingScheme = BY_INDEX;
  recvRequest.id = 11;
  completion.wr_id = ibReqWrIdEncode(11);
  completion.imm_data = htobe32(77);
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&recvComm.base, &completion, 0, &result), ncclSuccess);
  EXPECT_EQ(result, &recvRequest);

  // A receiver completion with no supported lookup encoding must return no request without
  // mutating the table.
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_SEND;
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&recvComm.base, &completion, 0, &result), ncclSuccess);
  EXPECT_EQ(result, nullptr);

  // A flush completion must map its bounded offset and reject the first out-of-range offset.
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RDMA_READ;
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + 4;
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&recvComm.base, &completion, 0, &result), ncclSuccess);
  EXPECT_EQ(result, &recvComm.base.reqs[4]);
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + NET_IB_MAX_REQUESTS;
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&recvComm.base, &completion, 0, &result), ncclInternalError);

  // Restore the default lookup policy so test order cannot affect later completion cases.
  testReceiverMatchingScheme = BY_ID;
}

/**
 * @brief Verifies ordinary and resiliency-adjusted request completion decisions.
 * @note Side effects: Mutates only process-local fake counters and local request fixtures.
 */
TEST(P2pRequestTest, CompletionDecision) {
  ResetFakes();
  ncclIbSendComm comm = MakeSendComm();
  ncclIbRequest request = {};
  request.base = &comm.base;

  // Four zero counters are complete without consulting resiliency.
  EXPECT_TRUE(ncclIbRequestIsComplete(&request));
  EXPECT_EQ(g_fake.resiliencyCompletionCalls, 0);

  // Positive outstanding work remains incomplete even when resiliency exists.
  ncclIbResiliency resiliency = {};
  comm.base.resiliency = &resiliency;
  request.events[0] = 1;
  EXPECT_FALSE(ncclIbRequestIsComplete(&request));
  EXPECT_EQ(g_fake.resiliencyCompletionCalls, 0);

  // A negative failover counter must delegate the final decision to resiliency.
  request.events[0] = -1;
  g_fake.resiliencyComplete = false;
  EXPECT_FALSE(ncclIbRequestIsComplete(&request));
  g_fake.resiliencyComplete = true;
  EXPECT_TRUE(ncclIbRequestIsComplete(&request));
  EXPECT_EQ(g_fake.resiliencyCompletionCalls, 2);
}

/**
 * @brief Verifies send and receive completion reporting and slot release.
 * @note Side effects: Mutates only local requests, output sizes, and communicator slot bookkeeping.
 */
TEST(P2pRequestTest, CompletionReporting) {
  // Completing the last send must report its size, clear the group, and free the request.
  ncclIbSendComm sendComm = MakeSendComm();
  ncclIbRequest& sendRequest = sendComm.base.reqs[3];
  sendRequest.base = &sendComm.base;
  sendRequest.type = NCCL_NET_IB_REQ_SEND;
  sendRequest.id = 3;
  sendRequest.send.size = 123;
  sendComm.sendReqs[3][0] = &sendRequest;
  sendComm.sendReqsCnt[3] = 1;
  int done = 0;
  int sizes[2] = {-1, -1};
  EXPECT_EQ(ncclIbRequestComplete(&sendRequest, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 123);
  EXPECT_EQ(sendComm.sendReqsCnt[3], 0);
  EXPECT_EQ(sendComm.sendReqs[3][0], nullptr);
  EXPECT_EQ(sendRequest.type, NCCL_NET_IB_REQ_UNUSED);

  // A resilient send must notify resiliency exactly once before the request is released.
  ResetFakes();
  ncclIbResiliency resiliency = {};
  sendComm.base.resiliency = &resiliency;
  sendRequest.base = &sendComm.base;
  sendRequest.type = NCCL_NET_IB_REQ_SEND;
  sendRequest.id = 3;
  sendComm.sendReqs[3][0] = &sendRequest;
  sendComm.sendReqsCnt[3] = 1;
  EXPECT_EQ(ncclIbRequestComplete(&sendRequest, &done, nullptr), ncclSuccess);
  EXPECT_EQ(g_fake.resiliencySendCompleteCalls, 1);
  EXPECT_EQ(sendRequest.type, NCCL_NET_IB_REQ_UNUSED);

  // A multi-receive must report each completion-record size and then free the request.
  ncclIbRecvComm recvComm = {};
  ncclIbRequestCompletionRecord record = {};
  record.sizes[0] = 40;
  record.sizes[1] = 50;
  ncclIbRequest recvRequest = {};
  recvRequest.base = &recvComm.base;
  recvRequest.type = NCCL_NET_IB_REQ_RECV;
  recvRequest.nreqs = 2;
  recvRequest.recv.cmplsRecords = &record;
  done = 0;
  EXPECT_EQ(ncclIbRequestComplete(&recvRequest, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 40);
  EXPECT_EQ(sizes[1], 50);
  EXPECT_EQ(recvRequest.type, NCCL_NET_IB_REQ_UNUSED);

  // A single receive without a published size must report its locally aggregated byte count.
  record = {};
  recvRequest = {};
  recvRequest.base = &recvComm.base;
  recvRequest.type = NCCL_NET_IB_REQ_RECV;
  recvRequest.nreqs = 1;
  recvRequest.recv.cmplsRecords = &record;
  recvRequest.recv.aggSize = 61;
  EXPECT_EQ(ncclIbRequestComplete(&recvRequest, &done, sizes), ncclSuccess);
  EXPECT_EQ(sizes[0], 61);
}

/**
 * @brief Verifies baseline send planning, FRM fallback, resilient dispatch, and provider errors.
 * @note Side effects: Mutates local fixtures and temporarily changes process-local fake-provider
 * and AR controls.
 */
TEST(P2pTransferTest, SendPlanningAndDispatch) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp verbsQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &verbsQp, &cq, 41);
  ncclIbSendComm comm = {};
  InitSendPath(&comm, &verbsQp, &cq);
  ncclIbRequest& request = comm.base.reqs[0];
  request = {};
  request.base = &comm.base;
  request.id = 0;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.nreqs = 1;
  request.send.size = 300;
  request.send.data = reinterpret_cast<void*>(0x1000);
  request.send.lkeys[0] = 17;
  request.send.wrIdToken = ibReqWrIdEncode(request.id);
  comm.sendReqs[0][0] = &request;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].addr = 0x2000;
  comm.ctsFifo[0][0].rkeys[0] = 23;

  // Baseline posting must publish one signaled immediate write for the exact clipped payload.
  ASSERT_EQ(ibMultiSendBaseline<false>(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 41u);
  EXPECT_EQ(g_fake.postedSends[0].wrId, ibReqWrIdEncode(0));
  EXPECT_EQ(g_fake.postedSends[0].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_NE(g_fake.postedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_fake.postedSends[0].remoteAddress, 0x2000u);
  EXPECT_EQ(g_fake.postedSends[0].remoteKey, 23u);
  EXPECT_EQ(g_fake.postedSends[0].numSge, 1);
  EXPECT_EQ(g_fake.postedSends[0].sge.addr, 0x1000u);
  EXPECT_EQ(g_fake.postedSends[0].sge.length, 300u);
  EXPECT_EQ(g_fake.postedSends[0].sge.lkey, 17u);
  EXPECT_TRUE(request.send.pendingData[0]);
  EXPECT_EQ(request.send.lastAttemptQpIndex[0], 0);

  // Adaptive Routing at the strict threshold must retain one signaled immediate data WR.
  ResetFakes();
  comm.ar = 1;
  testArThreshold = 128;
  request.send.size = 128;
  ASSERT_EQ(ibMultiSendBaseline<false>(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 41u);
  EXPECT_EQ(g_fake.postedSends[0].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_NE(g_fake.postedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_fake.postedSends[0].sge.length, 128u);

  // One byte above the AR threshold must append a separate signaled immediate WR.
  ResetFakes();
  comm.ar = 1;
  testArThreshold = 128;
  request.send.size = 129;
  ASSERT_EQ(ibMultiSendBaseline<false>(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 2);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 41u);
  EXPECT_EQ(g_fake.postedSends[1].qpNumber, 41u);
  EXPECT_EQ(g_fake.postedSends[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(g_fake.postedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_fake.postedSends[0].sge.length, 129u);
  EXPECT_EQ(g_fake.postedSends[1].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_NE(g_fake.postedSends[1].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_fake.postedSends[1].numSge, 0);
  request.send.size = 300;

  // Two requests must form two data WRs followed by one remote-size immediate WR.
  ResetFakes();
  comm.ar = 0;
  comm.remCmplsRecords.addr = 0x9000;
  comm.remCmplsRecords.rkeys[0] = 29;
  comm.devs[0].sge.lkey = 30;
  ncclIbRequest secondRequest = {};
  secondRequest.base = &comm.base;
  secondRequest.id = 0;
  secondRequest.type = NCCL_NET_IB_REQ_SEND;
  secondRequest.nreqs = 2;
  secondRequest.send.size = 180;
  secondRequest.send.data = reinterpret_cast<void*>(0x3000);
  secondRequest.send.lkeys[0] = 18;
  secondRequest.send.wrIdToken = request.send.wrIdToken;
  request.nreqs = 2;
  comm.sendReqs[0][1] = &secondRequest;
  comm.ctsFifo[0][0].nreqs = 2;
  comm.ctsFifo[0][1].addr = 0x4000;
  comm.ctsFifo[0][1].rkeys[0] = 24;
  ASSERT_EQ(ibMultiSendBaseline<false>(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 3);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 41u);
  EXPECT_EQ(g_fake.postedSends[1].qpNumber, 41u);
  EXPECT_EQ(g_fake.postedSends[2].qpNumber, 41u);
  EXPECT_EQ(g_fake.postedSends[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(g_fake.postedSends[0].sge.length, 300u);
  EXPECT_EQ(g_fake.postedSends[1].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(g_fake.postedSends[1].sge.length, 180u);
  EXPECT_EQ(g_fake.postedSends[2].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_EQ(g_fake.postedSends[2].remoteAddress, 0x9000u);
  EXPECT_EQ(g_fake.postedSends[2].remoteKey, 29u);
  EXPECT_EQ(g_fake.postedSends[2].sge.length, 2u * sizeof(int));

  // A zero-byte baseline request must keep its immediate completion WR but remove its payload SGE.
  ResetFakes();
  request.nreqs = 1;
  request.send.size = 0;
  comm.sendReqs[0][1] = nullptr;
  comm.ctsFifo[0][0].nreqs = 1;
  ASSERT_EQ(ibMultiSendBaseline<false>(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 41u);
  EXPECT_EQ(g_fake.postedSends[0].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_NE(g_fake.postedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_fake.postedSends[0].numSge, 0);
  request.send.size = 300;

  // Baseline resiliency must skip a delivered logical QP and preserve the next chunk's offsets.
  ResetFakes();
  ibv_qp secondVerbsQp = {};
  secondVerbsQp.context = &context;
  secondVerbsQp.qp_num = 42;
  comm.base.nqps = 2;
  comm.base.nDataQps = 2;
  comm.base.splitDataOnQps = 1;
  comm.base.qps[1].qp = &secondVerbsQp;
  comm.base.qps[1].devIndex = 0;
  comm.base.qps[1].remDevIdx = 0;
  comm.base.activeQps[1] = &comm.base.qps[1];
  ncclIbResiliency skipResiliency = {};
  comm.base.resiliency = &skipResiliency;
  request.nreqs = 1;
  request.send.pendingData[0] = false;
  request.send.pendingData[1] = true;
  comm.sendReqs[0][1] = nullptr;
  comm.ctsFifo[0][0].nreqs = 1;
  const uint32_t deliveredChunk =
      DIVUP(DIVUP(request.send.size, comm.base.nqps), IB_WRITE_CHUNK_ALIGNMENT) * IB_WRITE_CHUNK_ALIGNMENT;
  const uint32_t expectedTail = request.send.size - deliveredChunk;
  ASSERT_EQ(ibMultiSendBaseline<false>(&comm, 0), ncclSuccess)
      << "A replay with one delivered lane and one pending lane must remain postable";
  ASSERT_EQ(g_fake.postSendCalls, 1)
      << "The replay must submit exactly one post operation for the pending lane";
  ASSERT_EQ(g_fake.postedSendCount, 1)
      << "The replay must not repost data already delivered on the skipped lane";
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 42u)
      << "The remaining tail must be posted on the pending queue pair";
  EXPECT_EQ(g_fake.postedSends[0].sge.addr, 0x1000u + deliveredChunk)
      << "The local address must advance past the delivered logical chunk";
  EXPECT_EQ(g_fake.postedSends[0].remoteAddress, 0x2000u + deliveredChunk)
      << "The remote address must advance past the delivered logical chunk";
  EXPECT_EQ(g_fake.postedSends[0].sge.length, expectedTail)
      << "The replay must post only the undelivered tail, not a second full chunk";
  EXPECT_EQ(g_fake.postedSends[0].sge.addr + g_fake.postedSends[0].sge.length,
            0x1000u + request.send.size)
      << "The replay SGE must end exactly at the request boundary";
  EXPECT_EQ(request.send.lastAttemptQpIndex[0], -1)
      << "A skipped delivered lane must not acquire a posting-attempt index";
  EXPECT_EQ(request.send.lastAttemptQpIndex[1], 1)
      << "The posted pending lane must record its queue-pair index";
  EXPECT_FALSE(request.send.pendingData[0])
      << "Replaying another lane must not mark delivered data pending again";
  EXPECT_TRUE(request.send.pendingData[1])
      << "The lane used for the current replay must remain identified as pending";
  request.send.size = 300;

  // CIM with no scratch must dispatch through FRM and safely fall back to the baseline planner.
  ResetFakes();
  CIMState cim = {};
  comm.cim = &cim;
  comm.base.nqps = 1;
  comm.base.nDataQps = 1;
  comm.base.splitDataOnQps = 0;
  comm.base.resiliency = nullptr;
  comm.sendReqs[0][1] = nullptr;
  comm.ctsFifo[0][0].nreqs = 1;
  EXPECT_EQ(ncclIbMultiSend(&comm, 0), ncclSuccess);
  EXPECT_TRUE(comm.frmGateOpen);
  EXPECT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 41u);

  // Healthy resiliency must select the combined wrapper while preserving the same posting result.
  ResetFakes();
  ncclIbResiliency resiliency = {};
  resiliency.frmAllQpsHealthy = true;
  comm.base.resiliency = &resiliency;
  request.send.pendingData[0] = true;
  FrmScratchStorage resilientStorage = {};
  AttachFrmScratch(&comm, &resilientStorage);
  EXPECT_EQ(ncclIbMultiSend(&comm, 0), ncclSuccess);
  EXPECT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 41u);

  // An oversized descriptor must be rejected before any provider call.
  ResetFakes();
  comm.base.resiliency = nullptr;
  comm.ctsFifo[0][0].nreqs = NCCL_NET_IB_MAX_RECVS + 1;
  EXPECT_EQ(ibMultiSendBaseline<false>(&comm, 0), ncclInternalError);
  EXPECT_EQ(g_fake.postSendCalls, 0);

  // A provider post failure must propagate as ncclSystemError.
  comm.ctsFifo[0][0].nreqs = 1;
  g_fake.postSendError = 5;
  EXPECT_EQ(ibMultiSendBaseline<false>(&comm, 0), ncclSystemError);
  EXPECT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 41u);
}

/**
 * @brief Verifies resilient FRM rejects missing or undersized general planning scratch.
 * @note Side effects: Records one warning per rejected state without calling the provider.
 */
TEST(P2pTransferTest, ResilientFrmRequiresSufficientGeneralScratchStorage) {
  ResetFakes();
  ncclIbSendComm comm = {};
  comm.base.nqps = 1;
  comm.ctsFifo[0][0].nreqs = 1;

  EXPECT_EQ(ibFrmMultiSendCore<false>(&comm, 0, true), ncclInternalError);
  MultiSendBuffers buffers = {};
  AttachMultiSendBuffers(&comm, &buffers, 1, true);
  comm.base.nqps = 2;
  EXPECT_EQ(ibFrmMultiSendCore<false>(&comm, 0, true), ncclInternalError);

  EXPECT_EQ(g_fake.warningCalls, 2);
  EXPECT_NE(g_fake.lastWarning.find("has no replay-plan storage"), std::string::npos);
  EXPECT_EQ(g_fake.postSendCalls, 0);
}

/**
 * @brief Verifies resilient FRM requires both replay-plan arrays in otherwise valid scratch.
 * @note Side effects: Records one warning for each missing replay-plan component.
 */
TEST(P2pTransferTest, ResilientFrmRequiresCompleteReplayPlanStorage) {
  ResetFakes();
  ncclIbSendComm comm = {};
  comm.base.nqps = 1;
  comm.ctsFifo[0][0].nreqs = 1;
  MultiSendBuffers buffers = {};
  AttachMultiSendBuffers(&comm, &buffers, 1, true);

  comm.multiSendScratch.replayPlanBytes = nullptr;
  EXPECT_EQ(ibFrmMultiSendCore<false>(&comm, 0, true), ncclInternalError);
  comm.multiSendScratch.replayPlanBytes = buffers.replayPlans;
  comm.multiSendScratch.replayPlanFrozen = nullptr;
  EXPECT_EQ(ibFrmMultiSendCore<false>(&comm, 0, true), ncclInternalError);

  EXPECT_EQ(g_fake.warningCalls, 2);
  EXPECT_NE(g_fake.lastWarning.find("Invalid resilient replay-plan state"), std::string::npos);
  EXPECT_EQ(g_fake.postSendCalls, 0);
}

/**
 * @brief Verifies real two-QP FRM planning and resilient replay skipping with preallocated scratch.
 * @note Side effects: Mutates local fixtures and temporarily changes process-local fake-provider
 * and AR controls.
 */
TEST(P2pTransferTest, FrmPlanner) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp verbsQp0 = {};
  ibv_qp verbsQp1 = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &verbsQp0, &cq, 61);
  verbsQp1.context = &context;
  verbsQp1.qp_num = 62;
  ncclIbSendComm comm = {};
  InitSendPath(&comm, &verbsQp0, &cq);
  comm.base.nqps = 2;
  comm.base.nDataQps = 2;
  comm.base.splitDataOnQps = 1;
  comm.base.qps[1].qp = &verbsQp1;
  comm.base.qps[1].devIndex = 0;
  comm.base.qps[1].remDevIdx = 0;
  comm.base.activeQps[1] = &comm.base.qps[1];
  CIMState cim = {};
  cim.nqps = 2;
  cim.spread_eps_us = 1;
  comm.cim = &cim;
  comm.base.frmFields.send.qp_lat[0] = 10;
  comm.base.frmFields.send.qp_lat[1] = 30;
  FrmScratchStorage storage = {};
  AttachFrmScratch(&comm, &storage);
  ncclIbRequest& request = comm.base.reqs[0];
  request = {};
  request.base = &comm.base;
  request.id = 0;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.nreqs = 1;
  request.send.size = 1000;
  request.send.data = reinterpret_cast<void*>(0x1000);
  request.send.lkeys[0] = 17;
  request.send.wrIdToken = ibReqWrIdEncode(0);
  comm.sendReqs[0][0] = &request;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].addr = 0x2000;
  comm.ctsFifo[0][0].rkeys[0] = 23;

  // Different measured latencies must produce two contiguous posts whose chunks preserve the
  // exact total.
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, false), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 2);
  ASSERT_EQ(g_fake.postedSendCount, 2);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 61u);
  EXPECT_EQ(g_fake.postedSends[1].qpNumber, 62u);
  EXPECT_GT(g_fake.postedSends[0].sge.length, g_fake.postedSends[1].sge.length);
  EXPECT_EQ(g_fake.postedSends[0].sge.length + g_fake.postedSends[1].sge.length, 1000u);
  EXPECT_EQ(g_fake.postedSends[0].sge.addr, 0x1000u);
  EXPECT_EQ(g_fake.postedSends[1].sge.addr, 0x1000u + g_fake.postedSends[0].sge.length);
  EXPECT_EQ(g_fake.postedSends[0].remoteAddress, 0x2000u);
  EXPECT_EQ(g_fake.postedSends[1].remoteAddress, 0x2000u + g_fake.postedSends[0].sge.length);
  EXPECT_EQ(comm.base.qps[0].sig_head, 1u);
  EXPECT_EQ(comm.base.qps[1].sig_head, 1u);
  EXPECT_EQ(comm.base.qps[0].sig_bytes[0], g_fake.postedSends[0].sge.length);
  EXPECT_EQ(comm.base.qps[1].sig_bytes[0], g_fake.postedSends[1].sge.length);
  EXPECT_NEAR(storage.weights[0] + storage.weights[1], 1.0f, 1e-6f);

  // Split-zero full FRM must run the active-window planner and still preserve the exact payload
  // total.
  ResetFakes();
  comm.base.splitDataOnQps = 0;
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, false), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 2);
  ASSERT_EQ(g_fake.postedSendCount, 2);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 61u);
  EXPECT_EQ(g_fake.postedSends[1].qpNumber, 62u);
  EXPECT_EQ(g_fake.postedSends[0].sge.length + g_fake.postedSends[1].sge.length, 1000u);
  EXPECT_NEAR(storage.weights[0] + storage.weights[1], 1.0f, 1e-6f);

  // FRM Adaptive Routing must post one unsignaled data WR and one signaled immediate WR per QP.
  ResetFakes();
  comm.base.splitDataOnQps = 1;
  comm.ar = 1;
  testArThreshold = 128;
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, false), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 2);
  ASSERT_EQ(g_fake.postedSendCount, 4);
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(g_fake.postedSends[2 * i].qpNumber, 61u + i);
    EXPECT_EQ(g_fake.postedSends[2 * i + 1].qpNumber, 61u + i);
    EXPECT_EQ(g_fake.postedSends[2 * i].opcode, IBV_WR_RDMA_WRITE);
    EXPECT_EQ(g_fake.postedSends[2 * i].sendFlags & IBV_SEND_SIGNALED, 0);
    EXPECT_GT(g_fake.postedSends[2 * i].sge.length, 0u);
    EXPECT_EQ(g_fake.postedSends[2 * i + 1].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
    EXPECT_NE(g_fake.postedSends[2 * i + 1].sendFlags & IBV_SEND_SIGNALED, 0);
    EXPECT_EQ(g_fake.postedSends[2 * i + 1].numSge, 0);
  }

  // A zero-byte FRM fast-path request must post only signaled immediate WRs with no payload SGE.
  ResetFakes();
  comm.ar = 0;
  request.send.size = 0;
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, false), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 2);
  ASSERT_EQ(g_fake.postedSendCount, 2);
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(g_fake.postedSends[i].qpNumber, 61u + i);
    EXPECT_EQ(g_fake.postedSends[i].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
    EXPECT_NE(g_fake.postedSends[i].sendFlags & IBV_SEND_SIGNALED, 0);
    EXPECT_EQ(g_fake.postedSends[i].numSge, 0);
  }

  // Resilient FRM Adaptive Routing must build the same two-WR chain through the general planner.
  ResetFakes();
  ncclIbResiliency resiliency = {};
  comm.base.resiliency = &resiliency;
  comm.ar = 1;
  testArThreshold = 128;
  request.send.size = 1000;
  request.send.pendingData[0] = true;
  request.send.pendingData[1] = true;
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, true), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 2);
  ASSERT_EQ(g_fake.postedSendCount, 4);
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(g_fake.postedSends[2 * i].qpNumber, 61u + i);
    EXPECT_EQ(g_fake.postedSends[2 * i + 1].qpNumber, 61u + i);
    EXPECT_EQ(g_fake.postedSends[2 * i].opcode, IBV_WR_RDMA_WRITE);
    EXPECT_EQ(g_fake.postedSends[2 * i].sendFlags & IBV_SEND_SIGNALED, 0);
    EXPECT_EQ(g_fake.postedSends[2 * i + 1].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
    EXPECT_NE(g_fake.postedSends[2 * i + 1].sendFlags & IBV_SEND_SIGNALED, 0);
    EXPECT_EQ(g_fake.postedSends[2 * i + 1].numSge, 0);
  }

  // A zero-byte resilient request must retain one signaled immediate WR per pending logical QP.
  ResetFakes();
  comm.ar = 0;
  request.send.size = 0;
  request.send.pendingData[0] = true;
  request.send.pendingData[1] = true;
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, true), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 2);
  ASSERT_EQ(g_fake.postedSendCount, 2);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 61u);
  EXPECT_EQ(g_fake.postedSends[1].qpNumber, 62u);
  EXPECT_EQ(g_fake.postedSends[0].numSge, 0);
  EXPECT_EQ(g_fake.postedSends[1].numSge, 0);
  request.send.size = 1000;
  const uint32_t firstHeadBeforeReplay = comm.base.qps[0].sig_head;
  const uint32_t secondHeadBeforeReplay = comm.base.qps[1].sig_head;

  // Replay must skip an already-delivered first lane but still advance the second lane's local and
  // remote offsets.
  ResetFakes();
  comm.ar = 0;
  comm.base.resiliency = &resiliency;
  request.send.pendingData[0] = false;
  request.send.pendingData[1] = true;
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, true), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 62u);
  EXPECT_EQ(g_fake.postedSends[0].sge.addr, 0x1000u + storage.chunks[0][0]);
  EXPECT_EQ(g_fake.postedSends[0].remoteAddress, 0x2000u + storage.chunks[0][0]);
  EXPECT_EQ(g_fake.postedSends[0].sge.length, storage.chunks[0][1]);
  EXPECT_EQ(request.send.lastAttemptQpIndex[0], -1);
  EXPECT_EQ(request.send.lastAttemptQpIndex[1], 1);
  EXPECT_FALSE(request.send.pendingData[0]);
  EXPECT_TRUE(request.send.pendingData[1]);
  EXPECT_EQ(comm.base.qps[0].sig_head, firstHeadBeforeReplay);
  EXPECT_EQ(comm.base.qps[1].sig_head, secondHeadBeforeReplay + 1);
}

/**
 * @brief Verifies FRM multi-request chaining, zero-length suppression, and remote size publication.
 * @note Side effects: Mutates local send, scratch, CIM, and process-local fake-provider fixtures.
 */
TEST(P2pTransferTest, FrmMultiRequest) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp verbsQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &verbsQp, &cq, 63);
  ncclIbSendComm comm = {};
  InitSendPath(&comm, &verbsQp, &cq);
  comm.base.splitDataOnQps = 1;
  comm.remCmplsRecords.addr = 0x9000;
  comm.remCmplsRecords.rkeys[0] = 29;
  comm.devs[0].sge.lkey = 30;
  CIMState cim = {};
  cim.nqps = 1;
  comm.cim = &cim;
  FrmScratchStorage storage = {};
  AttachFrmScratch(&comm, &storage);
  ncclIbRequest& request0 = comm.base.reqs[0];
  ncclIbRequest& request1 = comm.base.reqs[1];
  request0 = {};
  request1 = {};
  request0.base = &comm.base;
  request1.base = &comm.base;
  request0.id = 0;
  request1.id = 0;
  request0.type = NCCL_NET_IB_REQ_SEND;
  request1.type = NCCL_NET_IB_REQ_SEND;
  request0.nreqs = 2;
  request1.nreqs = 2;
  request0.send.size = 100;
  request1.send.size = 0;
  request0.send.data = reinterpret_cast<void*>(0x1000);
  request1.send.data = reinterpret_cast<void*>(0x2000);
  request0.send.lkeys[0] = 11;
  request1.send.lkeys[0] = 12;
  request0.send.wrIdToken = ibReqWrIdEncode(0);
  request1.send.wrIdToken = ibReqWrIdEncode(0);
  comm.sendReqs[0][0] = &request0;
  comm.sendReqs[0][1] = &request1;
  comm.ctsFifo[0][0].nreqs = 2;
  comm.ctsFifo[0][0].addr = 0x3000;
  comm.ctsFifo[0][1].addr = 0x4000;
  comm.ctsFifo[0][0].rkeys[0] = 23;
  comm.ctsFifo[0][1].rkeys[0] = 24;

  // One live payload must be followed by one signaled immediate record; zero data needs no data WR.
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, false), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 2);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 63u);
  EXPECT_EQ(g_fake.postedSends[1].qpNumber, 63u);
  EXPECT_EQ(g_fake.postedSends[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(g_fake.postedSends[0].sge.length, 100u);
  EXPECT_EQ(g_fake.postedSends[0].remoteAddress, 0x3000u);
  EXPECT_EQ(g_fake.postedSends[1].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_NE(g_fake.postedSends[1].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_fake.postedSends[1].remoteAddress, 0x9000u);
  EXPECT_EQ(g_fake.postedSends[1].remoteKey, 29u);
  EXPECT_EQ(g_fake.postedSends[1].sge.length, 2u * sizeof(int));
  EXPECT_EQ(comm.remCmplsRecords.elems[0][0], 100);
  EXPECT_EQ(comm.remCmplsRecords.elems[0][1], 0);
  EXPECT_EQ(comm.base.qps[0].sig_bytes[0], 100u);

  // Multi-request replay must issue no chain when its only logical QP was already delivered.
  ResetFakes();
  ncclIbResiliency resiliency = {};
  comm.base.resiliency = &resiliency;
  request0.send.pendingData[0] = false;
  request1.send.pendingData[0] = false;
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, true), ncclSuccess);
  EXPECT_EQ(g_fake.postSendCalls, 0);
  EXPECT_EQ(g_fake.postedSendCount, 0);
  EXPECT_EQ(request0.send.lastAttemptQpIndex[0], -1);
  EXPECT_EQ(request1.send.lastAttemptQpIndex[0], -1);
  EXPECT_FALSE(request0.send.pendingData[0]);
  EXPECT_FALSE(request1.send.pendingData[0]);

  // Split-zero FRM must chain every live request on each QP and preserve both request totals.
  ResetFakes();
  ibv_qp secondVerbsQp = {};
  secondVerbsQp.context = &context;
  secondVerbsQp.qp_num = 64;
  comm.base.resiliency = nullptr;
  comm.base.nqps = 2;
  comm.base.nDataQps = 2;
  comm.base.splitDataOnQps = 0;
  comm.base.qps[1].qp = &secondVerbsQp;
  comm.base.qps[1].devIndex = 0;
  comm.base.qps[1].remDevIdx = 0;
  comm.base.activeQps[1] = &comm.base.qps[1];
  cim.nqps = 2;
  cim.spread_eps_us = 1;
  comm.base.frmFields.send.qp_lat[0] = 10;
  comm.base.frmFields.send.qp_lat[1] = 30;
  request0.send.size = 1000;
  request1.send.size = 800;
  AttachFrmScratch(&comm, &storage);
  ASSERT_EQ(ibFrmMultiSendCore<false>(&comm, 0, false), ncclSuccess);
  ASSERT_EQ(g_fake.postSendCalls, 2);
  ASSERT_EQ(g_fake.postedSendCount, 6);
  for (int qpIndex = 0; qpIndex < 2; qpIndex++) {
    EXPECT_EQ(g_fake.postedSends[3 * qpIndex].qpNumber, 63u + qpIndex);
    EXPECT_EQ(g_fake.postedSends[3 * qpIndex + 1].qpNumber, 63u + qpIndex);
    EXPECT_EQ(g_fake.postedSends[3 * qpIndex + 2].qpNumber, 63u + qpIndex);
    EXPECT_EQ(g_fake.postedSends[3 * qpIndex].opcode, IBV_WR_RDMA_WRITE);
    EXPECT_EQ(g_fake.postedSends[3 * qpIndex + 1].opcode, IBV_WR_RDMA_WRITE);
    EXPECT_EQ(g_fake.postedSends[3 * qpIndex + 2].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
    EXPECT_NE(g_fake.postedSends[3 * qpIndex + 2].sendFlags & IBV_SEND_SIGNALED, 0);
  }
  EXPECT_EQ(g_fake.postedSends[0].sge.length + g_fake.postedSends[3].sge.length, 1000u);
  EXPECT_EQ(g_fake.postedSends[1].sge.length + g_fake.postedSends[4].sge.length, 800u);
}

/**
 * @brief Verifies sender readiness, CTS waiting, matching, clipping, and request publication.
 * @note Side effects: Mutates local send fixtures and process-local fake provider records.
 */
TEST(P2pTransferTest, SendEntryPoint) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp verbsQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &verbsQp, &cq, 42);
  ncclIbSendComm comm = {};
  InitSendPath(&comm, &verbsQp, &cq);
  ibv_mr mr = {};
  mr.lkey = 31;
  ncclIbMrHandle memory = {};
  memory.mrs[0] = &mr;
  void* request = reinterpret_cast<void*>(0x1);

  // No CTS record means success-without-progress and a cleared request output.
  EXPECT_EQ(ncclIbIsend(&comm, reinterpret_cast<void*>(0x1000), 512, 9, &memory, nullptr, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(g_fake.postSendCalls, 0);

  // A nonmatching tag must also leave the FIFO and request pool untouched.
  comm.ctsFifo[0][0].idx = 1;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].tag = 8;
  request = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(ncclIbIsend(&comm, reinterpret_cast<void*>(0x1000), 512, 9, &memory, nullptr, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(comm.base.fifoHead, 0u);

  // A matching but structurally invalid CTS record must be rejected before allocating a request.
  comm.ctsFifo[0][0].tag = 9;
  comm.ctsFifo[0][0].size = 256;
  comm.ctsFifo[0][0].addr = 0;
  comm.ctsFifo[0][0].rkeys[0] = 32;
  EXPECT_EQ(ncclIbIsend(&comm, reinterpret_cast<void*>(0x1000), 512, 9, &memory, nullptr, &request),
            ncclInternalError);
  EXPECT_EQ(comm.base.fifoHead, 0u);
  EXPECT_EQ(comm.sendReqs[0][0], nullptr);

  // A matching CTS record must clip size to capacity, post the send, and advance the FIFO.
  comm.ctsFifo[0][0].addr = 0x2000;
  EXPECT_EQ(ncclIbIsend(&comm, reinterpret_cast<void*>(0x1000), 512, 9, &memory, nullptr, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  auto* sendRequest = static_cast<ncclIbRequest*>(request);
  EXPECT_EQ(sendRequest->type, NCCL_NET_IB_REQ_SEND);
  EXPECT_EQ(sendRequest->send.size, 256);
  EXPECT_EQ(sendRequest->send.lkeys[0], 31u);
  EXPECT_EQ(sendRequest->events[0], 1);
  EXPECT_EQ(comm.remCmplsRecords.elems[0][0], 256);
  EXPECT_EQ(comm.base.fifoHead, 1u);
  EXPECT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 42u);

  // A resilient send must clear stale replay arrays before marking only the active logical QP
  // pending.
  ResetFakes();
  ncclIbSendComm resilientComm = {};
  InitSendPath(&resilientComm, &verbsQp, &cq);
  ncclIbResiliency resiliency = {};
  resilientComm.base.resiliency = &resiliency;
  resilientComm.ctsFifo[0][0].idx = 1;
  resilientComm.ctsFifo[0][0].nreqs = 2;
  resilientComm.ctsFifo[0][0].tag = 9;
  resilientComm.ctsFifo[0][0].size = 256;
  resilientComm.ctsFifo[0][0].addr = 0x2000;
  resilientComm.ctsFifo[0][0].rkeys[0] = 32;
  resilientComm.ctsFifo[0][1].idx = 1;
  resilientComm.ctsFifo[0][1].tag = 10;
  std::memset(resilientComm.base.reqs[0].send.pendingData, 1,
              sizeof(resilientComm.base.reqs[0].send.pendingData));
  std::memset(resilientComm.base.reqs[0].send.lastAttemptQpIndex, 0,
              sizeof(resilientComm.base.reqs[0].send.lastAttemptQpIndex));
  std::memset(resilientComm.base.reqs[0].send.lastAttemptOrder, 0xff,
              sizeof(resilientComm.base.reqs[0].send.lastAttemptOrder));
  std::memset(resilientComm.base.reqs[0].send.ackedCompletions, 0xff,
              sizeof(resilientComm.base.reqs[0].send.ackedCompletions));
  ASSERT_EQ(ncclIbIsend(
                &resilientComm, reinterpret_cast<void*>(0x1000), 256, 9, &memory, nullptr, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  auto* resilientRequest = static_cast<ncclIbRequest*>(request);
  EXPECT_TRUE(resilientRequest->send.pendingData[0]);
  EXPECT_EQ(g_fake.postSendCalls, 0);
  EXPECT_EQ(resilientComm.sendReqsCnt[0], 1);
  EXPECT_EQ(resilientComm.base.fifoHead, 0u);
  for (int qpIndex = 0; qpIndex < NCCL_IB_MAX_QPS; qpIndex++) {
    if (qpIndex > 0) {
      EXPECT_FALSE(resilientRequest->send.pendingData[qpIndex]);
    }
    EXPECT_EQ(resilientRequest->send.lastAttemptQpIndex[qpIndex], -1);
    EXPECT_EQ(resilientRequest->send.lastAttemptOrder[qpIndex], 0u);
    EXPECT_EQ(resilientRequest->send.ackedCompletions[qpIndex], 0u);
  }

  // An unready communicator must return an error and clear the output pointer.
  comm.base.ready = 0;
  request = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(ncclIbIsend(&comm, nullptr, 0, 9, &memory, nullptr, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
}

/**
 * @brief Verifies CTS, receive, and flush entry points through fake provider callbacks.
 * @note Side effects: Mutates local receive fixtures and process-local fake provider records.
 */
TEST(P2pTransferTest, ReceiveAndFlushEntryPoints) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp verbsQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &verbsQp, &cq, 43);
  ncclIbRecvComm comm = {};
  InitRecvPath(&comm, &verbsQp, &cq);
  comm.remCtsFifo.addr = 0x4000;
  comm.base.remDevs[0].rkey = 44;
  comm.devs[0].sge.lkey = 45;
  ibv_mr mr = {};
  mr.rkey = 46;
  ncclIbMrHandle memory = {};
  memory.mrs[0] = &mr;
  void* data[1] = {reinterpret_cast<void*>(0x5000)};
  size_t receiveSizes[1] = {64};
  int tags[1] = {7};
  void* memories[1] = {&memory};
  void* request = nullptr;

  // An unready communicator must clear the request output, and an oversized receive must stop
  // before posting.
  comm.base.ready = 0;
  request = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(ncclIbIrecv(&comm, 1, data, receiveSizes, tags, memories, nullptr, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
  comm.base.ready = 1;
  request = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(ncclIbIrecv(
                &comm, NCCL_NET_IB_MAX_RECVS + 1, data, receiveSizes, tags, memories, nullptr, &request),
            ncclInternalError);
  EXPECT_EQ(request, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(g_fake.postRecvCalls, 0);
  EXPECT_EQ(g_fake.postSendCalls, 0);

  // Receive setup must post one receive WR and one CTS write with the advertised buffer metadata.
  ASSERT_EQ(ncclIbIrecv(&comm, 1, data, receiveSizes, tags, memories, nullptr, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  auto* recvRequest = static_cast<ncclIbRequest*>(request);
  EXPECT_EQ(recvRequest->type, NCCL_NET_IB_REQ_RECV);
  EXPECT_EQ(recvRequest->events[0], 1);
  EXPECT_EQ(g_fake.postRecvCalls, 1);
  EXPECT_EQ(g_fake.lastRecvWrId, ibReqWrIdEncode(0));
  EXPECT_EQ(g_fake.postSendCalls, 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 43u);
  EXPECT_EQ(g_fake.postedSends[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_NE(g_fake.postedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].addr, 0x5000u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].size, 64u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].tag, 7u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].rkeys[0], 46u);
  EXPECT_EQ(comm.base.fifoHead, 1u);

  // Preposted mode must create and advertise the next request without posting another receive WR.
  comm.prepostReceiveWorkRequests = true;
  ASSERT_EQ(ncclIbIrecv(&comm, 1, data, receiveSizes, tags, memories, nullptr, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(g_fake.postRecvCalls, 1);
  EXPECT_EQ(g_fake.postSendCalls, 2);
  ASSERT_EQ(g_fake.postedSendCount, 2);
  EXPECT_EQ(g_fake.postedSends[1].qpNumber, 43u);
  EXPECT_EQ(comm.base.fifoHead, 2u);

  // Resiliency must make every configured receive device available for later failover completion
  // handling.
  ResetFakes();
  ncclIbRecvComm resilientComm = {};
  InitRecvPath(&resilientComm, &verbsQp, &cq);
  ncclIbResiliency resiliency = {};
  resilientComm.base.vProps.ndevs = 2;
  resilientComm.base.resiliency = &resiliency;
  resilientComm.remCtsFifo.addr = 0x4000;
  resilientComm.base.remDevs[0].rkey = 44;
  resilientComm.devs[0].sge.lkey = 45;
  resilientComm.devs[1].base.cq = &cq;
  memory.mrs[1] = &mr;
  request = nullptr;
  ASSERT_EQ(ncclIbIrecv(
                &resilientComm, 1, data, receiveSizes, tags, memories, nullptr, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  auto* resilientRequest = static_cast<ncclIbRequest*>(request);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 43u);
  EXPECT_EQ(resilientRequest->devBases[0], &resilientComm.devs[0].base);
  EXPECT_EQ(resilientRequest->devBases[1], &resilientComm.devs[1].base);
  EXPECT_EQ(resilientComm.remCtsFifo.elems[0][0].rkeys[1], 46u);
  for (int devIndex = 2; devIndex < NCCL_IB_MAX_DEVS_PER_NIC; devIndex++) {
    EXPECT_EQ(resilientRequest->devBases[devIndex], nullptr);
  }

  // A non-device-matching slot must preserve inline flags without forcing a signaled CTS.
  ResetFakes();
  ncclIbRequest ctsRequest = {};
  ctsRequest.base = &comm.base;
  ctsRequest.id = 1;
  ctsRequest.nreqs = 1;
  EXPECT_EQ(ncclIbPostFifo(&comm, &ctsRequest, 1), ncclSuccess);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 43u);
  EXPECT_EQ(g_fake.postedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_fake.postedSends[0].wrId, 0u);

  // A disabled or empty flush must not publish a request or call the provider.
  ResetFakes();
  int flushSizes[1] = {64};
  request = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(ncclIbIflush(&comm, 1, data, flushSizes, memories, &request), ncclSuccess);
  EXPECT_EQ(request, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(g_fake.postSendCalls, 0);
  comm.flushEnabled = 1;
  flushSizes[0] = 0;
  EXPECT_EQ(ncclIbIflush(&comm, 1, data, flushSizes, memories, &request), ncclSuccess);
  EXPECT_EQ(g_fake.postSendCalls, 0);

  // An enabled nonempty flush must post one signaled RDMA read and track one completion event.
  flushSizes[0] = 64;
  EXPECT_EQ(ncclIbIflush(&comm, 1, data, flushSizes, memories, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  auto* flushRequest = static_cast<ncclIbRequest*>(request);
  EXPECT_EQ(flushRequest->type, NCCL_NET_IB_REQ_FLUSH);
  EXPECT_EQ(flushRequest->events[0], 1);
  ASSERT_EQ(g_fake.postedSendCount, 1);
  EXPECT_EQ(g_fake.postedSends[0].qpNumber, 43u);
  EXPECT_EQ(g_fake.postedSends[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_NE(g_fake.postedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_fake.postedSends[0].remoteAddress, 0x5000u);
  EXPECT_EQ(g_fake.postedSends[0].remoteKey, 46u);
}

/**
 * @brief Verifies successful send, receive, flush, stale, and invalid completion processing.
 * @note Side effects: Mutates local fixtures and temporarily changes the process-local matching
 * control.
 */
TEST(P2pCompletionTest, CompletionProcessing) {
  ResetFakes();

  // A single send completion must decrement exactly its live request event.
  ncclIbSendComm sendComm = MakeSendComm();
  ncclIbRequest sendRequest = {};
  sendRequest.base = &sendComm.base;
  sendRequest.type = NCCL_NET_IB_REQ_SEND;
  sendRequest.id = 2;
  sendRequest.nreqs = 1;
  sendRequest.events[0] = 1;
  sendComm.sendReqs[2][0] = &sendRequest;
  ibv_wc completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.wr_id = ibReqWrIdEncode(2);
  completion.opcode = IBV_WC_RDMA_WRITE;
  EXPECT_EQ(ncclIbCompletionEventProcess(&sendComm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(sendRequest.events[0], 0);
  EXPECT_EQ(ncclIbCompletionEventProcess(&sendComm.base, &completion, 0), ncclInternalError);

  // A sender completion bound to a non-send request must fail without changing its event count.
  ncclIbRequest wrongSendRequest = {};
  wrongSendRequest.base = &sendComm.base;
  wrongSendRequest.type = NCCL_NET_IB_REQ_RECV;
  wrongSendRequest.id = 6;
  wrongSendRequest.nreqs = 1;
  wrongSendRequest.events[0] = 1;
  sendComm.sendReqs[6][0] = &wrongSendRequest;
  completion.wr_id = ibReqWrIdEncode(6);
  EXPECT_EQ(ncclIbCompletionEventProcess(&sendComm.base, &completion, 0), ncclInternalError);
  EXPECT_EQ(wrongSendRequest.events[0], 1);

  // A grouped send completion must decrement every request represented by the shared WR-ID.
  ncclIbRequest grouped0 = {};
  ncclIbRequest grouped1 = {};
  grouped0.base = &sendComm.base;
  grouped1.base = &sendComm.base;
  grouped0.type = NCCL_NET_IB_REQ_SEND;
  grouped1.type = NCCL_NET_IB_REQ_SEND;
  grouped0.id = 5;
  grouped1.id = 5;
  grouped0.nreqs = 2;
  grouped1.nreqs = 2;
  grouped0.events[0] = 1;
  grouped1.events[0] = 1;
  sendComm.sendReqs[5][0] = &grouped0;
  sendComm.sendReqs[5][1] = &grouped1;
  completion.wr_id = ibReqWrIdEncode(5);
  EXPECT_EQ(ncclIbCompletionEventProcess(&sendComm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(grouped0.events[0], 0);
  EXPECT_EQ(grouped1.events[0], 0);

  // A duplicate grouped completion must detect the first exhausted request instead of underflowing
  // it.
  EXPECT_EQ(ncclIbCompletionEventProcess(&sendComm.base, &completion, 0), ncclInternalError);
  EXPECT_EQ(grouped0.events[0], 0);
  EXPECT_EQ(grouped1.events[0], 0);

  // A receive immediate completion must aggregate bytes and decrement exactly one event.
  ncclIbRecvComm recvComm = {};
  ncclIbRequestCompletionRecord record = {};
  ncclIbRequest recvRequest = {};
  recvRequest.base = &recvComm.base;
  recvRequest.type = NCCL_NET_IB_REQ_RECV;
  recvRequest.id = 3;
  recvRequest.nreqs = 1;
  recvRequest.events[0] = 1;
  recvRequest.recv.cmplsRecords = &record;
  recvComm.recvReqs[3] = &recvRequest;
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
  completion.imm_data = htobe32(3);
  completion.byte_len = 71;
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(recvRequest.recv.aggSize, 71);
  EXPECT_EQ(recvRequest.events[0], 0);

  // Preposted receive mode must credit the physical QP and replenish one receive WR.
  ResetFakes();
  ibv_context context = {};
  ibv_qp verbsQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &verbsQp, &cq, 73);
  InitRecvPath(&recvComm, &verbsQp, &cq);
  recvComm.prepostReceiveWorkRequests = true;
  qpMapBuild(&recvComm.base);
  recvRequest.events[0] = 1;
  completion.qp_num = 73;
  completion.byte_len = 9;
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(recvRequest.recv.aggSize, 80);
  EXPECT_EQ(recvRequest.events[0], 0);
  EXPECT_EQ(record.completions[0], 1u);
  EXPECT_EQ(g_fake.postRecvCalls, 1);

  // A flush read completion must resolve by offset and decrement exactly its event.
  ncclIbRequest& flushRequest = recvComm.base.reqs[4];
  flushRequest.base = &recvComm.base;
  flushRequest.type = NCCL_NET_IB_REQ_FLUSH;
  flushRequest.events[0] = 1;
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RDMA_READ;
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + 4;
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(flushRequest.events[0], 0);

  // A CTS completion may name a live receive but must not consume its data-completion event.
  recvRequest.events[0] = 1;
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.wr_id = ibReqWrIdEncode(3);
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(recvRequest.events[0], 1);

  // A data completion bound to a non-receive request must fail when resiliency is disabled.
  recvComm.prepostReceiveWorkRequests = false;
  ncclIbRequest wrongRecvRequest = {};
  wrongRecvRequest.base = &recvComm.base;
  wrongRecvRequest.type = NCCL_NET_IB_REQ_SEND;
  wrongRecvRequest.id = 7;
  wrongRecvRequest.nreqs = 1;
  wrongRecvRequest.events[0] = 1;
  wrongRecvRequest.recv.cmplsRecords = &record;
  recvComm.recvReqs[7] = &wrongRecvRequest;
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
  completion.imm_data = htobe32(7);
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclInternalError);
  EXPECT_EQ(wrongRecvRequest.events[0], 1);

  // Resiliency must accept a stale data completion that still resolves to an unused receive slot.
  ncclIbResiliency recvResiliency = {};
  recvComm.base.resiliency = &recvResiliency;
  ncclIbRequest unusedRecvRequest = {};
  unusedRecvRequest.base = &recvComm.base;
  unusedRecvRequest.type = NCCL_NET_IB_REQ_UNUSED;
  unusedRecvRequest.id = 8;
  recvComm.recvReqs[8] = &unusedRecvRequest;
  completion.imm_data = htobe32(8);
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclSuccess);

  // BY_INDEX must write the immediate byte count directly and consume exactly one receive event.
  recvComm.base.resiliency = nullptr;
  testReceiverMatchingScheme = BY_INDEX;
  record.sizes[0] = 0;
  recvRequest.events[0] = 1;
  completion.wr_id = ibReqWrIdEncode(3);
  completion.imm_data = htobe32(99);
  completion.byte_len = 17;
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(record.sizes[0], 99);
  EXPECT_EQ(recvRequest.recv.aggSize, 80);
  EXPECT_EQ(recvRequest.events[0], 0);

  // A stale CTS completion bound to an unused slot is accepted without changing request state.
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.wr_id = ibReqWrIdEncode(8);
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(unusedRecvRequest.type, NCCL_NET_IB_REQ_UNUSED);

  // An unsupported receiver opcode with a live binding must be rejected as an unknown completion.
  completion.opcode = IBV_WC_SEND;
  completion.wr_id = ibReqWrIdEncode(3);
  EXPECT_EQ(ncclIbCompletionEventProcess(&recvComm.base, &completion, 0), ncclInternalError);
  EXPECT_EQ(recvRequest.events[0], 0);

  // A missing request is an error normally but is an accepted stale completion under resiliency.
  completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.wr_id = 0;
  EXPECT_EQ(ncclIbCompletionEventProcess(&sendComm.base, &completion, 0), ncclInternalError);
  ncclIbResiliency resiliency = {};
  sendComm.base.resiliency = &resiliency;
  EXPECT_EQ(ncclIbCompletionEventProcess(&sendComm.base, &completion, 0), ncclSuccess);

  // Restore the default lookup policy so test order cannot affect later completion cases.
  testReceiverMatchingScheme = BY_ID;
}

/**
 * @brief Verifies latency normalization boundaries and signature consumption without exact timer
 * values.
 * @note Side effects: Mutates only local telemetry and signature-ring fixtures.
 */
TEST(P2pCompletionTest, LatencyStamping) {
  // Small transfers skip MTU scaling, while boundary and full-MTU transfers use the documented
  // formula.
  EXPECT_EQ(norm_lat_us_per_mtu(0, 0), 7u);
  EXPECT_EQ(norm_lat_us_per_mtu(7, FRM_LAT_NORM_HALF_MTU_BYTES - 1), 7u);
  EXPECT_EQ(norm_lat_us_per_mtu(17, FRM_LAT_NORM_HALF_MTU_BYTES), 21u);
  EXPECT_EQ(norm_lat_us_per_mtu(17, FRM_LAT_NORM_MTU_BYTES), 11u);

  // Unsupported or unmapped completions must leave both the ring and telemetry unchanged.
  ncclIbSendComm comm = {};
  comm.base.nqps = 1;
  ncclIbRequest request = {};
  request.base = &comm.base;
  ibv_wc completion = {};
  completion.opcode = IBV_WC_RDMA_READ;
  ibQpSigPush(&comm.base.qps[0], 1, 4096);
  ibFrmUpdateLatencyStamp<false>(&request, &completion, true);
  EXPECT_EQ(comm.base.qps[0].sig_tail, 0u);
  EXPECT_EQ(comm.base.frmFields.send.qp_lat[0], 0u);

  // A mapped write must consume one signature and publish its exact byte count.
  qpMapClear(&comm.base.frmFields.completion.qpHash);
  qpMapInsert(&comm.base.frmFields.completion.qpHash, 55, 0);
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.qp_num = 55;
  ibFrmUpdateLatencyStamp<false>(&request, &completion, true);
  EXPECT_EQ(comm.base.qps[0].sig_tail, 1u);
  EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 4096u);
  EXPECT_GT(comm.base.frmFields.send.qp_lat[0], 0u);
  EXPECT_GT(comm.base.frmCold.qp_raw_lat[0], 0u);

  // The default two-sample zero-byte holdoff must suppress two samples and accept the third.
  comm.base.frmFields.send.qp_lat[0] = 0;
  for (int i = 0; i < 3; i++) {
    ibQpSigPush(&comm.base.qps[0], 1, 0);
    ibFrmUpdateLatencyStamp<false>(&request, &completion, false);
    if (i < 2) {
      EXPECT_EQ(comm.base.frmFields.send.qp_lat[0], 0u);
    }
  }
  EXPECT_GT(comm.base.frmFields.send.qp_lat[0], 0u);
  EXPECT_EQ(comm.base.frmFields.completion.qp_zero_stamp_holdoff[0], 0u);
}

/**
 * @brief Verifies standby latency stamping rejects both invalid logical-lane boundaries.
 * @note Side effects: Leaves queued signatures and telemetry unchanged.
 */
TEST(P2pCompletionTest, StandbyLatencyStampRejectsInvalidLogicalLane) {
  ncclIbSendComm comm = {};
  comm.base.nqps = 1;
  comm.base.nPhysicalQps = 2;
  comm.base.hasStandbyPaths = true;
  qpMapInsert(&comm.base.frmFields.completion.qpHash, 55, 0);

  ncclIbRequest request = {};
  request.base = &comm.base;
  ibv_wc completion = {};
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.qp_num = 55;

  for (int invalidLogicalLane : {-1, comm.base.nqps}) {
    SCOPED_TRACE(testing::Message() << "logicalLane=" << invalidLogicalLane);
    ncclIbQp* qp = &comm.base.qps[0];
    qp->logicalLane = invalidLogicalLane;
    qp->sig_head = 0;
    qp->sig_tail = 0;
    ibQpSigPush(qp, 1, 4096);
    comm.base.frmFields.send.qp_lat[0] = 17;
    comm.base.frmCold.qp_lat_bytes[0] = 23;

    ibFrmUpdateLatencyStamp<true>(&request, &completion, true);

    EXPECT_EQ(qp->sig_head, 1u);
    EXPECT_EQ(qp->sig_tail, 0u);
    EXPECT_EQ(comm.base.frmFields.send.qp_lat[0], 17u);
    EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 23u);
  }
}

/**
 * @brief Verifies standby latency stamping maps a physical QP to its logical telemetry lane.
 * @note Side effects: Consumes one standby-QP signature and updates logical-lane telemetry.
 */
TEST(P2pCompletionTest, StandbyLatencyStampUsesLogicalLane) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp primaryQp = {};
  ibv_qp standbyQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &primaryQp, &cq, 57);
  standbyQp.context = &context;
  standbyQp.qp_num = 58;

  ncclIbSendComm comm = {};
  InitSendPath(&comm, &primaryQp, &cq);
  AddStandbySendPathForTest(&comm, &standbyQp);
  ncclIbRequest request = {};
  request.base = &comm.base;
  ibv_wc completion = {};
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.qp_num = standbyQp.qp_num;

  ibQpSigPush(&comm.base.qps[1], timer_get_cycles(), 4096);
  ibFrmUpdateLatencyStamp<true>(&request, &completion, true);

  EXPECT_EQ(comm.base.qps[0].sig_tail, 0u)
      << "A standby completion must not consume the primary QP signature ring";
  EXPECT_EQ(comm.base.qps[1].sig_tail, 1u)
      << "A standby completion must consume its physical QP signature";
  EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 4096u)
      << "Standby telemetry must be published under the QP's logical lane";
  EXPECT_GT(comm.base.frmFields.send.qp_lat[0], 0u)
      << "The logical lane must receive normalized latency from its standby path";
}

/**
 * @brief Verifies baseline polling completes a resilient send from a physical standby QP.
 * @note Side effects: Consumes one fake CQE and releases one communicator-owned request.
 */
TEST(P2pCompletionTest, StandbyBaselinePollingCompletesRequest) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp primaryQp = {};
  ibv_qp standbyQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &primaryQp, &cq, 57);
  standbyQp.context = &context;
  standbyQp.qp_num = 58;

  ncclIbSendComm comm = {};
  InitSendPath(&comm, &primaryQp, &cq);
  AddStandbySendPathForTest(&comm, &standbyQp);
  CIMState cim = {};
  cim.nqps = 1;
  comm.cim = &cim;
  ncclIbResiliency resiliency = {};
  resiliency.frmAllQpsHealthy = true;
  comm.base.resiliency = &resiliency;
  ncclIbRequest* request = PrepareSingleSendRequestForTest(&comm, 256);
  ibQpSigPush(&comm.base.qps[1], timer_get_cycles(), 256);

  g_fake.completions[0].status = IBV_WC_SUCCESS;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = standbyQp.qp_num;
  g_fake.completionCount = 1;
  int done = -1;
  int size = -1;

  ASSERT_EQ(ncclIbTest(request, &done, &size), ncclSuccess)
      << "A healthy resilient request must use standby-aware baseline polling";
  EXPECT_EQ(done, 1);
  EXPECT_EQ(size, 256);
  EXPECT_EQ(request->type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(g_fake.resiliencySendCompleteCalls, 1)
      << "Baseline standby completion must notify resiliency ownership";
  EXPECT_EQ(comm.base.qps[1].sig_tail, 1u)
      << "Baseline polling must consume the completing standby QP signature";
  EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 256u)
      << "Baseline standby polling must publish bytes under the logical lane";
}

/**
 * @brief Verifies FRM polling completes a non-resilient send from a physical standby QP.
 * @note Side effects: Consumes one fake CQE and releases one communicator-owned request.
 */
TEST(P2pCompletionTest, StandbyFrmPollingCompletesRequest) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp primaryQp = {};
  ibv_qp standbyQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &primaryQp, &cq, 57);
  standbyQp.context = &context;
  standbyQp.qp_num = 58;

  ncclIbSendComm comm = {};
  InitSendPath(&comm, &primaryQp, &cq);
  AddStandbySendPathForTest(&comm, &standbyQp);
  CIMState cim = {};
  cim.nqps = 1;
  comm.cim = &cim;
  ncclIbRequest* request = PrepareSingleSendRequestForTest(&comm, 512);
  ibQpSigPush(&comm.base.qps[1], timer_get_cycles(), 512);

  g_fake.completions[0].status = IBV_WC_SUCCESS;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = standbyQp.qp_num;
  g_fake.completionCount = 1;
  int done = -1;
  int size = -1;

  ASSERT_EQ(ncclIbTest(request, &done, &size), ncclSuccess)
      << "A non-resilient CIM request must use standby-aware FRM polling";
  EXPECT_EQ(done, 1);
  EXPECT_EQ(size, 512);
  EXPECT_EQ(request->type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(comm.base.qps[1].sig_tail, 1u)
      << "FRM polling must consume the completing standby QP signature";
  EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 512u)
      << "FRM standby polling must publish bytes under the logical lane";
}

/**
 * @brief Verifies public and direct FRM polling across completion and provider-error paths.
 * @note Side effects: Mutates local request and communicator state plus fake polling records.
 */
TEST(P2pCompletionTest, PollingAndErrors) {
  ResetFakes();
  ibv_context context = {};
  ibv_qp verbsQp = {};
  ibv_cq cq = {};
  InitFakeVerbs(&context, &verbsQp, &cq, 57);
  ibv_device device = {};
  std::strcpy(device.name, "fake-hca");
  context.device = &device;
  ibv_pd pd = {};
  pd.context = &context;
  ncclIbSendComm comm = {};
  InitSendPath(&comm, &verbsQp, &cq);
  comm.devs[0].base.pd = &pd;
  CIMState cim = {};
  ncclIbRequest& request = comm.base.reqs[0];
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.id = 0;
  request.nreqs = 1;
  request.send.size = 88;
  request.events[0] = 1;
  request.devBases[0] = &comm.devs[0].base;
  comm.sendReqs[0][0] = &request;
  comm.sendReqsCnt[0] = 1;
  int done = -1;
  int size = -1;
  std::vector<unsigned char> commBefore(sizeof(comm));

  // An unused second device must be skipped while an empty active CQ reports the request pending.
  comm.base.vProps.ndevs = 2;
  EXPECT_EQ(ibTestBaseline<false>(&request, &done, &size), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_SEND);
  EXPECT_EQ(request.events[0], 1);
  EXPECT_EQ(request.devBases[1], nullptr);
  EXPECT_EQ(g_fake.pollCalls, 1);
  comm.base.vProps.ndevs = 1;

  // One successful CQE must complete the request, report size, and release its slot.
  g_fake.completions[0] = {};
  g_fake.completions[0].status = IBV_WC_SUCCESS;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = 57;
  g_fake.completionCount = 1;
  EXPECT_EQ(ncclIbTest(&request, &done, &size), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(size, 88);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(comm.sendReqs[0][0], nullptr);

  // A negative provider poll result must propagate as ncclSystemError while leaving the request
  // live.
  ResetFakes();
  request = {};
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.id = 0;
  request.nreqs = 1;
  request.events[0] = 1;
  request.devBases[0] = &comm.devs[0].base;
  comm.sendReqs[0][0] = &request;
  g_fake.pollError = -1;
  EXPECT_EQ(ncclIbTest(&request, &done, nullptr), ncclSystemError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_SEND);

  // A baseline failed CQE without resiliency must report a nonrecoverable remote error.
  ResetFakes();
  comm.cim = nullptr;
  g_fake.completions[0] = {};
  g_fake.completions[0].status = IBV_WC_GENERAL_ERR;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = 57;
  g_fake.completionCount = 1;
  EXPECT_EQ(ncclIbTest(&request, &done, nullptr), ncclRemoteError);
  EXPECT_EQ(done, 0);

  // A live FRM request with no CQE must skip an unused device and return success without
  // completion.
  ResetFakes();
  comm.cim = &cim;
  comm.base.vProps.ndevs = 2;
  EXPECT_EQ(ncclIbTest(&request, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_SEND);
  EXPECT_EQ(request.events[0], 1);
  EXPECT_EQ(g_fake.pollCalls, 1);
  EXPECT_EQ(request.devBases[1], nullptr);

  comm.base.vProps.ndevs = 1;

  // A failed FRM CQE without resiliency must return ncclRemoteError after logging RoCE GID details.
  ResetFakes();
  comm.devs[0].base.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  comm.devs[0].base.gidInfo.localGid.raw[15] = 1;
  comm.base.remDevs[0].remoteGid.raw[15] = 2;
  g_fake.completions[0] = {};
  g_fake.completions[0].status = IBV_WC_GENERAL_ERR;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = 57;
  g_fake.completionCount = 1;
  EXPECT_EQ(ncclIbTest(&request, &done, nullptr), ncclRemoteError);
  EXPECT_EQ(done, 0);

  /** @brief Defines one direct FRM polling outcome that must preserve a live request. */
  struct DirectFrmPollCase {
    const char* name;
    int pollError;
    bool hasCompletion;
    ibv_wc_status status;
    uint64_t wrId;
    ncclResult_t expected;
  };
  const DirectFrmPollCase directCases[] = {
      {"pending", 0, false, IBV_WC_SUCCESS, 0, ncclSuccess},
      {"provider error", -1, false, IBV_WC_SUCCESS, 0, ncclSystemError},
      {"failed CQE", 0, true, IBV_WC_GENERAL_ERR, ibReqWrIdEncode(0), ncclRemoteError},
      {"unmatched CQE", 0, true, IBV_WC_SUCCESS, ibReqWrIdEncode(17), ncclInternalError},
  };
  for (const DirectFrmPollCase& directCase : directCases) {
    SCOPED_TRACE(testing::Message() << "direct FRM case=" << directCase.name);
    ResetFakes();
    g_fake.pollError = directCase.pollError;
    if (directCase.hasCompletion) {
      g_fake.completions[0].status = directCase.status;
      g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
      g_fake.completions[0].wr_id = directCase.wrId;
      g_fake.completions[0].qp_num = 57;
      g_fake.completionCount = 1;
    }
    done = -1;
    size = -1;
    std::memcpy(commBefore.data(), &comm, sizeof(comm));
    EXPECT_EQ(ibTestFrm<false>(&request, &done, &size), directCase.expected)
        << "Direct FRM polling must return the status defined by the CQ/provider outcome";
    EXPECT_EQ(done, 0) << "A non-completing direct FRM outcome must clear the done output";
    EXPECT_EQ(size, -1) << "A non-completing direct FRM outcome must preserve the size output";
    EXPECT_EQ(std::memcmp(commBefore.data(), &comm, sizeof(comm)), 0)
        << "A non-completing direct FRM outcome must preserve the complete communicator";
  }

  // Public ncclIbTest must route resilient failures to baseline recovery and keep the request
  // pending.
  ResetFakes();
  ncclIbResiliency resiliency = {};
  resiliency.inProgress = true;
  resiliency.frmAllQpsHealthy = false;
  comm.base.resiliency = &resiliency;
  g_fake.completions[0].status = IBV_WC_GENERAL_ERR;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = 57;
  g_fake.completionCount = 1;
  EXPECT_EQ(ncclIbTest(&request, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(g_fake.resiliencyProgressCalls, 1);
  EXPECT_EQ(g_fake.resiliencyErrorCalls, 1);

  // Baseline completion with an open FRM gate must update latency telemetry before releasing the
  // request.
  ResetFakes();
  comm.base.resiliency = nullptr;
  request.events[0] = 1;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.devBases[0] = &comm.devs[0].base;
  comm.sendReqs[0][0] = &request;
  comm.sendReqsCnt[0] = 1;
  qpMapBuild(&comm.base);
  ibQpSigPush(&comm.base.qps[0], 1, 64);
  g_fake.completions[0].status = IBV_WC_SUCCESS;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = 57;
  g_fake.completionCount = 1;
  EXPECT_EQ(ibTestBaseline<false>(&request, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 64u);

  // A non-resilient CIM request must use the FRM polling path and consume its matching latency
  // signature.
  ResetFakes();
  comm.base.resiliency = nullptr;
  comm.cim = &cim;
  request.events[0] = 1;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.devBases[0] = &comm.devs[0].base;
  comm.sendReqs[0][0] = &request;
  comm.sendReqsCnt[0] = 1;
  qpMapBuild(&comm.base);
  const uint32_t tailBeforeFrmCompletion = comm.base.qps[0].sig_tail;
  ibQpSigPush(&comm.base.qps[0], 1, 128);
  g_fake.completions[0].status = IBV_WC_SUCCESS;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = 57;
  g_fake.completionCount = 1;
  EXPECT_EQ(ncclIbTest(&request, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(comm.base.qps[0].sig_tail, tailBeforeFrmCompletion + 1);
  EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 128u);

  // A direct FRM call must publish every completion output and preserve immutable request fields.
  ResetFakes();
  request = {};
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.id = 0;
  request.nreqs = 1;
  request.send.size = 88;
  request.events[0] = 1;
  request.devBases[0] = &comm.devs[0].base;
  comm.sendReqs[0][0] = &request;
  comm.sendReqsCnt[0] = 1;
  qpMapBuild(&comm.base);
  const uint32_t tailBeforeDirectCompletion = comm.base.qps[0].sig_tail;
  ibQpSigPush(&comm.base.qps[0], 1, 88);
  g_fake.completions[0].status = IBV_WC_SUCCESS;
  g_fake.completions[0].opcode = IBV_WC_RDMA_WRITE;
  g_fake.completions[0].wr_id = ibReqWrIdEncode(0);
  g_fake.completions[0].qp_num = 57;
  g_fake.completionCount = 1;
  done = -1;
  size = -1;
  EXPECT_EQ(ibTestFrm<false>(&request, &done, &size), ncclSuccess)
      << "Direct FRM polling must complete a valid CQE";
  EXPECT_EQ(done, 1) << "A valid CQE must set the direct FRM completion output";
  EXPECT_EQ(size, 88) << "A completed direct FRM send must report its submitted byte count";
  EXPECT_EQ(request.events[0], 0) << "A valid CQE must consume the request's device event";
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED)
      << "A completed direct FRM request must release its pool entry";
  EXPECT_EQ(request.base, &comm.base)
      << "Completion must preserve the request's communicator association";
  EXPECT_EQ(request.id, 0u) << "Completion must preserve the logical request ID";
  EXPECT_EQ(request.nreqs, 1) << "Completion must preserve the request group size";
  EXPECT_EQ(request.send.size, 88) << "Completion must preserve the submitted byte count";
  EXPECT_EQ(comm.sendReqs[0][0], nullptr)
      << "Completing the final grouped send must clear its lookup slot";
  EXPECT_EQ(comm.base.qps[0].sig_tail, tailBeforeDirectCompletion + 1)
      << "Direct FRM completion must consume its latency signature";
  EXPECT_EQ(comm.base.frmCold.qp_lat_bytes[0], 88u)
      << "Direct FRM completion must publish the signature's exact byte count";
}

/**
 * @brief Verifies a stale successful CTS completion cannot modify a reused request slot.
 */
TEST(NetIbP2pTest,
     SuccessfulNonResilientCtsAfterSlotReuseDoesNotTouchReplacement) {
  ncclIbRecvComm receive = {};
  ncclIbNetCommBase* base = &receive.base;
  base->isSend = false;
  ASSERT_EQ(base->resiliency, nullptr);

  constexpr uint64_t staleRequestId = 23;
  constexpr uint64_t replacementRequestId =
      staleRequestId + NET_IB_MAX_REQUESTS;
  constexpr int requestSlot = staleRequestId % NET_IB_MAX_REQUESTS;
  ncclIbRequest* replacement = &base->reqs[requestSlot];
  replacement->base = base;
  replacement->id = replacementRequestId;
  replacement->type = NCCL_NET_IB_REQ_RECV;
  replacement->nreqs = 2;
  replacement->events[0] = 9;
  receive.recvReqs[requestSlot] = replacement;
  unsigned char replacementBefore[sizeof(*replacement)];
  std::memcpy(replacementBefore, replacement, sizeof(replacementBefore));

  ibv_wc completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.wr_id = ibReqWrIdEncode(staleRequestId);
  completion.qp_num = 1201;

  ASSERT_EQ(ncclIbCompletionEventProcess(base, &completion, 0), ncclSuccess);
  EXPECT_EQ(std::memcmp(replacementBefore, replacement,
                        sizeof(replacementBefore)), 0);
  EXPECT_EQ(replacement->type, NCCL_NET_IB_REQ_RECV);
  EXPECT_EQ(replacement->id, replacementRequestId);
  EXPECT_EQ(replacement->nreqs, 2);
  EXPECT_EQ(replacement->events[0], 9);
  EXPECT_EQ(receive.recvReqs[requestSlot], replacement);
}

/** @brief Verifies CTS selection follows resiliency-remapped active QPs. */
TEST(NetIbP2pTest, CtsSelectionFollowsRemappedActiveQps) {
  ncclIbRecvComm receive = {};
  receive.base.vProps.ndevs = 2;
  receive.base.nqps = 2;
  receive.base.nPhysicalQps = 4;
  receive.base.hasStandbyPaths = true;
  receive.base.activeQps[0] = &receive.base.qps[2];
  receive.base.activeQps[1] = &receive.base.qps[3];

  ncclIbQp* qp = nullptr;
  ASSERT_EQ(ncclIbRecvCommGetQpForCts(&receive, 4, &qp), ncclSuccess);
  EXPECT_EQ(qp, &receive.base.qps[2]);

  ASSERT_EQ(ncclIbRecvCommGetQpForCts(&receive, 5, &qp), ncclSuccess);
  EXPECT_EQ(qp, &receive.base.qps[3]);
}

/**
 * @brief Verifies direct CTS selection rotates devices across multiple logical QPs.
 * @note Side effects: Resets and records process-local verbs fakes.
 */
TEST(NetIbP2pTest, DirectCtsSelectionRetainsDeviceRotationWithMultipleQps) {
  ResetVerbsFakes();
  ncclIbRecvComm receive = {};
  receive.base.vProps.ndevs = 2;
  receive.base.nqps = 8;
  receive.base.nPhysicalQps = 8;
  receive.base.nRemDevs = 2;
  for (int qpIndex = 0; qpIndex < receive.base.nqps; qpIndex++) {
    receive.base.qps[qpIndex].qp = &verbsQps[qpIndex];
    receive.base.qps[qpIndex].devIndex = qpIndex % 2;
    receive.base.qps[qpIndex].remDevIdx = qpIndex % 2;
    receive.base.qps[qpIndex].logicalLane = qpIndex;
    receive.base.activeQps[qpIndex] = &receive.base.qps[qpIndex];
  }
  receive.base.remDevs[0].rkey = 0x50;
  receive.base.remDevs[1].rkey = 0x51;
  receive.remCtsFifo.addr = 0x100000;

  ncclIbQp* qp = nullptr;
  ASSERT_EQ(ncclIbRecvCommGetQpForCts(&receive, 6, &qp), ncclSuccess);
  EXPECT_EQ(qp, &receive.base.qps[0]);
  ASSERT_EQ(ncclIbRecvCommGetQpForCts(&receive, 7, &qp), ncclSuccess);
  EXPECT_EQ(qp, &receive.base.qps[1]);

  ncclIbRequest request = {};
  request.base = &receive.base;
  request.id = 7;
  request.nreqs = 1;
  ASSERT_EQ(ncclIbPostFifo(&receive, &request, 1), ncclSuccess);
  ASSERT_EQ(postedRanges.size(), 1u);
  EXPECT_EQ(postedRanges[0].qp, &verbsQps[1]);
  EXPECT_NE(postedRanges[0].sendFlags & IBV_SEND_SIGNALED, 0u);

  postedRanges.clear();
  ASSERT_EQ(ncclIbPostFifo(&receive, &request, 3), ncclSuccess);
  ASSERT_EQ(postedRanges.size(), 1u);
  EXPECT_EQ(postedRanges[0].qp, &verbsQps[1]);
  EXPECT_EQ(postedRanges[0].sendFlags & IBV_SEND_SIGNALED, 0u);
}

/**
 * @brief Verifies receive completion credits the physical standby QP that completed it.
 * @note Side effects: Resets and records process-local verbs fakes.
 */
TEST(NetIbP2pTest, ReceiveCompletionCreditsPhysicalStandbyQp) {
  ResetVerbsFakes();
  ncclIbRecvComm receive = {};
  receive.base.isSend = false;
  receive.base.vProps.ndevs = 2;
  receive.base.nqps = 2;
  receive.base.nPhysicalQps = 4;
  receive.prepostReceiveWorkRequests = true;
  for (int physicalIndex = 0; physicalIndex < 4; physicalIndex++) {
    receive.base.qps[physicalIndex].qp = &verbsQps[physicalIndex];
    receive.base.qps[physicalIndex].devIndex = physicalIndex / 2;
    receive.base.qps[physicalIndex].logicalLane = physicalIndex % 2;
  }
  qpMapBuild(&receive.base);

  constexpr uint64_t requestId = 7;
  ncclIbRequestCompletionRecord completionRecord = {};
  ncclIbRequest* request = &receive.base.reqs[requestId % NET_IB_MAX_REQUESTS];
  request->base = &receive.base;
  request->id = requestId;
  request->type = NCCL_NET_IB_REQ_RECV;
  request->nreqs = 1;
  request->events[1] = 1;
  request->recv.cmplsRecords = &completionRecord;
  receive.recvReqs[requestId % NET_IB_MAX_REQUESTS] = request;

  ibv_wc completion = {};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
  completion.imm_data = htobe32(static_cast<uint32_t>(requestId));
  completion.byte_len = 512;
  completion.qp_num = verbsQps[3].qp_num;

  ASSERT_EQ(ncclIbCompletionEventProcess(&receive.base, &completion, 1), ncclSuccess);
  EXPECT_EQ(completionRecord.completions[3], 1);
  EXPECT_EQ(completionRecord.completions[1], 0);
  EXPECT_EQ(request->events[1], 0);
  EXPECT_EQ(postRecvCalls, 1);
}

/**
 * @brief Verifies completion lookup follows each QP's peer-device mapping.
 * @note Side effects: Resets process-local verbs fakes.
 */
TEST(NetIbP2pTest, CompletionRemoteDeviceFollowsQpPeerMapping) {
  ResetVerbsFakes();
  auto comm = MakeSendComm();
  comm.base.vProps.ndevs = 3;
  comm.base.nRemDevs = 2;
  comm.base.nqps = 1;
  comm.base.nPhysicalQps = 1;
  comm.base.qps[0].qp = &verbsQps[0];
  comm.base.qps[0].devIndex = 2;
  comm.base.qps[0].remDevIdx = 0;
  qpMapBuild(&comm.base);

  ibv_wc completion = {};
  completion.qp_num = verbsQps[0].qp_num;
  EXPECT_EQ(occlIbCompletionRemoteDeviceIndex(&comm.base, &completion, 2), 0);

  completion.qp_num++;
  EXPECT_EQ(occlIbCompletionRemoteDeviceIndex(&comm.base, &completion, 1), 1);
  EXPECT_EQ(occlIbCompletionRemoteDeviceIndex(&comm.base, &completion, 2), -1);
}

/**
 * @brief Verifies direct two- and four-QP splits retain their logical lane geometry.
 * @note Side effects: Resets and records process-local verbs fakes.
 */
TEST(NetIbP2pTest, DirectSplitQpsTwoAndFourRetainLegacyLaneGeometry) {
  for (int qpsPerConnection : {2, 4}) {
    ResetVerbsFakes();
    auto comm = MakeSendComm();
    const int laneCount = 2 * qpsPerConnection;
    comm.base.vProps.ndevs = 2;
    comm.base.nqps = laneCount;
    comm.base.nPhysicalQps = laneCount;
    comm.base.hasStandbyPaths = false;
    comm.base.nDataQps = 2;
    comm.base.splitDataOnQps = 1;
    for (int lane = 0; lane < laneCount; lane++) {
      comm.base.qps[lane].qp = &verbsQps[lane];
      comm.base.qps[lane].devIndex = lane % 2;
      comm.base.qps[lane].remDevIdx = lane % 2;
      comm.base.qps[lane].logicalLane = lane;
      comm.base.activeQps[lane] = &comm.base.qps[lane];
    }

    ncclIbResiliency resiliency = {};
    resiliency.baseComm = &comm.base;
    resiliency.frmAllQpsHealthy = true;
    comm.base.resiliency = &resiliency;

    CIMState congestion = {};
    congestion.nqps = laneCount;
    congestion.spread_eps_us = 0;
    comm.cim = &congestion;
    for (int lane = 0; lane < laneCount; lane++) {
      comm.base.frmFields.send.qp_lat[lane] = 10 + lane;
    }

    MultiSendBuffers buffers = {};
    AttachMultiSendBuffers(&comm, &buffers, laneCount, true);
    ASSERT_NE(comm.multiSendScratch.replayPlanBytes, nullptr);
    ASSERT_NE(comm.multiSendScratch.replayPlanFrozen, nullptr);

    constexpr int slot = 0;
    constexpr uint32_t sendBytes = 16384;
    alignas(64) uint8_t localStorage[sendBytes] = {};
    const uintptr_t localBase = reinterpret_cast<uintptr_t>(localStorage);
    constexpr uint64_t remoteBase = 0x180000;
    comm.ctsFifo[slot][0].nreqs = 1;
    comm.ctsFifo[slot][0].addr = remoteBase;
    comm.ctsFifo[slot][0].rkeys[0] = 0x10;
    comm.ctsFifo[slot][0].rkeys[1] = 0x11;

    ncclIbRequest& request = comm.base.reqs[0];
    request.base = &comm.base;
    request.id = 0;
    request.nreqs = 1;
    request.type = NCCL_NET_IB_REQ_SEND;
    request.send.wrIdToken = 1;
    request.send.size = sendBytes;
    request.send.data = localStorage;
    request.send.lkeys[0] = 0x20;
    request.send.lkeys[1] = 0x21;
    for (int lane = 0; lane < laneCount; lane++) {
      request.send.pendingData[lane] = true;
    }
    comm.sendReqs[slot][0] = &request;

    ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
    ASSERT_EQ(postedRanges.size(), (size_t)laneCount)
        << "qpsPerConnection=" << qpsPerConnection;
    uint32_t coveredBytes = 0;
    for (int lane = 0; lane < laneCount; lane++) {
      const PostedRange& range = postedRanges[lane];
      EXPECT_EQ(range.qp, &verbsQps[lane]);
      EXPECT_EQ(range.localAddress, localBase + coveredBytes);
      EXPECT_EQ(range.remoteAddress, remoteBase + coveredBytes);
      coveredBytes += range.bytes;
    }
    EXPECT_EQ(coveredBytes, sendBytes);

    uint32_t* replayPlan = occlIbMultiSendScratchReplayPlanRow(&comm.multiSendScratch, 0);
    ASSERT_EQ(comm.multiSendScratch.replayPlanFrozen[0], 1);
    ASSERT_NE(replayPlan[0], replayPlan[1]);
    request.send.pendingData[0] = false;
    resiliency.inProgress = true;
    postedRanges.clear();

    ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
    ASSERT_EQ(postedRanges.size(), (size_t)laneCount - 1);
    uint32_t replayOffset = replayPlan[0];
    for (int lane = 1; lane < laneCount; lane++) {
      const PostedRange& range = postedRanges[lane - 1];
      EXPECT_EQ(range.qp, &verbsQps[lane]);
      EXPECT_EQ(range.bytes, replayPlan[lane]);
      EXPECT_EQ(range.localAddress, localBase + replayOffset);
      EXPECT_EQ(range.remoteAddress, remoteBase + replayOffset);
      replayOffset += replayPlan[lane];
    }
    EXPECT_EQ(replayOffset, sendBytes);
  }
}

/**
 * @brief Verifies resilient weighted replay preserves each logical lane's byte range.
 * @note Side effects: Resets and records process-local verbs fakes.
 */
TEST(NetIbP2pTest, ResilientWeightedReplayKeepsLogicalByteRanges) {
  ResetVerbsFakes();
  auto comm = MakeSendComm();
  comm.base.vProps.ndevs = 2;
  comm.base.nqps = 2;
  comm.base.nPhysicalQps = 4;
  comm.base.hasStandbyPaths = true;
  comm.base.nDataQps = 1;
  comm.base.splitDataOnQps = 1;
  for (int physicalIndex = 0; physicalIndex < 4; physicalIndex++) {
    comm.base.qps[physicalIndex].qp = &verbsQps[physicalIndex];
    comm.base.qps[physicalIndex].devIndex = physicalIndex / 2;
    comm.base.qps[physicalIndex].remDevIdx = physicalIndex / 2;
    comm.base.qps[physicalIndex].logicalLane = physicalIndex % 2;
  }
  comm.base.activeQps[0] = &comm.base.qps[0];
  comm.base.activeQps[1] = &comm.base.qps[1];

  ncclIbResiliency resiliency = {};
  resiliency.baseComm = &comm.base;
  resiliency.frmAllQpsHealthy = true;
  comm.base.resiliency = &resiliency;

  CIMState congestion = {};
  congestion.nqps = 2;
  congestion.spread_eps_us = 0;
  comm.cim = &congestion;
  comm.base.frmFields.send.qp_lat[0] = 10;
  comm.base.frmFields.send.qp_lat[1] = 20;

  MultiSendBuffers buffers = {};
  AttachMultiSendBuffers(&comm, &buffers, 2, true);

  constexpr int slot = 0;
  constexpr uint32_t sendBytes = 4096;
  alignas(64) uint8_t localStorage[sendBytes] = {};
  const uintptr_t localBase = reinterpret_cast<uintptr_t>(localStorage);
  constexpr uint64_t remoteBase = 0x200000;
  comm.ctsFifo[slot][0].nreqs = 1;
  comm.ctsFifo[slot][0].addr = remoteBase;
  comm.ctsFifo[slot][0].rkeys[0] = 0x10;
  comm.ctsFifo[slot][0].rkeys[1] = 0x11;

  constexpr int requestIndex = 5;
  ncclIbRequest* request = &comm.base.reqs[requestIndex];
  request->base = &comm.base;
  request->id = 0;
  request->nreqs = 1;
  request->type = NCCL_NET_IB_REQ_SEND;
  request->send.wrIdToken = 1;
  request->send.size = sendBytes;
  request->send.data = localStorage;
  request->send.lkeys[0] = 0x20;
  request->send.lkeys[1] = 0x21;
  request->send.pendingData[0] = true;
  request->send.pendingData[1] = true;
  comm.sendReqs[slot][0] = request;

  ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
  ASSERT_NE(comm.multiSendScratch.replayPlanFrozen, nullptr);
  ASSERT_EQ(comm.multiSendScratch.replayPlanFrozen[requestIndex], 1);
  uint32_t* replayPlan = occlIbMultiSendScratchReplayPlanRow(
      &comm.multiSendScratch, requestIndex);
  ASSERT_EQ(postedRanges.size(), 2u);
  const uint32_t firstPlan[] = {replayPlan[0], replayPlan[1]};
  ASSERT_NE(firstPlan[0], firstPlan[1]);
  ASSERT_EQ(firstPlan[0] + firstPlan[1], sendBytes);
  const std::vector<PostedRange> firstAttempt = postedRanges;
  EXPECT_EQ(firstAttempt[0].bytes, firstPlan[0]);
  EXPECT_EQ(firstAttempt[1].bytes, firstPlan[1]);
  EXPECT_EQ(firstAttempt[0].localAddress, localBase);
  EXPECT_EQ(firstAttempt[1].localAddress, localBase + firstPlan[0]);
  EXPECT_EQ(firstAttempt[0].remoteAddress, remoteBase);
  EXPECT_EQ(firstAttempt[1].remoteAddress, remoteBase + firstPlan[0]);

  comm.base.activeQps[0] = &comm.base.qps[2];
  comm.base.activeQps[1] = &comm.base.qps[3];
  comm.base.frmFields.send.qp_lat[0] = 20;
  comm.base.frmFields.send.qp_lat[1] = 10;
  resiliency.inProgress = true;
  postedRanges.clear();

  ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
  ASSERT_EQ(postedRanges.size(), 2u);
  EXPECT_EQ(replayPlan[0], firstPlan[0]);
  EXPECT_EQ(replayPlan[1], firstPlan[1]);
  EXPECT_EQ(postedRanges[0].qp, &verbsQps[2]);
  EXPECT_EQ(postedRanges[1].qp, &verbsQps[3]);
  EXPECT_EQ(postedRanges[0].bytes, firstAttempt[0].bytes);
  EXPECT_EQ(postedRanges[1].bytes, firstAttempt[1].bytes);
  EXPECT_EQ(postedRanges[0].localAddress, firstAttempt[0].localAddress);
  EXPECT_EQ(postedRanges[1].localAddress, firstAttempt[1].localAddress);
  EXPECT_EQ(postedRanges[0].remoteAddress, firstAttempt[0].remoteAddress);
  EXPECT_EQ(postedRanges[1].remoteAddress, firstAttempt[1].remoteAddress);

  // A first post made while the FRM gate is closed freezes its equal plan.
  comm.multiSendScratch.replayPlanFrozen[requestIndex] = 0;
  postedRanges.clear();
  ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
  ASSERT_EQ(comm.multiSendScratch.replayPlanFrozen[requestIndex], 1);
  ASSERT_EQ(replayPlan[0] + replayPlan[1], sendBytes);
  EXPECT_EQ(replayPlan[0], replayPlan[1]);
  const std::vector<PostedRange> gateClosedAttempt = postedRanges;

  // Reopening FRM must still consume the plan owned by that first attempt.
  resiliency.inProgress = false;
  resiliency.frmAllQpsHealthy = true;
  postedRanges.clear();
  ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
  ASSERT_EQ(postedRanges.size(), gateClosedAttempt.size());
  for (size_t lane = 0; lane < postedRanges.size(); lane++) {
    EXPECT_EQ(postedRanges[lane].bytes, gateClosedAttempt[lane].bytes);
    EXPECT_EQ(postedRanges[lane].localAddress, gateClosedAttempt[lane].localAddress);
    EXPECT_EQ(postedRanges[lane].remoteAddress, gateClosedAttempt[lane].remoteAddress);
  }
}

/**
 * @brief Verifies both send planners preserve partial replay across two active ports.
 * @note Side effects: Repeatedly resets and records process-local verbs fakes.
 */
TEST(NetIbP2pTest, TwoActivePortsPreservePartialReplayWithBothSendPlanners) {
  const int devicePairs[][2] = {{2, 2}, {4, 4}, {8, 8}, {2, 4}, {4, 2}};
  const uint32_t payloadSizes[] = {16384, 128, 1, 129, 16385};
  for (const auto& devices : devicePairs) {
    for (int qpsPerPort : {1, 2, 4}) {
      for (int split : {0, 1}) {
        for (int featureMode = 0; featureMode < 4 * (int)(sizeof(payloadSizes) / sizeof(payloadSizes[0])); featureMode++) {
          const bool enableFrm = (featureMode & 1) != 0;
          const bool enableResiliency = (featureMode & 2) != 0;
          const uint32_t sendBytes = payloadSizes[featureMode / 4];
          for (int failedDevice = 0; failedDevice < (enableResiliency ? 2 : 1); failedDevice++) {
            SCOPED_TRACE(::testing::Message() << "local=" << devices[0]
                << " remote=" << devices[1] << " qps=" << qpsPerPort
                << " split=" << split << " frm=" << enableFrm
                << " resiliency=" << enableResiliency
                << " bytes=" << sendBytes
                << " failed=" << failedDevice);
            ResetVerbsFakes();
            auto comm = MakeSendComm();
            const int lanes = 2 * qpsPerPort;
            const int paths = std::max(devices[0], devices[1]);
            comm.base.vProps.ndevs = devices[0];
            comm.base.nRemDevs = devices[1];
            comm.base.nqps = lanes;
            comm.base.nDataQps = 2;
            comm.base.nPhysicalQps = lanes * paths;
            comm.base.hasStandbyPaths = true;
            comm.base.splitDataOnQps = split;
            ibv_qp dataQps[64] = {};
            ASSERT_LE(comm.base.nPhysicalQps, 64);
            for (int path = 0; path < paths; path++) {
              for (int lane = 0; lane < lanes; lane++) {
                const int physical = occlIbCommPhysicalQpIndex(&comm.base, lane, path);
                dataQps[physical].context = &verbsContext;
                dataQps[physical].qp_num = 1000 + physical;
                auto* qp = &comm.base.qps[physical];
                qp->qp = &dataQps[physical];
                qp->devIndex = (path + lane % 2) % devices[0];
                qp->remDevIdx = (path + lane % 2) % devices[1];
                qp->logicalLane = lane;
                qp->coldIdx = physical;
                if (path == 0) comm.base.activeQps[lane] = qp;
              }
            }
            ncclIbResiliency resiliency = {};
            resiliency.baseComm = &comm.base;
            resiliency.ndevs = devices[0];
            resiliency.frmAllQpsHealthy = true;
            comm.base.resiliency = enableResiliency ? &resiliency : nullptr;
            CIMState congestion = {};
            congestion.nqps = lanes;
            congestion.spread_eps_us = 0;
            comm.cim = enableFrm ? &congestion : nullptr;
            for (int lane = 0; lane < lanes; lane++) {
              comm.base.frmFields.send.qp_lat[lane] = 10 + 10 * lane;
            }
            MultiSendBuffers buffers = {};
            AttachMultiSendBuffers(&comm, &buffers, lanes, true);

            // Start on the second active port and wrap the logical-lane ring.
            const int requestId = lanes - 1;
            const int slot = requestId % NET_IB_MAX_REQUESTS;
            constexpr int requestIndex = 5;
            alignas(64) uint8_t localStorage[16385] = {};
            const uintptr_t localBase = reinterpret_cast<uintptr_t>(localStorage);
            constexpr uint64_t remoteBase = 0x300000;
            comm.ctsFifo[slot][0].nreqs = 1;
            comm.ctsFifo[slot][0].addr = remoteBase;
            for (int device = 0; device < devices[1]; device++) {
              comm.ctsFifo[slot][0].rkeys[device] = 0x10 + device;
            }
            auto* request = &comm.base.reqs[requestIndex];
            request->base = &comm.base;
            request->id = requestId;
            request->nreqs = 1;
            request->type = NCCL_NET_IB_REQ_SEND;
            request->send.wrIdToken = ibReqWrIdEncode(requestId);
            request->send.size = sendBytes;
            request->send.data = localStorage;
            for (int device = 0; device < devices[0]; device++) {
              request->send.lkeys[device] = 0x20 + device;
            }
            comm.sendReqs[slot][0] = request;
            const int requestLanes = ncclIbCommBaseGetNqpsPerRequest(&comm.base);
            for (int i = 0; i < requestLanes; i++) {
              request->send.pendingData[(requestId + i) % lanes] = true;
            }

            ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
            ASSERT_EQ(postedRanges.size(), (size_t)requestLanes);
            const auto firstAttempt = postedRanges;
            uint32_t offset = 0;
            int postsPerPort[2] = {};
            for (int i = 0; i < requestLanes; i++) {
              const int lane = (requestId + i) % lanes;
              const auto& range = firstAttempt[i];
              EXPECT_EQ(range.qp, &dataQps[lane]);
              EXPECT_EQ(range.immediatePosts, 1u);
              EXPECT_EQ(range.signaledPosts, 1u);
              if (range.bytes > 0) {
                EXPECT_EQ(range.localAddress, localBase + offset);
                EXPECT_EQ(range.remoteAddress, remoteBase + offset);
                EXPECT_EQ(range.localKey, 0x20u + lane % 2);
              } else {
                // Zero-length lanes still post an immediate completion. They
                // have no payload SGE and still participate in replay credit.
                EXPECT_EQ(range.emptyImmediatePosts, 1u);
              }
              EXPECT_EQ(range.immediateRkey, 0x10u + lane % 2);
              postsPerPort[lane % 2]++;
              offset += range.bytes;
            }
            EXPECT_EQ(offset, sendBytes);
            EXPECT_EQ(postsPerPort[0], requestLanes / 2);
            EXPECT_EQ(postsPerPort[1], requestLanes / 2);
            if (!enableResiliency) {
              EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen[requestIndex], 0);
              continue;
            }
            auto* replayPlan = occlIbMultiSendScratchReplayPlanRow(
                &comm.multiSendScratch, requestIndex);
            if (enableFrm) {
              ASSERT_EQ(comm.multiSendScratch.replayPlanFrozen[requestIndex], 1);
              for (int i = 0; i < requestLanes; i++) {
                EXPECT_EQ(replayPlan[(requestId + i) % lanes], firstAttempt[i].bytes);
              }
            }

            // Probe credited the healthy port. Only failed-port logical chunks
            // remain pending, now on their own preconnected standby paths.
            for (int lane = 0; lane < lanes; lane++) {
              if (!request->send.pendingData[lane]) continue;
              if (lane % 2 != failedDevice) {
                request->send.pendingData[lane] = false;
                continue;
              }
              for (int path = 1; path < paths; path++) {
                const int physical = occlIbCommPhysicalQpIndex(&comm.base, lane, path);
                if (comm.base.qps[physical].devIndex == failedDevice) continue;
                comm.base.activeQps[lane] = &comm.base.qps[physical];
                break;
              }
              ASSERT_NE(comm.base.activeQps[lane]->devIndex, failedDevice);
            }
            for (int lane = 0; lane < lanes; lane++) {
              comm.base.frmFields.send.qp_lat[lane] = 1000 - 10 * lane;
            }
            resiliency.inProgress = true;
            for (int attempt = 0; attempt < 2; attempt++) {
              if (attempt == 1) {
                // Peer-ready failback and reopening FRM must not re-plan a
                // partially delivered request or resurrect delivered chunks.
                for (int lane = 0; lane < lanes; lane++) {
                  comm.base.activeQps[lane] = &comm.base.qps[lane];
                }
                resiliency.inProgress = false;
              }
              postedRanges.clear();
              ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
              ASSERT_EQ(postedRanges.size(), (size_t)requestLanes / 2);
              size_t post = 0;
              for (int i = 0; i < requestLanes; i++) {
                const int lane = (requestId + i) % lanes;
                if (lane % 2 != failedDevice) {
                  EXPECT_FALSE(request->send.pendingData[lane]);
                  EXPECT_EQ(request->send.lastAttemptQpIndex[lane], -1);
                  continue;
                }
                const auto* qp = comm.base.activeQps[lane];
                const auto& replay = postedRanges[post++];
                EXPECT_EQ(replay.qp, qp->qp);
                EXPECT_EQ(qp->logicalLane, lane);
                EXPECT_EQ(replay.bytes, firstAttempt[i].bytes);
                EXPECT_EQ(replay.immediatePosts, 1u);
                EXPECT_EQ(replay.signaledPosts, 1u);
                if (firstAttempt[i].bytes > 0) {
                  EXPECT_EQ(replay.localAddress, firstAttempt[i].localAddress);
                  EXPECT_EQ(replay.remoteAddress, firstAttempt[i].remoteAddress);
                  EXPECT_EQ(replay.localKey, 0x20u + qp->devIndex);
                } else {
                  EXPECT_EQ(replay.emptyImmediatePosts, 1u);
                  EXPECT_EQ(replay.bytes, 0u);
                }
                EXPECT_TRUE(request->send.pendingData[lane]);
                EXPECT_EQ(replay.immediateRkey, 0x10u + qp->remDevIdx);
                EXPECT_EQ(request->send.lastAttemptQpIndex[lane], qp - comm.base.qps);
                if (enableFrm) {
                  EXPECT_EQ(replayPlan[lane], firstAttempt[i].bytes);
                }
              }
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Verifies direct partial replay preserves small and unequal grouped tail ranges.
 * @note Side effects: Repeatedly resets and records process-local verbs fakes.
 */
TEST(NetIbP2pTest, DirectPartialReplayKeepsSmallAndUnequalGroupedTailRanges) {
  for (int requestCount : {1, 2}) {
    for (uint32_t firstSize : {1u, 128u, 129u}) {
      for (int split : {0, 1}) {
        SCOPED_TRACE(::testing::Message() << "requests=" << requestCount
            << " firstBytes=" << firstSize << " split=" << split);
        ResetVerbsFakes();
        auto comm = MakeSendComm();
        comm.base.vProps.ndevs = 2;
        comm.base.nRemDevs = 2;
        comm.base.nqps = 2;
        comm.base.nPhysicalQps = 2;
        comm.base.nDataQps = 2;
        comm.base.splitDataOnQps = split;
        ASSERT_FALSE(comm.base.hasStandbyPaths);
        ASSERT_EQ(comm.cim, nullptr);
        for (int lane = 0; lane < 2; lane++) {
          auto* qp = &comm.base.qps[lane];
          qp->qp = &verbsQps[lane];
          qp->devIndex = lane;
          qp->remDevIdx = lane;
          qp->logicalLane = lane;
          comm.base.activeQps[lane] = qp;
          comm.remCmplsRecords.rkeys[lane] = 0x60 + lane;
        }
        ncclIbResiliency resiliency = {};
        resiliency.baseComm = &comm.base;
        resiliency.ndevs = 2;
        comm.base.resiliency = &resiliency;
        comm.ctsFifo[0][0].nreqs = requestCount;
        comm.remCmplsRecords.addr = 0x900000;
        alignas(64) uint8_t localStorage[2][512] = {};
        uint32_t sizes[2] = {};
        uint32_t firstChunks[2] = {};
        for (int r = 0; r < requestCount; r++) {
          sizes[r] = firstSize + 128 * r;
          firstChunks[r] = std::min(sizes[r], ((sizes[r] + 1) / 2 + 127) / 128 * 128);
          auto* request = &comm.base.reqs[r];
          request->base = &comm.base;
          request->id = 0;
          request->type = NCCL_NET_IB_REQ_SEND;
          request->nreqs = requestCount;
          request->send.wrIdToken = ibReqWrIdEncode(0);
          request->send.data = localStorage[r];
          request->send.size = sizes[r];
          for (int device = 0; device < 2; device++) {
            request->send.lkeys[device] = 0x20 + 2 * r + device;
            request->send.pendingData[device] = true;
            comm.ctsFifo[0][r].rkeys[device] = 0x40 + 2 * r + device;
          }
          comm.ctsFifo[0][r].addr = 0x500000 + r * 0x1000;
          comm.sendReqs[0][r] = request;
        }
        ASSERT_EQ(ncclIbMultiSend(&comm, 0), ncclSuccess);
        ASSERT_EQ(postedRanges.size(), 2u);
        const auto firstAttempt = postedRanges;
        for (int r = 0; r < requestCount; r++) {
          ASSERT_GT(firstAttempt[0].writes.size(), (size_t)r);
          ASSERT_GT(firstAttempt[1].writes.size(), (size_t)r);
          EXPECT_EQ(firstAttempt[0].writes[r].bytes, firstChunks[r]);
          EXPECT_EQ(firstAttempt[1].writes[r].bytes, sizes[r] - firstChunks[r]);
          comm.base.reqs[r].send.pendingData[0] = false;
        }

        // Direct geometry may borrow a healthy physical QP for another lane.
        // Its delivered first chunk must consume only its actual, clamped size.
        comm.base.activeQps[1] = &comm.base.qps[0];
        resiliency.inProgress = true;
        postedRanges.clear();
        ASSERT_EQ(ncclIbMultiSend(&comm, 0), ncclSuccess);
        ASSERT_EQ(postedRanges.size(), 1u);
        const auto& replay = postedRanges[0];
        EXPECT_EQ(replay.qp, &verbsQps[0]);
        EXPECT_EQ(replay.immediatePosts, 1u);
        EXPECT_EQ(replay.signaledPosts, 1u);
        ASSERT_EQ(replay.writes.size(), (size_t)(requestCount + (requestCount > 1)));
        for (int r = 0; r < requestCount; r++) {
          const auto& tail = replay.writes[r];
          const uint32_t tailBytes = sizes[r] - firstChunks[r];
          EXPECT_EQ(tail.bytes, tailBytes);
          EXPECT_EQ(tail.bytes, firstAttempt[1].writes[r].bytes);
          EXPECT_EQ(tail.numSge, tailBytes > 0 ? 1 : 0);
          if (tailBytes > 0) {
            EXPECT_EQ(tail.localAddress, (uintptr_t)localStorage[r] + firstChunks[r]);
            EXPECT_EQ(tail.remoteAddress, 0x500000u + r * 0x1000 + firstChunks[r]);
            EXPECT_EQ(tail.localKey, 0x20u + 2 * r);
            EXPECT_EQ(tail.remoteKey, 0x40u + 2 * r);
          }
          EXPECT_FALSE(comm.base.reqs[r].send.pendingData[0]);
          EXPECT_TRUE(comm.base.reqs[r].send.pendingData[1]);
          EXPECT_EQ(comm.base.reqs[r].send.lastAttemptQpIndex[0], -1);
          EXPECT_EQ(comm.base.reqs[r].send.lastAttemptQpIndex[1], 0);
        }
      }
    }
  }
}

/**
 * @brief Verifies request-pool pointer lookup returns exact first, interior, and final indices.
 * @note Side effects: None.
 */
TEST(P2pRequestTest, ResolvesRequestPoolIndicesDirectly) {
  SCOPED_TRACE("Request-pool indexing must reflect the request's exact position without mutation.");
  auto comm = MakeSendComm();
  std::vector<unsigned char> commBefore(sizeof(comm));

  for (int expectedIndex : {0, 17, NET_IB_MAX_REQUESTS - 1}) {
    SCOPED_TRACE(testing::Message() << "expectedIndex=" << expectedIndex);
    ncclIbRequest* const request = &comm.base.reqs[expectedIndex];
    request->id = static_cast<uint64_t>(NET_IB_MAX_REQUESTS + expectedIndex);
    request->type = NCCL_NET_IB_REQ_SEND;
    request->events[0] = expectedIndex + 1;
    std::memcpy(commBefore.data(), &comm, sizeof(comm));

    EXPECT_EQ(occlIbRequestPoolIndex(&comm, request), expectedIndex)
        << "Pool indexing must return the exact offset of every valid request pointer";
    EXPECT_EQ(std::memcmp(commBefore.data(), &comm, sizeof(comm)), 0)
        << "Read-only pool indexing must preserve the complete send communicator";
  }
}

/** @brief Verifies replay-plan ownership resets by request-pool index. */
TEST(NetIbP2pTest, ResilientReplayOwnershipResetsByRequestPoolIndex) {
  auto comm = MakeSendComm();
  comm.base.nqps = 2;
  comm.base.hasStandbyPaths = false;
  MultiSendBuffers buffers = {};
  AttachMultiSendBuffers(&comm, &buffers, 2, true);

  constexpr int fifoSlot = 7;
  ncclIbRequest* request = &comm.base.reqs[0];
  comm.multiSendScratch.replayPlanFrozen[0] = 1;
  comm.multiSendScratch.replayPlanFrozen[fifoSlot] = 1;
  occlIbReplayPlanReset(&comm, request);

  EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen[0], 0);
  EXPECT_EQ(comm.multiSendScratch.replayPlanFrozen[fifoSlot], 1);
}

/**
 * @brief Verifies multi-receive size writes use the selected remote device's key.
 * @note Side effects: Repeatedly resets and records process-local verbs fakes.
 */
TEST(NetIbP2pTest, MultiReceiveSizeWriteUsesRemoteDeviceRkey) {
  for (int planner : {0, 1, 2}) {
    ResetVerbsFakes();
    auto comm = MakeSendComm();
    const bool useResiliency = planner == 1;
    const bool useFrm = planner == 2;
    comm.base.vProps.ndevs = 3;
    comm.base.nqps = 1;
    comm.base.nPhysicalQps = 1;
    comm.base.nDataQps = 1;
    comm.base.splitDataOnQps = 0;
    comm.base.qps[0].qp = &verbsQps[0];
    comm.base.qps[0].devIndex = 2;
    comm.base.qps[0].remDevIdx = 0;
    comm.base.qps[0].logicalLane = 0;
    comm.base.activeQps[0] = &comm.base.qps[0];

    MultiSendBuffers buffers = {};
    CIMState congestion = {};
    ncclIbResiliency resiliency = {};
    if (useResiliency) comm.base.resiliency = &resiliency;
    if (useFrm) {
      AttachMultiSendBuffers(&comm, &buffers);
      congestion.nqps = 1;
      comm.cim = &congestion;
    }

    constexpr int slot = 0;
    constexpr int requestCount = 2;
    alignas(64) uint8_t localStorage[requestCount][256] = {};
    ncclIbRequest requests[requestCount] = {};
    comm.ctsFifo[slot][0].nreqs = requestCount;
    comm.remCmplsRecords.addr = 0x300000;
    comm.remCmplsRecords.rkeys[0] = 0xa0;
    comm.remCmplsRecords.rkeys[2] = 0xa2;

    for (int requestIndex = 0; requestIndex < requestCount; requestIndex++) {
      comm.ctsFifo[slot][requestIndex].addr = 0x400000 + requestIndex * 0x1000;
      comm.ctsFifo[slot][requestIndex].rkeys[0] = 0xb0 + requestIndex;
      requests[requestIndex].base = &comm.base;
      requests[requestIndex].id = 0;
      requests[requestIndex].nreqs = requestCount;
      requests[requestIndex].type = NCCL_NET_IB_REQ_SEND;
      requests[requestIndex].send.wrIdToken = 1;
      requests[requestIndex].send.size = sizeof(localStorage[requestIndex]);
      requests[requestIndex].send.data = localStorage[requestIndex];
      requests[requestIndex].send.lkeys[2] = 0xc0 + requestIndex;
      if (useResiliency) requests[requestIndex].send.pendingData[0] = true;
      comm.sendReqs[slot][requestIndex] = &requests[requestIndex];
    }

    ASSERT_EQ(ncclIbMultiSend(&comm, slot), ncclSuccess);
    ASSERT_EQ(postedRanges.size(), 1u);
    EXPECT_EQ(postedRanges[0].immediateRkey, comm.remCmplsRecords.rkeys[0])
        << "planner=" << planner;
  }
}
