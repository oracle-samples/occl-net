/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "p2p.h"
#include "common.h"
#include "compiler.h"
#include "p2p_resiliency.h"

enum ncclIbRequestMatchingScheme {
  BY_INDEX=0,
  BY_ID=1,
};


NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
// By default, use ncclIbRequestMatchingScheme::BY_INDEX matching scheme.
NCCL_PARAM(IbReceiverSideMatchingScheme, "IB_RECEIVER_SIDE_MATCHING_SCHEME", 0);
NCCL_PARAM(FrmSpreadEpsUs, "FRM_SPREAD_EPS_US", 2);
NCCL_PARAM(FrmZeroStampHoldoff, "FRM_ZERO_STAMP_HOLDOFF", 3);

#define FRM_DEBUG 0
#define FRM_LAT_NORM_MTU_BYTES 4096u
#define FRM_LAT_NORM_HALF_MTU_BYTES 2048u

static constexpr int kIbMultiSendAlign = 128;

static inline uint64_t wrid_pack_with_qp(uint64_t req_pack, uint32_t qpIndex) {
  (void)qpIndex;
  // OCCL resiliency already uses the high bits for epoch/slot tracking.
  // Keep the token unchanged and rebuild FRM QP mapping from wc->qp_num.
  return req_pack;
}

static inline int wrid_unpack_qp(uint64_t wr_id, int* has_qp) {
  (void)wr_id;
  if (has_qp) *has_qp = 0;
  return -1;
}

static inline uint64_t ibFrmSpreadEpsUs() {
  static const uint64_t eps = []() -> uint64_t {
    const int64_t value = ncclParamFrmSpreadEpsUs();
    return value < 0 ? 2ULL : (uint64_t)value;
  }();
  return eps;
}

static inline bool ibFrmResiliencyHealthy(const struct ncclIbNetCommBase* base) {
  if (base == NULL || base->resiliency == NULL) return true;
  if (base->resiliency->inProgress) return false;
  for (int qpIndex = 0; qpIndex < base->nqps; qpIndex++) {
    if (base->resiliency->qpInfo[qpIndex].state != ncclIbResiliencyQpStateOk) return false;
  }
  return true;
}

static inline void ibFrmFlushTelemetry(struct ncclIbSendComm* comm) {
  if (comm == NULL) return;

  const bool dbgLatEnabled = (comm->cim != NULL) && (FRM_LAT_PRINT_MS > 0);
  if (dbgLatEnabled) {
    (void)__atomic_add_fetch(&comm->cim->dbg_seq, 1u, __ATOMIC_RELEASE);
  }

  memset(comm->base.frmFields.qp_lat, 0, sizeof(comm->base.frmFields.qp_lat));
  memset(comm->base.frmFields.qp_raw_lat, 0, sizeof(comm->base.frmFields.qp_raw_lat));
  memset(comm->base.frmFields.qp_lat_bytes, 0, sizeof(comm->base.frmFields.qp_lat_bytes));
  memset(comm->base.frmFields.qp_zero_stamp_holdoff, 0, sizeof(comm->base.frmFields.qp_zero_stamp_holdoff));
  for (int qpIndex = 0; qpIndex < comm->base.nqps; qpIndex++) {
    ncclIbQp* qp = &comm->base.qps[qpIndex];
    qp->sig_tail = qp->sig_head;
    qp->sig_req_tail = qp->sig_head;
    memset(qp->sig_tsc, 0, sizeof(qp->sig_tsc));
    memset(qp->sig_bytes, 0, sizeof(qp->sig_bytes));
    memset(qp->sig_req, 0, sizeof(qp->sig_req));
  }

  if (comm->cim != NULL) {
    memset(comm->cim->lw, 0, sizeof(comm->cim->lw));
    memset(comm->cim->latency, 0, sizeof(comm->cim->latency));
    memset(comm->cim->raw_latency, 0, sizeof(comm->cim->raw_latency));
    memset(comm->cim->lat_bytes, 0, sizeof(comm->cim->lat_bytes));
  }

  if (dbgLatEnabled) {
    (void)__atomic_add_fetch(&comm->cim->dbg_seq, 1u, __ATOMIC_RELEASE);
  }
}

static inline bool ibFrmGateUpdate(struct ncclIbSendComm* comm) {
  bool enabled = ncclParamOciFrmEnable() &&
      comm != NULL &&
      comm->base.nqps > 0 &&
      comm->cim != NULL &&
      comm->multiSendScratch.storage != NULL &&
      ibFrmResiliencyHealthy(&comm->base);

  if (comm != NULL) {
    if (!enabled && comm->frmGateOpen) {
      ibFrmFlushTelemetry(comm);
    } else if (enabled && !comm->frmGateOpen) {
      qpMapBuild(&comm->base);
    }
    comm->frmGateOpen = enabled;
  }
  return enabled;
}

static inline bool ibFrmEnabledForSend(struct ncclIbSendComm* comm) {
  return ibFrmGateUpdate(comm);
}

static inline bool ibFrmEnabledForTest(struct ncclIbRequest* req) {
  if (req == NULL || req->base == NULL || !req->base->isSend) return false;
  return ibFrmGateUpdate((struct ncclIbSendComm*)req->base);
}

static inline void ibFrmComputeLw(struct ncclIbSendComm* comm, int nqps, float* lwArr, bool split, int curQpIndex) {
  const int baseN = comm->base.nqps;
  const int nDataQ = comm->base.nDataQps;

  if (comm->cim == NULL) {
    const float eq = 1.0f / (float)nqps;
    for (int q = 0; q < nqps; q++) lwArr[q] = eq;
    return;
  }

  int is_equal = 0;
  if (split) {
    CIM_latencyToWeightsDirect(comm->base.frmFields.qp_lat, nqps, lwArr, &is_equal);
    if (is_equal) {
      return;
    }
    return;
  }

  CIM_computeActiveWindowWeights(comm->base.frmFields.qp_lat, baseN, curQpIndex, nDataQ, lwArr);

  float sumLw = 0.0f;
  for (int i = 0; i < nqps; i++) sumLw += lwArr[i];
  if (sumLw > 0.0f && (sumLw < 0.999f || sumLw > 1.001f)) {
    float inv = 1.0f / sumLw;
    for (int i = 0; i < nqps; i++) {
      lwArr[i] *= inv;
    }
  }
}

static_assert((QP_SIG_RING & (QP_SIG_RING - 1)) == 0, "QP_SIG_RING must be a power of two");

static inline void ibQpSigPush(ncclIbQp* qp, struct ncclIbRequest* req, uint64_t tsc, uint32_t bytes) {
  uint32_t head = qp->sig_head;
  uint32_t tail = qp->sig_tail;
  uint32_t req_tail = qp->sig_req_tail;
  if (__builtin_expect((head - tail) >= QP_SIG_RING, 0)) {
    qp->sig_tail = tail + 1;
  }
  if (__builtin_expect((head - req_tail) >= QP_SIG_RING, 0)) {
    qp->sig_req_tail = req_tail + 1;
  }
  const uint32_t slot = head & (QP_SIG_RING - 1u);
  qp->sig_tsc[slot] = tsc;
  qp->sig_bytes[slot] = bytes;
  qp->sig_req[slot] = req;
  qp->sig_head = head + 1;
}

static inline int ibQpSigPop(ncclIbQp* qp, uint64_t* out_t0, uint32_t* out_bytes) {
  const uint32_t tail = qp->sig_tail;
  if (__builtin_expect(tail == qp->sig_head, 0)) return 0;
  const uint32_t slot = tail & (QP_SIG_RING - 1u);
  *out_t0 = qp->sig_tsc[slot];
  *out_bytes = qp->sig_bytes[slot];
  qp->sig_tail = tail + 1;
  return 1;
}

static inline int ibQpSigPopReq(ncclIbQp* qp, struct ncclIbRequest** out_req) {
  const uint32_t tail = qp->sig_req_tail;
  if (__builtin_expect(tail == qp->sig_head, 0)) return 0;
  const uint32_t slot = tail & (QP_SIG_RING - 1u);
  *out_req = qp->sig_req[slot];
  qp->sig_req[slot] = NULL;
  qp->sig_req_tail = tail + 1;
  return 1;
}

static inline void ibFrmComputeChunksAll(const float* lwArr, int nqps, size_t totalSize, int align, size_t* chunkSize) {
  if (chunkSize == NULL || nqps <= 0) return;

  size_t sum = 0;
  const size_t a = (size_t)align;
  const double totalSizeD = (double)totalSize;
  for (int q = 0; q < nqps; q++) {
    size_t cs = (size_t)(totalSizeD * (double)lwArr[q]);
    if (cs != 0) {
      if (cs < a) {
        cs = a;
      } else {
        cs = DIVUP(cs, a) * a;
      }
      if (cs > totalSize) cs = totalSize;
    }

    chunkSize[q] = cs;
    sum += cs;
  }

  if (sum > totalSize) {
    size_t extra = sum - totalSize;
    for (int q = nqps - 1; q >= 0 && extra > 0; q--) {
      size_t take = (chunkSize[q] > extra) ? extra : chunkSize[q];
      chunkSize[q] -= take;
      extra -= take;
    }
  } else if (sum < totalSize) {
    chunkSize[nqps - 1] += (totalSize - sum);
  }
}

const char* ncclIbReqTypeStr[] = { "Unused", "Send", "Recv", "Flush", "IPut" };

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      r->id = 0;
      r->nreqs = 0;
      r->connectionId = 0;
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->events, 0, sizeof(r->events));
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex) {
  struct ncclIbNetCommDevBase* base = ncclIbGetNetCommDevBase(req->base, devIndex);
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

void ncclIbSendReqsRefreshWrIdToken(struct ncclIbSendComm* comm, int slot, int nreqs) {
  assert(comm != NULL);
  assert(slot >= 0 && slot < NET_IB_MAX_REQUESTS);
  assert(nreqs > 0 && nreqs <= NCCL_NET_IB_MAX_RECVS);

  struct ncclIbRequest* req = comm->sendReqs[slot][0];
  assert(req != NULL);
  uint64_t wrId = ibReqWrIdEncode(req->id);
  for (int r = 0; r < nreqs; r++) {
    assert(comm->sendReqs[slot][r] != NULL);
    comm->sendReqs[slot][r]->send.wrIdToken = wrId;
  }
}


#ifdef ENABLE_TRACE
static ncclResult_t ncclIbPrintWr(struct ibv_send_wr* wr, char* wrStr) {
  if (wr == NULL) {
    sprintf(wrStr, "wr=NULL");
    return ncclSuccess;
  }
  const char* opcodeStr = ibvWrOpcodeStr(wr->opcode);
  switch (wr->opcode) {
    case IBV_WR_RDMA_WRITE:
      sprintf(wrStr, "wr=%p, wr_id=%ld, opcode=%s, num_sge=%d, sge[0].length=%" PRIu32 ", sge[0].addr=0x%016" PRIx64 ", rdma.remote_addr=0x%016" PRIx64 ", rdma.rkey=0x%x",
        wr,
        wr->wr_id,
        opcodeStr,
        wr->num_sge,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->length : 0,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->addr : 0,
        wr->wr.rdma.remote_addr,
        wr->wr.rdma.rkey);
      break;
    case IBV_WR_RDMA_WRITE_WITH_IMM:
      sprintf(wrStr, "wr=%p, wr_id=%ld, opcode=%s, num_sge=%d, sge[0].length=%" PRIu32 ", sge[0].addr=0x%016" PRIx64 ", rdma.remote_addr=0x%016" PRIx64 ",  rdma.rkey=0x%x, imm_data=0x%x",
        wr,
        wr->wr_id,
        opcodeStr,
        wr->num_sge,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->length : 0,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->addr : 0,
        wr->wr.rdma.remote_addr,
        wr->wr.rdma.rkey,
        wr->imm_data);
      break;
    default:
      WARN("NET/IB: %s: No format specified for opcode=%d", __func__, wr->opcode);
      return ncclInternalError;
  }
  return ncclSuccess;
}
#endif // ENABLE_TRACE

// The alignment for IB writes that is required to make LL and LL128 protocols work
#define IB_WRITE_CHUNK_ALIGNMENT 128

static ncclResult_t ibMultiSendBaseline(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  TRACE(NCCL_NET, "NET/IB: %s: Posting a send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs);

  int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
  uint8_t attemptOrder[NCCL_IB_MAX_QPS] = {0};
  if (comm->base.resiliency) {
    for (int r = 0; r < nreqs; r++) {
      memset(reqs[r]->send.lastAttemptQpIndex, 0xff, sizeof(reqs[r]->send.lastAttemptQpIndex));
      memset(reqs[r]->send.lastAttemptOrder, 0, sizeof(reqs[r]->send.lastAttemptOrder));
    }
  }
  uint64_t wr_id = reqs[0]->send.wrIdToken;
  assert(wr_id != 0);
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr->wr_id = wr_id;
#ifdef NCCL_ENABLE_NET_PROFILING
    reqs[r]->pInfo[0].nEventHandles = 0;
#endif

    // Every request is chunked equally across all QPs that are used to transfer
    // the request (in case of a single QP, the chunk is the size of the request).
    // The chunk size of each request determined solely by the send size and the
    // number of QPs used to transfer the request. If the send size is not big
    // enough, starting from some QP there might be no data left to send and the
    // length will be zeroed.
    sge->length = DIVUP(DIVUP(reqs[r]->send.size, nqps), IB_WRITE_CHUNK_ALIGNMENT) * IB_WRITE_CHUNK_ALIGNMENT;
    wr->sg_list = sge;
    wr->num_sge = 1;
  }

  // For ID-based matching scheme, immData carries the request ID.
  // For index-based matching scheme, immData carries the send size:
  // - nreqs == 1
  //      It's the send size.
  // - nreqs > 1
  //      Send size is still sent but receiver ignores it since the sizes are
  //      written to directly to remote completion records array
  uint32_t immData = ncclParamIbReceiverSideMatchingScheme() == BY_ID ? (uint32_t)(reqs[0]->id % UINT32_MAX) : reqs[0]->send.size;

  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    // When Adaptive Routing is enabled, send the bulk of the data first as an
    // RDMA Write.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // Write remote sizes array
      lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot*sizeof(struct ncclIbRequestCompletionRecord);
      lastWr->num_sge = 1;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = htobe32(immData);
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  uint32_t sendOffsets[NCCL_NET_IB_MAX_RECVS] = {0};
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, reqs[0]->id, i, &qp, &qpIndex));

    TRACE(NCCL_NET, "NET/IB: %s: Posting send (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld) on QP (qp_num=%u, devIndex=%d, logicalQpIndex=%d)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs, wr_id, qp->qp->qp_num, qp->devIndex, qpIndex);

    // Skip logical chunks that were already credited by the probing path.
    if (comm->base.resiliency && reqs[0]->send.pendingData[qpIndex] == false) {
      for (int r=0; r<nreqs; r++) {
        comm->wrs[r].sg_list->addr += comm->wrs[r].sg_list->length;
        comm->wrs[r].wr.rdma.remote_addr += comm->wrs[r].sg_list->length;
      }
      TRACE(NCCL_NET, "NET/IB: %s: Skipping logical QP index %d (req=%p, slot=%d) as it was already delivered.", __func__, qpIndex, reqs[0], slot);
      continue;
    }

    int devIndex = qp->devIndex;
    int physicalQpIndex = qp - comm->base.qps;
    assert(physicalQpIndex >= 0 && physicalQpIndex < comm->base.nqps);
    for (int r=0; r<nreqs; r++) {
      // Track this event for completion
      //ncclIbAddEvent(reqs[r], devIndex);

      // Select proper rkey (needed even for 0-size send)
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      // Check the data left to send. If the send is too small, it might be
      // that on the current QP there is no data left to be sent.
      comm->wrs[r].sg_list->length = std::min(reqs[r]->send.size-sendOffsets[r], comm->wrs[r].sg_list->length);
      if (comm->wrs[r].sg_list->length == 0) {
        comm->wrs[r].num_sge = 0;
      } else {
        comm->wrs[r].sg_list->lkey = reqs[r]->send.lkeys[devIndex];
      }
    }

    if (nreqs > 1) {
      // Populating the correct gather information based on the device and
      // slot used.
      // Note that the lkey is already correct from the initialization phase.
      lastWr->sg_list = &(comm->devs[devIndex].sge);
      lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
      lastWr->sg_list[0].length = nreqs*sizeof(int);
      // Populate the correct RKey based on the device used
      lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
    }

    struct ibv_send_wr* bad_wr;
#ifdef NCCL_ENABLE_NET_PROFILING
    // QP profiling loop
    for (int r=0; r<nreqs; r++) {
      // Store the qpIndex for this request
      int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      reqs[r]->pInfo[0].data.type = ncclProfileQp;
      reqs[r]->pInfo[0].data.qp.device = devIndex;
      reqs[r]->pInfo[0].data.qp.wr_id = comm->wrs[r].wr_id;
      reqs[r]->pInfo[0].data.qp.opcode = comm->wrs[r].opcode;
      reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      reqs[r]->pInfo[0].data.qp.length = comm->sges[r].length;
      void* pHandle = reqs[r]->pInfo[0].pHandle;
      NCCLCHECK(ncclProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, pHandle, pluginId, &reqs[r]->pInfo[0].data));
      reqs[r]->pInfo[0].nEventHandles++;
    }
#endif
#ifdef ENABLE_TRACE
    for (int r = 0; r < nreqs; r++) {
      TRACE(NCCL_NET, "NET/IB: %s: Posting send work request on QP (qpn=%u, devIndex=%d, qpIndex=%d) (slot=%d, req[r=%d]=%p)", __func__, qp->qp->qp_num, qp->devIndex, qpIndex, slot, r, reqs[r]);
    }
    int wrIdx = 0;
    char wrStr[1024];
    struct ibv_send_wr* currWr = comm->wrs;
    while (currWr != NULL) {
      NCCLCHECK(ncclIbPrintWr(currWr, wrStr));
      TRACE(NCCL_NET, "NET/IB: %s: slot=%d, wrIdx[%d], %s", __func__, slot, wrIdx, wrStr);
      wrIdx++;
      currWr = currWr->next;
    }
#endif // ENABLE_TRACE
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));
    for (int r = 0; r < nreqs; r++) {
      reqs[r]->send.lastAttemptQpIndex[qpIndex] = physicalQpIndex;
      reqs[r]->send.lastAttemptOrder[qpIndex] = attemptOrder[physicalQpIndex];
    }
    attemptOrder[physicalQpIndex]++;

    // Update the send offset and addresses for the next QP according to the
    // actual data size that was sent on the current QP, for every request
    for (int r=0; r<nreqs; r++) {
      sendOffsets[r] = std::min<uint32_t>(sendOffsets[r] + comm->wrs[r].sg_list->length, reqs[r]->send.size);
      comm->wrs[r].sg_list->addr += comm->wrs[r].sg_list->length;
      comm->wrs[r].wr.rdma.remote_addr += comm->wrs[r].sg_list->length;
      TRACE(NCCL_NET, "NET/IB: %s: Send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, reqIdx=%d, wr_id=%ld) posted %d bytes on logical QP index %d via physical QP index %d (devIndex=%d, qp_num=%u), total posted %d/%d bytes", __func__, reqs[r], reqs[0]->base, reqs[r]->id, slot, nreqs, r, comm->wrs[r].wr_id, comm->wrs[r].sg_list->length, qpIndex, physicalQpIndex, devIndex, qp->qp->qp_num, sendOffsets[r], reqs[r]->send.size);
      reqs[r]->send.pendingData[qpIndex] = true;
    }
  }

  TRACE(NCCL_NET, "NET/IB: %s: Send request posted (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs, wr_id);

  return ncclSuccess;
}

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  static int g_lat_src_set = 0;

  if (!ibFrmEnabledForSend(comm)) return ibMultiSendBaseline(comm, slot);

  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  struct ncclIbMultiSendScratch* scratch = &comm->multiSendScratch;
  if (scratch->storage == NULL || scratch->nqps < comm->base.nqps) {
    return ibMultiSendBaseline(comm, slot);
  }

  TRACE(NCCL_NET, "NET/IB: %s: Posting FRM send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d)",
        __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs);

  const int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
  const bool split = comm->base.splitDataOnQps == 1;
  const int activeBaseIx = reqs[0]->id % comm->base.nqps;
  const bool dbgLatEnabled = (comm->cim != NULL) && (FRM_LAT_PRINT_MS > 0);
  uint8_t attemptOrder[NCCL_IB_MAX_QPS] = {0};
  int physicalQpIndexByPost[NCCL_IB_MAX_QPS] = {0};
#if FRM_DEBUG
  int requestOrderByPost[NCCL_IB_MAX_QPS] = {0};
#endif

  if (comm->base.resiliency) {
    for (int r = 0; r < nreqs; r++) {
      memset(reqs[r]->send.lastAttemptQpIndex, 0xff, sizeof(reqs[r]->send.lastAttemptQpIndex));
      memset(reqs[r]->send.lastAttemptOrder, 0, sizeof(reqs[r]->send.lastAttemptOrder));
    }
  }

  float* lwArr = scratch->lwPerQp;
  bool bypassEqual = false;
  {
    uint64_t minLat = UINT64_MAX;
    uint64_t maxLat = 0;
    const int count = split ? nqps : comm->base.nDataQps;
    for (int i = 0; i < count; i++) {
      int q = split ? i : ((activeBaseIx + i) % comm->base.nqps);
      uint64_t v = comm->base.frmFields.qp_lat[q];
      if (v == 0) continue;
      if (v < minLat) minLat = v;
      if (v > maxLat) maxLat = v;
    }
    if (minLat == UINT64_MAX || (maxLat - minLat) <= ibFrmSpreadEpsUs()) {
      bypassEqual = true;
    }
  }
  const bool useFrmNow = !bypassEqual;

  if (useFrmNow) {
    ibFrmComputeLw(comm, nqps, lwArr, split, activeBaseIx);
  } else {
    const float eq = 1.0f / (float)nqps;
    for (int q = 0; q < nqps; q++) lwArr[q] = eq;
  }

  const uint64_t wr_id = reqs[0]->send.wrIdToken;
  assert(wr_id != 0);
  for (int r = 0; r < nreqs; r++) {
#ifdef NCCL_ENABLE_NET_PROFILING
    reqs[r]->pInfo[0].nEventHandles = 0;
#endif
    if (useFrmNow) {
      ibFrmComputeChunksAll(lwArr, nqps, reqs[r]->send.size, kIbMultiSendAlign,
                            ncclIbMultiSendScratchChunkRow(scratch, r));
    } else {
      scratch->equalChunkSize[r] = DIVUP(DIVUP(reqs[r]->send.size, nqps), kIbMultiSendAlign) * kIbMultiSendAlign;
    }
    scratch->reqOffsetShadow[r] = 0;
    scratch->sgeAddrShadow[r] = (uintptr_t)reqs[r]->send.data;
    scratch->remoteAddrShadow[r] = slots[r].addr;
  }

  const uint32_t immData = ncclParamIbReceiverSideMatchingScheme() == BY_ID ?
      (uint32_t)(reqs[0]->id % UINT32_MAX) : reqs[0]->send.size;
  const uint32_t immDataBe = htobe32(immData);
  const bool needImmWr = (nreqs > 1) || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold());
  const uint64_t remSizesRemoteAddr = comm->remCmplsRecords.addr + slot*sizeof(struct ncclIbRequestCompletionRecord);
  if (nreqs > 1) {
    int* sizes = comm->remCmplsRecords.elems[slot];
    for (int r = 0; r < nreqs; r++) sizes[r] = reqs[r]->send.size;
  }

  int postCount = 0;
  if (dbgLatEnabled) {
    (void)__atomic_add_fetch(&comm->cim->dbg_seq, 1u, __ATOMIC_RELEASE);
  }

  if (nreqs == 1) {
    struct ncclIbRequest* const req = reqs[0];
    size_t* const chunkRow = useFrmNow ? ncclIbMultiSendScratchChunkRow(scratch, 0) : NULL;
    int reqOffset = scratch->reqOffsetShadow[0];
    uintptr_t sgeAddr = scratch->sgeAddrShadow[0];
    uint64_t remoteAddr = scratch->remoteAddrShadow[0];

    for (int i = 0; i < nqps; i++) {
      int logicalQpIndex = -1;
      ncclIbQp* qp = NULL;
      NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, req->id, i, &qp, &logicalQpIndex));
      int physicalQpIndex = qp - comm->base.qps;
      assert(physicalQpIndex >= 0 && physicalQpIndex < comm->base.nqps);

      size_t remaining = (size_t)(req->send.size - reqOffset);
      size_t cs = !useFrmNow ? (size_t)scratch->equalChunkSize[0] : (split ? chunkRow[logicalQpIndex] : chunkRow[i]);
      if (cs > remaining) cs = remaining;
      const int length = (int)cs;
      const size_t sendSz = (size_t)(length > 0 ? length : 0);
      const uintptr_t curSgeAddr = sgeAddr;
      const uint64_t curRemoteAddr = remoteAddr;
      reqOffset += (int)sendSz;
      sgeAddr += sendSz;
      remoteAddr += sendSz;

      if (comm->base.resiliency && req->send.pendingData[logicalQpIndex] == false) {
        continue;
      }

      struct ibv_send_wr* postWrs = ncclIbMultiSendScratchWrRow(scratch, postCount);
      struct ibv_sge* postSges = ncclIbMultiSendScratchSgeRow(scratch, postCount);
      struct ibv_send_wr* postImmWr = &postWrs[NCCL_NET_IB_MAX_RECVS];
      size_t* sendSzRow = ncclIbMultiSendScratchSendSizeRow(scratch, postCount);
      struct ibv_send_wr* signaledWr = &postWrs[0];
      const uint64_t wridForPost = wrid_pack_with_qp(wr_id, (uint32_t)logicalQpIndex);
      const int devIndex = qp->devIndex;

      scratch->qpPerPost[postCount] = qp;
      scratch->qpIndexPerPost[postCount] = logicalQpIndex;
      physicalQpIndexByPost[postCount] = physicalQpIndex;
#if FRM_DEBUG
      requestOrderByPost[postCount] = i;
#endif
      scratch->chainBytesPerPost[postCount] = (sendSz > (size_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)sendSz;
      sendSzRow[0] = sendSz;

      signaledWr->wr.rdma.rkey = slots[0].rkeys[qp->remDevIdx];
      signaledWr->wr.rdma.remote_addr = curRemoteAddr;
      signaledWr->imm_data = immDataBe;
      postSges[0].addr = curSgeAddr;

      if (needImmWr) {
        signaledWr->wr_id = wridForPost;
        signaledWr->opcode = IBV_WR_RDMA_WRITE;
        signaledWr->send_flags = 0;
        signaledWr->next = postImmWr;

        postImmWr->wr_id = wridForPost;
        postImmWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        postImmWr->imm_data = immDataBe;
        postImmWr->next = NULL;
        postImmWr->send_flags = IBV_SEND_SIGNALED;
        postImmWr->sg_list = NULL;
        postImmWr->num_sge = 0;
        postImmWr->wr.rdma.rkey = 0;
        postImmWr->wr.rdma.remote_addr = 0;
      } else {
        signaledWr->wr_id = wridForPost;
        signaledWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        signaledWr->next = NULL;
        signaledWr->send_flags = IBV_SEND_SIGNALED;
      }

      if (length <= 0) {
        signaledWr->sg_list = NULL;
        signaledWr->num_sge = 0;
      } else {
        postSges[0].lkey = req->send.lkeys[devIndex];
        postSges[0].length = length;
        signaledWr->sg_list = postSges;
        signaledWr->num_sge = 1;
      }

#ifdef NCCL_ENABLE_NET_PROFILING
      int nEventHandles = req->pInfo[0].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      req->pInfo[0].qpIndex[nEventHandles] = logicalQpIndex;
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      req->pInfo[0].data.type = ncclProfileQp;
      req->pInfo[0].data.qp.device = devIndex;
      req->pInfo[0].data.qp.wr_id = signaledWr->wr_id;
      req->pInfo[0].data.qp.opcode = signaledWr->opcode;
      req->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      req->pInfo[0].data.qp.length = (uint32_t)sendSzRow[0];
      void* pHandle = req->pInfo[0].pHandle;
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart,
                                     pHandle, pluginId, &req->pInfo[0].data));
      req->pInfo[0].nEventHandles++;
#endif

      if (dbgLatEnabled) {
        const float postLw = split ? lwArr[logicalQpIndex] : lwArr[i];
        comm->cim->lw[logicalQpIndex] = postLw;
        comm->cim->latency[logicalQpIndex] = comm->base.frmFields.qp_lat[logicalQpIndex];
        comm->cim->raw_latency[logicalQpIndex] = comm->base.frmFields.qp_raw_lat[logicalQpIndex];
        comm->cim->lat_bytes[logicalQpIndex] = comm->base.frmFields.qp_lat_bytes[logicalQpIndex];
      }

      postCount++;
    }
  } else {
    for (int i = 0; i < nqps; i++) {
      int logicalQpIndex = -1;
      ncclIbQp* qp = NULL;
      NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, reqs[0]->id, i, &qp, &logicalQpIndex));
      int physicalQpIndex = qp - comm->base.qps;
      assert(physicalQpIndex >= 0 && physicalQpIndex < comm->base.nqps);

      const bool skipPost = comm->base.resiliency && reqs[0]->send.pendingData[logicalQpIndex] == false;
      for (int r = 0; r < nreqs; r++) {
        uintptr_t curSgeAddr = scratch->sgeAddrShadow[r];
        uint64_t curRemoteAddr = scratch->remoteAddrShadow[r];
        size_t remaining = (size_t)(reqs[r]->send.size - scratch->reqOffsetShadow[r]);
        size_t cs = !useFrmNow ?
            (size_t)scratch->equalChunkSize[r] :
            ncclIbMultiSendScratchChunkRow(scratch, r)[split ? logicalQpIndex : i];
        if (cs > remaining) cs = remaining;
        const int length = (int)cs;
        const size_t sendSz = (size_t)(length > 0 ? length : 0);

        scratch->reqOffsetShadow[r] += (int)sendSz;
        scratch->sgeAddrShadow[r] += sendSz;
        scratch->remoteAddrShadow[r] += sendSz;
      }

      if (skipPost) continue;

      struct ibv_send_wr* postWrs = ncclIbMultiSendScratchWrRow(scratch, postCount);
      struct ibv_sge* postSges = ncclIbMultiSendScratchSgeRow(scratch, postCount);
      struct ibv_send_wr* postImmWr = &postWrs[NCCL_NET_IB_MAX_RECVS];
      size_t* sendSzRow = ncclIbMultiSendScratchSendSizeRow(scratch, postCount);
      size_t chainBytes = 0;
      struct ibv_send_wr* lastDataWr = NULL;
      int liveDataWrCount = 0;
#ifdef NCCL_ENABLE_NET_PROFILING
      int dataWrSlotByReq[NCCL_NET_IB_MAX_RECVS];
      for (int r = 0; r < nreqs; r++) dataWrSlotByReq[r] = -1;
#endif

      for (int r = 0; r < nreqs; r++) {
        uintptr_t curSgeAddr = scratch->sgeAddrShadow[r];
        uint64_t curRemoteAddr = scratch->remoteAddrShadow[r];
        size_t remaining = (size_t)(reqs[r]->send.size - scratch->reqOffsetShadow[r]);
        size_t cs = !useFrmNow ?
            (size_t)scratch->equalChunkSize[r] :
            ncclIbMultiSendScratchChunkRow(scratch, r)[split ? logicalQpIndex : i];
        if (cs > remaining) cs = remaining;
        const int length = (int)cs;
        const size_t sendSz = (size_t)(length > 0 ? length : 0);

        sendSzRow[r] = sendSz;
        chainBytes += sendSz;
        if (length <= 0) continue;

        const int wrSlot = liveDataWrCount++;
        struct ibv_send_wr* postWr = &postWrs[wrSlot];
        struct ibv_sge* postSge = &postSges[wrSlot];
#ifdef NCCL_ENABLE_NET_PROFILING
        dataWrSlotByReq[r] = wrSlot;
#endif

        postWr->wr_id = wrid_pack_with_qp(wr_id, (uint32_t)logicalQpIndex);
        postWr->opcode = IBV_WR_RDMA_WRITE;
        postWr->send_flags = 0;
        postWr->next = NULL;
        postWr->wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];
        postWr->wr.rdma.remote_addr = curRemoteAddr;
        postSge->addr = curSgeAddr;
        postSge->lkey = reqs[r]->send.lkeys[qp->devIndex];
        postSge->length = length;
        postWr->sg_list = postSge;
        postWr->num_sge = 1;

        if (lastDataWr != NULL) lastDataWr->next = postWr;
        lastDataWr = postWr;
      }

      scratch->qpPerPost[postCount] = qp;
      scratch->qpIndexPerPost[postCount] = logicalQpIndex;
      physicalQpIndexByPost[postCount] = physicalQpIndex;
#if FRM_DEBUG
      requestOrderByPost[postCount] = i;
#endif
      scratch->chainBytesPerPost[postCount] = (chainBytes > (size_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)chainBytes;

      if (lastDataWr != NULL) lastDataWr->next = postImmWr;
      postImmWr->wr_id = wrid_pack_with_qp(wr_id, (uint32_t)logicalQpIndex);
      postImmWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      postImmWr->imm_data = immDataBe;
      postImmWr->next = NULL;
      postImmWr->send_flags = IBV_SEND_SIGNALED;
      postImmWr->sg_list = &(comm->devs[qp->devIndex].sge);
      postImmWr->num_sge = 1;
      postImmWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
      postImmWr->sg_list[0].length = nreqs*sizeof(int);
      postImmWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[qp->devIndex];
      postImmWr->wr.rdma.remote_addr = remSizesRemoteAddr;

#ifdef NCCL_ENABLE_NET_PROFILING
      for (int r = 0; r < nreqs; r++) {
        int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
        assert(nEventHandles < MAX_QPS_PER_REQ);
        reqs[r]->pInfo[0].qpIndex[nEventHandles] = logicalQpIndex;
        int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
        int wrSlot = dataWrSlotByReq[r];
        reqs[r]->pInfo[0].data.type = ncclProfileQp;
        reqs[r]->pInfo[0].data.qp.device = qp->devIndex;
        reqs[r]->pInfo[0].data.qp.wr_id = (wrSlot >= 0) ? postWrs[wrSlot].wr_id : 0ULL;
        reqs[r]->pInfo[0].data.qp.opcode = (wrSlot >= 0) ? postWrs[wrSlot].opcode : IBV_WR_RDMA_WRITE;
        reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
        reqs[r]->pInfo[0].data.qp.length = (uint32_t)sendSzRow[r];
        void* pHandle = reqs[r]->pInfo[0].pHandle;
        NCCLCHECK(ncclProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart,
                                       pHandle, pluginId, &reqs[r]->pInfo[0].data));
        reqs[r]->pInfo[0].nEventHandles++;
      }
#endif

      if (dbgLatEnabled) {
        const float postLw = split ? lwArr[logicalQpIndex] : lwArr[i];
        comm->cim->lw[logicalQpIndex] = postLw;
        comm->cim->latency[logicalQpIndex] = comm->base.frmFields.qp_lat[logicalQpIndex];
        comm->cim->raw_latency[logicalQpIndex] = comm->base.frmFields.qp_raw_lat[logicalQpIndex];
        comm->cim->lat_bytes[logicalQpIndex] = comm->base.frmFields.qp_lat_bytes[logicalQpIndex];
      }

      postCount++;
    }
  }

  if (dbgLatEnabled) {
    (void)__atomic_add_fetch(&comm->cim->dbg_seq, 1u, __ATOMIC_RELEASE);
  }

  struct ibv_send_wr* bad_wr = NULL;
  uint64_t spt0 = timer_get_cycles();
  for (int p = 0; p < postCount; p++) {
    ncclIbQp* qp = scratch->qpPerPost[p];
    int logicalQpIndex = scratch->qpIndexPerPost[p];
    int physicalQpIndex = physicalQpIndexByPost[p];
    struct ibv_send_wr* postWrs = ncclIbMultiSendScratchWrRow(scratch, p);
    struct ibv_send_wr* postImmWr = &postWrs[NCCL_NET_IB_MAX_RECVS];
    struct ibv_send_wr* postHeadWr = (scratch->chainBytesPerPost[p] == 0) ? postImmWr : &postWrs[0];

    ibQpSigPush(qp, reqs[0], spt0, scratch->chainBytesPerPost[p]);
    NCCLCHECK(wrap_ibv_post_send(qp->qp, postHeadWr, &bad_wr));

    if (comm->base.resiliency) {
      for (int r = 0; r < nreqs; r++) {
        reqs[r]->send.lastAttemptQpIndex[logicalQpIndex] = physicalQpIndex;
        reqs[r]->send.lastAttemptOrder[logicalQpIndex] = attemptOrder[physicalQpIndex];
        reqs[r]->send.pendingData[logicalQpIndex] = true;
      }
      attemptOrder[physicalQpIndex]++;
    }

#if FRM_DEBUG
    size_t* sendSzRow = ncclIbMultiSendScratchSendSizeRow(scratch, p);
    for (int r = 0; r < nreqs; r++) {
      const float postLw = split ? lwArr[logicalQpIndex] : lwArr[requestOrderByPost[p]];
      printf("FRM post: qp=%d phys=%d request=%d send=%zu lw=%.6f lat=%lu raw_lat=%lu\n",
             logicalQpIndex, physicalQpIndex, r, sendSzRow[r], postLw,
             comm->base.frmFields.qp_lat[logicalQpIndex], comm->base.frmFields.qp_raw_lat[logicalQpIndex]);
      fflush(stdout);
    }
#endif
  }

  if (dbgLatEnabled && !g_lat_src_set) {
    g_lat_src_set = 1;
    CIM_DebugLatPrinterSetWeightSource("lw", (volatile float*)comm->cim->lw, comm->base.nqps);
    CIM_DebugLatPrinterSetSource("qp_lat(us)", (volatile uint64_t*)comm->cim->latency,
                                 (volatile uint64_t*)comm->cim->raw_latency, comm->base.nqps);
    CIM_DebugLatPrinterSetByteSource("bytes", (volatile uint32_t*)comm->cim->lat_bytes, comm->base.nqps);
    CIM_DebugLatPrinterSetSyncSeq(&comm->cim->dbg_seq);
  }

  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIsend() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  slots = comm->ctsFifo[slot];
  uint64_t idx = comm->base.fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  std::atomic_thread_fence(std::memory_order_seq_cst); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    if (size > slots[r].size) size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %ld addr %lx rkeys[0]=%x",
        r, nreqs, tag, ncclSocketToString(&addr, line), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
      return ncclInternalError;
    }

    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->id = comm->base.fifoHead;
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.wrIdToken = 0;
    req->send.size = size;
    req->send.data = data;
    if (comm->base.resiliency) {
      memset(req->send.pendingData, 0, sizeof(req->send.pendingData));
      memset(req->send.lastAttemptQpIndex, 0xff, sizeof(req->send.lastAttemptQpIndex));
      memset(req->send.lastAttemptOrder, 0, sizeof(req->send.lastAttemptOrder));
      memset(req->send.ackedCompletions, 0, sizeof(req->send.ackedCompletions));
    }
#ifdef NCCL_ENABLE_NET_PROFILING
    req->pInfo[0].pHandle = phandle;
#endif

    // Populate events
    int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
    int qpIndex = -1;
    ncclIbQp* qp = NULL;
    for (int i = 0; i < nqps; i++) {
      NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, req->id, i, &qp, &qpIndex));
      ncclIbAddEvent(req, qp->devIndex);
      if (comm->base.resiliency) req->send.pendingData[qpIndex] = true;
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    // In case the sender will write the size of the send directly to the
    // receiver's memory, prepare the source buffer which will hold the sizes
    // and be sent to the receiver.
    comm->remCmplsRecords.elems[slot][r] = req->send.size;

    TRACE(NCCL_NET, "NET/IB: %s: Send request created (req=%p, comm=%p, id=%ld, slot=%d, reqIdx=%d, nreqs=%d, tag=%x, size=%ld, data=0x%016" PRIx64 ", mhandle=%p, size=%ld)", __func__, req, req->base, req->id, slot, r, nreqs, tag, size, (uint64_t)data, mhandle, size);

    *request = reqs[r] = req;

    comm->sendReqsCnt[slot]++;
    // If this is a multi-recv, send only when all requests have matched.
    if (comm->sendReqsCnt[slot] < nreqs) return ncclSuccess;

    ncclIbSendReqsRefreshWrIdToken(comm, slot, nreqs);
    TIME_START(0);
    NCCLCHECK(ncclIbMultiSend(comm, slot));

    comm->base.fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot) {
  ncclIbQp* ctsQp = NULL;;
  NCCLCHECK(ncclIbRecvCommGetQpForCts(comm, req->id, &ctsQp));

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr.rdma.remote_addr = comm->remCtsFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct rkey
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;

  // Populating the correct gather information based on the device and user
  // provided information
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
  wr.sg_list = &(comm->devs[ctsQp->devIndex].sge);
  wr.sg_list[0].addr = (uint64_t)localElem;
  wr.sg_list[0].length = req->nreqs*sizeof(struct ncclIbSendFifo);
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remCtsFifo.flags; // IBV_SEND_INLINE

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  // slot == devIndex - When writing to CTS FIFO slot N, and this QP lives on device index N, it should send signalled.
  // This works out that each CTS posting QP gets drained
  if (slot == ctsQp->devIndex || comm->base.resiliency) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = ibReqWrIdEncode(req->id);
  }

  TRACE(NCCL_NET, "NET/IB: %s: Posting a CTS (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d, qp_num=%u)", __func__, req, req->base, req->id, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags, ctsQp->qp->qp_num);

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));

  TRACE(NCCL_NET, "NET/IB: %s: CTS posted (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d, qp_num=%u)", __func__, req, req->base, req->id, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags, ctsQp->qp->qp_num);

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIrecv() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  req->id = comm->base.fifoHead;
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;
  if (comm->base.resiliency) {
    // When resiliency is enabled, a recv request can be served by any device.
    for (int devIndex = 0; devIndex < comm->base.vProps.ndevs; devIndex++) {
      req->devBases[devIndex] = ncclIbGetNetCommDevBase(&comm->base, devIndex);
    }
  }
  TRACE(NCCL_NET, "NET/IB: %s: Recv request created (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, tag[0]=%x)", __func__, req, req->base, req->id, slot, n, tags[0]);

#ifdef NCCL_ENABLE_NET_PROFILING
  for (int r = 0; r < n && phandles; r++) req->pInfo[r].nEventHandles = 0;
#endif

  // Store the request in a table for easy retrieval by ID.
  comm->recvReqs[req->id % NET_IB_MAX_REQUESTS] = req;

  TIME_START(1);
  const int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, req->id, i, &qp, &qpIndex));
    ncclIbAddEvent(req, qp->devIndex);
    if (comm->prepostReceiveWorkRequests) {
      continue;
    }
    // Post receive work request on the QP
    comm->ibRecvWorkRequest.wr_id = ibReqWrIdEncode(req->id);
    NCCLCHECK(ncclIbPostRecvWorkRequest(qp->qp, &comm->ibRecvWorkRequest));
#ifdef NCCL_ENABLE_NET_PROFILING
    // Start a QP event for every request in the multirecv and every qp
    for (int r = 0; r < n; r++) {
      int nEventHandles = req->pInfo[r].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      req->pInfo[r].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      req->pInfo[r].data.type = ncclProfileQp;
      req->pInfo[r].data.qp.device = qp->devIndex;
      req->pInfo[r].data.qp.wr_id = comm->ibRecvWorkRequest.wr_id;
      req->pInfo[r].data.qp.qpNum = qp->qp->qp_num;
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[r].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, phandles[r], pluginId, &req->pInfo[r].data));
      req->pInfo[r].nEventHandles++;
    }
#endif
  }
  TIME_STOP(1);

  req->recv.aggSize = 0;
  req->recv.cmplsRecords = &comm->cmplsRecords[slot];
  memset(req->recv.cmplsRecords->sizes, 0, sizeof(int)*n);
  memset(req->recv.cmplsRecords->completions, 0, sizeof(req->recv.cmplsRecords->completions));
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];
    // Send all applicable rkeys
    for (int j = 0; j < comm->base.vProps.ndevs; j++) {
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;
    }
    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->base.fifoHead+1;
  }

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, req, slot));
  comm->base.fifoHead++;
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*) mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (req - comm->base.reqs) + NCCL_IB_FLUSH_REQ_WR_ID_OFFSET;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    ncclIbAddEvent(req, i);

    TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
  }

  *request = req;
  return ncclSuccess;
}

#define HCA_NAME(req, index) ((req)->devBases[(index)]->pd->context->device->name)

#ifdef NCCL_ENABLE_NET_PROFILING
static int getReqQpIndex(struct ncclIbRequest* req, int request, int qpNumber) {
  for (int i = 0; i < MAX_QPS_PER_REQ; i++) {
    int qpIndex = req->pInfo[request].qpIndex[i];
    if (req->base->qps[qpIndex].qp->qp_num == qpNumber) return i;
  }
  return 0;
}
#endif

static inline struct ncclIbRequest* ibSendReqPopByQp(struct ncclIbSendComm* sendComm, uint32_t qpNum, int devIndex) {
  if (sendComm == NULL || !sendComm->frmGateOpen) return NULL;

  ncclIbQp* qp = NULL;
  if (ncclIbCommBaseGetQpByQpNum(&sendComm->base, devIndex, qpNum, &qp, NULL) != ncclSuccess || qp == NULL) {
    return NULL;
  }

  struct ncclIbRequest* req = NULL;
  if (!ibQpSigPopReq(qp, &req)) return NULL;
  return req;
}

static inline ncclResult_t ncclIbRequestRetrieveFromCompletion(struct ncclIbNetCommBase* base, ibv_wc* wc, int devIndex, ncclIbRequest** req) {
  assert(req != NULL);
  assert(wc != NULL);

  // In case of a completion with error, there is no guarantee that all fields
  // of the completion are valid.
  assert(wc->status == IBV_WC_SUCCESS);

  TRACE(NCCL_NET, "NET/IB: %s: Retrieving a %s request (wr_id=%ld, opcode=%s)", __func__, base->isSend ? "send" : "recv", wc->wr_id, ibvWcOpcodeStr(wc->opcode));

  if (!base->isSend && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM && ncclParamIbReceiverSideMatchingScheme() == BY_ID) {
    TRACE(NCCL_NET, "NET/IB: %s: Retrieving a receive request (wr_id=%ld, opcode=%s, imm_data=%d, byte_len=%d)", __func__, wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len);
    struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)base;
    uint32_t immData = be32toh(wc->imm_data);
    *req = recvComm->recvReqs[immData % NET_IB_MAX_REQUESTS];
    if (*req != NULL && ((*req)->id % UINT32_MAX) != immData) {
      *req = NULL;
    }
  } else if (!base->isSend && wc->opcode == IBV_WC_RDMA_READ) { // Flush request completion
    NCCLCHECK(ncclIbRequestRetrieveAsIndex(base->reqs, (wc->wr_id - NCCL_IB_FLUSH_REQ_WR_ID_OFFSET), req));
  } else if (!base->isSend && ibReqWrIdIsRequest(wc->wr_id)) {
    struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)base;
    *req = ibRecvReqFind(recvComm, ibReqWrIdDecode(wc->wr_id));
  } else if (base->isSend) {
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)base;
    *req = ibSendReqPopByQp(sendComm, wc->qp_num, devIndex);
    if (*req == NULL) *req = ibSendReqFindByWrId(sendComm, wc->wr_id);
  } else {
    *req = NULL;
  }
  if (*req != NULL) {
    TRACE(NCCL_NET, "NET/IB: %s: Retrieved a %s request (req=%p, comm=%p, id=%ld, type=%s, wc.wr_id=%ld, wc.opcode=%s, wc.imm_data=%d, wc.byte_len=%d, wc.qp_num=%u)", __func__, base->isSend ? "send" : "recv", *req, (*req)->base, (*req)->id, ncclIbReqTypeStr[(*req)->type], wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len, wc->qp_num);
  } else {
    TRACE(NCCL_NET, "NET/IB: %s: No %s request matched completion (wr_id=%ld, opcode=%s, imm_data=%d, byte_len=%d, qp_num=%u)", __func__, base->isSend ? "send" : "recv", wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len, wc->qp_num);
  }
  return ncclSuccess;
}

static inline bool ncclIbRequestIsComplete(struct ncclIbRequest *request) {
  bool complete = (request->events[0] == 0 && request->events[1] == 0 && request->events[2] == 0 && request->events[3] == 0);
  if (!complete && request->base->resiliency) {
    NCCLCHECK(ncclIbResiliencyRequestIsComplete(request, &complete));
  }
  return complete;
}

static inline ncclResult_t ncclIbRequestComplete(struct ncclIbRequest* r, int* done, int* sizes) {
  TRACE(NCCL_NET, "NET/IB: %s: %s request completed (req=%p, comm=%p, id=%ld, type=%s)", __func__, r->base->isSend ? "Send" : "Recv", r, r->base, r->id, ncclIbReqTypeStr[r->type]);
  *done = 1;
  if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
    TRACE(NCCL_NET, "NET/IB: %s: Recv request completed (req=%p, comm=%p, id=%ld, type=%s, nreqs=%d)", __func__, r, r->base, r->id, ncclIbReqTypeStr[r->type], r->nreqs);
    int *sizesToReport = (r->nreqs > 1 || r->recv.cmplsRecords->sizes[0] > 0) ? r->recv.cmplsRecords->sizes : &(r->recv.aggSize);
    for (int i=0; i<r->nreqs; i++) {
      sizes[i] = sizesToReport[i];
#ifdef NCCL_ENABLE_NET_PROFILING
      for (int j = 0; j < r->pInfo[i].nEventHandles; j++) {
        NCCLCHECK(ncclProfilerFunction(&r->pInfo[i].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
      }
#endif
    }
  }
  if (r->type == NCCL_NET_IB_REQ_SEND) {
    TRACE(NCCL_NET, "NET/IB: %s: Send request completed (req=%p, comm=%p, id=%ld)", __func__, r, r->base, r->id);
    if (sizes) {
      sizes[0] = r->send.size;
  #ifdef NCCL_ENABLE_NET_PROFILING
      for (int j = 0; j < r->pInfo[0].nEventHandles; j++) {
        NCCLCHECK(ncclProfilerFunction(&r->pInfo[0].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
      }
  #endif
    }
    int slot = r->id % NET_IB_MAX_REQUESTS;
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)r->base;
    sendComm->sendReqsCnt[slot]--;
    if (sendComm->sendReqsCnt[slot] == 0) {
      // Only after completing the last send of a multi-recv, allow accepting
      // following send requests on the same slot.
      memset(&sendComm->sendReqs[slot], 0, sizeof(sendComm->sendReqs[slot]));
    }
    if (r->base->resiliency) {
      NCCLCHECK(ncclIbResiliencySendRequestComplete(r->base->resiliency, r));
    }
  }
  // Stop all remaining Qp events for this event
  NCCLCHECK(ncclIbFreeRequest(r));
  return ncclSuccess;
}

// Log the details of a completion with error. The provided devIndex is the index
// of the IB device on which the completion was received.
static ncclResult_t ncclIbLogCompletionWithError(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(commBase, devIndex);
  char localGidString[INET6_ADDRSTRLEN] = "";
  char remoteGidString[INET6_ADDRSTRLEN] = "";
  const char* localGidStr = NULL, *remoteGidStr = NULL;
  if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
    remoteGidStr = ibvGetGidStr(&commBase->remDevs[devIndex].remoteGid, remoteGidString, sizeof(remoteGidString));
  }

  char sockStr[SOCKET_NAME_MAXLEN+1];
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);
  ncclSocketToString(&addr, sockStr);
  char *hcaName = devBase->pd->context->device->name;
  WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) vendor_err=%u %s%s%s%s hca %s",
      sockStr, ibvWcStatusStr(wc->status), wc->status,
      ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->vendor_err,
      localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
  return ncclSuccess;
}

static inline ncclResult_t ncclIbCompletionEventProcess(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);

  struct ncclIbRequest* req = NULL;
  NCCLCHECK(ncclIbRequestRetrieveFromCompletion(commBase, wc, devIndex, &req));
  if (req == NULL) {
    if (commBase->resiliency != NULL) {
      INFO(NCCL_NET, "NET/IB: %s: Ignoring stale successful completion with no live request binding (comm=%p, wc.wr_id=%ld, opcode=%d, qp_num=%u).",
           __func__, commBase, wc->wr_id, wc->opcode, wc->qp_num);
      return ncclSuccess;
    }
    WARN("NET/IB: %s: %s comm could not retreive a request found for a successful completion (comm=%p, wc.wr_id=%ld, opcode=%d, qp_num=%u)", __func__, commBase->isSend ? "Send" : "Recv", commBase, wc->wr_id, wc->opcode, wc->qp_num);
    return ncclInternalError;
  }

  #ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d,%d,%d}, devIndex=%d",
    ncclSocketToString(&addr, line), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], req->events[2], req->events[3], devIndex);
  #endif

  if (commBase->isSend) {
    if (req->type != NCCL_NET_IB_REQ_SEND) {
      WARN("NET/IB: %s: Sender expected a 'send' request but got '%s' (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u)", __func__, ncclIbReqTypeStr[req->type], req, commBase, req->id, wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
      return ncclInternalError;
    }
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)commBase;
    struct ncclIbRequest* sendReq = NULL;
    int slot = req->id % NET_IB_MAX_REQUESTS;
    for (int j = 0; j < req->nreqs; j++) {
      sendReq = sendComm->sendReqs[slot][j];
      if (!commBase->resiliency && (sendReq->events[devIndex] <= 0)) {
        WARN("NET/IB: sendReq(%p)->events={%d,%d,%d,%d}, devIndex=%d, reqIdx=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], sendReq->events[2], sendReq->events[3], devIndex, j);
        return ncclInternalError;
      }
      sendReq->events[devIndex]--;
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a send request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, sendReq, sendReq->base, sendReq->id, devIndex, wc->qp_num);
#ifdef NCCL_ENABLE_NET_PROFILING
      // Stop Qp event for sendReq
      int qpIndex = getReqQpIndex(sendReq, j, wc->qp_num);
      NCCLCHECK(ncclProfilerFunction(&sendReq->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
#endif
    }
  } else {
    if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      if (req->type == NCCL_NET_IB_REQ_UNUSED && commBase->resiliency) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver got a completion for a data transfer but retrieved an 'unused' request (req=%p, comm=%p, id=%ld, wc.status=%s(%d), wc.wr_id=%ld, wc.imm_data=%d, wc.opcode=%s(%d), wc.qp_num=%u)", __func__, req, commBase, req->id, ibvWcStatusStr(wc->status), wc->status, wc->wr_id, be32toh(wc->imm_data), ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
        return ncclSuccess;
      }
      if (req->type != NCCL_NET_IB_REQ_RECV && !commBase->resiliency) {
        WARN("NET/IB: %s: Receiver expected a 'recv' request but got '%s' (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.status=%s(%d) wc.opcode=%s(%d), wc.qp_num=%u)", __func__, ncclIbReqTypeStr[req->type], req, req->base, req->id, wc->wr_id, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
        return ncclInternalError;
      }
      if (req->nreqs == 1) {
        if (ncclParamIbReceiverSideMatchingScheme() == BY_INDEX) {
          req->recv.cmplsRecords->sizes[0] = be32toh(wc->imm_data);
        } else if (req->recv.cmplsRecords->sizes[0] == 0) {
          req->recv.aggSize+= wc->byte_len;
        }
      }
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a recv request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)commBase;
      if (recvComm->prepostReceiveWorkRequests) {
        // Post another receive work request on the QP
        ncclIbQp* qp = NULL;
        int qpIndex = -1;
        NCCLCHECK(ncclIbCommBaseGetQpByQpNum(commBase, devIndex, wc->qp_num, &qp, &qpIndex));
        if (req->recv.cmplsRecords->completions[qpIndex] != UINT16_MAX) {
          req->recv.cmplsRecords->completions[qpIndex]++;
        }
        ncclIbPostRecvWorkRequest(qp->qp, &recvComm->ibRecvWorkRequest);
      }
      req->events[devIndex]--;
    } else if (wc->opcode == IBV_WC_RDMA_READ) {
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a flush request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      req->events[devIndex]--;
    } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
      // This is a CTS completion
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a CTS (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      if (req->type == NCCL_NET_IB_REQ_UNUSED) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver got a completion for a CTS but retrieved an 'unused' request (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u, wc.imm_data=%d)", __func__, req, req->base, req->id, wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num, be32toh(wc->imm_data));
        return ncclSuccess;
      }
    } else {
      WARN("NET/IB: %s: Unknown completion (req=%p, comm=%p, id=%ld, devIndex=%d, req->type=%s, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u, wc.imm_data=%d)", __func__, req, commBase, req ? req->id : -1, devIndex, ncclIbReqTypeStr[req->type], wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num, be32toh(wc->imm_data));
      return ncclInternalError;
    }
#ifdef NCCL_ENABLE_NET_PROFILING
    // Stop Qp event for workFifo
    for (int j = 0; j < req->nreqs; j++) {
      int qpIndex = getReqQpIndex(req, j, wc->qp_num);
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
    }
#endif
  }
  return ncclSuccess;
}

// Fabric constant propagation delay (us)
#define FABRIC_PROP_DELAY_US 7ULL

static inline uint64_t norm_lat_us_per_mtu(uint64_t raw_us, uint32_t bytes) {
  uint64_t adj_us = (raw_us > FABRIC_PROP_DELAY_US) ? (raw_us - FABRIC_PROP_DELAY_US) : FABRIC_PROP_DELAY_US;

  // Tail fragments can otherwise explode adj_us*mtu/bytes and get misread as congestion.
  if (bytes < FRM_LAT_NORM_HALF_MTU_BYTES) return adj_us;
  return ((adj_us * (uint64_t)FRM_LAT_NORM_MTU_BYTES) / (uint64_t)bytes) + 1;
}

static inline void ibFrmUpdateLatencyStamp(struct ncclIbRequest* req, struct ibv_wc* wc, bool captureRawLat) {
  if (req == NULL || req->base == NULL || wc == NULL) return;
  if (wc->opcode != IBV_WC_RDMA_WRITE) return;

  int has_qp_flag = 0;
  int qpIndex = wrid_unpack_qp(wc->wr_id, &has_qp_flag);
  if (!has_qp_flag) {
    qpIndex = qpMapFind(&req->base->frmFields.qpHash, wc->qp_num);
  }
  if ((unsigned)qpIndex >= (unsigned)req->base->nqps) return;

  static const uint8_t kZeroStampHoldoff = []() -> uint8_t {
    const int64_t value = ncclParamFrmZeroStampHoldoff();
    return (uint8_t)(value <= 0 ? 0 : (value >= 255 ? 255 : value));
  }();

  ncclIbQp* qp = &req->base->qps[qpIndex];
  uint64_t t0 = 0;
  uint32_t bytes = 0;
  if (!ibQpSigPop(qp, &t0, &bytes)) return;
  if (bytes == 0 && kZeroStampHoldoff != 0) {
    uint8_t skip = req->base->frmFields.qp_zero_stamp_holdoff[qpIndex];
    if (skip < kZeroStampHoldoff) {
      req->base->frmFields.qp_zero_stamp_holdoff[qpIndex] = (uint8_t)(skip + 1);
      return;
    }
    req->base->frmFields.qp_zero_stamp_holdoff[qpIndex] = 0;
  } else {
    req->base->frmFields.qp_zero_stamp_holdoff[qpIndex] = 0;
  }

  uint64_t t1 = timer_get_cycles();
  if (t0 == 0 || t1 <= t0) return;

  const double inv = (g_inv_cyc_per_us > 0.0) ? g_inv_cyc_per_us : 0.0;
  uint64_t us = (inv > 0.0) ? (uint64_t)((double)(t1 - t0) * inv) : 0;
  if (captureRawLat) {
    req->base->frmFields.qp_raw_lat[qpIndex] = us;
  }
  req->base->frmFields.qp_lat[qpIndex] = norm_lat_us_per_mtu(us, bytes);
  req->base->frmFields.qp_lat_bytes[qpIndex] = bytes;
}

static ncclResult_t ibTestBaseline(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  if (r->base->resiliency && r->base->resiliency->inProgress) {
    NCCLCHECK(ncclIbResiliencyProgress(r->base->resiliency));
  }

  int totalWrDone = 0;
  int wrDone = 0;
  struct ibv_wc wcs[4];
  do {
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats,__func__));
    if (ncclIbRequestIsComplete(r)) {
      NCCLCHECK(ncclIbRequestComplete(r, done, sizes));
      return ncclSuccess;
    }

    totalWrDone = 0;
    for (int i = 0; i < r->base->vProps.ndevs; i++) {
      // Reasons to skip polling this device:
      // 1. When resiliency is enabled events counters might reach negative values.
      // 2. On the sender side, a request might not use all devices (e.g., upon
      //    submission of the send request, a device was not available)
      if (!r->devBases[i] || (r->events[i] == 0 && !r->base->resiliency)) {
        continue;
      }
      TIME_START(3);
      NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
      if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
      if (wrDone == 0) continue;
      totalWrDone += wrDone;
      for (int w=0; w<wrDone; w++) {
        struct ibv_wc *wc = wcs+w;
        if (wc->status != IBV_WC_SUCCESS) {
          if (r->base->resiliency == NULL) {
            WARN("NET/IB: %s: Got CQE with error (devIndex=%d, req=%p, comm=%p (%s), wr_id=%lu, qp_num=%d)", __func__, i, r, r->base, r->base->isSend ? "send" : "recv", wc->wr_id, wc->qp_num);
            WARN("NET/IB: %s: Resiliency is disabled for comm=%p. NET/IB cannot fail over this request; set NCCL_IB_RESILIENCY_PORT_FAILOVER=1 to enable NIC failover before rerunning.", __func__, r->base);
            ncclIbLogCompletionWithError(r->base, wc, i);
            // If resiliency is not enabled, we cannot recover from any error.
            return ncclRemoteError;
          }
          NCCLCHECK(ncclIbResiliencyHandleCompletionError(r->base->resiliency, wc, i));
        } else {
          TRACE(NCCL_NET, "NET/IB: %s: Processing a completion event (devIndex=%d, comm=%p (%s), req=%p, wr_id=%lu, qp_num=%d)", __func__, i, r->base, r->base->isSend ? "send" : "recv", r, wc->wr_id, wc->qp_num);
          NCCLCHECK(ncclIbCompletionEventProcess(r->base, wc, i));
        }
      }
      // Once the IB fatal event is reported in the async thread, we want to propagate this error
      // to communicator and prevent further polling to reduce error pollution.
      NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats,__func__));
    }
  } while (totalWrDone > 0);

  // If no (more) CQEs found on any device, return and come back later
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest* r = (struct ncclIbRequest*)request;
  *done = 0;

  if (!ibFrmEnabledForTest(r)) return ibTestBaseline(request, done, sizes);

  struct ncclIbSendComm* frmComm = (struct ncclIbSendComm*)r->base;
  constexpr bool captureRawLat = (FRM_LAT_PRINT_MS > 0);
  bool frmTelemetryEnabled = true;
  int totalWrDone = 0;
  int wrDone = 0;
  struct ibv_wc wcs[4];
  do {
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats,__func__));
    if (ncclIbRequestIsComplete(r)) {
      NCCLCHECK(ncclIbRequestComplete(r, done, sizes));
      return ncclSuccess;
    }

    totalWrDone = 0;
    for (int i = 0; i < r->base->vProps.ndevs; i++) {
      if (!r->devBases[i] || (r->events[i] == 0 && !r->base->resiliency)) {
        continue;
      }
      TIME_START(3);
      NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
      if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
      if (wrDone == 0) continue;
      totalWrDone += wrDone;
      for (int w = 0; w < wrDone; w++) {
        struct ibv_wc* wc = wcs + w;
        if (wc->status != IBV_WC_SUCCESS) {
          if (r->base->resiliency == NULL) {
            WARN("NET/IB: %s: Got CQE with error (devIndex=%d, req=%p, comm=%p (%s), wr_id=%lu, qp_num=%d)",
                 __func__, i, r, r->base, r->base->isSend ? "send" : "recv", wc->wr_id, wc->qp_num);
            ncclIbLogCompletionWithError(r->base, wc, i);
            return ncclRemoteError;
          }
          NCCLCHECK(ncclIbResiliencyHandleCompletionError(r->base->resiliency, wc, i));
          (void)ibFrmGateUpdate(frmComm);
          frmTelemetryEnabled = false;
          continue;
        }

        TRACE(NCCL_NET, "NET/IB: %s: Processing FRM completion event (devIndex=%d, comm=%p (%s), req=%p, wr_id=%lu, qp_num=%d)",
              __func__, i, r->base, r->base->isSend ? "send" : "recv", r, wc->wr_id, wc->qp_num);
        NCCLCHECK(ncclIbCompletionEventProcess(r->base, wc, i));
        if (!frmTelemetryEnabled) continue;

        ibFrmUpdateLatencyStamp(r, wc, captureRawLat);
      }
      NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats,__func__));
    }
  } while (totalWrDone > 0);

  return ncclSuccess;
}
