/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "p2p_resiliency.h"
#include "p2p.h" // For replay (ncclIbMultiSend() and ncclIbPostFifo())
#include "connect.h" // For ncclIbCreateQp()

NCCL_PARAM(IbResiliencyPortFailover, "IB_RESILIENCY_PORT_FAILOVER", 0);
NCCL_PARAM(IbResiliencyPortFailoverMaxAttempts, "IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS", 1);
NCCL_PARAM(IbResiliencyPortFailoverProbeDelay, "IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY", 10); // In milliseconds

#define MSEC_TO_NSEC 1000000ULL
#define IB_RESILIENCY_READY_QP_WORDS ((NCCL_IB_MAX_QPS + 63) / 64)
#define IB_RESILIENCY_PEER_READY_WR_ID ((uint64_t)(NET_IB_MAX_REQUESTS + 0x2000))
#define IB_RESILIENCY_QP_REJOIN_MAX_FAILURES 16

static ncclResult_t ncclIbResiliencyProbeHandleCompletionEvent(struct ncclIbResiliencySend* sendResCtx, struct ibv_wc* probeWc, int devIndex);
static ncclResult_t ncclIbResiliencyHandleDeviceFailure(struct ncclIbResiliency* resCtx, int devIndex);

static inline bool ncclIbResiliencyQpIsPendingRecovery(enum ncclIbResiliencyQpState state) {
  return state == ncclIbResiliencyQpStateFailed || state == ncclIbResiliencyQpStateLocalReady;
}

static inline bool ncclIbResiliencyQpIsTrafficReady(enum ncclIbResiliencyQpState state) {
  return state == ncclIbResiliencyQpStateOk;
}

static inline void ibResiliencySetQpState(struct ncclIbResiliency* resCtx, int qpIndex,
                                          enum ncclIbResiliencyQpState newState) {
  assert(resCtx != NULL);
  assert(resCtx->baseComm != NULL);
  assert(qpIndex >= 0 && qpIndex < resCtx->baseComm->nqps);

  struct ncclIbResiliencyQpInfo* qpInfo = &resCtx->qpInfo[qpIndex];
  enum ncclIbResiliencyQpState oldState = qpInfo->state;
  if (oldState == newState) return;

  if (oldState == ncclIbResiliencyQpStateOk) {
    resCtx->nonOkQps++;
  }
  if (newState == ncclIbResiliencyQpStateOk) {
    resCtx->nonOkQps--;
  }
  assert(resCtx->nonOkQps >= 0);
  assert(resCtx->nonOkQps <= resCtx->baseComm->nqps);

  qpInfo->state = newState;
}

static inline void ncclIbResiliencyResetPeerReadyReadState(struct ncclIbResiliency* resCtx) {
  assert(resCtx != NULL);
  resCtx->peerReadyReadInFlight = false;
  resCtx->peerReadyReadQpIndex = -1;
  resCtx->peerReadyReadDevIndex = -1;
}

static inline bool ncclIbResiliencyIsReceiverDummyFlushCompletion(struct ncclIbResiliency* resCtx, const struct ibv_wc* wc) {
  return resCtx != NULL &&
         wc != NULL &&
         !resCtx->baseComm->isSend &&
         wc->wr_id == NCCL_IB_RECV_WR_ID_DUMMY &&
         wc->status == IBV_WC_WR_FLUSH_ERR;
}

// Checks if the error indicated in the given work completion is fatal or not.
static ncclResult_t ncclIbResiliencyCheckErrorNotFatal(struct ncclIbResiliency* resCtx, struct ibv_wc *wc, int devIndex) {
  int nFailedDevices = 0;
  bool fatalCompletionStatus = true;
  const char* failureReason = NULL;
  for (int i = 0; i < resCtx->ndevs; i++) {
    if (i == devIndex || resCtx->devs[i].state != ncclIbResiliencyDevStateOk) {
      nFailedDevices++;
    }
  }

  switch (wc->status) {
    case IBV_WC_WR_FLUSH_ERR:
    case IBV_WC_RETRY_EXC_ERR:
      fatalCompletionStatus = false;
      break;
    default:
      WARN("NET/IB: %s: Unsupported completion status %s (%d)", __func__, ibvWcStatusStr(wc->status), wc->status);
      break;
  }

  if ((nFailedDevices < resCtx->ndevs) && !fatalCompletionStatus) {
    INFO(NCCL_NET, "NET/IB: %s: The error is not fatal. Trying to continue...", __func__);
    return ncclSuccess;
  }

  if (nFailedDevices == resCtx->ndevs) {
    failureReason = "No functional devices left";
  } else {
    failureReason = "Fatal error status in work completion";
  }

  WARN("NET/IB: %s: The error is fatal (%s). Cannot continue.", __func__, failureReason);
  return ncclRemoteError;
}

// Function to replace all QPs associated with a given failed device.
static ncclResult_t ncclIbResiliencyReplaceQps(struct ncclIbResiliency* resCtx, int failedDevIndex) {
  // Iterate over all active QPs and replace the ones that are associated with
  // the failed device
  struct ncclIbQp** activeQps = resCtx->baseComm->activeQps;
  for (int qpIndex = 0; qpIndex < resCtx->baseComm->nqps; qpIndex++) {
    struct ncclIbQp* failedQp = activeQps[qpIndex];
    if (failedQp->devIndex != failedDevIndex) {
      // This is not a failed QP
      continue;
    }
    // Find the failed QP's index in the baseComm.qps[] array
    int failedQpIndex = failedQp - resCtx->baseComm->qps;
    assert(failedQpIndex >= 0 && failedQpIndex < resCtx->baseComm->nqps);

    // After finding a failed QP, iterate over the QPs to find a replacement QP
    // that is not associated with a failed device.
    bool replaced = false;
    int newQpIndex = -1;
    int newDevIndex = -1;
    int offset = 1;
    enum ncclIbResiliencyDevState newDevState = ncclIbResiliencyDevStateError;
    do {
      newQpIndex = (failedQpIndex + offset) % resCtx->baseComm->nqps;

      // Check if the new QP is on a functional device
      newDevIndex = resCtx->baseComm->qps[newQpIndex].devIndex;
      newDevState = resCtx->devs[newDevIndex].state;
      enum ncclIbResiliencyQpState newQpState = resCtx->qpInfo[newQpIndex].state;
      if (newDevState != ncclIbResiliencyDevStateOk || !ncclIbResiliencyQpIsTrafficReady(newQpState)) {
        offset++;
        WARN("NET/IB: %s: Cannot replace QP with qpIndex=%d because the candidate QP (qpIndex=%d, devIndex=%d, devState=%d, qpState=%d) is not traffic-ready", __func__, failedQpIndex, newQpIndex, newDevIndex, newDevState, newQpState);
        continue;
      }

      INFO(NCCL_NET, "NET/IB: %s: Replacing QP: qpIndex=%d (qp_num=%u, devIndex=%d) to qpIndex=%d (qp_num=%u, devIndex=%d) on %s communicator (comm=%p)", __func__, failedQpIndex, resCtx->baseComm->qps[failedQpIndex].qp->qp_num, resCtx->baseComm->qps[failedQpIndex].devIndex, newQpIndex, resCtx->baseComm->qps[newQpIndex].qp->qp_num, resCtx->baseComm->qps[newQpIndex].devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
      activeQps[qpIndex] = &resCtx->baseComm->qps[newQpIndex];
      replaced = true;

    } while (replaced == false && newQpIndex != failedQpIndex);

    if (!replaced) {
      WARN("NET/IB: %s: Could not find a replacement QP for the failed QP with qpIndex=%d (devIndex=%d)", __func__, failedQpIndex, failedQp->devIndex);
      return ncclInternalError;
    }

  }
  return ncclSuccess;
}

// Returns true if at least one QP on this communicator is currently marked
// failed and still waiting to rejoin.
static inline bool ncclIbResiliencyHasPendingQpRecovery(struct ncclIbResiliency* resCtx) {
  for (int qpIndex = 0; qpIndex < resCtx->baseComm->nqps; qpIndex++) {
    enum ncclIbResiliencyQpState state = resCtx->qpInfo[qpIndex].state;
    if (ncclIbResiliencyQpIsPendingRecovery(state)) return true;
  }
  return false;
}

static inline void ncclIbResiliencyReadyQpSetBit(uint64_t* mask, int qpIndex, bool value) {
  assert(mask != NULL);
  assert(qpIndex >= 0 && qpIndex < NCCL_IB_MAX_QPS);
  int word = qpIndex / 64;
  int bit = qpIndex % 64;
  uint64_t bitMask = 1ULL << bit;
  if (value) {
    mask[word] |= bitMask;
  } else {
    mask[word] &= ~bitMask;
  }
}

static inline bool ncclIbResiliencyReadyQpGetBit(const uint64_t* mask, int qpIndex) {
  assert(mask != NULL);
  assert(qpIndex >= 0 && qpIndex < NCCL_IB_MAX_QPS);
  int word = qpIndex / 64;
  int bit = qpIndex % 64;
  return (mask[word] & (1ULL << bit)) != 0;
}

// Uses existing probe delay as a conservative cooldown for QP rehabilitation.
// User requested no extra tuning knobs, so this intentionally reuses the
// already-available delay parameter.
static inline uint64_t ncclIbResiliencyQpRejoinCooldownNs() {
  int delayMs = ncclParamIbResiliencyPortFailoverProbeDelay();
  if (delayMs <= 0) delayMs = 1;
  return (uint64_t)delayMs * MSEC_TO_NSEC;
}

static inline uint64_t ncclIbResiliencyPeerReadyBackoffNs(int failureCount) {
  int boundedFailures = std::max(1, std::min(failureCount, IB_RESILIENCY_QP_REJOIN_MAX_FAILURES));
  return ncclIbResiliencyQpRejoinCooldownNs() * boundedFailures;
}

// Verifies the underlying IB port is active before trying to rebuild QP state.
static ncclResult_t ncclIbResiliencyIsLocalPortActive(struct ncclIbNetCommBase* baseComm, int devIndex, bool* active, enum ibv_mtu* mtu) {
  assert(baseComm != NULL);
  assert(active != NULL);
  assert(mtu != NULL);
  *active = false;
  *mtu = IBV_MTU_256;
  if (devIndex < 0 || devIndex >= baseComm->vProps.ndevs) return ncclInternalError;

  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(baseComm, devIndex);
  ncclIbDev* ibDev = &ncclIbDevs[devBase->ibDevN];
  struct ibv_port_attr portAttr;
  ncclResult_t res = wrap_ibv_query_port(ibDev->context, ibDev->portNum, &portAttr);
  if (res != ncclSuccess) {
    WARN("NET/IB: %s: Failed to query port status for devIndex=%d (ibDevN=%d, comm=%p).", __func__, devIndex, devBase->ibDevN, baseComm);
    return ncclSuccess;
  }
  *active = (portAttr.state == IBV_PORT_ACTIVE);
  *mtu = portAttr.active_mtu;
  return ncclSuccess;
}

// Explicitly moves QP to RESET before rebuilding RTR/RTS state.
static ncclResult_t ncclIbResiliencyQpToReset(struct ibv_qp* qp) {
  assert(qp != NULL);
  struct ibv_qp_attr qpAttr = {};
  qpAttr.qp_state = IBV_QPS_RESET;
  return wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE);
}

static ncclResult_t ncclIbResiliencyMarkQpRejoinFailure(struct ncclIbResiliency* resCtx, int qpIndex, const char* stage) {
  assert(resCtx != NULL);
  assert(stage != NULL);
  assert(qpIndex >= 0 && qpIndex < resCtx->baseComm->nqps);
  struct ncclIbResiliencyQpInfo* qpInfo = &resCtx->qpInfo[qpIndex];
  struct ncclIbQp* qp = &resCtx->baseComm->qps[qpIndex];

  if (qpInfo->state == ncclIbResiliencyQpStateFailedPermanent) {
    return ncclSuccess;
  }

  ncclIbResiliencyReadyQpSetBit(resCtx->localReadyQpMask, qpIndex, false);
  ncclIbResiliencyReadyQpSetBit(resCtx->remoteReadyQpMask, qpIndex, false);
  ibResiliencySetQpState(resCtx, qpIndex, ncclIbResiliencyQpStateFailed);
  qpInfo->localPortRetryLogged = false;
  qpInfo->failCount++;
  resCtx->inProgress = true;

  if (qpInfo->failCount >= IB_RESILIENCY_QP_REJOIN_MAX_FAILURES) {
    ibResiliencySetQpState(resCtx, qpIndex, ncclIbResiliencyQpStateFailedPermanent);
    WARN("NET/IB: %s: Rejoin step %s exhausted retry budget for QP index %d (qp_num=%u, devIndex=%d, failCount=%d, comm=%p). Leaving the QP on failover permanently.",
         __func__, stage, qpIndex, qp->qp->qp_num, qp->devIndex, qpInfo->failCount, resCtx->baseComm);
    return ncclSuccess;
  }

  INFO(NCCL_NET, "NET/IB: %s: Rejoin step %s failed for QP index %d (qp_num=%u, devIndex=%d, failCount=%d/%d, comm=%p).",
       __func__, stage, qpIndex, qp->qp->qp_num, qp->devIndex, qpInfo->failCount, IB_RESILIENCY_QP_REJOIN_MAX_FAILURES, resCtx->baseComm);
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyHandlePeerReadyReadFailure(struct ncclIbResiliency* resCtx, int devIndex, const char* reason, const struct ibv_wc* wc) {
  assert(resCtx != NULL);
  assert(reason != NULL);
  if (devIndex < 0 || devIndex >= resCtx->ndevs) return ncclInternalError;

  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  int probingQpIndex = resCtx->peerReadyReadQpIndex;
  uint64_t retryAfterNs = clockNano() + ncclIbResiliencyPeerReadyBackoffNs(resDev->peerReadyFailureCount + 1);

  ncclIbResiliencyResetPeerReadyReadState(resCtx);
  memset(resCtx->remoteReadyQpMask, 0, sizeof(resCtx->remoteReadyQpMask));

  resDev->peerReadyFailureCount++;
  resDev->peerReadyRetryAfterNs = retryAfterNs;

  if (wc != NULL) {
    WARN("NET/IB: %s: Peer-ready read %s on devIndex=%d (probingQpIndex=%d, status=%s(%d), wr_id=%ld, qp_num=%u, failCount=%d). Scoping this to peer-ready backoff until %llu ns instead of failing the data device (comm=%p).",
         __func__, reason, devIndex, probingQpIndex, ibvWcStatusStr(wc->status), wc->status, wc->wr_id, wc->qp_num, resDev->peerReadyFailureCount,
         (unsigned long long)retryAfterNs, resCtx->baseComm);
  } else {
    WARN("NET/IB: %s: Peer-ready read %s on devIndex=%d (probingQpIndex=%d, failCount=%d). Scoping this to peer-ready backoff until %llu ns instead of failing the data device (comm=%p).",
         __func__, reason, devIndex, probingQpIndex, resDev->peerReadyFailureCount, (unsigned long long)retryAfterNs, resCtx->baseComm);
  }
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyHandlePeerReadyReadSuccess(struct ncclIbResiliency* resCtx, int devIndex) {
  assert(resCtx != NULL);
  if (devIndex < 0 || devIndex >= resCtx->ndevs) return ncclInternalError;

  ncclIbResiliencyResetPeerReadyReadState(resCtx);
  resCtx->devs[devIndex].peerReadyFailureCount = 0;
  resCtx->devs[devIndex].peerReadyRetryAfterNs = 0;
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyReconcileDeviceStates(struct ncclIbResiliency* resCtx) {
  assert(resCtx != NULL);
  for (int devIndex = 0; devIndex < resCtx->ndevs; devIndex++) {
    int nTrafficReadyQps = 0;
    int nPendingQps = 0;
    int nPermanentFailedQps = 0;
    for (int qpIndex = 0; qpIndex < resCtx->baseComm->nqps; qpIndex++) {
      if (resCtx->baseComm->qps[qpIndex].devIndex != devIndex) continue;
      switch (resCtx->qpInfo[qpIndex].state) {
        case ncclIbResiliencyQpStateOk:
          nTrafficReadyQps++;
          break;
        case ncclIbResiliencyQpStateFailed:
        case ncclIbResiliencyQpStateLocalReady:
          nPendingQps++;
          break;
        case ncclIbResiliencyQpStateFailedPermanent:
          nPermanentFailedQps++;
          break;
      }
    }

    enum ncclIbResiliencyDevState newState = ncclIbResiliencyDevStateOk;
    if (nPendingQps > 0) {
      newState = ncclIbResiliencyDevStateError;
    } else if (nTrafficReadyQps > 0) {
      newState = ncclIbResiliencyDevStateOk;
    } else if (nPermanentFailedQps > 0) {
      newState = ncclIbResiliencyDevStateErrorPermanent;
    }

    enum ncclIbResiliencyDevState oldState = resCtx->devs[devIndex].state;
    resCtx->devs[devIndex].state = newState;
    if (newState == oldState) continue;

    if (newState == ncclIbResiliencyDevStateErrorPermanent) {
      WARN("NET/IB: %s: Device %d moved to permanent error state (trafficReadyQps=%d, pendingQps=%d, permanentFailedQps=%d, comm=%p).",
           __func__, devIndex, nTrafficReadyQps, nPendingQps, nPermanentFailedQps, resCtx->baseComm);
    } else {
      INFO(NCCL_NET, "NET/IB: %s: Device %d state %d -> %d (trafficReadyQps=%d, pendingQps=%d, permanentFailedQps=%d, comm=%p).",
           __func__, devIndex, oldState, newState, nTrafficReadyQps, nPendingQps, nPermanentFailedQps, resCtx->baseComm);
    }
  }
  return ncclSuccess;
}

// Restores receive queue depth for a rejoined receive-side QP when pre-posting
// is enabled.
static ncclResult_t ncclIbResiliencyRefillRecvQueueForQp(struct ncclIbResiliency* resCtx, struct ncclIbQp* qp) {
  assert(resCtx != NULL);
  assert(qp != NULL);
  if (resCtx->baseComm->isSend) return ncclSuccess;

  struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)resCtx->baseComm;
  if (!recvComm->prepostReceiveWorkRequests) return ncclSuccess;

  uint32_t nRecvWorkRequestsPerQp = NET_IB_MAX_REQUESTS;
  NCCLCHECK(ncclIbResiliencyDataRqSizeGet(resCtx, qp->devIndex, &nRecvWorkRequestsPerQp));
  for (uint32_t i = 0; i < nRecvWorkRequestsPerQp; i++) {
    ncclResult_t res = ncclIbPostRecvWorkRequest(qp->qp, &recvComm->ibRecvWorkRequest);
    if (res != ncclSuccess) {
      WARN("NET/IB: %s: Failed to pre-post receive WQE %u/%u on rejoined QP (qp_num=%u, devIndex=%d, comm=%p).",
           __func__, i + 1, nRecvWorkRequestsPerQp, qp->qp->qp_num, qp->devIndex, resCtx->baseComm);
      return res;
    }
  }
  return ncclSuccess;
}

// Poll completion for an in-flight peer-ready bitmap RDMA read.
static ncclResult_t ncclIbResiliencyPeerReadyReadPoll(struct ncclIbResiliency* resCtx) {
  assert(resCtx != NULL);
  if (!resCtx->peerReadyReadInFlight) return ncclSuccess;

  int devIndex = resCtx->peerReadyReadDevIndex;
  if (devIndex < 0 || devIndex >= resCtx->ndevs) return ncclInternalError;

  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  struct ibv_wc wc;
  int nCompletions = 0;
  NCCLCHECK(wrap_ibv_poll_cq(resDev->probingCq, 1, &wc, &nCompletions));
  if (nCompletions == 0) return ncclSuccess;

  if (resDev->nOutstandingProbes > 0) resDev->nOutstandingProbes--;

  if (wc.wr_id != IB_RESILIENCY_PEER_READY_WR_ID && resCtx->baseComm->isSend) {
    // Completion belongs to regular sender probing path; hand it over.
    struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
    NCCLCHECK(ncclIbResiliencyProbeHandleCompletionEvent(sendResCtx, &wc, devIndex));
    return ncclSuccess;
  }

  if (wc.wr_id != IB_RESILIENCY_PEER_READY_WR_ID) {
    WARN("NET/IB: %s: Unexpected completion while polling peer-ready read (wr_id=%lu, devIndex=%d, comm=%p).",
         __func__, wc.wr_id, devIndex, resCtx->baseComm);
    return ncclSuccess;
  }

  if (wc.status == IBV_WC_SUCCESS) return ncclIbResiliencyHandlePeerReadyReadSuccess(resCtx, devIndex);
  return ncclIbResiliencyHandlePeerReadyReadFailure(resCtx, devIndex, "completion failed", &wc);
}

// Post a non-blocking RDMA read of peer-ready bitmap using a probing QP.
static ncclResult_t ncclIbResiliencyPeerReadyReadPost(struct ncclIbResiliency* resCtx) {
  assert(resCtx != NULL);
  if (resCtx->peerReadyReadInFlight) return ncclSuccess;
  if (resCtx->remoteReadyQpMaskAddr == 0) return ncclSuccess;
  if (resCtx->nProbingQps == 0) return ncclSuccess;
  bool needPeerReadyRead = false;
  for (int qpIndex = 0; qpIndex < resCtx->baseComm->nqps; qpIndex++) {
    if (resCtx->qpInfo[qpIndex].state == ncclIbResiliencyQpStateLocalReady) {
      needPeerReadyRead = true;
      break;
    }
  }
  if (!needPeerReadyRead) {
    resCtx->peerReadyReadLogged = false;
    return ncclSuccess;
  }

  uint64_t now = clockNano();
  uint64_t elapsed = now - resCtx->lastPeerReadyReadNs;
  uint64_t minInterval = ncclIbResiliencyQpRejoinCooldownNs();
  if (elapsed < minInterval) return ncclSuccess;

  int probingQpIndex = -1;
  int devIndex = -1;
  int remDevIdx = -1;
  uint32_t remoteRkey = 0;
  for (int offset = 0; offset < resCtx->nProbingQps; offset++) {
    int candidateQpIndex = (resCtx->nextPeerReadyQpIndex + offset) % resCtx->nProbingQps;
    struct ncclIbQp* qp = &resCtx->probingQps[candidateQpIndex];
    if (qp->qp == NULL) continue;
    if (qp->devIndex < 0 || qp->devIndex >= resCtx->ndevs) continue;
    // Keep the peer-ready control read on the known-good failover path. Using a
    // probing QP on a device that is itself still recovering can stall the
    // handshake even though replay traffic is already succeeding elsewhere.
    if (resCtx->devs[qp->devIndex].state != ncclIbResiliencyDevStateOk) continue;
    if (resCtx->devs[qp->devIndex].peerReadyRetryAfterNs > now) continue;
    if (qp->remDevIdx < 0 || qp->remDevIdx >= resCtx->baseComm->nRemDevs) continue;
    if (resCtx->remoteReadyQpMaskRkeys[qp->remDevIdx] == 0) continue;
    probingQpIndex = candidateQpIndex;
    devIndex = qp->devIndex;
    remDevIdx = qp->remDevIdx;
    remoteRkey = resCtx->remoteReadyQpMaskRkeys[remDevIdx];
    break;
  }
  if (probingQpIndex < 0) return ncclSuccess;

  struct ncclIbQp* probingQp = &resCtx->probingQps[probingQpIndex];

  struct ibv_sge sge = {};
  sge.addr = (uint64_t)&resCtx->remoteReadyQpMask[0];
  sge.length = IB_RESILIENCY_READY_QP_WORDS * sizeof(uint64_t);
  sge.lkey = resCtx->devs[devIndex].peerReadyQpMaskMr->lkey;

  struct ibv_send_wr wr = {};
  wr.wr_id = IB_RESILIENCY_PEER_READY_WR_ID;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = resCtx->remoteReadyQpMaskAddr;
  wr.wr.rdma.rkey = remoteRkey;

  struct ibv_send_wr* bad_wr = NULL;
  resCtx->peerReadyReadQpIndex = probingQpIndex;
  resCtx->peerReadyReadDevIndex = devIndex;
  resCtx->nextPeerReadyQpIndex = (probingQpIndex + 1) % resCtx->nProbingQps;
  ncclResult_t res = wrap_ibv_post_send(probingQp->qp, &wr, &bad_wr);
  if (res != ncclSuccess) {
    return ncclIbResiliencyHandlePeerReadyReadFailure(resCtx, devIndex, "post failed", NULL);
  }
  resCtx->peerReadyReadInFlight = true;
  resCtx->lastPeerReadyReadNs = now;
  resCtx->devs[devIndex].nOutstandingProbes++;
  if (!resCtx->peerReadyReadLogged) {
    INFO(NCCL_NET, "NET/IB: %s: Posted peer-ready read using probingQpIndex=%d (qp_num=%u, devIndex=%d, remDevIdx=%d, comm=%p). Further retries while waiting for peer-ready will be silent.",
         __func__, probingQpIndex, probingQp->qp->qp_num, devIndex, remDevIdx, resCtx->baseComm);
    resCtx->peerReadyReadLogged = true;
  }
  return ncclSuccess;
}

// Activate only QPs that reached local-ready and are also marked ready by peer.
static ncclResult_t ncclIbResiliencyTryActivatePeerReadyQps(struct ncclIbResiliency* resCtx) {
  assert(resCtx != NULL);
  for (int qpIndex = 0; qpIndex < resCtx->baseComm->nqps; qpIndex++) {
    struct ncclIbResiliencyQpInfo* qpInfo = &resCtx->qpInfo[qpIndex];
    if (qpInfo->state != ncclIbResiliencyQpStateLocalReady) continue;
    if (!ncclIbResiliencyReadyQpGetBit(resCtx->localReadyQpMask, qpIndex)) continue;
    if (!ncclIbResiliencyReadyQpGetBit(resCtx->remoteReadyQpMask, qpIndex)) continue;

    resCtx->baseComm->activeQps[qpIndex] = &resCtx->baseComm->qps[qpIndex];
    ibResiliencySetQpState(resCtx, qpIndex, ncclIbResiliencyQpStateOk);
    qpInfo->failCount = 0;
    qpInfo->lastAttemptNs = 0;
    qpInfo->localPortRetryLogged = false;
    INFO(NCCL_NET, "NET/IB: %s: Activated peer-ready QP index %d (qp_num=%u, devIndex=%d, comm=%p).",
         __func__, qpIndex, resCtx->baseComm->qps[qpIndex].qp->qp_num, resCtx->baseComm->qps[qpIndex].devIndex, resCtx->baseComm);
  }
  return ncclSuccess;
}

// Attempt a single QP rehabilitation for a failed QP.
// The sequence mirrors the original connect path:
// RESET -> INIT -> ECE -> RTR -> RTS, then publish local-ready for peer synchronization.
static ncclResult_t ncclIbResiliencyTryRejoinSingleQp(struct ncclIbResiliency* resCtx, int qpIndex) {
  assert(resCtx != NULL);
  assert(qpIndex >= 0 && qpIndex < resCtx->baseComm->nqps);

  struct ncclIbResiliencyQpInfo* qpInfo = &resCtx->qpInfo[qpIndex];
  struct ncclIbQp* qp = &resCtx->baseComm->qps[qpIndex];
  uint64_t now = clockNano();
  uint64_t elapsed = now - qpInfo->lastAttemptNs;
  if (elapsed < ncclIbResiliencyQpRejoinCooldownNs()) return ncclSuccess;
  qpInfo->lastAttemptNs = now;

  if (qp->remDevIdx < 0 || qp->remDevIdx >= resCtx->baseComm->nRemDevs || qp->remQpn == 0) {
    WARN("NET/IB: %s: Skipping QP rejoin due to incomplete remote info (qpIndex=%d, remDevIdx=%d, remQpn=%u, comm=%p).",
         __func__, qpIndex, qp->remDevIdx, qp->remQpn, resCtx->baseComm);
    return ncclIbResiliencyMarkQpRejoinFailure(resCtx, qpIndex, "REMOTE_INFO");
  }

  bool localPortActive = false;
  enum ibv_mtu localPortMtu = IBV_MTU_256;
  NCCLCHECK(ncclIbResiliencyIsLocalPortActive(resCtx->baseComm, qp->devIndex, &localPortActive, &localPortMtu));
  if (!localPortActive) {
    if (!qpInfo->localPortRetryLogged) {
      INFO(NCCL_NET, "NET/IB: %s: Local port is not active for QP index %d (qp_num=%u, devIndex=%d, comm=%p). Rejoin will keep retrying silently until the port becomes active again.",
           __func__, qpIndex, qp->qp->qp_num, qp->devIndex, resCtx->baseComm);
      qpInfo->localPortRetryLogged = true;
    }
    return ncclSuccess;
  }
  qpInfo->localPortRetryLogged = false;

  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(resCtx->baseComm, qp->devIndex);
  struct ncclIbDevInfo remDevInfo = resCtx->baseComm->remDevs[qp->remDevIdx];
  remDevInfo.mtu = std::min(remDevInfo.mtu, localPortMtu);
  bool fifoTc = !resCtx->baseComm->isSend;

  ncclResult_t res = ncclIbResiliencyQpToReset(qp->qp);
  if (res != ncclSuccess) {
    return ncclIbResiliencyMarkQpRejoinFailure(resCtx, qpIndex, "RESET");
  }
  res = ibQpToInit(qp);
  if (res != ncclSuccess) {
    return ncclIbResiliencyMarkQpRejoinFailure(resCtx, qpIndex, "INIT");
  }
  if (qp->remEceSupported) {
    int eceSupported = qp->remEceSupported;
    INFO(NCCL_NET, "NET/IB: %s: Reapplying cached ECE while rejoining QP index %d (qp_num=%u, devIndex=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x, comm=%p).",
         __func__, qpIndex, qp->qp->qp_num, qp->devIndex, qp->remEce.vendor_id, qp->remEce.options, qp->remEce.comp_mask, resCtx->baseComm);
    res = wrap_ibv_set_ece(qp->qp, &qp->remEce, &eceSupported);
    if (res != ncclSuccess) {
      return ncclIbResiliencyMarkQpRejoinFailure(resCtx, qpIndex, "ECE");
    }
    if (eceSupported == 0) {
      INFO(NCCL_NET, "NET/IB: %s: Cached ECE is not supported during rejoin of QP index %d (qp_num=%u, devIndex=%d, comm=%p). Continuing without reduced ECE.",
           __func__, qpIndex, qp->qp->qp_num, qp->devIndex, resCtx->baseComm);
    }
  }
  res = ncclIbRtrQp(qp->qp, &devBase->gidInfo, qp->remQpn, &remDevInfo, fifoTc, resCtx->baseComm->connTc, resCtx->baseComm->connSl);
  if (res != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB: %s: RTR failed while rejoining QP index %d (qp_num=%u, devIndex=%d, remQpn=%u, comm=%p).",
         __func__, qpIndex, qp->qp->qp_num, qp->devIndex, qp->remQpn, resCtx->baseComm);
    return ncclIbResiliencyMarkQpRejoinFailure(resCtx, qpIndex, "RTR");
  }
  res = ncclIbRtsQp(qp->qp);
  if (res != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB: %s: RTS failed while rejoining QP index %d (qp_num=%u, devIndex=%d, comm=%p).",
         __func__, qpIndex, qp->qp->qp_num, qp->devIndex, resCtx->baseComm);
    return ncclIbResiliencyMarkQpRejoinFailure(resCtx, qpIndex, "RTS");
  }

  res = ncclIbResiliencyRefillRecvQueueForQp(resCtx, qp);
  if (res != ncclSuccess) {
    return ncclIbResiliencyMarkQpRejoinFailure(resCtx, qpIndex, "RQ_REFILL");
  }

  // Publish local-ready; activation itself is gated on peer-ready intersection.
  ncclIbResiliencyReadyQpSetBit(resCtx->remoteReadyQpMask, qpIndex, false);
  ncclIbResiliencyReadyQpSetBit(resCtx->localReadyQpMask, qpIndex, true);
  ibResiliencySetQpState(resCtx, qpIndex, ncclIbResiliencyQpStateLocalReady);
  resCtx->peerReadyReadLogged = false;
  INFO(NCCL_NET, "NET/IB: %s: Locally rebuilt QP index %d (qp_num=%u, devIndex=%d, remQpn=%u, comm=%p). Waiting for peer-ready.",
       __func__, qpIndex, qp->qp->qp_num, qp->devIndex, qp->remQpn, resCtx->baseComm);
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencySendRequestInit(struct ncclIbResiliencySend* sendResCtx, ncclIbRequest* request, int devIndex) {
  int slot = request->id % NET_IB_MAX_REQUESTS;
  struct ncclIbResiliencyRequestSend* failedSendRequest = &sendResCtx->failedRequests[slot];

  // Check if the request is/was already tracked
  if (failedSendRequest->id == request->id + 1) {
    if (failedSendRequest->request != NULL) {
      // It might be that a different send request that is part of this
      // multi-send request already got an error and is being replayed.
      // No need to initate a new tracking.
      INFO(NCCL_NET, "NET/IB: %s: No need to add this failed request (req=%p, comm=%p, id=%ld) while another request (req=%p, comm=%p, id=%ld) is already being tracked in the same slot (slot=%d).", __func__, request, request->base, request->id, failedSendRequest->request, failedSendRequest->request->base, failedSendRequest->request->id, slot);
      return ncclSuccess;
    } else {
      // The request was already replayed and released. The CQE should be ignored.
      INFO(NCCL_NET, "NET/IB: %s: Attempting to initiate a replay protocol but the failed request was already handled (req=%p, comm=%p, id=%ld, slot=%d, req.type=%s).", __func__, request, request->base, request->id, slot, ncclIbReqTypeStr[request->type]);
      return ncclSuccess;
    }
  }

  if (request->id + 1 <= failedSendRequest->id) {
    WARN("NET/IB: %s: Attempting to initiate a replay using an old request (req=%p, comm=%p, id=%ld, slot=%d, failedSendRequest.id=%ld).", __func__, request, request->base, request->id, slot, failedSendRequest->id);
    return ncclInternalError;
  }

  if (request->type != NCCL_NET_IB_REQ_SEND) {
    WARN("NET/IB: %s: Attempting to initiate a failed request using a '%s' request while expecting a 'send' request (req=%p, comm=%p, id=%ld, slot=%d, failedSendRequest.id=%ld).", __func__, ncclIbReqTypeStr[request->type], request, request->base, request->id, slot, failedSendRequest->id);
    return ncclInternalError;
  }

  failedSendRequest->state = ncclIbResiliencyRequestStatePending;
  failedSendRequest->request = request;
  failedSendRequest->errorInfo.devIndex = devIndex;
  failedSendRequest->errorInfo.time = clockNano();
  failedSendRequest->failedAttempts = 0;
  failedSendRequest->id = request->id+1;
  sendResCtx->base.outstandingRequests++;
  sendResCtx->base.inProgress = true;
  INFO(NCCL_NET, "NET/IB: %s: Tracking a new failed send request (req=%p, comm=%p, id=%ld, slot=%d, devIndex=%d, time=%ld, total tracked requests: %d).", __func__, request, request->base, request->id, slot, devIndex, failedSendRequest->errorInfo.time, sendResCtx->base.outstandingRequests);
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencySendRequestFree(struct ncclIbResiliencySend* sendResCtx, struct ncclIbResiliencyRequestSend* failedSendRequest) {
  assert(failedSendRequest != NULL);
  if (failedSendRequest->request == NULL) {
    int slot = failedSendRequest - sendResCtx->failedRequests;
    WARN("NET/IB: %s: Attempting to free a non-existent failed request (slot=%d).", __func__, slot);
    return ncclInternalError;
  }
  INFO(NCCL_NET, "NET/IB: %s: Done handling failed send request (req=%p, comm=%p, id=%ld, slot=%ld).", __func__, failedSendRequest->request, failedSendRequest->request->base, failedSendRequest->request->id, failedSendRequest->request->id % NET_IB_MAX_REQUESTS);

  // Note that ID is not reset to allow ignoring old CQEs that might still be
  // in the CQ even after the failed send request was handled completely and
  // freed.
  failedSendRequest->state = ncclIbResiliencyRequestStatePending;
  failedSendRequest->request = NULL;
  failedSendRequest->errorInfo = {0};
  failedSendRequest->failedAttempts = 0;

  sendResCtx->base.outstandingRequests--;
  return ncclSuccess;
}

// Function to repost a given request.
static ncclResult_t ncclIbResiliencyRepostRequest(struct ncclIbRequest* request) {
  if (request->type == NCCL_NET_IB_REQ_UNUSED) {
    WARN("NET/IB: %s: Attempting to repost an unused request (id=%ld).", __func__, request->id);
    return ncclInternalError;
  }
  int slot = request->id % NET_IB_MAX_REQUESTS;
  if (request->type == NCCL_NET_IB_REQ_SEND) {
      struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)request->base;
      struct ncclIbRequest** sendReqs = sendComm->sendReqs[slot];
      struct ncclIbRequest* representative = sendReqs[0];
      for (int r = 0; r < request->nreqs; r++) {
        // Clear all event counters and later on increment only the required
        // ones based on the probing results on which QP a retransmission is
        // required.
        memset(sendReqs[r]->events, 0, sizeof(sendReqs[r]->events));

        // Populate events
        int nqps = ncclIbCommBaseGetNqpsPerRequest(sendReqs[r]->base);
        int qpIndex = -1;
        ncclIbQp* qp = NULL;
        for (int i = 0; i < nqps; i++) {
          NCCLCHECK(ncclIbCommBaseGetQpForRequest(sendReqs[r]->base, sendReqs[r]->id, i, &qp, &qpIndex));

          // Only retransmit logical chunks that are still pending after probing
          // credited all receiver-observed physical completions.
          if (representative->send.pendingData[qpIndex] == false) {
            INFO(NCCL_NET, "NET/IB: %s: Skipping retransmission on logical QP index %d (req=%p, comm=%p, id=%ld, slot=%d) as it was already delivered.", __func__, qpIndex, sendReqs[r], sendReqs[r]->base, sendReqs[r]->id, slot);
            continue;
          }

          int physicalQpIndex = qp - sendReqs[r]->base->qps;
          assert(physicalQpIndex >= 0 && physicalQpIndex < sendReqs[r]->base->nqps);
          INFO(NCCL_NET, "NET/IB: %s: Retransmitting reqIndex=%d on qp_num=%u for logical QP index %d via physical QP index %d (req=%p, comm=%p, id=%ld, slot=%d) as it was not delivered.", __func__, r, qp->qp->qp_num, qpIndex, physicalQpIndex, sendReqs[r], sendReqs[r]->base, sendReqs[r]->id, slot);
          ncclIbAddEvent(sendReqs[r], qp->devIndex);
        }
      }
      INFO(NCCL_NET, "NET/IB: %s: Reposting send request (request=%p, comm=%p, id=%ld, slot=%ld, nreqs=%d)", __func__, request, request->base, request->id, request->id % NET_IB_MAX_REQUESTS, request->nreqs);
      NCCLCHECK(ncclIbMultiSend((struct ncclIbSendComm*)request->base, slot));
  } else if (request->type == NCCL_NET_IB_REQ_RECV) {
    INFO(NCCL_NET, "NET/IB: %s: Reposting CTS (request=%p, comm=%p, id=%ld, slot=%ld)", __func__, request, request->base, request->id, request->id % NET_IB_MAX_REQUESTS);
    NCCLCHECK(ncclIbPostFifo((struct ncclIbRecvComm*)request->base, request, slot));
  } else {
    WARN("NET/IB: %s: Unsupported type of request reposting (type=%d, id=%ld).", __func__, request->type, request->id);
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyHandleCompletionErrorReceiver(struct ncclIbResiliency* resCtx, struct ibv_wc* wc, int devIndex) {
  INFO(NCCL_NET,"NET/IB: %s: Handling an error on the receiver side (comm %p)", __func__, resCtx->baseComm);
  bool inRecvRange = (wc->wr_id < NET_IB_MAX_REQUESTS);
  bool inFlushRange = (wc->wr_id >= NCCL_IB_FLUSH_REQ_WR_ID_OFFSET && wc->wr_id < (NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + NET_IB_MAX_REQUESTS));
  if (!inRecvRange && !inFlushRange && (wc->wr_id != NCCL_IB_RECV_WR_ID_DUMMY)) {
    WARN("NET/IB: %s: Invalid wr_id (%ld). Unable to retrieve a request on the receiver side (comm=%p)", __func__, wc->wr_id, resCtx->baseComm);
    return ncclInternalError;
  }
  if (wc->wr_id == NCCL_IB_RECV_WR_ID_DUMMY) {
    // The only case where the HW will produce an error on the receiver side on
    // a receive queue is a flush error. The flush error indicates that the QP
    // transitioned into an error state and all outstanding work requests are
    // now flushed.
    assert(wc->status == IBV_WC_WR_FLUSH_ERR);
    // In this case, there is nothing left to do.
    INFO(NCCL_NET, "NET/IB: %s: Ignoring flush error on a QP (comm=%p, wc.wr_id=%ld, wc.status=%s(%d)).", __func__, resCtx->baseComm, wc->wr_id, ibvWcStatusStr(wc->status), wc->status);
    return ncclSuccess;
  }

  ncclIbRequest* request = NULL;
  if (inFlushRange) {
    // Completion for a flush request is offset by NCCL_IB_FLUSH_REQ_WR_ID_OFFSET
    ncclIbRequestRetrieveAsIndex(resCtx->baseComm->reqs, wc->wr_id - NCCL_IB_FLUSH_REQ_WR_ID_OFFSET, &request);
  } else {
    struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)resCtx->baseComm;
    request = recvComm->recvReqs[wc->wr_id];
  }

  INFO(NCCL_NET, "NET/IB: %s: The receiver side request that got an error is %p (req=%p, comm=%p, id=%ld)", __func__, request, request, request->base, request->id);

  switch (request->type) {
    case NCCL_NET_IB_REQ_FLUSH:
      // When a flush request encounters an error, it's ignored and the event
      // counter on that device is set to zero, so the flush request could be
      // completed on other devices if needed.
      request->events[devIndex] = 0;
      INFO(NCCL_NET, "NET/IB: %s: Ignoring error on flush request (req=%p, comm=%p, id=%ld) on device index %d", __func__, request, request->base, request->id, devIndex);
      break;
    case NCCL_NET_IB_REQ_RECV:
      // Assert it's a CTS message that got an error.
      // When error occurs the CQE's opcode is not valid and cannot be read!
      // The only valid fields are: wr_id, status, qp_num, and vendor_err.
      // From: https://www.rdmamojo.com/2013/02/15/ibv_poll_cq/
      // Assert the CQE belongs to a CTS and not a data transfer.
      assert(wc->wr_id != NCCL_IB_RECV_WR_ID_DUMMY);
      // CTS is reposted immediately
      NCCLCHECK(ncclIbResiliencyRepostRequest(request));
      break;
    case (NCCL_NET_IB_REQ_UNUSED):
      // This might happen for a CTS message. Consider a case where a HW ack
      // failed for a CTS message but before the receiver got a CQE with error
      // because of a HW timeout, the sender already completed the data transfer
      // and receiver completed the receive request. Note that receiver does not
      // verify if the CTS was completed for every receive request before
      // completing a receive request.
      // When error occurs the CQE's opcode is not valid and cannot be read!
      WARN("NET/IB: %s: Unrecognized request. It might be a CTS message for which the request was already completed. Continue.", __func__);
      break;
    default:
      WARN("NET/IB: %s: Unrecognized request type. request->type=%d", __func__, request->type);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyHandleCompletionErrorSender(struct ncclIbResiliency* resCtx, struct ibv_wc* wc, int devIndex) {
  ncclResult_t res;
  ncclIbRequest* request = NULL;

  uint64_t slot = (wc->wr_id & 0xff);
  struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)resCtx->baseComm;
  request = sendComm->sendReqs[slot][0];

  if (request == NULL) {
    WARN("NET/IB: %s: Encountered a stale CQE with error for slot=%ld. Slot was already handled (comm=%p, wc.wr_id=%ld, wc.status=%s(%d), wc.opcode=%s(%d)).", __func__, slot, resCtx->baseComm, wc->wr_id, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode);
    return ncclSuccess;
  }

  struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
  res = ncclIbResiliencySendRequestInit(sendResCtx, request, devIndex);
  if (res != ncclSuccess) {
    WARN("NET/IB: %s: Failed to initialize a resiliency send request (req=%p, comm=%p, id=%ld, type=%s, wc.wr_id=%ld, wc.status=%s(%d), wc.opcode=%s(%d), slot=%ld).", __func__, request, request->base, request->id, ncclIbReqTypeStr[request->type], wc->wr_id, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, slot);
    return res;
  }

  // The request will be progressed in the main progress function.
  return ncclSuccess;
}

// Mark the device as failed and replace its QPs.
static ncclResult_t ncclIbResiliencyHandleDeviceFailure(struct ncclIbResiliency* resCtx, int devIndex) {
  if (resCtx->devs[devIndex].state == ncclIbResiliencyDevStateOk) {
    WARN("NET/IB: %s: Device %d marked as failed. (%s comm=%p)", __func__, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
  } else {
    INFO(NCCL_NET, "NET/IB: %s: Device %d was already marked as failed (state=%d). Refreshing active-QP failover mapping.", __func__, devIndex, resCtx->devs[devIndex].state);
  }
  if (resCtx->devs[devIndex].state != ncclIbResiliencyDevStateErrorPermanent) {
    resCtx->devs[devIndex].state = ncclIbResiliencyDevStateError;
  }
  NCCLCHECK(ncclIbResiliencyReplaceQps(resCtx, devIndex));
  for (int qpIndex = 0; qpIndex < resCtx->baseComm->nqps; qpIndex++) {
    if (resCtx->baseComm->qps[qpIndex].devIndex == devIndex) {
      NCCLCHECK(ncclIbResiliencyMarkQpFailed(resCtx, qpIndex));
    }
  }
  return ncclSuccess;
}

// -----------------------------
// Probing related functions
// -----------------------------

// After a probe is completed, handle the results of the probe. If the probe
// shows that the request was completed successfully on all QPs, the request
// is completed. Otherwise, the request is reposted.
static ncclResult_t ncclIbResiliencyHandleProbeCompleted(struct ncclIbResiliencySend* sendResCtx, struct ncclIbResiliencyRequestSend* failedRequest) {
  int slot = failedRequest->request->id % NET_IB_MAX_REQUESTS;
  struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)failedRequest->request->base;
  struct ncclIbRequest** sendReqs = sendComm->sendReqs[slot];
  uint16_t newlyCreditable[NCCL_IB_MAX_QPS] = {0};
  bool missingData = false;
  for (int qpIndex = 0; qpIndex < failedRequest->request->base->nqps; qpIndex++) {
    uint16_t observedCompletions = sendResCtx->probingResults[slot][qpIndex];
    uint16_t creditedCompletions = failedRequest->request->send.ackedCompletions[qpIndex];
    if (observedCompletions > creditedCompletions) {
      newlyCreditable[qpIndex] = observedCompletions - creditedCompletions;
    }
    for (int r = 0; r < failedRequest->request->nreqs; r++) {
      sendReqs[r]->send.ackedCompletions[qpIndex] = std::max(sendReqs[r]->send.ackedCompletions[qpIndex], observedCompletions);
    }
  }

  for (int physicalQpIndex = 0; physicalQpIndex < failedRequest->request->base->nqps; physicalQpIndex++) {
    while (newlyCreditable[physicalQpIndex] > 0) {
      int deliveredLogicalQpIndex = -1;
      int bestOrder = failedRequest->request->base->nqps + 1;
      for (int logicalQpIndex = 0; logicalQpIndex < failedRequest->request->base->nqps; logicalQpIndex++) {
        if (failedRequest->request->send.pendingData[logicalQpIndex] == false) continue;
        if (failedRequest->request->send.lastAttemptQpIndex[logicalQpIndex] != physicalQpIndex) continue;
        int order = failedRequest->request->send.lastAttemptOrder[logicalQpIndex];
        if (deliveredLogicalQpIndex == -1 || order < bestOrder) {
          deliveredLogicalQpIndex = logicalQpIndex;
          bestOrder = order;
        }
      }

      if (deliveredLogicalQpIndex < 0) {
        INFO(NCCL_NET, "NET/IB: %s: Probe observed %u extra completion(s) on physical QP index %d with no pending logical chunk to credit (req=%p, slot=%d).", __func__, newlyCreditable[physicalQpIndex], physicalQpIndex, failedRequest->request, slot);
        break;
      }

      for (int r = 0; r < failedRequest->request->nreqs; r++) {
        sendReqs[r]->send.pendingData[deliveredLogicalQpIndex] = false;
        sendReqs[r]->send.lastAttemptQpIndex[deliveredLogicalQpIndex] = -1;
        sendReqs[r]->send.lastAttemptOrder[deliveredLogicalQpIndex] = 0;
      }
      INFO(NCCL_NET, "NET/IB: %s: Credited delivery of logical QP index %d from physical QP index %d (req=%p, slot=%d).", __func__, deliveredLogicalQpIndex, physicalQpIndex, failedRequest->request, slot);
      newlyCreditable[physicalQpIndex]--;
    }
  }

  for (int logicalQpIndex = 0; logicalQpIndex < failedRequest->request->base->nqps; logicalQpIndex++) {
    if (failedRequest->request->send.pendingData[logicalQpIndex] == false) continue;
    missingData = true;
    INFO(NCCL_NET, "NET/IB: %s: Missing data from logical QP index %d (req=%p, slot=%d, lastAttemptPhysicalQp=%d).", __func__, logicalQpIndex, failedRequest->request, slot, failedRequest->request->send.lastAttemptQpIndex[logicalQpIndex]);
    break;
  }
  if (!missingData) {
    // All data was delivered to the sender and no need for retransmission.
    INFO(NCCL_NET, "NET/IB: %s: Probing conclusion: All data was delivered for request %p (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d). Completing the request.", __func__, failedRequest->request, failedRequest->request, failedRequest->request->base, failedRequest->request->id, slot, failedRequest->request->nreqs);
    // Clear all events on the request so it could be completed towards the
    // user as well. Note that in case of a multi-send requests, all requests
    // are also cleared.
    for (int r = 0; r < failedRequest->request->nreqs; r++) {
      memset(sendReqs[r]->events, 0, sizeof(sendReqs[r]->events));
      INFO(NCCL_NET, "NET/IB: %s: Clearing events on send request %p (req=%p, comm=%p, id=%ld, slot=%d, reqIdx=%d)", __func__, sendReqs[r], sendReqs[r], sendReqs[r]->base, sendReqs[r]->id, slot, r);
    }
    return ncclSuccess;
  } else {
    // Repost the send request
    NCCLCHECK(ncclIbResiliencyRepostRequest(failedRequest->request));
  }
  return ncclSuccess;
}

// Posts a probe (RDMA Read) operation to check the status of a send request
// that encountered an error.
static ncclResult_t ncclIbResiliencyProbePost(struct ncclIbResiliencySend* sendResCtx, struct ncclIbResiliencyRequestSend* failedSendRequest) {
  assert(failedSendRequest->state == ncclIbResiliencyRequestStatePending);

  if (failedSendRequest->failedAttempts > ncclParamIbResiliencyPortFailoverMaxAttempts()) {
    WARN("NET/IB: %s: Maximum number of probing attempts (%ld) reached for request %p (id=%ld). Cannot post another probe.", __func__, ncclParamIbResiliencyPortFailoverMaxAttempts(), failedSendRequest->request, failedSendRequest->request->id);
    return ncclRemoteError;
  }

  // Get current time and calculate elapsed time since error
  uint64_t currentTime = clockNano();
  uint64_t elapsedTime = currentTime - failedSendRequest->errorInfo.time;
  uint64_t requiredWaitTime = ncclParamIbResiliencyPortFailoverProbeDelay() * MSEC_TO_NSEC;

  if (elapsedTime < requiredWaitTime) {
    return ncclSuccess;
  }

  INFO(NCCL_NET, "NET/IB: %s: Probe delay elapsed (%llu ms) for request %p, posting the probe", __func__, elapsedTime / MSEC_TO_NSEC, failedSendRequest->request);

  int devIndex = 0;
  for (devIndex = 0; devIndex < sendResCtx->base.ndevs; devIndex++) {
    if (sendResCtx->base.devs[devIndex].state == ncclIbResiliencyDevStateOk) {
      // This device is functional. Use it to post the probe.
      break;
    }
  }

  if (devIndex == sendResCtx->base.ndevs) {
    WARN("NET/IB: %s: Could not find a functional device to post the probe for request %p (id=%ld)", __func__, failedSendRequest->request, failedSendRequest->request->id);
    return ncclInternalError;
  }

  struct ncclIbResiliency* resCtx = &sendResCtx->base;
  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  struct ncclIbQp* probingQp = &resCtx->probingQps[devIndex];

  struct ibv_send_wr probeWr = {0};
  struct ibv_sge sge = {0};

  int slot = failedSendRequest->request->id % NET_IB_MAX_REQUESTS;

  // The length of the RDMA Read and where to write the response to
  sge.addr = (uint64_t)&(sendResCtx->probingResults[slot]);
  sge.length = sizeof(sendResCtx->probingResults[slot]);
  sge.lkey = resDev->probingResultMr->lkey;

  // Format the RDMA Read WR
  probeWr.sg_list = &sge;
  probeWr.num_sge = 1;
  probeWr.opcode = IBV_WR_RDMA_READ;
  probeWr.wr_id = slot;
  probeWr.send_flags = IBV_SEND_SIGNALED;
  probeWr.next = NULL;

  uint64_t remoteAddr = sendResCtx->remCmplRecordsInfo[probingQp->remDevIdx].addr;
  // Skip "slot" number of completion records
  remoteAddr += sizeof(struct ncclIbRequestCompletionRecord) * slot;
  // Skip the "sizes" array to point to the "completions" array.
  remoteAddr += sizeof(int) * NCCL_NET_IB_MAX_RECVS;
  probeWr.wr.rdma.remote_addr = remoteAddr;
  probeWr.wr.rdma.rkey = sendResCtx->remCmplRecordsInfo[probingQp->remDevIdx].rkey;

  INFO(NCCL_NET, "NET/IB: %s: Posting probe (slot=%d, id=%ld, devIndex=%d, qp_num=%u)", __func__, slot, failedSendRequest->request->id, devIndex, probingQp->qp->qp_num);

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(probingQp->qp, &probeWr, &bad_wr));
  failedSendRequest->state = ncclIbResiliencyRequestStateProbePosted;
  resDev->nOutstandingProbes++;
  assert(resDev->nOutstandingProbes <= NET_IB_MAX_REQUESTS);
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyProbeHandleCompletionEvent(struct ncclIbResiliencySend* sendResCtx, struct ibv_wc* probeWc, int devIndex) {

  INFO(NCCL_NET, "NET/IB: %s: Got probing completion (devIndex=%d, wc->status=%d, wc->opcode=%d, wc->wr_id=%ld, wc->qp_num=%u)", __func__, devIndex, probeWc->status, probeWc->opcode, probeWc->wr_id, probeWc->qp_num);

  struct ncclIbResiliencyRequestSend* failedRequest = &sendResCtx->failedRequests[probeWc->wr_id % NET_IB_MAX_REQUESTS];

  if (probeWc->status == IBV_WC_SUCCESS) {
    // The probing was successful. Mark the probe as completed.
    failedRequest->state = ncclIbResiliencyRequestStateProbeCompleted;
    // Further processing will be done in the main progress function.
    return ncclSuccess;
  }

  NCCLCHECK(ncclIbResiliencyCheckErrorNotFatal(&sendResCtx->base, probeWc, devIndex));

  NCCLCHECK(ncclIbResiliencyHandleDeviceFailure(&sendResCtx->base, devIndex));

  // The probe will reposted upon the next call to progress resiliency
  failedRequest->state = ncclIbResiliencyRequestStatePending;
  failedRequest->failedAttempts++;
  return ncclSuccess;
}

#define MAX_PROBE_WC 4


static ncclResult_t ncclIbResiliencyProbeProgress(struct ncclIbResiliencySend* sendResCtx) {

  struct ibv_wc wcs[MAX_PROBE_WC];
  int nCompletions = 0;
  struct ncclIbResiliencyDev *resDev = NULL;
  for (int devIndex = 0; devIndex < sendResCtx->base.ndevs; devIndex++) {
    resDev = &sendResCtx->base.devs[devIndex];
    // Note that even if the device failed, we still want to drain all the CQEs
    // out of it. So the only check before polling is whether there are
    // outstanding probing requests on this CQ or not.
    if (resDev->nOutstandingProbes == 0) {
      continue;
    }
    do {
      NCCLCHECK(wrap_ibv_poll_cq(resDev->probingCq, MAX_PROBE_WC, wcs, &nCompletions));
      if (nCompletions == 0) {
        break;
      }
      assert(nCompletions <= resDev->nOutstandingProbes);
      resDev->nOutstandingProbes -= nCompletions;
      for (int i = 0; i < nCompletions; i++) {
        if (wcs[i].wr_id == IB_RESILIENCY_PEER_READY_WR_ID) {
          if (wcs[i].status != IBV_WC_SUCCESS) {
            NCCLCHECK(ncclIbResiliencyHandlePeerReadyReadFailure(&sendResCtx->base, devIndex, "completion failed", &wcs[i]));
          } else {
            NCCLCHECK(ncclIbResiliencyHandlePeerReadyReadSuccess(&sendResCtx->base, devIndex));
          }
          continue;
        }
        NCCLCHECK(ncclIbResiliencyProbeHandleCompletionEvent(sendResCtx, &wcs[i], devIndex));
      }
    } while (nCompletions > 0);
  }
  return ncclSuccess;
}

// -----------------------------
// Implementation of entry point functions
// -----------------------------

ncclResult_t ncclIbResiliencyInit(struct ncclIbNetCommBase* baseComm, struct ncclIbResiliency** resCtx) {
  assert(baseComm != NULL);
  assert(resCtx != NULL);
  if (ncclParamIbResiliencyPortFailover() == 0) {
    INFO(NCCL_NET, "NET/IB: %s: Resiliency is disabled on the %s communicator (comm=%p) because NCCL_IB_RESILIENCY_PORT_FAILOVER=0. Set NCCL_IB_RESILIENCY_PORT_FAILOVER=1 to enable NIC failover.",
         __func__, baseComm->isSend ? "send" : "recv", baseComm);
    *resCtx = NULL;
    return ncclSuccess;
  }
  size_t sizeToAlloc = 0;
  if (baseComm->isSend) {
    sizeToAlloc = sizeof(struct ncclIbResiliencySend);
  } else {
    sizeToAlloc = sizeof(struct ncclIbResiliency);
  }
  // TODO: No real need to use IB malloc as this whole memory is not registered,
  // only the probing results should be allocated in memory that should be
  // registered
  NCCLCHECK(ncclIbMalloc((void**)resCtx, sizeToAlloc));
  struct ncclIbResiliency* baseCtx = *resCtx;
  baseCtx->baseComm = baseComm;
  baseCtx->inProgress = false;
  baseCtx->outstandingRequests = 0;
  baseCtx->nonOkQps = 0;
  memset(baseCtx->localReadyQpMask, 0, sizeof(baseCtx->localReadyQpMask));
  memset(baseCtx->remoteReadyQpMask, 0, sizeof(baseCtx->remoteReadyQpMask));
  memset(baseCtx->remoteReadyQpMaskRkeys, 0, sizeof(baseCtx->remoteReadyQpMaskRkeys));
  baseCtx->remoteReadyQpMaskAddr = 0;
  ncclIbResiliencyResetPeerReadyReadState(baseCtx);
  baseCtx->nextPeerReadyQpIndex = 0;
  baseCtx->lastPeerReadyReadNs = 0;
  for (int i = 0; i < NCCL_IB_MAX_QPS; i++) {
    baseCtx->qpInfo[i].state = ncclIbResiliencyQpStateOk;
    baseCtx->qpInfo[i].lastAttemptNs = 0;
    baseCtx->qpInfo[i].localPortRetryLogged = false;
    baseCtx->qpInfo[i].failCount = 0;
  }
  baseCtx->peerReadyReadLogged = false;
  if (baseComm->isSend) {
    struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)baseCtx;
    memset(sendResCtx->failedRequests, 0, sizeof(sendResCtx->failedRequests));
    memset(sendResCtx->probingResults, 0, sizeof(sendResCtx->probingResults));
    memset(sendResCtx->remCmplRecordsInfo, 0, sizeof(sendResCtx->remCmplRecordsInfo));
  }
  INFO(NCCL_NET, "NET/IB: %s: Resiliency context was initialized on the %s communicator (%p)", __func__, baseComm->isSend ? "send" : "recv", baseComm);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyMarkQpFailed(struct ncclIbResiliency* resCtx, int qpIndex) {
  assert(resCtx != NULL);
  if (qpIndex < 0 || qpIndex >= resCtx->baseComm->nqps) {
    WARN("NET/IB: %s: Invalid QP index %d for comm=%p.", __func__, qpIndex, resCtx->baseComm);
    return ncclInternalError;
  }
  struct ncclIbResiliencyQpInfo* qpInfo = &resCtx->qpInfo[qpIndex];
  if (qpInfo->state == ncclIbResiliencyQpStateFailed || qpInfo->state == ncclIbResiliencyQpStateFailedPermanent) return ncclSuccess;

  ibResiliencySetQpState(resCtx, qpIndex, ncclIbResiliencyQpStateFailed);
  qpInfo->lastAttemptNs = clockNano();
  qpInfo->localPortRetryLogged = false;
  qpInfo->failCount++;
  ncclIbResiliencyReadyQpSetBit(resCtx->localReadyQpMask, qpIndex, false);
  ncclIbResiliencyReadyQpSetBit(resCtx->remoteReadyQpMask, qpIndex, false);
  // Keep background progress active until QP rehabilitation converges.
  resCtx->inProgress = true;
  INFO(NCCL_NET, "NET/IB: %s: Marked QP index %d as failed (qp_num=%u, devIndex=%d, comm=%p).",
       __func__, qpIndex, resCtx->baseComm->qps[qpIndex].qp->qp_num, resCtx->baseComm->qps[qpIndex].devIndex, resCtx->baseComm);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyTryRejoinQps(struct ncclIbResiliency* resCtx) {
  assert(resCtx != NULL);

  for (int qpIndex = 0; qpIndex < resCtx->baseComm->nqps; qpIndex++) {
    struct ncclIbResiliencyQpInfo* qpInfo = &resCtx->qpInfo[qpIndex];
    if (qpInfo->state != ncclIbResiliencyQpStateFailed) continue;

    struct ncclIbQp* qp = &resCtx->baseComm->qps[qpIndex];
    int devIndex = qp->devIndex;
    if (devIndex < 0 || devIndex >= resCtx->ndevs) continue;
    if (resCtx->devs[devIndex].state == ncclIbResiliencyDevStateErrorPermanent) continue;

    NCCLCHECK(ncclIbResiliencyTryRejoinSingleQp(resCtx, qpIndex));
  }

  NCCLCHECK(ncclIbResiliencyPeerReadyReadPoll(resCtx));
  NCCLCHECK(ncclIbResiliencyTryActivatePeerReadyQps(resCtx));
  NCCLCHECK(ncclIbResiliencyReconcileDeviceStates(resCtx));
  NCCLCHECK(ncclIbResiliencyPeerReadyReadPost(resCtx));

  // Keep inProgress active while there are unresolved QP recoveries.
  if (ncclIbResiliencyHasPendingQpRecovery(resCtx)) {
    resCtx->inProgress = true;
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDestroy(struct ncclIbResiliency** resCtx) {
  if (resCtx == NULL || *resCtx == NULL) {
    return ncclSuccess;
  }
  free(*resCtx);
  *resCtx = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDevInit(struct ncclIbResiliency* resCtx, uint devIndex, ncclIbDev* ibDev) {
  assert(resCtx != NULL);
  INFO(NCCL_NET, "NET/IB: %s: Initializing resiliency context on devIndex %d for %s communicator (comm=%p)", __func__, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
  assert(devIndex < resCtx->ndevs);
  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  resDev->state = ncclIbResiliencyDevStateOk;
  resDev->peerReadyRetryAfterNs = 0;
  resDev->peerReadyFailureCount = 0;
  void* cqContext = (void*)&resCtx->baseComm->stats;
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(resCtx->baseComm, devIndex);
  NCCLCHECK(wrap_ibv_reg_mr(&resDev->readyQpMaskMr, devBase->pd, &resCtx->localReadyQpMask, sizeof(resCtx->localReadyQpMask), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
  NCCLCHECK(wrap_ibv_reg_mr(&resDev->peerReadyQpMaskMr, devBase->pd, &resCtx->remoteReadyQpMask, sizeof(resCtx->remoteReadyQpMask), IBV_ACCESS_LOCAL_WRITE));
  int cqSize = -1;
  if (resCtx->baseComm->isSend) {
    cqSize = NET_IB_MAX_REQUESTS;
    struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
    NCCLCHECK(wrap_ibv_reg_mr(&resDev->probingResultMr, devBase->pd, &sendResCtx->probingResults, sizeof(sendResCtx->probingResults), IBV_ACCESS_LOCAL_WRITE));
    INFO(NCCL_NET, "NET/IB: %s: Registered probing results memory (%p) on device %d for resiliency context (comm=%p)", __func__, &sendResCtx->probingResults, devIndex, resCtx->baseComm);
  } else {
    // It's not allowed to create a CQ with size 0 so set it to 1 although
    // no CQEs are expected to be generated on this CQ (on the receiver side).
    cqSize = 1;
  }
  INFO(NCCL_NET, "NET/IB: %s: Creating probing CQ on device %d for resiliency context (%s comm=%p, cq_size=%d)", __func__, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm, cqSize);
  NCCLCHECK(wrap_ibv_create_cq(&resDev->probingCq, ibDev->context, cqSize, cqContext, NULL, 0));
  INFO(NCCL_NET, "NET/IB: %s: Created probing CQ (cq=%p) on device %d for resiliency context (%s comm=%p, cq_size=%d)", __func__, resDev->probingCq, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm, cqSize);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDevDestroy(struct ncclIbResiliency* resCtx, uint devIndex) {
  assert(resCtx != NULL);
  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  NCCLCHECK(wrap_ibv_destroy_cq(resDev->probingCq));
  INFO(NCCL_NET, "NET/IB: %s: Destroyed probing CQ (cq=%p) on device %d for resiliency context (comm=%p)", __func__, resDev->probingCq, devIndex, resCtx->baseComm);
  if (resCtx->baseComm->isSend) {
    struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
    NCCLCHECK(wrap_ibv_dereg_mr(resDev->probingResultMr));
    INFO(NCCL_NET, "NET/IB: %s: Deregistered probing results memory (%p) on device %d for resiliency context (comm=%p)", __func__, &sendResCtx->probingResults, devIndex, resCtx->baseComm);
  }

  NCCLCHECK(wrap_ibv_dereg_mr(resDev->peerReadyQpMaskMr));
  NCCLCHECK(wrap_ibv_dereg_mr(resDev->readyQpMaskMr));
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDataCqSizeGet(struct ncclIbResiliency* resCtx, uint devIndex, int* cqSize) {
  assert(cqSize != NULL);
  *cqSize = 0;
  struct ncclIbNetCommBase* baseComm = resCtx->baseComm;
  if (baseComm->isSend) {
    // Every send request generates one completion on every QP it uses.
    *cqSize = NET_IB_MAX_REQUESTS * ncclIbCommBaseGetNqpsPerRequest(baseComm);
  } else {
    // In the worst case, a receive is not a multi-receive request, so every
    // request generates two completions (one for the CTS messages and one for
    // the receive request).
    *cqSize = NET_IB_MAX_REQUESTS * 2 * ncclIbCommBaseGetNqpsPerRequest(baseComm);
  }
  // In the worst case, all devices failed except for one device, so the single
  // device remaining must bear all be able to accommodate for all the
  // completions of all other requests.
  assert(resCtx->ndevs > 0);
  *cqSize = (*cqSize) * resCtx->ndevs;
  INFO(NCCL_NET, "NET/IB: %s: CQ size should be %d on device %d for %s communicator (comm=%p)", __func__, *cqSize, devIndex, baseComm->isSend ? "send" : "recv", baseComm);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDataRqSizeGet(struct ncclIbResiliency* resCtx, uint devIndex, uint32_t* rqSize) {
  assert(rqSize != NULL);
  // This API should only be called on the receiver side.
  assert(resCtx->baseComm->isSend == 0);

  struct ncclIbNetCommBase* baseComm = resCtx->baseComm;
  // The size of a single RQ should accommodate all the receive requests.
  // When resiliency is enabled, the RQ size should accommodate receive requests
  // assuming all other devices have failed so instead of transferring every
  // request on all QPs, a single QP is used and this QP should bear the load.
  *rqSize = NET_IB_MAX_REQUESTS * ncclIbCommBaseGetNqpsPerRequest(baseComm);
  assert(*rqSize > 0);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDeviceNumSet(struct ncclIbResiliency* resCtx, int nLocalDevs, int nRemDevs) {
  assert(resCtx != NULL);
  assert(nLocalDevs > 0);
  assert(nRemDevs > 0);

  if (nLocalDevs <= 1) {
    INFO(NCCL_NET, "NET/IB: %s: Resiliency is being enabled on a communicator (comm=%p) with a single local device. In case of failure there will be no device to fail-over to.", __func__, resCtx->baseComm);
  }

  INFO(NCCL_NET, "NET/IB: %s: Resiliency context (comm=%p) is configured with %d local devices and %d remote devices", __func__, resCtx->baseComm, nLocalDevs, nRemDevs);

  resCtx->ndevs = nLocalDevs;
  if (resCtx->baseComm->isSend) {
    // On the sender side, the number of probing QPs will be the number of the
    // local devices. The reason for that is that sender should be able to
    // probe on all its devices.
    resCtx->nProbingQps = nLocalDevs;
  } else {
    // On the receiver side, the number of probing QPs should match the number
    // of QPs created on the sender side. So every QP on the sender side is
    // connected to a QP on the receiver side. If the number of devices on the
    // receiver side is less than the number of devices on the sender side,
    // some devices on the receiver side will have more than one probing QP.
    // If the number of devices on the receiver side is more than the number
    // of devices on the sender side, some devices on the receiver side will not
    // have any probing QPs since the sender will not probe on these devices.
    resCtx->nProbingQps = nRemDevs;
  }
  // In any case, the number of probing QPs cannot exceed the maximum number
  // of devices supported.
  assert(resCtx->nProbingQps <= NCCL_IB_MAX_DEVS_PER_NIC);
  if (resCtx->nProbingQps <= 1) {
    WARN("NET/IB: %s: Resiliency is enabled on a %s communicator (comm=%p) with a single device. This does not make sense since there is no other device to fail over to.", __func__, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
  }
  INFO(NCCL_NET, "NET/IB: %s: Resiliency context (comm=%p) is configured with %d probing QPs", __func__, resCtx->baseComm, resCtx->nProbingQps);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencySenderCreateQps(struct ncclIbResiliency* resCtx, struct ncclIbResiliencyInfo* localResiliencyInfo) {
  ncclIbSendComm* sendComm = (ncclIbSendComm*)resCtx->baseComm;
  void* qpContext = (void*)&sendComm->base.stats;
  struct ncclIbQpCreateAttr qpCreateAttrs = {0};
  qpCreateAttrs.type = IBV_QPT_RC;
  // Both peers may issue ready-mask RDMA reads during rejoin, so sender probing
  // QPs must also accept inbound remote reads.
  qpCreateAttrs.accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
  qpCreateAttrs.maxRecvWorkRequest = 0;
  // Every send request can initiate at most one probing request.
  qpCreateAttrs.maxSendWorkRequest = NET_IB_MAX_REQUESTS;
  for (int localQpIndex = 0; localQpIndex < resCtx->nProbingQps; localQpIndex++) {
    // Sender creates a single probing QP per local device.
    int localDevIndex = localQpIndex;
    ncclIbSendCommDev* sendCommDev = &sendComm->devs[localDevIndex];
    ncclIbDev* ibDev = &ncclIbDevs[sendCommDev->base.ibDevN];
    ncclIbQp* localQp = &resCtx->probingQps[localQpIndex];
    qpCreateAttrs.ibPort = ibDev->portNum;
    qpCreateAttrs.cq = resCtx->devs[localDevIndex].probingCq;
    qpCreateAttrs.pd = sendCommDev->base.pd;
    NCCLCHECK(ncclIbCreateQp(&qpCreateAttrs, qpContext, localQp));
    localQp->devIndex = localDevIndex;
    // Populate the info that will be delivered to the remote receiver peer
    ncclIbQpInfo* localQpInfo = &localResiliencyInfo->probingQpsInfo[localQpIndex];
    localQpInfo->qpn = localQp->qp->qp_num;
    localQpInfo->devIndex = localDevIndex;
  }
  localResiliencyInfo->readyQpMaskAddr = (uint64_t)&resCtx->localReadyQpMask[0];
  for (int i = 0; i < resCtx->ndevs; i++) {
    localResiliencyInfo->readyQpMaskRkeys[i] = resCtx->devs[i].readyQpMaskMr->rkey;
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencySenderQpsToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo) {
  ncclIbSendComm* sendComm = (ncclIbSendComm*)resCtx->baseComm;
  ncclIbQp* localQp = NULL;
  ncclIbQpInfo* remQpInfo = NULL;
  for (int localQpIndex = 0; localQpIndex < resCtx->nProbingQps; localQpIndex++) {
    int localDevIndex = localQpIndex;
    ncclIbSendCommDev* sendCommDev = &sendComm->devs[localDevIndex];
    ncclIbDev* ibDev = &ncclIbDevs[sendCommDev->base.ibDevN];
    localQp = &resCtx->probingQps[localQpIndex];
    remQpInfo = &(remInfo->resiliencyInfo.probingQpsInfo[localQpIndex]);
    localQp->remDevIdx = remQpInfo->devIndex;
    // It might be that the remote side has a different number of devices, so
    // finding the correct remote device information is done by checking the
    // remote QP info.
    ncclIbDevInfo* remDevInfo = &remInfo->devs[remQpInfo->devIndex];
    remDevInfo->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu); // TODO: This is bad practice!
    NCCLCHECK(ncclIbRtrQp(localQp->qp, &sendCommDev->base.gidInfo, remQpInfo->qpn, remDevInfo, false, remInfo->tc, remInfo->sl));
    NCCLCHECK(ncclIbRtsQp(localQp->qp));
  }
  resCtx->remoteReadyQpMaskAddr = remInfo->resiliencyInfo.readyQpMaskAddr;
  for (int i = 0; i < resCtx->baseComm->nRemDevs; i++) {
    resCtx->remoteReadyQpMaskRkeys[i] = remInfo->resiliencyInfo.readyQpMaskRkeys[i];
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyReceiverQpsCreateToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo, struct ncclIbResiliencyInfo* localResiliencyInfo) {
  ncclIbRecvComm* recvComm = (ncclIbRecvComm*)resCtx->baseComm;
  void* qpContext = (void*)&recvComm->base.stats;
  struct ncclIbQpCreateAttr qpCreateAttrs = {0};
  qpCreateAttrs.type = IBV_QPT_RC;
  // On the receiver side, probing QPs do not need to send/receive any messages.
  // They are only used as targets of RDMA Read operations.
  qpCreateAttrs.accessFlags = IBV_ACCESS_REMOTE_READ;
  qpCreateAttrs.maxRecvWorkRequest = 0;
  // Enable send queue so receiver can issue peer-ready RDMA reads.
  qpCreateAttrs.maxSendWorkRequest = NET_IB_MAX_REQUESTS;
  for (int localQpIndex = 0; localQpIndex < resCtx->nProbingQps; localQpIndex++) {
    // When number of QPs on the receiver is larger than the number of devices
    // it has, the probing QPs on the receiver side are created in a "striped"
    // manner.
    int localDevIndex = localQpIndex % recvComm->base.vProps.ndevs;
    ncclIbRecvCommDev* recvCommDev = &recvComm->devs[localDevIndex];
    ncclIbDev* ibDev = &ncclIbDevs[recvCommDev->base.ibDevN];
    ncclIbQp* localQp = &resCtx->probingQps[localQpIndex];
    qpCreateAttrs.ibPort = ibDev->portNum;
    qpCreateAttrs.cq = resCtx->devs[localDevIndex].probingCq;
    qpCreateAttrs.pd = recvCommDev->base.pd;
    NCCLCHECK(ncclIbCreateQp(&qpCreateAttrs, qpContext, localQp));
    localQp->devIndex = localDevIndex;
    localResiliencyInfo->probingQpsInfo[localQpIndex].qpn = localQp->qp->qp_num;
    localResiliencyInfo->probingQpsInfo[localQpIndex].devIndex = localDevIndex;

    ncclIbQpInfo* remQpInfo = &remInfo->resiliencyInfo.probingQpsInfo[localQpIndex];
    ncclIbDevInfo* remDevInfo = &remInfo->devs[remQpInfo->devIndex];
    localQp->remDevIdx = remQpInfo->devIndex;
    NCCLCHECK(ncclIbRtrQp(localQp->qp, &recvCommDev->base.gidInfo, remQpInfo->qpn, remDevInfo, true, remInfo->tc, remInfo->sl));
    NCCLCHECK(ncclIbRtsQp(localQp->qp));
  }
  localResiliencyInfo->readyQpMaskAddr = (uint64_t)&resCtx->localReadyQpMask[0];
  for (int i = 0; i < resCtx->ndevs; i++) {
    localResiliencyInfo->readyQpMaskRkeys[i] = resCtx->devs[i].readyQpMaskMr->rkey;
  }
  resCtx->remoteReadyQpMaskAddr = remInfo->resiliencyInfo.readyQpMaskAddr;
  for (int i = 0; i < resCtx->baseComm->nRemDevs; i++) {
    resCtx->remoteReadyQpMaskRkeys[i] = remInfo->resiliencyInfo.readyQpMaskRkeys[i];
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyClose(struct ncclIbResiliency* resCtx) {
  if (resCtx == NULL) {
    return ncclSuccess;
  }
  INFO(NCCL_NET, "NET/IB: %s: Destroying %d probing QPs for resiliency context (comm=%p)", __func__, resCtx->nProbingQps, resCtx->baseComm);
  for (int qpIndex = 0; qpIndex < resCtx->nProbingQps; qpIndex++) {
    struct ncclIbQp* probingQp = &resCtx->probingQps[qpIndex];
    assert(probingQp != NULL);
    assert(probingQp->qp != NULL);
    INFO(NCCL_NET, "NET/IB: %s: Destroying probing QP (index=%d, qp=%p, qp_num=%u) for resiliency context (comm=%p)", __func__, qpIndex, probingQp->qp, probingQp->qp->qp_num, resCtx->baseComm);
    NCCLCHECK(wrap_ibv_destroy_qp(probingQp->qp));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyRemoteCompletionRecordsSet(struct ncclIbResiliency* resCtx, uint32_t cmplsRecordsRkey, uint64_t cmplsRecordsAddr, uint devIndex) {
  assert(resCtx != NULL);
  assert(resCtx->baseComm->isSend);
  assert(devIndex <= resCtx->nProbingQps);
  struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
  sendResCtx->remCmplRecordsInfo[devIndex].rkey = cmplsRecordsRkey;
  sendResCtx->remCmplRecordsInfo[devIndex].addr = cmplsRecordsAddr;
  INFO(NCCL_NET, "NET/IB: %s: Set remote completion records info (comm=%p, addr=0x%lx, rkey=0x%x)", __func__, &resCtx->baseComm, cmplsRecordsAddr, cmplsRecordsRkey);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyRequestIsComplete(struct ncclIbRequest *request, bool *isComplete) {
  assert(isComplete != NULL);

  if (request == NULL) {
    *isComplete = false;
    return ncclSuccess;
  }

  struct ncclIbNetCommBase* baseComm = request->base;
  struct ncclIbResiliency* resCtx = baseComm->resiliency;
  assert(resCtx != NULL);

  int remainingEventsSum = 0;
  int negativeEventsSum = 0;

  // This loop sums the total number of events a request was updated with.
  // Each device on which the request was expected to generate an event was
  // added with the number of events expected on this device.
  // During the request processing, if a device was found to be faulty, the
  // request might have been processed on a different device, generating more
  // events on that device than expected initiatlly, causing the events counter
  // on that device to reach a negative value.
  for (int devIndex = 0; devIndex < resCtx->ndevs; devIndex++) {
    if (request->events[devIndex] < 0) {
      // If the counter < 0 it means that this device was serving requests
      // for a device that failed.
      negativeEventsSum += request->events[devIndex];
    } else {
      remainingEventsSum += request->events[devIndex];
    }
  }
  // A request is considered complete if (1) on every device the number of
  // events is zero, or (2) the absolute value of the sum of all the counters
  // with negative values – is equal to the total remaining completions for a
  // request.
  *isComplete = ((remainingEventsSum == 0) || (abs(negativeEventsSum) == remainingEventsSum));
#ifdef ENABLE_TRACE
  if (*isComplete == 1) {
    TRACE(NCCL_NET, "NET/IB: %s: request=%p, remainingEventsSum=%d, negativeEventsSum=%d, completed? %s", __func__, request, remainingEventsSum, negativeEventsSum, *isComplete ? "yes" : "no");
  }
#endif // ENABLE_TRACE
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyHandleCompletionError(struct ncclIbResiliency* resCtx, struct ibv_wc* wc, int devIndex) {
  INFO(NCCL_NET, "NET/IB: %s: Got completion with error (devIndex=%d, wc->status=(%s)%d, wc->opcode=(%s)%d, wc->wr_id=%ld, wc->qp_num=%u, wc->byte_len=%d)", __func__, devIndex, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->wr_id, wc->qp_num, wc->byte_len);
  NCCLCHECK(ncclIbResiliencyCheckErrorNotFatal(resCtx, wc, devIndex));

  // Receiver-side dummy RQ completions are emitted while draining a QP that has
  // already been moved into error. Once the device is already marked failed,
  // consume those drain CQEs without re-running the device failover path.
  if (ncclIbResiliencyIsReceiverDummyFlushCompletion(resCtx, wc) &&
      resCtx->devs[devIndex].state != ncclIbResiliencyDevStateOk) {
    return ncclIbResiliencyHandleCompletionErrorReceiver(resCtx, wc, devIndex);
  }

  // Before handling the request that got an error, first the device is
  // transitioned to a "failure" state to make sure that any potential replay
  // will not be done on the failed device.
  NCCLCHECK(ncclIbResiliencyHandleDeviceFailure(resCtx, devIndex));

  if (resCtx->baseComm->isSend) {
    NCCLCHECK(ncclIbResiliencyHandleCompletionErrorSender(resCtx, wc, devIndex));
  } else {
    NCCLCHECK(ncclIbResiliencyHandleCompletionErrorReceiver(resCtx, wc, devIndex));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyProgress(struct ncclIbResiliency* resCtx) {
  if (resCtx->inProgress == false) {
    // No operations needs to be done
    return ncclSuccess;
  }

  // Always try to rejoin failed QPs first. This now applies to send and recv
  // communicators so receiver-side QPs can also re-enter service.
  NCCLCHECK(ncclIbResiliencyTryRejoinQps(resCtx));

  if (!resCtx->baseComm->isSend) {
    // Receiver side currently only needs QP rejoin progression.
    if (!ncclIbResiliencyHasPendingQpRecovery(resCtx)) {
      resCtx->inProgress = false;
    }
    return ncclSuccess;
  }

  struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;

  NCCLCHECK(ncclIbResiliencyProbeProgress(sendResCtx));

  // Iterate over all failed requests and progress them.
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbResiliencyRequestSend* failedRequest = &sendResCtx->failedRequests[i];
    if (failedRequest->request == NULL) {
      continue;
    }
    if (failedRequest->state == ncclIbResiliencyRequestStatePending) {
      NCCLCHECK(ncclIbResiliencyProbePost(sendResCtx, failedRequest));
      continue;
    }
    if (failedRequest->state == ncclIbResiliencyRequestStateProbeCompleted) {
      NCCLCHECK(ncclIbResiliencyHandleProbeCompleted(sendResCtx, failedRequest));
      NCCLCHECK(ncclIbResiliencySendRequestFree(sendResCtx, failedRequest));
      continue;
    }
  }
  if (resCtx->outstandingRequests == 0 && !ncclIbResiliencyHasPendingQpRecovery(resCtx)) {
    INFO(NCCL_NET, "NET/IB: %s: All resiliency operations are completed for resiliency context (%s comm=%p). Marking progress as completed.", __func__, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
    resCtx->inProgress = false;
  }
  return ncclSuccess;
}
