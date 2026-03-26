/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "p2p_resiliency.h"

char ncclIbIfName[MAX_IF_NAME_SIZE+1];
union ncclSocketAddress ncclIbIfAddr;

int ncclNMergedIbDevs = -1;
int ncclNIbDevs = -1;
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
int ncclIbRelaxedOrderingEnabled = 0;

ncclProfilerCallback_t ncclProfilerFunction;

NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);
NCCL_PARAM(IbPrepostReceiveWorkRequests, "IB_PREPOST_RECEIVE_WORK_REQUESTS", 0);
NCCL_PARAM(IbAsyncEvents,"IB_RETURN_ASYNC_EVENTS",1);

ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbAsyncEvents() && COMPILER_ATOMIC_LOAD(&stat->fatalErrorCount, std::memory_order_relaxed)) {
    WARN("communicator encountered a fatal error (detected in %s)", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*) base;
    return &sComm->devs[devIndex].base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*) base;
    return &rComm->devs[devIndex].base;
  }
}

static size_t ncclIbFrmRoundUpCacheline(size_t bytes) {
  const size_t cacheline = 64;
  return ((bytes + cacheline - 1) / cacheline) * cacheline;
}

void ncclIbSendCommFreeMultiSendScratch(struct ncclIbSendComm* comm) {
  if (comm == NULL) return;

  struct ncclIbMultiSendScratch* scratch = &comm->multiSendScratch;
  free(scratch->storage);
  scratch->storage = NULL;
  scratch->nqps = 0;
  scratch->chunkSizePerReq = NULL;
  scratch->qpPerPost = NULL;
  scratch->wrsPerPost = NULL;
  scratch->sgesPerPost = NULL;
  scratch->sendSzPerPost = NULL;
}

ncclResult_t ncclIbSendCommInitMultiSendScratch(struct ncclIbSendComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm == NULL) return ncclSuccess;

  struct ncclIbMultiSendScratch* scratch = &comm->multiSendScratch;
  const size_t nqps = (size_t)(comm->base.nqps > 0 ? comm->base.nqps : 0);
  ncclIbSendCommFreeMultiSendScratch(comm);
  scratch->nqps = (int)nqps;
  if (nqps == 0) return ncclSuccess;

  const size_t qpReqCount = nqps * NCCL_NET_IB_MAX_RECVS;
  const size_t wrCount = nqps * (NCCL_NET_IB_MAX_RECVS + 1);

  const size_t chunkBytes = ncclIbFrmRoundUpCacheline(qpReqCount * sizeof(size_t));
  const size_t qpPtrBytes = ncclIbFrmRoundUpCacheline(nqps * sizeof(ncclIbQp*));
  const size_t wrBytes = ncclIbFrmRoundUpCacheline(wrCount * sizeof(struct ibv_send_wr));
  const size_t sgeBytes = ncclIbFrmRoundUpCacheline(qpReqCount * sizeof(struct ibv_sge));
  const size_t sendSzBytes = ncclIbFrmRoundUpCacheline(qpReqCount * sizeof(size_t));
  const size_t totalBytes = chunkBytes + qpPtrBytes + wrBytes + sgeBytes + sendSzBytes;

  NCCLCHECKGOTO(ncclIbMalloc((void**)&scratch->storage, totalBytes), ret, fail);
  memset(scratch->storage, 0, totalBytes);
  {
    char* ptr = (char*)scratch->storage;
    scratch->chunkSizePerReq = (size_t*)ptr;
    ptr += chunkBytes;
    scratch->qpPerPost = (ncclIbQp**)ptr;
    ptr += qpPtrBytes;
    scratch->wrsPerPost = (struct ibv_send_wr*)ptr;
    ptr += wrBytes;
    scratch->sgesPerPost = (struct ibv_sge*)ptr;
    ptr += sgeBytes;
    scratch->sendSzPerPost = (size_t*)ptr;
  }

  for (size_t i = 0; i < nqps; i++) {
    struct ibv_send_wr* wrRow = ncclIbMultiSendScratchWrRow(scratch, (int)i);
    struct ibv_sge* sgeRow = ncclIbMultiSendScratchSgeRow(scratch, (int)i);

    for (int r = 0; r < NCCL_NET_IB_MAX_RECVS; r++) {
      wrRow[r].sg_list = &sgeRow[r];
      wrRow[r].num_sge = 1;
      wrRow[r].opcode = IBV_WR_RDMA_WRITE;
      wrRow[r].send_flags = 0;
      wrRow[r].next = (r == NCCL_NET_IB_MAX_RECVS - 1) ? &wrRow[NCCL_NET_IB_MAX_RECVS] : &wrRow[r + 1];
    }

    wrRow[NCCL_NET_IB_MAX_RECVS].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wrRow[NCCL_NET_IB_MAX_RECVS].send_flags = IBV_SEND_SIGNALED;
    wrRow[NCCL_NET_IB_MAX_RECVS].next = NULL;
  }

  return ncclSuccess;
fail:
  ncclIbSendCommFreeMultiSendScratch(comm);
  return ret;
}

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend) {
  memset(&baseComm->frmFields, 0, sizeof(baseComm->frmFields));
  for (int i = 0; i < NCCL_IB_MAX_QPS; i++) {
    baseComm->qps[i].ibPort = 0;
    baseComm->qps[i].accessFlags = 0;
    baseComm->qps[i].devIndex= -1;
    baseComm->qps[i].remDevIdx= -1;
    baseComm->qps[i].remQpn = 0;
    memset(&baseComm->qps[i].remEce, 0, sizeof(baseComm->qps[i].remEce));
    baseComm->qps[i].remEceSupported = 0;
    baseComm->qps[i].sig_head = 0;
    baseComm->qps[i].sig_tail = 0;
    memset(baseComm->qps[i].sig_tsc, 0, sizeof(baseComm->qps[i].sig_tsc));
    memset(baseComm->qps[i].sig_bytes, 0, sizeof(baseComm->qps[i].sig_bytes));
    baseComm->activeQps[i] = &baseComm->qps[i];
  }
  baseComm->nqps = -1;
  baseComm->splitDataOnQps = ncclParamIbSplitDataOnQps();
  baseComm->nDataQps = -1;
  baseComm->connTc = 0;
  baseComm->connSl = 0;
  baseComm->isSend = isSend;
  baseComm->ready = 0;

  NCCLCHECK(ncclIbResiliencyInit(baseComm, &baseComm->resiliency));

  return ncclSuccess;
}

ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm) {
  NCCLCHECK(ncclIbBaseCommInit(&recvComm->base, false));
  recvComm->ibRecvWorkRequest = {
    .wr_id = NCCL_IB_RECV_WR_ID_DUMMY,
    .next = NULL,
    .sg_list = NULL,
    .num_sge = 0
  };
  if (recvComm->base.resiliency) {
    if (ncclParamIbPrepostReceiveWorkRequests() == 0) {
      WARN("NET/IB: %s: Resiliency requires pre-posted receive work requests. Enabling pre-posting.", __func__);
    }
    recvComm->prepostReceiveWorkRequests = true;
  } else {
    recvComm->prepostReceiveWorkRequests = (ncclParamIbPrepostReceiveWorkRequests() == 1);
  }
  INFO(NCCL_NET, "NET/IB: %s: Receive work requests will be %s", __func__, recvComm->prepostReceiveWorkRequests ? "pre-posted" : "posted on-demand");
  return ncclSuccess;
}

ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm) {
  NCCLCHECK(ncclIbBaseCommInit(&sendComm->base, true));
  sendComm->cim = NULL;
  memset(&sendComm->multiSendScratch, 0, sizeof(sendComm->multiSendScratch));
  sendComm->frmGateOpen = false;
  return ncclSuccess;
}

std::thread ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args) {
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      ncclIbDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      // the above is a CQ fatal error
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      ncclIbCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // the above are QP fatal errors
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      ncclIbQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in NCCL
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_GID_CHANGE:
      WARN("NET/IB : %s:%d GID table changed", dev->devName, dev->portNum);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got non-fatal async event: %s(%d)", dev->devName, dev->portNum, str, event.event_type);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

ncclNet_t ncclNetIb = {
  "IB",
  ncclIbInit,
  ncclIbDevices,
  ncclIbGetProperties,
  ncclIbListen,
  ncclIbConnect,
  ncclIbAccept,
  ncclIbRegMr,
  ncclIbRegMrDmaBuf,
  ncclIbDeregMr,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbIflush,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  ncclIbMakeVDevice,
  ncclIbFinalize,
  ncclIbSetNetAttr,
};
