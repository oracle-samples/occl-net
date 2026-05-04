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

static size_t ibFrmRoundUp(size_t bytes, size_t alignment) {
  return ((bytes + alignment - 1) / alignment) * alignment;
}

void ncclIbSendCommFreeMultiSendScratch(struct ncclIbSendComm* comm) {
  if (comm == NULL) return;

  struct ncclIbMultiSendScratch* scratch = &comm->multiSendScratch;
  free(scratch->storage);
  scratch->storage = NULL;
  scratch->nqps = 0;
  scratch->chunkRowBytes = 0;
  scratch->wrRowBytes = 0;
  scratch->sgeRowBytes = 0;
#if FRM_DEBUG
  scratch->sendSzRowBytes = 0;
#endif
  scratch->lwPerQp = NULL;
  scratch->equalChunkSize = NULL;
  scratch->reqOffsetShadow = NULL;
  scratch->sgeAddrShadow = NULL;
  scratch->remoteAddrShadow = NULL;
  scratch->chunkSizePerReq = NULL;
  scratch->qpPerPost = NULL;
  scratch->qpIndexPerPost = NULL;
  scratch->chainBytesPerPost = NULL;
  scratch->wrsPerPost = NULL;
  scratch->sgesPerPost = NULL;
#if FRM_DEBUG
  scratch->sendSzPerPost = NULL;
#endif
}

ncclResult_t ncclIbSendCommInitMultiSendScratch(struct ncclIbSendComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm == NULL) return ncclSuccess;

  struct ncclIbMultiSendScratch* scratch = &comm->multiSendScratch;
  const size_t nqps = (size_t)(comm->base.nqps > 0 ? comm->base.nqps : 0);
  const size_t cachelineBytes = 64;
  const size_t sectorBytes = 128;
  ncclIbSendCommFreeMultiSendScratch(comm);
  scratch->nqps = (int)nqps;
  if (nqps == 0) return ncclSuccess;

  const size_t lwBytes = ibFrmRoundUp(nqps * sizeof(float), cachelineBytes);
  const size_t equalChunkBytes = ibFrmRoundUp(NCCL_NET_IB_MAX_RECVS * sizeof(int), cachelineBytes);
  const size_t reqOffsetBytes = ibFrmRoundUp(NCCL_NET_IB_MAX_RECVS * sizeof(int), cachelineBytes);
  const size_t sgeAddrBytes = ibFrmRoundUp(NCCL_NET_IB_MAX_RECVS * sizeof(uintptr_t), cachelineBytes);
  const size_t remoteAddrBytes = ibFrmRoundUp(NCCL_NET_IB_MAX_RECVS * sizeof(uint64_t), cachelineBytes);
  const size_t chunkRowBytes = ibFrmRoundUp(nqps * sizeof(uint32_t), sectorBytes);
  const size_t chunkBytes = NCCL_NET_IB_MAX_RECVS * chunkRowBytes;
  const size_t qpPtrBytes = ibFrmRoundUp(nqps * sizeof(ncclIbQp*), cachelineBytes);
  const size_t qpIndexBytes = ibFrmRoundUp(nqps * sizeof(int), cachelineBytes);
  const size_t chainBytesBytes = ibFrmRoundUp(nqps * sizeof(uint32_t), cachelineBytes);
  const size_t wrRowBytes = ibFrmRoundUp((NCCL_NET_IB_MAX_RECVS + 1) * sizeof(struct ibv_send_wr), sectorBytes);
  const size_t wrBytes = nqps * wrRowBytes;
  const size_t sgeRowBytes = ibFrmRoundUp(NCCL_NET_IB_MAX_RECVS * sizeof(struct ibv_sge), sectorBytes);
  const size_t sgeBytes = nqps * sgeRowBytes;
#if FRM_DEBUG
  const size_t sendSzRowBytes = ibFrmRoundUp(NCCL_NET_IB_MAX_RECVS * sizeof(uint32_t), sectorBytes);
  const size_t sendSzBytes = nqps * sendSzRowBytes;
#endif

  assert(chunkRowBytes <= UINT32_MAX);
  assert(wrRowBytes <= UINT32_MAX);
  assert(sgeRowBytes <= UINT32_MAX);
#if FRM_DEBUG
  assert(sendSzRowBytes <= UINT32_MAX);
#endif
  scratch->chunkRowBytes = (uint32_t)chunkRowBytes;
  scratch->wrRowBytes = (uint32_t)wrRowBytes;
  scratch->sgeRowBytes = (uint32_t)sgeRowBytes;
#if FRM_DEBUG
  scratch->sendSzRowBytes = (uint32_t)sendSzRowBytes;
#endif

  size_t totalBytes = 0;
  const size_t lwOffset = totalBytes;
  totalBytes += lwBytes;
  const size_t equalChunkOffset = totalBytes;
  totalBytes += equalChunkBytes;
  const size_t reqOffsetOffset = totalBytes;
  totalBytes += reqOffsetBytes;
  const size_t sgeAddrOffset = totalBytes;
  totalBytes += sgeAddrBytes;
  const size_t remoteAddrOffset = totalBytes;
  totalBytes += remoteAddrBytes;
  totalBytes = ibFrmRoundUp(totalBytes, sectorBytes);
  const size_t chunkOffset = totalBytes;
  totalBytes += chunkBytes;
  const size_t qpPtrOffset = totalBytes;
  totalBytes += qpPtrBytes;
  const size_t qpIndexOffset = totalBytes;
  totalBytes += qpIndexBytes;
  const size_t chainOffset = totalBytes;
  totalBytes += chainBytesBytes;
  totalBytes = ibFrmRoundUp(totalBytes, sectorBytes);
  const size_t wrOffset = totalBytes;
  totalBytes += wrBytes;
  totalBytes = ibFrmRoundUp(totalBytes, sectorBytes);
  const size_t sgeOffset = totalBytes;
  totalBytes += sgeBytes;
#if FRM_DEBUG
  totalBytes = ibFrmRoundUp(totalBytes, sectorBytes);
  const size_t sendSzOffset = totalBytes;
  totalBytes += sendSzBytes;
#endif

  NCCLCHECKGOTO(ncclIbMalloc((void**)&scratch->storage, totalBytes), ret, fail);
  memset(scratch->storage, 0, totalBytes);
  {
    char* base = (char*)scratch->storage;
    scratch->lwPerQp = (float*)(base + lwOffset);
    scratch->equalChunkSize = (int*)(base + equalChunkOffset);
    scratch->reqOffsetShadow = (int*)(base + reqOffsetOffset);
    scratch->sgeAddrShadow = (uintptr_t*)(base + sgeAddrOffset);
    scratch->remoteAddrShadow = (uint64_t*)(base + remoteAddrOffset);
    scratch->chunkSizePerReq = (uint32_t*)(base + chunkOffset);
    scratch->qpPerPost = (ncclIbQp**)(base + qpPtrOffset);
    scratch->qpIndexPerPost = (int*)(base + qpIndexOffset);
    scratch->chainBytesPerPost = (uint32_t*)(base + chainOffset);
    scratch->wrsPerPost = (struct ibv_send_wr*)(base + wrOffset);
    scratch->sgesPerPost = (struct ibv_sge*)(base + sgeOffset);
#if FRM_DEBUG
    scratch->sendSzPerPost = (uint32_t*)(base + sendSzOffset);
#endif
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
  memset(&baseComm->frmCold, 0, sizeof(baseComm->frmCold));
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
  sendComm->sendWrIdCounter = 0;
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
