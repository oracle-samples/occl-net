/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_P2P_H_
#define NET_IB_P2P_H_

#include "common.h"

#define NCCL_IB_FLUSH_REQ_WR_ID_OFFSET 0x1000
static_assert(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET > NET_IB_MAX_REQUESTS, "wr_id offset for flush requests must be greater than NET_IB_MAX_REQUESTS");
static_assert(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET <= UINT64_MAX - NET_IB_MAX_REQUESTS, "wr_id for flush requests must fit in 64 bits since ibv_send_wr::wr_id is 64 bits");
#define IB_REQ_WR_ID_FLAG (1ULL << 63)
#define IB_SEND_REQ_WR_ID_SLOT_BITS 8
#define IB_SEND_REQ_WR_ID_SLOT_MASK ((1ULL << IB_SEND_REQ_WR_ID_SLOT_BITS) - 1)
#define IB_SEND_REQ_WR_ID_EPOCH_BITS 24
#define IB_SEND_REQ_WR_ID_EPOCH_VALUES ((1ULL << IB_SEND_REQ_WR_ID_EPOCH_BITS) - 1)
static_assert(NET_IB_MAX_REQUESTS <= (1 << IB_SEND_REQ_WR_ID_SLOT_BITS), "send wr_id slot encoding must cover all request slots");
static_assert((IB_SEND_REQ_WR_ID_SLOT_BITS + IB_SEND_REQ_WR_ID_EPOCH_BITS) <= 32, "send wr_id must remain stable if CQEs preserve only low 32 bits");

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot);
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot);
void ncclIbSendReqsRefreshWrIdToken(struct ncclIbSendComm* comm, int slot, int nreqs);

static inline uint64_t ibReqWrIdEncode(uint64_t reqId) {
  assert((reqId & IB_REQ_WR_ID_FLAG) == 0);
  return IB_REQ_WR_ID_FLAG | reqId;
}

static inline bool ibReqWrIdIsRequest(uint64_t wrId) {
  return wrId != NCCL_IB_RECV_WR_ID_DUMMY && (wrId & IB_REQ_WR_ID_FLAG);
}

static inline uint64_t ibReqWrIdDecode(uint64_t wrId) {
  assert(ibReqWrIdIsRequest(wrId));
  return wrId & ~IB_REQ_WR_ID_FLAG;
}

static inline uint64_t ibSendReqWrIdEncode(uint64_t epoch, uint64_t slot) {
  assert(slot < NET_IB_MAX_REQUESTS);
  // Keep send WR-IDs entirely in the low 32 bits. Some stacks appear to
  // preserve only the lower portion of the WR-ID in successful send CQEs.
  const uint64_t epochTag = (epoch % IB_SEND_REQ_WR_ID_EPOCH_VALUES) + 1ULL;
  return (epochTag << IB_SEND_REQ_WR_ID_SLOT_BITS) | slot;
}

static inline uint64_t ibSendReqWrIdDecodeSlot(uint64_t wrId) {
  return wrId & IB_SEND_REQ_WR_ID_SLOT_MASK;
}

static inline struct ncclIbRequest* ibSendReqFind(struct ncclIbSendComm* sendComm, uint64_t reqId) {
  struct ncclIbRequest* req = sendComm->sendReqs[reqId % NET_IB_MAX_REQUESTS][0];
  return (req != NULL && req->id == reqId) ? req : NULL;
}

static inline struct ncclIbRequest* ibSendReqFindByWrId(struct ncclIbSendComm* sendComm, uint64_t wrId) {
  struct ncclIbRequest* req = sendComm->sendReqs[ibSendReqWrIdDecodeSlot(wrId)][0];
  if (req == NULL || req->type != NCCL_NET_IB_REQ_SEND) return NULL;
  const uint32_t token = (uint32_t)req->send.wrIdToken;
  return (token != 0 && token == (uint32_t)wrId) ? req : NULL;
}

static inline struct ncclIbRequest* ibRecvReqFind(struct ncclIbRecvComm* recvComm, uint64_t reqId) {
  struct ncclIbRequest* req = recvComm->recvReqs[reqId % NET_IB_MAX_REQUESTS];
  return (req != NULL && req->id == reqId) ? req : NULL;
}

static inline ncclResult_t ncclIbRecvCommGetQpForCts(struct ncclIbRecvComm* recvComm, uint32_t id, ncclIbQp** qp) {
  int devIndex = id % recvComm->base.vProps.ndevs;
  // CTS message is always posted the first QP on the device
  int qpIndex = 0;
  ncclIbCommBaseGetQpByIndex(&recvComm->base, devIndex, qpIndex, qp);
  assert(*qp != NULL);
  return ncclSuccess;
}

static inline ncclResult_t ncclIbRequestRetrieveAsIndex(ncclIbRequest* reqs, uint32_t reqIndex, ncclIbRequest** req) {
  if (reqIndex < 0 || reqIndex >= NET_IB_MAX_REQUESTS) {
    WARN("NET/IB: %s: Invalid request index %d. Not in the range [%d, %d). Cannot retrieve request.", __func__, reqIndex, 0, NET_IB_MAX_REQUESTS);
    return ncclInternalError;
  }
  *req = &reqs[reqIndex];
  return ncclSuccess;
}


#endif // NET_IB_P2P_H_
