/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_COMMON_H_
#define NET_IB_COMMON_H_

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "plugin/nccl_net.h"
#include "utils.h"
#include "param.h"
#include "profiler/net_ib.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"
#include "cim.h"

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"

#ifndef FRM_LAT_PRINT_MS
// Build-time trace knob for CIM latency/weight snapshots.
// 0 disables tracing; any positive value enables it and sets the print interval in ms.
#define FRM_LAT_PRINT_MS 0
#endif

#ifndef FRM_DEBUG
#define FRM_DEBUG 0
#endif

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)
extern char ncclIbIfName[MAX_IF_NAME_SIZE+1];
extern union ncclSocketAddress ncclIbIfAddr;

struct ncclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr *mr;
};



struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

extern int ncclNMergedIbDevs;
#define NCCL_IB_MAX_DEVS_PER_NIC 4
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
struct alignas(64) ncclIbMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

struct ncclIbStats {
  int fatalErrorCount;
};

enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

extern int ncclNIbDevs;
struct alignas(64) ncclIbDev {
  std::mutex mutex;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char fullPciPath[PATH_MAX];
  char* pciPath;
  int realPort;
  int maxQp;
  float latency;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
  enum ncclIbProvider ibProvider;
  union {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
};

#define MAX_IB_DEVS  32
#define MAX_IB_VDEVS MAX_IB_DEVS*8
extern struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
extern struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
extern int ncclIbRelaxedOrderingEnabled;

#define NCCL_IB_LLSTR(ll) (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

// Per-Dev connection metadata
struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE and IB Rounter
  union ibv_gid gid;

  // The key used for remote access to the addr exchanged by the peers
  // in ncclIbConnectionMetadata::addr
  // This member is populated differently on the sender and on the receiver
  // side.
  // The sender side populates this member with the RKey obtained after it
  // registered the CTS FIFO (on the specific device).
  // The receiver side populates this member with the RKey obtained after it
  // registered the completion records structure (on the specific device).
  uint32_t rkey;

  //remote dev info
  union ibv_gid remoteGid;
};

// Retain local RoCE address for error logging
struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

#define MAX_QPS_PER_REQ 8
struct ncclProfilerInfo {
  void* qpEventHandles[MAX_QPS_PER_REQ];
  int qpIndex[MAX_QPS_PER_REQ];
  int nEventHandles;
  ncclProfilerNetIbDescr_v1_t data;
  void* pHandle;
};

#define NCCL_NET_IB_MAX_RECVS 8

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
#define NCCL_NET_IB_REQ_GIN_IPUT 4
extern const char* ncclIbReqTypeStr[];

// Maximal number of QPs a communicator can have for data transfers
#define NCCL_IB_MAX_QPS 128

// Tracks data transfers between sender and receiver. A multi-recv/send uses a
// single record.
struct ncclIbRequestCompletionRecord {
  // This array communicates data transfer sizes from the sender to the
  // receiver. The sender writes the size of each completed data transfer to
  // this array. The receiver reads these sizes before reporting completion of
  // the corresponding receive request to the user.
  int sizes[NCCL_NET_IB_MAX_RECVS];
  // The receiver fills this array with the number of successfully received data
  // completions per physical QP. The sender can then reconcile those physical
  // counts against its logical request-to-QP mapping across failovers/replays
  // without assuming that logical slot indices still match the physical QP
  // indices that produced the completions.
  uint16_t completions[NCCL_IB_MAX_QPS];
};

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  // Array of counters. Each element in the array is populated with the expected
  // number of completion events that the request is expecting to be generated
  // on the device corresponding to the index of the element. After the request
  // is initialized and posted, for every completion event generated by a
  // device, the corresponding counter is decremented. When the counter reaches
  // zero it means that the request was fully completed on that device.
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to the per-device base structures to make it easier to
  // poll the device's CQ when the request is tested for progress.
  // The pointers are initialized only for the devices that the request expects
  // to receive completions from.
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct ncclProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];
#endif
  uint64_t id;
  int nreqs;
  union {
    struct {
      // Current send-attempt CQE token. Reposts refresh this so late CQEs from
      // older attempts cannot masquerade as fresh failures on the same request.
      uint64_t wrIdToken;
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      // Tracks logical request chunks that still need delivery.
      bool pendingData[NCCL_IB_MAX_QPS];
      // Physical QP index used for each logical chunk in the most recent send
      // attempt. Entries remain -1 for logical chunks that were not attempted.
      int lastAttemptQpIndex[NCCL_IB_MAX_QPS];
      // In-order position of the logical chunk within the physical QP stream
      // for the most recent send attempt.
      uint8_t lastAttemptOrder[NCCL_IB_MAX_QPS];
      // Number of receiver completions already credited per physical QP.
      uint16_t ackedCompletions[NCCL_IB_MAX_QPS];
    } send;
    struct {
      struct ncclIbRequestCompletionRecord* cmplsRecords;
      // Aggregates the size of a send request when sender does not write to the
      // completion records array.
      int aggSize;
    } recv;
    struct {
      int rank;
    } iput;
  };
  int connectionId;
};

struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
};

struct ncclIbSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
  char padding[16];
};

#define QP_SIG_RING NCCL_NET_IB_MAX_RECVS

struct alignas(128) ncclIbQp {
  struct ibv_qp* qp;
  // The index of the device on which this QP was created on.
  int devIndex;
  // The index of the device on the remote side to which this QP is connected
  // to.
  int remDevIdx;
  // FRM completion ordering keeps only timestamp and byte-count state here.
  // Request recovery for send CQEs now comes uniformly from wr_id lookup.
  uint32_t sig_head;
  uint32_t sig_tail;
  uint64_t sig_tsc[QP_SIG_RING];
  uint32_t sig_bytes[QP_SIG_RING];
  // Cached local QP attributes used to restore INIT state during rejoin.
  unsigned int accessFlags;
  // Remote QP number paired with this local QP.
  // This is required to rebuild QP state (RTR/RTS) after failures.
  uint32_t remQpn;
  int remEceSupported;
  uint8_t ibPort;
  // Cached remote reduced ECE parameters for data-QP rejoin.
  struct ibv_ece remEce;
};
static_assert(alignof(struct ncclIbQp) == 128, "ncclIbQp must be 128-byte aligned");
static_assert((sizeof(struct ncclIbQp) % 128) == 0, "ncclIbQp size must be a 128-byte multiple");
static_assert(offsetof(struct ncclIbQp, qp) == 0, "ncclIbQp::qp offset changed");
static_assert(offsetof(struct ncclIbQp, sig_tsc) == 24, "ncclIbQp::sig_tsc should stay in the first 128B region");
static_assert(offsetof(struct ncclIbQp, sig_bytes) == 88, "ncclIbQp::sig_bytes should stay in the first 128B region");
static_assert(offsetof(struct ncclIbQp, accessFlags) == 120, "ncclIbQp::accessFlags should stay in the first 128B region");

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

// Structure to describe the completion records on the sender side.
struct ncclIbRemCompletionsRecords {
  // A "shadow" structure of the receiver's completion records in which the
  // sender tracks the completion records locally on its side. Sender uses this
  // memory to place the records it writes/reads to/from the receiver's
  // completion records.
  int elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  // Address in memory of the completion records structure on the receiver side.
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the sender chooses the
  // RKey (depending on the device being used) when it accesses the receiver's
  // completion records structure.
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
};

// A per-dev struct for netIbSendComm
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* ctsFifoMr;
  struct ibv_mr* putSignalScratchpadMr;
  struct ibv_mr* cmplsRecordsMr;
  struct ibv_sge sge;
};


// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

#define QP_MAP_BITS 7
#define QP_MAP_SIZE (1u << QP_MAP_BITS)
#define QP_MAP_MASK (QP_MAP_SIZE - 1u)

typedef struct {
  uint32_t keys[QP_MAP_SIZE];
  uint16_t vals[QP_MAP_SIZE];
} ncclIbQpNumHash;

// Forward declaration
struct ncclIbResiliency;

struct alignas(64) ncclIbNetCommFrmFields {
  uint64_t qp_lat[NCCL_IB_MAX_QPS];
  ncclIbQpNumHash qpHash;
  uint8_t qp_zero_stamp_holdoff[NCCL_IB_MAX_QPS];
};
static_assert(alignof(struct ncclIbNetCommFrmFields) == 64, "ncclIbNetCommFrmFields must be 64-byte aligned");
static_assert((sizeof(struct ncclIbNetCommFrmFields) % 64) == 0, "ncclIbNetCommFrmFields size must be 64-byte multiple");

struct alignas(64) ibFrmColdFields {
  // Raw latency and byte-count telemetry are updated on the completion/debug
  // side and are not consulted when the send path computes FRM weights.
  uint64_t qp_raw_lat[NCCL_IB_MAX_QPS];
  uint32_t qp_lat_bytes[NCCL_IB_MAX_QPS];
};
static_assert(alignof(struct ibFrmColdFields) == 64, "ibFrmColdFields must be 64-byte aligned");
static_assert((sizeof(struct ibFrmColdFields) % 64) == 0, "ibFrmColdFields size must be 64-byte multiple");

struct alignas(128) ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;
  bool isSend;
  struct ncclIbRequest reqs[NET_IB_MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
  uint64_t fifoHead;
  int nqps;
  int splitDataOnQps;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // statistics about the comm
  struct ncclIbStats stats;
  int nDataQps;
  struct ncclIbNetCommFrmFields frmFields;
  // Array of pointers to the "actual" QPs that are used for data transfers.
  // The pointers point to QPs in the ncclIbNetCommBase::qps[] array and may be
  // remapped away from the physical qps[] order during failover/recovery.
  struct ncclIbQp* activeQps[NCCL_IB_MAX_QPS];
  // Cached traffic-class and service-level used during QP bring-up.
  // Rejoin logic reuses these values when moving a failed QP back to RTR/RTS.
  int connTc;
  int connSl;
  struct ncclIbResiliency* resiliency;
  struct ibFrmColdFields frmCold;
};
static_assert(alignof(struct ncclIbNetCommBase) == 128, "ncclIbNetCommBase must be 128-byte aligned");
static_assert(offsetof(struct ncclIbNetCommBase, reqs) < offsetof(struct ncclIbNetCommBase, frmFields), "reqs should stay ahead of frmFields");
static_assert(offsetof(struct ncclIbNetCommBase, qps) < offsetof(struct ncclIbNetCommBase, frmFields), "qps should stay ahead of frmFields");
static_assert(offsetof(struct ncclIbNetCommBase, remDevs) < offsetof(struct ncclIbNetCommBase, frmFields), "remDevs should stay ahead of frmFields");
static_assert(offsetof(struct ncclIbNetCommBase, stats) < offsetof(struct ncclIbNetCommBase, frmFields), "stats should stay ahead of frmFields");
static_assert(offsetof(struct ncclIbNetCommBase, activeQps) > offsetof(struct ncclIbNetCommBase, frmFields), "activeQps should stay behind frmFields");
static_assert(offsetof(struct ncclIbNetCommBase, frmCold) > offsetof(struct ncclIbNetCommBase, activeQps), "frmCold should stay behind activeQps");

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex);

// qpIndex is the index relative to a device.
// For example, if a device has 2 QPs, qpIndex can be 0 or 1.
static inline ncclResult_t ncclIbCommBaseGetQpByIndex(struct ncclIbNetCommBase* commBase, int devIndex, int qpIndex, ncclIbQp** qp) {
  assert(devIndex >= 0 && devIndex < commBase->vProps.ndevs);
  *qp = commBase->activeQps[commBase->vProps.ndevs*qpIndex + devIndex];
  return ncclSuccess;
}

// The function selects the QP to be used for the request. The QP selected
// based on the request ID and also based on the provided QP index. A request
// can be posted on multiple QPs. For example, if a request is posted on 4
// QPs, this function should be called 4 times, each time with a different
// qpIndex, ranging from 0 to 3.
// The function outputs the selected QP in the outQp argument and populates the
// outQpIndex argument with the index of the selected QP. Note that the
// outQpIndex is the index of the QP in the base::qps[] array.
static inline ncclResult_t ncclIbCommBaseGetQpForRequest(struct ncclIbNetCommBase* baseComm, const uint64_t id, const uint8_t qpIndex, ncclIbQp** outQp, int* outQpIndex) {
  *outQpIndex = (id + qpIndex) % baseComm->nqps;
  *outQp = baseComm->activeQps[*outQpIndex];
  assert(*outQp != NULL);
  return ncclSuccess;
}

// Get a QP object from a QP number. If not NULL, qpIndex will also return the
// index of the QP in the ncclIbNetCommBase::qps[] array.
static inline ncclResult_t ncclIbCommBaseGetQpByQpNum(struct ncclIbNetCommBase* commBase, int devIndex, uint32_t qpNum, ncclIbQp** qp, int* qpIndex) {
  assert(devIndex >= 0 && devIndex < commBase->vProps.ndevs);
  assert(qp != NULL);
  TRACE(NCCL_NET, "NET/IB: %s: Looking for QP num %u on devIndex %d among %d QPs", __func__, qpNum, devIndex, commBase->nqps / commBase->vProps.ndevs);
  for (int qpIndexInDev = 0; qpIndexInDev < (commBase->nqps / commBase->vProps.ndevs); qpIndexInDev++) {
    *qp = &(commBase->qps[commBase->vProps.ndevs*qpIndexInDev + devIndex]);
    if ((*qp)->qp->qp_num == qpNum) {
      if (qpIndex != NULL) {
        *qpIndex = *qp - commBase->qps;
      }
      return ncclSuccess;
    }
  }
  *qp = NULL;
  return ncclInternalError;
}

// Each request is transfered over all devices, and depending on the
// "splitDataOnQps" configuration parameter, a request may be transffered over
// a single QP per device or on all QPs of each device.
static inline int ncclIbCommBaseGetNqpsPerRequest(struct ncclIbNetCommBase* baseComm) {
  assert(baseComm->nDataQps != -1);
  assert(baseComm->nqps != -1);
  return (baseComm->splitDataOnQps == 1) ? baseComm->nqps : baseComm->nDataQps;
}

static inline ncclResult_t ncclIbPostRecvWorkRequest(struct ibv_qp* qp, struct ibv_recv_wr* wr) {
  struct ibv_recv_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_recv(qp, wr, &bad_wr));
  return ncclSuccess;
}

struct ncclIbMultiSendScratch {
  void* storage;
  int nqps;
  // These row strides are padded to 128B so each row start stays aligned with
  // the targeted cache sector boundary.
  uint32_t chunkRowBytes;
  uint32_t wrRowBytes;
  uint32_t sgeRowBytes;
#if FRM_DEBUG
  uint32_t sendSzRowBytes;
#endif
  float* lwPerQp;
  int* equalChunkSize;
  int* reqOffsetShadow;
  uintptr_t* sgeAddrShadow;
  uint64_t* remoteAddrShadow;
  uint32_t* chunkSizePerReq;
  ncclIbQp** qpPerPost;
  int* qpIndexPerPost;
  uint32_t* chainBytesPerPost;
  struct ibv_send_wr* wrsPerPost;
  struct ibv_sge* sgesPerPost;
#if FRM_DEBUG
  uint32_t* sendSzPerPost;
#endif
};

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // Start with CTS FIFO and ibv structs as they have alignment restrictions

  // CTS FIFO from which the sender reads the Clear-to-Send (CTS) messages that
  // are written by the receiver (The receiver side writes into it upon
  // issuing a (multi-)receive request). Each row in the 2D array corresponds
  // to a single CTS message but can describe multiple recv-requests issued
  // on the receiver side.
  struct ncclIbSendFifo ctsFifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the send requests for faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array based on the "slot" they
  // are associated with.
  struct ncclIbRequest* sendReqs[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];

  // Counter per "slot" on how many send request were called for a multi-recv
  int sendReqsCnt[NET_IB_MAX_REQUESTS];
  uint64_t sendWrIdCounter;
  struct ncclIbRemCompletionsRecords remCmplsRecords;
  int ar; // Use adaptive routing when all merged devices have it enabled
  uint64_t putSignalScratchpad;
  CIMState* cim;
  struct ncclIbMultiSendScratch multiSendScratch;
  bool frmGateOpen;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure ctsFifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, ctsFifo) % 32) == 0, "ncclIbSendComm ctsFifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

static inline uint32_t* ncclIbMultiSendScratchChunkRow(struct ncclIbMultiSendScratch* scratch, int reqIndex) {
  return (uint32_t*)((char*)scratch->chunkSizePerReq + (size_t)reqIndex * (size_t)scratch->chunkRowBytes);
}

static inline struct ibv_send_wr* ncclIbMultiSendScratchWrRow(struct ncclIbMultiSendScratch* scratch, int qpSlot) {
  return (struct ibv_send_wr*)((char*)scratch->wrsPerPost + (size_t)qpSlot * (size_t)scratch->wrRowBytes);
}

static inline struct ibv_sge* ncclIbMultiSendScratchSgeRow(struct ncclIbMultiSendScratch* scratch, int qpSlot) {
  return (struct ibv_sge*)((char*)scratch->sgesPerPost + (size_t)qpSlot * (size_t)scratch->sgeRowBytes);
}

#if FRM_DEBUG
static inline uint32_t* ncclIbMultiSendScratchSendSizeRow(struct ncclIbMultiSendScratch* scratch, int qpSlot) {
  return (uint32_t*)((char*)scratch->sendSzPerPost + (size_t)qpSlot * (size_t)scratch->sendSzRowBytes);
}
#endif

struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ncclIbQp qp;
};

// This structure describes the FIFO which the receiver uses when it sends CTS
// messages to the sender.
struct ncclIbRemCtsFifo {
  // A "shadow" structure of the sender's CTS FIFO in which the receiver tracks
  // the CTS FIFO locally on its side. Receiver uses this memory to place the
  // CTS messages and populates the RDMA message "gather address" with the
  // memory of the CTS message that is sent.
  struct ncclIbSendFifo elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the receiver chooses the
  // RKey (depending on the device being used) when it posts a CTS to the
  // sender
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
};

struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
  // MR that is obtained after registering the "shadow" CTS FIFO on the
  // receiver's side. The LKey of this MR allows RDMA operations on the receiver
  // side to gather CTS messages (formatted by the receiver) and write them to
  // the sender's CTS FIFO.
  struct ibv_mr* ctsFifoMr;
  // MR that is obtained after registering the completion records on the
  // receiver side. The RKey of this MR is provided to the sender side, to allow
  // the sender side to access receiver's completion records using RDMA
  // operations.
  struct ibv_mr* cmplsRecordsMr;
  // SGE to avoid allocation of SGE structures on the stack when receiver
  // posts RDMA operations. The SGE is populated by the address of the memory
  // in which the CTS message formatted on the receiver is placed.
  struct ibv_sge sge;
};

#define NCCL_IB_RECV_WR_ID_DUMMY UINT64_MAX

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the recv requests to allow faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array using a hash (modulo) on
  // their ID.
  struct ncclIbRequest* recvReqs[NET_IB_MAX_REQUESTS];
  // Structure to hold all the related structures regarding the CTS FIFO
  // structure.
  struct ncclIbRemCtsFifo remCtsFifo;
  // Structure to hold all the completion records of all the outstanding
  // receive requests on the receiver side.
  struct ncclIbRequestCompletionRecord cmplsRecords[NET_IB_MAX_REQUESTS];
  int gpuFlushHostMem;
  int flushEnabled;
  bool prepostReceiveWorkRequests;
  // To avoid allocation and memset on the data-path a single structure is used
  // and only the wr_id is updated before posting a receive work request.
  struct ibv_recv_wr ibRecvWorkRequest;
};
static_assert((offsetof(struct ncclIbRecvComm, remCtsFifo) % 32) == 0, "ncclIbRecvComm ctsFifo must be 32-byte aligned");

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend);
ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm);
ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm);

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage* stage;
};

static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  COMPILER_ATOMIC_STORE(&stat->fatalErrorCount, 0, std::memory_order_relaxed);
  return ncclSuccess;
}
static void ncclIbStatsFatalError(struct ncclIbStats* stat){
  COMPILER_ATOMIC_FETCH_ADD(&stat->fatalErrorCount, 1, std::memory_order_relaxed);
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}
ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName);

extern ncclProfilerCallback_t ncclProfilerFunction;

extern std::thread ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args);

ncclResult_t ncclIbGdrSupport();
ncclResult_t ncclIbPeerMemSupport();
ncclResult_t ncclIbDmaBufSupport(int dev);

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex);
ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex);
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req);
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r);
ncclResult_t ncclIbSendCommInitMultiSendScratch(struct ncclIbSendComm* comm);
void ncclIbSendCommFreeMultiSendScratch(struct ncclIbSendComm* comm);

ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle);

// Net IB plugin entry functions.
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbDevices(int* ndev);
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);
ncclResult_t ncclIbConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/);
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/);
ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle);
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request);
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request);
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t ncclIbTest(void* request, int* done, int* sizes);
ncclResult_t ncclIbCloseSend(void* sendComm);
ncclResult_t ncclIbCloseRecv(void* recvComm);
ncclResult_t ncclIbCloseListen(void* listenComm);
ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props);
ncclResult_t ncclIbFinalizeDevices(void);
ncclResult_t ncclIbFinalize(void* ctx);
ncclResult_t ncclIbSetNetAttr(void *ctx, ncclNetAttr_t *netAttr);

static inline uint32_t qpHash32(uint32_t x) {
  // Fold upper bits into the low bits we keep after QP_MAP_MASK, then use one multiply.
  x ^= x >> 16;
  return x * 0x9e3779b1U;
}
static inline void qpMapClear(ncclIbQpNumHash* m) {
  memset(m->keys, 0, sizeof(m->keys));
  // vals can be left uninitialized; keys==0 denotes empty.
}

static inline void qpMapInsert(ncclIbQpNumHash* m, uint32_t qp_num, uint16_t qpIndex) {
  // qp_num should never be 0; if it is, shift it to 1.
  if (qp_num == 0) qp_num = 1;

  uint32_t h = qpHash32(qp_num) & QP_MAP_MASK;
  for (uint32_t probes = 0; probes < QP_MAP_SIZE; probes++) {
    uint32_t k = m->keys[h];
    if (k == 0 || k == qp_num) {
      m->keys[h] = qp_num;
      m->vals[h] = qpIndex;
    #if FRM_DEBUG
      fprintf(stdout, "inserted this qp_num=%u for qp_index=%d\n", qp_num, qpIndex);
      fflush(stdout);
    #endif
      return;
    }
    h = (h + 1) & QP_MAP_MASK; // linear probing
  }

  // If you ever hit this, table is too small or someone inserted garbage.
  // In NCCL plugin style, you could WARN() + abort/return error from init.
}

static inline int qpMapFind(const ncclIbQpNumHash* m, uint32_t qp_num) {
  if (qp_num == 0) qp_num = 1;
  uint32_t h = qpHash32(qp_num) & QP_MAP_MASK;
  uint32_t k = m->keys[h];
  if (k == qp_num) return (int)m->vals[h];
  if (k == 0) return -1;

  for (uint32_t probes = 1; probes < QP_MAP_SIZE; probes++) {
    h = (h + 1) & QP_MAP_MASK;
    k = m->keys[h];
    if (k == qp_num) return (int)m->vals[h];
    if (k == 0) return -1;
  }
  return -1;
}

static inline void qpMapBuild(ncclIbNetCommBase* base) {
  if (base == NULL) return;
  qpMapClear(&base->frmFields.qpHash);
#if FRM_DEBUG
  fprintf(stdout,"Building qmap\n");
  fflush(stdout);
#endif
  for (int i = 0; i < base->nqps; i++) {
    if (base->qps[i].qp == NULL) continue;
    uint32_t qp_num = base->qps[i].qp->qp_num;
    qpMapInsert(&base->frmFields.qpHash, qp_num, (uint16_t)i);
  }
}

int64_t ncclParamOciFrmEnable();
int64_t ncclParamIbSplitDataOnQps();
extern bool g_ncclIbUseCim;
/**
 * Cached conversion factor: cycles -> microseconds (hot path).
 *
 * IMPORTANT: must have external linkage (single global) so init.cc can
 * initialize it once and p2p.cc can read it.
 */
extern double g_inv_cyc_per_us;

#endif
