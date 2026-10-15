/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * Modifications by Oracle and/or its affiliates are licensed under the
 * Universal Permissive License (UPL), Version 1.0. See LICENSE.txt.
 * Modified by Oracle and/or its affiliates on 2026-03-06.
 *************************************************************************/

/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See third_party/licenses/Apache-2.0.txt for more license information
 *************************************************************************/

#ifndef NET_IB_P2P_RESILIENCY_H_
#define NET_IB_P2P_RESILIENCY_H_



#include "common.h"
#include "connect.h"

enum ncclIbResiliencyDevState {
  // The device is operating normally.
  ncclIbResiliencyDevStateOk = 0,
  // The device encountered an error and and can be tried to be recovered.
  ncclIbResiliencyDevStateError,
  // The device was determined to be failed and will not be used anymore.
  ncclIbResiliencyDevStateErrorPermanent
};

struct alignas(64) ncclIbResiliencyDev {
  enum ncclIbResiliencyDevState state;
  // CQ to get CQEs on the sender side for probing operations.
  // Receiver side is not expected to get any CQEs on the this CQ but Verbs
  // requires a CQ to be associated with a QP.
  struct ibv_cq* probingCq;
  int nOutstandingProbes;
  // MR to allow the RDMA Read to write to local memory on the sender side.
  // Note that this memory registration is not required to be communicated to
  // the receiver side as the memory is only used as a target for RDMA Read
  // operations initiated by the sender side.
  struct ibv_mr* probingResultMr;
  // MR exposing local ready-state snapshot to peer via RDMA read.
  struct ibv_mr* readyStateMr;
  // MR used as local write target for RDMA-read of peer ready-state snapshot.
  struct ibv_mr* peerReadyStateMr;
  // Temporary backoff window for auxiliary peer-ready reads on this device.
  uint64_t peerReadyRetryAfterNs;
  int peerReadyFailureCount;
};
static_assert(alignof(struct ncclIbResiliencyDev) == 64, "ncclIbResiliencyDev must be 64-byte aligned");
static_assert((sizeof(struct ncclIbResiliencyDev) % 64) == 0, "ncclIbResiliencyDev size must be a 64-byte multiple");

enum ncclIbResiliencyQpState {
  // QP is healthy and can be selected for traffic.
  ncclIbResiliencyQpStateOk = 0,
  // QP failed and is excluded from active traffic.
  ncclIbResiliencyQpStateFailed,
  // QP was rebuilt locally and is waiting for peer-ready confirmation.
  ncclIbResiliencyQpStateLocalReady,
  // QP exceeded the local retry budget and will remain on its failover path.
  ncclIbResiliencyQpStateFailedPermanent,
};

struct ncclIbResiliencyQpInfo {
  enum ncclIbResiliencyQpState state;
  uint64_t lastAttemptNs;
  // Once a failed QP is observed on a locally inactive port, suppress repeated
  // retry logs until the port becomes active again.
  bool localPortRetryLogged;
  int failCount;
};

struct alignas(64) ibReadyStateSnapshot {
  uint64_t mask[(NCCL_IB_MAX_QPS + 63) / 64];
  uint32_t epoch[NCCL_IB_MAX_QPS];
};
static_assert(alignof(struct ibReadyStateSnapshot) == 64, "ibReadyStateSnapshot must be 64-byte aligned");
static_assert((sizeof(struct ibReadyStateSnapshot) % 64) == 0, "ibReadyStateSnapshot size must be a 64-byte multiple");

struct alignas(128) ncclIbResiliency {
  // Back pointer to the base communicator.
  struct ncclIbNetCommBase* baseComm;

  struct ncclIbResiliencyDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  int ndevs;

  // QPs used for probing of data transfers in case of QP/device failures.
  // Both peers create max(local devices, remote devices) probing paths so their
  // path indices agree for asymmetric groups. Paths are placed modulo the local
  // device count, which leaves equal-width configurations unchanged.
  struct ncclIbQp probingQps[NCCL_IB_MAX_DEVS_PER_NIC];
  int nProbingQps;
  // Failed auxiliary RC QPs cannot self-heal. Keep them quarantined for this
  // communicator without changing the associated data-device state.
  uint64_t failedProbeQpMask;

  // As long as this variable is true, it means that resiliency operations are
  // in progress.
  bool inProgress;

  // Number of requests that are currently being handled by the resiliency
  // module.
  int outstandingRequests;

  // Cached summary for the FRM gate so the send completion path does not need
  // to rescan every QP state on each successful CQE.
  bool frmAllQpsHealthy;

  // Keep per-QP rejoin state dense: progress scans the whole array, so
  // per-entry cache-line alignment would reduce useful residency.
  struct ncclIbResiliencyQpInfo qpInfo[NCCL_IB_MAX_QPS];

  // Local and remote ready-state snapshots used for peer-ready synchronization.
  // The epoch array makes each recovery cycle distinguishable so a later
  // rejoin cannot be satisfied by a stale ready bit left over from an earlier
  // recovery.
  struct ibReadyStateSnapshot localReady;
  struct ibReadyStateSnapshot remoteReady;
  // Last remote ready epoch that was consumed when a local QP was activated.
  uint32_t activatedRemoteReadyEpoch[NCCL_IB_MAX_QPS];
  // Remote memory handle for peer-ready state RDMA reads.
  uint64_t remoteReadyStateAddr;
  uint32_t remoteReadyStateRkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  // Bookkeeping for non-blocking RDMA read of peer-ready state.
  bool peerReadyReadInFlight;
  int peerReadyReadQpIndex;
  int peerReadyReadDevIndex;
  bool peerReadyReadLogged;
  int nextPeerReadyQpIndex;
  uint64_t lastPeerReadyReadNs;
};
static_assert(alignof(struct ncclIbResiliency) == 128, "ncclIbResiliency must be 128-byte aligned");
static_assert((sizeof(struct ncclIbResiliency) % 128) == 0, "ncclIbResiliency size must be a 128-byte multiple");
static_assert((offsetof(struct ncclIbResiliency, devs) % 64) == 0, "ncclIbResiliency::devs must stay 64-byte aligned");
static_assert((offsetof(struct ncclIbResiliency, probingQps) % 128) == 0, "ncclIbResiliency::probingQps must stay 128-byte aligned");
static_assert((offsetof(struct ncclIbResiliency, localReady) % 64) == 0, "ncclIbResiliency::localReady must stay 64-byte aligned");
static_assert((offsetof(struct ncclIbResiliency, remoteReady) % 64) == 0, "ncclIbResiliency::remoteReady must stay 64-byte aligned");

enum ncclIbResiliencyRequestSendState {
  // The request just encountered an error and waiting for probing to be
  // posted.
  ncclIbResiliencyRequestStatePending = 0,
  // A probe was posted for this request.
  ncclIbResiliencyRequestStateProbePosted,
  // The probe was completed. After this state, the request is either completed
  // or replayed.
  ncclIbResiliencyRequestStateProbeCompleted,
  // The request was replayed and is waiting for either completion or a new
  // error before another recovery cycle is armed.
  ncclIbResiliencyRequestStateReplayPosted
};

struct ncclIbResiliencyErrorInfo {
  // The index of the device on which the error occurred.
  int devIndex;
  // The time on which the error occurred. Can be used to verify if sufficient
  // time has elapsed since the error to send the read probe.
  uint64_t time;
};

// Structure to hold information about a send request that encountered an
// error.
struct ncclIbResiliencyRequestSend {
  enum ncclIbResiliencyRequestSendState state;
  struct ncclIbRequest* request;
  struct ncclIbResiliencyErrorInfo errorInfo;
  int failedAttempts;

  // Tracks the last send-attempt CQE token associated with this slot. Reposts
  // refresh the token so late CQEs from an older send attempt can be ignored
  // without blocking a later error on the same still-live request from
  // re-entering replay.
  uint64_t id;
  // Unique WR ID for the currently posted probe, if any.
  uint64_t probeWrId;
};

struct ncclIbResiliencyRemoteCompletionRecordsInfo {
  // The address of the completion records structure on the receiver side.
  uint64_t addr;
  // For accessing the completion records structure on the receiver side.
  uint32_t rkey;
};

struct alignas(128) ncclIbResiliencySend {
  struct ncclIbResiliency base;
  // Keep failed-request entries dense: the replay path scans the whole array,
  // and one-cache-line-per-entry padding would inflate the steady-state slab.
  // Array of requests that encountered an error. Requests in this array can be
  // waiting for probing and then possibly replay. Any request that encounters
  // an error is added to this array. If a replay is posted, the entry stays
  // bound to the live request so a later error on the same request can arm a
  // new recovery cycle while old CQEs are still filtered correctly. The
  // request pointer is finally cleared when the request itself completes.
  // Note that multi-recv/send requests are considered as a single request.
  // Meaning, if one send request of a multi-send request encounters an error,
  // only the first send request is added here as a "representative" of the
  // whole multi-send request.
  // Note that iterating in order to progress these requests is not efficient
  // but on the other hand, if there are requests here, it means that requests
  // are in a transient state and might be replayed over different devices.
  // At a "steady state" there should be no requests here.
  struct ncclIbResiliencyRequestSend failedRequests[NET_IB_MAX_REQUESTS];

  // Stores the cumulative receiver-side completion counts per physical QP for
  // this request slot.
  // Note, this memory must be registered with an MR to allow RDMA Read
  // operations to write to it. Although each 1D array in this 2D array
  // "belongs" to a request, a 2D array is allocated to have a contiguous
  // memory allocation, so that the MR registration is easier and can be done
  // *simply* under a single MR (per device).
  uint16_t probingResults[NET_IB_MAX_REQUESTS][NCCL_IB_MAX_QPS];

  // Keep these tiny per-device remote handles packed; the whole array is read
  // as one small bundle when choosing the probing access path.
  // Stores information, per-device, used to access the completion records
  // structure on the receiver side.
  struct ncclIbResiliencyRemoteCompletionRecordsInfo remCmplRecordsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
  // Monotonic probe wr_id base; the low slot bits stay clear so each probe can
  // encode its failed-request slot directly in the posted wr_id.
  uint64_t nextProbeWrId;
};
static_assert(alignof(struct ncclIbResiliencySend) == 128, "ncclIbResiliencySend must be 128-byte aligned");
static_assert((sizeof(struct ncclIbResiliencySend) % 128) == 0, "ncclIbResiliencySend size must be a 128-byte multiple");
static_assert((offsetof(struct ncclIbResiliencySend, failedRequests) % 128) == 0, "ncclIbResiliencySend::failedRequests must stay 128-byte aligned");

// -----------------------------
// Data path APIs
// -----------------------------

ncclResult_t ncclIbResiliencyRequestIsComplete(struct ncclIbRequest *request, bool *isComplete);

// First checks if the error is recoverable or not. If yes, performs QPs
// replacement on the communicator for all QPs that are associated
// with the given device index and also initiates probing if needed.
// Note: The error might occur on a "new" request or on a request that is
// already waiting for a probe posting/completion or replay is already ongoing.
ncclResult_t ncclIbResiliencyHandleCompletionError(struct ncclIbResiliency* resCtx, struct ibv_wc* wc, int devIndex);

// Progresses all operations on the resiliency context.
ncclResult_t ncclIbResiliencyProgress(struct ncclIbResiliency* resCtx);

// -----------------------------
// Control path APIs
// -----------------------------

// Initializes/Destroys the resiliency context.
ncclResult_t ncclIbResiliencyInit(struct ncclIbNetCommBase* baseComm, struct ncclIbResiliency** resCtx);
ncclResult_t ncclIbResiliencyDestroy(struct ncclIbResiliency** resCtx);

// Initializes/Destroys device-related resources.
ncclResult_t ncclIbResiliencyDevInit(struct ncclIbResiliency* resCtx, uint devIndex, ncclIbDev* ibDev);
ncclResult_t ncclIbResiliencyDevDestroy(struct ncclIbResiliency* resCtx, uint devIndex);

// Gets the size of the CQ that is associated with the data QPs. This CQ size
// accomodates the number of devices that are supported for failover.
ncclResult_t ncclIbResiliencyDataCqSizeGet(struct ncclIbResiliency* resCtx, uint devIndex, int* cqSize);
// Gets the size of the Receive Queue (RQ) that is expected to receive data.
// The RQ size accomodates the number of devices that are supported for
// failover.
ncclResult_t ncclIbResiliencyDataRqSizeGet(struct ncclIbResiliency* resCtx, uint devIndex, uint32_t* rqSize);

// Set the number of local devices and remote devices for the resiliency
// context. This function must be called BEFORE creating the resiliency QPs
ncclResult_t ncclIbResiliencyDeviceNumSet(struct ncclIbResiliency* resCtx, int nLocalDevs, int nRemDevs);

// The local info should be populated by the function with the information of
// the QPs created so it could be passed to the receiver side.
ncclResult_t ncclIbResiliencySenderCreateQps(struct ncclIbResiliency* resCtx, struct ncclIbResiliencyInfo* localResiliencyInfo);
// The remote info should be used for modifying the QPs required for resiliency
// on the sender side to RTS state.
ncclResult_t ncclIbResiliencySenderQpsToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo);
// The local info should be populated with the information of the QPs created
// so it could be passed to the sender side.
ncclResult_t ncclIbResiliencyReceiverQpsCreateToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo, struct ncclIbResiliencyInfo* localResiliencyInfo);

ncclResult_t ncclIbResiliencyClose(struct ncclIbResiliency* resCtx);

// Allow resiliency context to reuse the memory registration
// Receiver side registers the memory for completion records and sends the
// memory info to the sender side. This function should be called on the sender
// side to allow the resiliency context to access the completion records
// structure on the receiver side.
ncclResult_t ncclIbResiliencyRemoteCompletionRecordsSet(struct ncclIbResiliency* resCtx, uint32_t cmplsRecordsRkey, uint64_t cmplsRecordsAddr, uint devIndex);

// Rejoin APIs used by progress/error paths to mark and rehabilitate failed QPs.
ncclResult_t ncclIbResiliencyMarkQpFailed(struct ncclIbResiliency* resCtx, int qpIndex);
ncclResult_t ncclIbResiliencyTryRejoinQps(struct ncclIbResiliency* resCtx);
ncclResult_t ncclIbResiliencySendRequestComplete(struct ncclIbResiliency* resCtx, struct ncclIbRequest* request);

#endif // NET_IB_P2P_RESILIENCY_H_
