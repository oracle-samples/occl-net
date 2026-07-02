/*
Copyright (c) 2026 OCCL-net.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "test_net_ib_state.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

const char* ncclIbReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut", "IGet"};

int64_t ncclParamIbPkey() {
  return 0;
}

int64_t ncclParamIbRetryCnt() {
  return 7;
}

int64_t ncclParamIbTimeout() {
  return 20;
}

int64_t ncclParamIbFifoTc() {
  return -1;
}

ncclResult_t ncclIbPortRecoveryInit(struct ncclIbResiliency*) {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoveryClose(struct ncclIbResiliency*) {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoveryDevInit(struct ncclIbResiliency*, int, ncclIbDev*) {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoveryDevDestroy(struct ncclIbResiliency*, int) {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoverySenderQpsCreate(struct ncclIbResiliency*, struct ncclIbQpInfo*, int) {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoverySenderQpsToRts(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*, int) {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoveryReceiverQpsCreateToRts(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*, struct ncclIbQpInfo*, int) {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoveryQpsDestroy(struct ncclIbResiliency*, int) {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoveryHandleFailure(struct ncclIbResiliency*, int) {
  return ncclSuccess;
}

ncclResult_t ncclIbPostReceiveWorkRequestsOnQp(struct ncclIbRecvComm*, ncclIbQp*) {
  return ncclSuccess;
}

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm*, int) {
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm*, struct ncclIbRequest*, int) {
  return ncclSuccess;
}

void ncclIbSendReqsRefreshWrIdToken(struct ncclIbSendComm*, int, int) {
}

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex) {
  req->events[devIndex]++;
}

ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) return &reinterpret_cast<ncclIbSendComm*>(base)->devs[devIndex].base;
  return &reinterpret_cast<ncclIbRecvComm*>(base)->devs[devIndex].base;
}

ncclResult_t wrap_ibv_query_port(struct ibv_context*, uint8_t, struct ibv_port_attr* portAttr) {
  portAttr->state = IBV_PORT_ACTIVE;
  portAttr->active_mtu = IBV_MTU_1024;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_create_qp(struct ibv_qp**, struct ibv_pd*, struct ibv_qp_init_attr*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_set_ece(struct ibv_qp*, struct ibv_ece*, int*) {
  return ncclSuccess;
}

ncclResult_t ncclIbQpReset(struct ncclIbQp*) {
  return ncclSuccess;
}

ncclResult_t ncclIbQpInit(struct ncclIbQp*) {
  return ncclSuccess;
}

ncclResult_t ncclIbQpRtr(struct ncclIbQp*) {
  return ncclSuccess;
}

ncclResult_t ncclIbQpRts(struct ncclIbQp*) {
  return ncclSuccess;
}

#include "../p2p_resiliency.cc"  // NOLINT(build/include)

namespace {

struct ResiliencyFixture : public ::testing::Test {
  void SetUp() override {
    std::memset(&base, 0, sizeof(base));
    std::memset(&ctx, 0, sizeof(ctx));
    for (int i = 0; i < 4; i++) {
      verbsQps[i].qp_num = 100 + i;
      base.qps[i].qp = &verbsQps[i];
      base.qps[i].devIndex = i % 2;
      base.qps[i].remDevIdx = i % 2;
      base.qps[i].remQpn = 200 + i;
      base.activeQps[i] = &base.qps[i];
      ctx.qpInfo[i].state = ncclIbResiliencyQpStateOk;
    }
    base.nqps = 4;
    base.nDataQps = 2;
    base.vProps.ndevs = 2;
    base.nRemDevs = 2;
    ctx.baseComm = &base;
    ctx.ndevs = 2;
    ctx.nProbingQps = 2;
    ctx.devs[0].state.store(ncclIbResiliencyDevStateOk, std::memory_order_release);
    ctx.devs[1].state.store(ncclIbResiliencyDevStateOk, std::memory_order_release);
  }

  ncclIbNetCommBase base = {};
  ncclIbResiliency ctx = {};
  ibv_qp verbsQps[4] = {};
};

}  // namespace

TEST(NetIbResiliencyTest, DisabledInitLeavesContextNull) {
  ncclIbNetCommBase base = {};
  base.nqps = 2;
  base.nDataQps = 1;
  ncclIbResiliency* ctx = reinterpret_cast<ncclIbResiliency*>(0x1);

  ASSERT_EQ(ncclIbResiliencyInit(&base, &ctx), ncclSuccess);

  EXPECT_EQ(ctx, nullptr);
}

TEST(NetIbResiliencyTest, DeviceNumSetTracksLocalAndRemoteDevices) {
  ncclIbNetCommBase base = {};
  ncclIbResiliency ctx = {};
  ctx.baseComm = &base;

  ASSERT_EQ(ncclIbResiliencyDeviceNumSet(&ctx, 2, 3), ncclSuccess);

  EXPECT_EQ(ctx.ndevs, 2);
  EXPECT_EQ(ctx.nProbingQps, 3);
}

TEST(NetIbResiliencyTest, RequestCompletionUsesAllDeviceEvents) {
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

TEST(NetIbResiliencyTest, MarkFailedQpUpdatesStateAndProgressFlag) {
  ncclIbResiliency ctx = {};
  ncclIbNetCommBase base = {};
  base.nqps = 2;
  ctx.baseComm = &base;
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateOk;

  ASSERT_EQ(ncclIbResiliencyMarkQpFailed(&ctx, 1), ncclSuccess);

  EXPECT_TRUE(ctx.inProgress);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailed);
}

TEST_F(ResiliencyFixture, StatePredicatesMasksAndBackoffHelpersAreStable) {
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

TEST_F(ResiliencyFixture, ErrorFatalityAndReplacementRespectOneFailedDevice) {
  ibv_wc wc = {};
  wc.status = IBV_WC_WR_FLUSH_ERR;
  ctx.devs[1].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);

  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&ctx, &wc, 1), ncclSuccess);

  ctx.devs[0].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&ctx, &wc, 1), ncclRemoteError);

  ctx.devs[0].state.store(ncclIbResiliencyDevStateOk, std::memory_order_release);
  ctx.devs[1].state.store(ncclIbResiliencyDevStateOk, std::memory_order_release);
  ASSERT_EQ(ncclIbResiliencyReplaceQps(&ctx, 1), ncclSuccess);
  EXPECT_EQ(base.activeQps[1]->devIndex, 0);
  EXPECT_EQ(base.activeQps[3]->devIndex, 0);
}

TEST_F(ResiliencyFixture, PendingRecoveryAndDeviceStateReconcileTrackQpStates) {
  EXPECT_FALSE(ncclIbResiliencyHasPendingQpRecovery(&ctx));
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateFailed;
  EXPECT_TRUE(ncclIbResiliencyHasPendingQpRecovery(&ctx));

  ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].state.load(std::memory_order_acquire), ncclIbResiliencyDevStateError);

  ctx.qpInfo[1].state = ncclIbResiliencyQpStateFailedPermanent;
  ctx.qpInfo[3].state = ncclIbResiliencyQpStateFailedPermanent;
  ASSERT_EQ(ncclIbResiliencyReconcileDeviceStates(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].state.load(std::memory_order_acquire), ncclIbResiliencyDevStateErrorPermanent);
}

TEST_F(ResiliencyFixture, RejoinFailureAndPeerReadyHandlersResetBookkeeping) {
  ctx.localReady.mask[0] = UINT64_MAX;
  ctx.remoteReady.mask[0] = UINT64_MAX;

  ASSERT_EQ(ncclIbResiliencyMarkQpRejoinFailure(&ctx, 1, "test"), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateFailed);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);
  EXPECT_TRUE(ctx.inProgress);
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.localReady.mask, 1));
  EXPECT_FALSE(ncclIbResiliencyReadyQpGetBit(ctx.remoteReady.mask, 1));

  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadQpIndex = 2;
  ctx.peerReadyReadDevIndex = 1;
  ctx.remoteReady.mask[0] = UINT64_MAX;
  ASSERT_EQ(ncclIbResiliencyHandlePeerReadyReadFailure(&ctx, 1, "unit", nullptr), ncclSuccess);
  EXPECT_FALSE(ctx.peerReadyReadInFlight);
  EXPECT_EQ(ctx.remoteReady.mask[0], 0u);
  EXPECT_EQ(ctx.devs[1].peerReadyFailureCount, 1);
  EXPECT_GT(ctx.devs[1].peerReadyRetryAfterNs, 0u);

  ASSERT_EQ(ncclIbResiliencyHandlePeerReadyReadSuccess(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.devs[1].peerReadyFailureCount, 0);
  EXPECT_EQ(ctx.devs[1].peerReadyRetryAfterNs, 0u);
}

TEST_F(ResiliencyFixture, SendRequestStateMachineCoversInitFreeAndReplayPosted) {
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

TEST_F(ResiliencyFixture, PeerReadyPostPollAndActivationHandleNoopPaths) {
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclSuccess);
  ctx.peerReadyReadInFlight = true;
  ctx.peerReadyReadDevIndex = 9;
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPoll(&ctx), ncclInternalError);
  ctx.peerReadyReadInFlight = false;

  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);
  ctx.remoteReadyStateAddr = 0x1000;
  ctx.qpInfo[1].state = ncclIbResiliencyQpStateOk;
  EXPECT_EQ(ncclIbResiliencyPeerReadyReadPost(&ctx), ncclSuccess);

  ctx.qpInfo[1].state = ncclIbResiliencyQpStateLocalReady;
  ctx.localReady.epoch[1] = 2;
  ctx.remoteReady.epoch[1] = 3;
  ncclIbResiliencyReadyQpSetBit(ctx.localReady.mask, 1, true);
  ncclIbResiliencyReadyQpSetBit(ctx.remoteReady.mask, 1, true);
  base.activeQps[1] = &base.qps[0];

  ASSERT_EQ(ncclIbResiliencyTryActivatePeerReadyQps(&ctx), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].state, ncclIbResiliencyQpStateOk);
  EXPECT_EQ(base.activeQps[1], &base.qps[1]);
  EXPECT_EQ(ctx.activatedRemoteReadyEpoch[1], 3u);
}

TEST_F(ResiliencyFixture, CompletionIgnoreAndHandlePathsTreatStaleEntriesAsSafe) {
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
  ctx.devs[0].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);
  wc.wr_id = NCCL_IB_RECV_WR_ID_DUMMY;
  EXPECT_EQ(ncclIbResiliencyShouldIgnoreReceiverCompletionError(&ctx, &wc, 0, &ignore), ncclSuccess);
  EXPECT_TRUE(ignore);
  EXPECT_EQ(ncclIbResiliencyHandleCompletionErrorReceiver(&ctx, &wc, 0), ncclSuccess);
}

TEST_F(ResiliencyFixture, PublicHelpersCoverDataSizesRemoteRecordsCloseAndProgress) {
  int cqSize = 0;
  uint32_t rqSize = 0;
  ctx.recoveryEnabled = false;

  ASSERT_EQ(ncclIbResiliencyDataCqSizeGet(&ctx, 0, &cqSize), ncclSuccess);
  EXPECT_GT(cqSize, 0);
  ASSERT_EQ(ncclIbResiliencyDataRqSizeGet(&ctx, 0, &rqSize), ncclSuccess);
  EXPECT_GT(rqSize, 0u);

  ncclIbResiliencySend sendCtx = {};
  sendCtx.base.baseComm = &base;
  sendCtx.base.nProbingQps = 2;
  base.isSend = true;
  EXPECT_EQ(ncclIbResiliencyRemoteCompletionRecordsSet(&sendCtx.base, 0x77, 0x1234, 1), ncclSuccess);
  EXPECT_EQ(sendCtx.remCmplRecordsInfo[1].rkey, 0x77u);
  EXPECT_EQ(sendCtx.remCmplRecordsInfo[1].addr, 0x1234u);

  ctx.qpInfo[1].state = ncclIbResiliencyQpStateFailed;
  base.qps[1].remQpn = 0;
  EXPECT_EQ(ncclIbResiliencyTryRejoinSingleQp(&ctx, 1), ncclSuccess);
  EXPECT_EQ(ctx.qpInfo[1].failCount, 1);

  EXPECT_EQ(ncclIbResiliencyClose(&ctx), ncclSuccess);

  ctx.qpInfo[1].state = ncclIbResiliencyQpStateOk;
  ctx.inProgress = false;
  EXPECT_EQ(ncclIbResiliencyProgress(&ctx), ncclSuccess);
}
