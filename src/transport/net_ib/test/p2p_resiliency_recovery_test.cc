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

#include <cstring>
#include <list>
#include <vector>

namespace {

ibv_context fakeContext = {};
ibv_qp fakeQps[NCCL_IB_MAX_QPS] = {};
ibv_cq fakeCq = {};
int fakeNextQp = 0;
int fakePostRecvCalls = 0;
int fakePostSendCalls = 0;
int fakePollCompletions = 0;
ibv_wc fakePollCompletion = {};

int FakePostRecv(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**) {
  fakePostRecvCalls++;
  return 0;
}

int FakePostSend(ibv_qp*, ibv_send_wr*, ibv_send_wr**) {
  fakePostSendCalls++;
  return 0;
}

int FakePollCq(ibv_cq*, int, ibv_wc* wc) {
  if (fakePollCompletions <= 0) return 0;
  *wc = fakePollCompletion;
  fakePollCompletions--;
  return 1;
}

void ResetRecoveryFakes() {
  std::memset(&fakeContext, 0, sizeof(fakeContext));
  std::memset(fakeQps, 0, sizeof(fakeQps));
  std::memset(&fakeCq, 0, sizeof(fakeCq));
  std::memset(&fakePollCompletion, 0, sizeof(fakePollCompletion));
  fakeContext.ops.post_recv = FakePostRecv;
  fakeContext.ops.post_send = FakePostSend;
  fakeContext.ops.poll_cq = FakePollCq;
  fakeCq.context = &fakeContext;
  fakeNextQp = 0;
  fakePostRecvCalls = 0;
  fakePostSendCalls = 0;
  fakePollCompletions = 0;
  for (int i = 0; i < NCCL_IB_MAX_QPS; i++) {
    fakeQps[i].context = &fakeContext;
    fakeQps[i].qp_num = 500 + i;
  }
}

}  // namespace

int64_t ncclParamIbPkey() {
  return 0;
}

ncclResult_t ncclIbQpError(struct ncclIbQp*) {
  return ncclSuccess;
}

ncclResult_t ncclIbQpCreate(struct ncclIbQp* qp, struct ncclIbQpCreateAttr*) {
  qp->qp = &fakeQps[fakeNextQp++];
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

ncclResult_t ncclIbQpReset(struct ncclIbQp*) {
  return ncclSuccess;
}

ncclResult_t ncclIbPostReceiveWorkRequestsOnQp(struct ncclIbRecvComm*, ncclIbQp*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_create_cq(struct ibv_cq** cq, struct ibv_context*, int, void*, struct ibv_comp_channel*, int) {
  *cq = &fakeCq;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_set_ece(struct ibv_qp*, struct ibv_ece*, int*) {
  return ncclSuccess;
}

#include "../p2p_resiliency_recovery.cc"  // NOLINT(build/include)

namespace {

struct RecoveryFixture : public ::testing::Test {
  void SetUp() override {
    ResetRecoveryFakes();
    std::memset(&sendComm, 0, sizeof(sendComm));
    std::memset(&recvComm, 0, sizeof(recvComm));
    std::memset(&ctx, 0, sizeof(ctx));
    std::memset(&remInfo, 0, sizeof(remInfo));
    sendComm.base.isSend = true;
    sendComm.base.nqps = 2;
    sendComm.base.vProps.ndevs = 1;
    sendComm.base.nRemDevs = 1;
    sendComm.base.qps[0].qp = &fakeQps[0];
    sendComm.base.qps[0].devIndex = 0;
    sendComm.base.qps[1].qp = &fakeQps[1];
    sendComm.base.qps[1].devIndex = 0;
    sendComm.devs[0].base.ibDevN = 0;
    sendComm.devs[0].base.pd = reinterpret_cast<ibv_pd*>(0x1);

    recvComm.base.isSend = false;
    recvComm.base.nqps = 2;
    recvComm.base.vProps.ndevs = 1;
    recvComm.base.nRemDevs = 1;
    recvComm.base.qps[0].qp = &fakeQps[2];
    recvComm.base.qps[0].devIndex = 0;
    recvComm.base.qps[1].qp = &fakeQps[3];
    recvComm.base.qps[1].devIndex = 0;
    recvComm.devs[0].base.ibDevN = 0;
    recvComm.devs[0].base.pd = reinterpret_cast<ibv_pd*>(0x1);

    ctx.baseComm = &sendComm.base;
    ctx.ndevs = 1;
    ctx.recoveryEnabled = true;
    ctx.devs[0].portRecoveryCq = &fakeCq;
    ctx.portRecoveryQps[0].qp = &fakeQps[4];
    ctx.portRecoveryQps[0].devIndex = 0;

    ncclIbDevs[0].portNum = 1;
    ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_1024;
    remInfo.resiliencyInfo.portRecoveryQpsInfo[0].qpn = 777;
    remInfo.resiliencyInfo.portRecoveryQpsInfo[0].devIndex = 0;
    remInfo.devs[0].mtu = IBV_MTU_1024;
    remInfo.devs[0].link_layer = IBV_LINK_LAYER_INFINIBAND;
    remInfo.devs[0].ib_port = 1;
  }

  ncclIbSendComm sendComm = {};
  ncclIbRecvComm recvComm = {};
  ncclIbResiliency ctx = {};
  ncclIbConnectionMetadata remInfo = {};
};

}  // namespace

TEST(NetIbRecoveryTest, DisabledThreadStartAndStopAreNoops) {
  EXPECT_EQ(ncclIbPortRecoveryThreadStart(), ncclSuccess);
  EXPECT_EQ(ncclIbPortRecoveryThreadStop(), ncclSuccess);
}

TEST(NetIbRecoveryTest, DevInitCreatesRecoveryCqWhenEnabledByDirectCall) {
  ResetRecoveryFakes();
  ncclIbResiliency ctx = {};
  ncclIbNetCommBase base = {};
  ncclIbDev dev = {};
  ctx.ndevs = 1;
  ctx.recoveryEnabled = true;
  ctx.baseComm = &base;

  ASSERT_EQ(ncclIbPortRecoveryDevInit(&ctx, 0, &dev), ncclSuccess);

  EXPECT_EQ(ctx.devs[0].portRecoveryCq, &fakeCq);
}

TEST_F(RecoveryFixture, ParamWrappersExposeExpectedDefaults) {
  EXPECT_EQ(ncclParamIbResiliencyPortRecovery(), 0);
  EXPECT_EQ(ncclParamIbResiliencyPortRecoveryStartDelay(), 200);
  EXPECT_EQ(ncclParamIbResiliencyPortRecoveryAliveMsgBatchInterval(), 500);
  EXPECT_EQ(ncclParamIbResiliencyPortRecoveryAliveMsgBatchSize(), 5);
  EXPECT_EQ(ncclParamIbResiliencyPortRecoveryAliveMsgSequenceSize(), 5);
  EXPECT_EQ(ncclParamIbResiliencyPortRecoveryAliveMsgTimeout(), 4000);
  EXPECT_EQ(ncclParamIbResiliencyPortRecoveryAckTimeout(), 5000);
  EXPECT_EQ(ncclParamIbResiliencyPortRecoveryAttemptsMax(), 5);
}

TEST_F(RecoveryFixture, ContextInitAndDestroyCoverSendAndReceiveSetup) {
  ncclIbPortRecoveryContext* recovery = nullptr;
  ASSERT_EQ(ncclIbPortRecoveryContextInit(&ctx, 0, &recovery), ncclSuccess);
  ASSERT_NE(recovery, nullptr);
  EXPECT_EQ(recovery->state, ncclIbPortRecoveryStateInit);
  EXPECT_EQ(recovery->recoveryCq, &fakeCq);
  EXPECT_EQ(fakePostRecvCalls, 1);
  EXPECT_EQ(ncclIbPortRecoveryContextDestroy(recovery), ncclSuccess);

  ResetRecoveryFakes();
  ctx.baseComm = &recvComm.base;
  ctx.portRecoveryQps[0].qp = &fakeQps[4];
  recovery = nullptr;
  ASSERT_EQ(ncclIbPortRecoveryContextInit(&ctx, 0, &recovery), ncclSuccess);
  ASSERT_NE(recovery, nullptr);
  EXPECT_EQ(fakePostRecvCalls, NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
  EXPECT_EQ(ncclIbPortRecoveryContextDestroy(recovery), ncclSuccess);
}

TEST_F(RecoveryFixture, DrainPollAndCompletionHandlersAdvanceLocalState) {
  ncclIbPortRecoveryContext recovery = {};
  recovery.resCtx = &ctx;
  recovery.devIndex = 0;
  recovery.recoveryCq = &fakeCq;
  recovery.state = ncclIbPortRecoveryStateAliveMessages;
  bool success = false;

  fakePollCompletions = 2;
  ASSERT_EQ(ncclIbPortRecoveryDrainCqAndPostReceiveWRs(&recovery, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(fakePostRecvCalls, 2);

  ibv_wc sendDone = {};
  sendDone.status = IBV_WC_SUCCESS;
  sendDone.opcode = IBV_WC_SEND;
  ASSERT_EQ(ncclIbPortRecoveryHandleCompletionSender(&recovery, sendDone, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_TRUE(recovery.send.aliveMsgCompleted);

  recovery.resCtx->baseComm = &recvComm.base;
  recovery.ackPosted = false;
  recovery.recv.nInOrderMsgsReceived = 0;
  ibv_wc recvDone = {};
  recvDone.status = IBV_WC_SUCCESS;
  recvDone.opcode = IBV_WC_RECV;
  recvDone.imm_data = 0;
  ASSERT_EQ(ncclIbPortRecoveryHandleCompletionReceiver(&recovery, recvDone, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(recovery.recv.nInOrderMsgsReceived, 1);

  fakePollCompletion = recvDone;
  fakePollCompletions = 1;
  ASSERT_EQ(ncclIbPortRecoveryPollCq(&recovery, &success), ncclSuccess);
  EXPECT_TRUE(success);
}

TEST_F(RecoveryFixture, QpCreateToRtsDestroyAndRestoreUseCachedMetadata) {
  ncclIbQpInfo localInfo[1] = {};

  ASSERT_EQ(ncclIbPortRecoverySenderQpsCreate(&ctx, localInfo, 1), ncclSuccess);
  EXPECT_EQ(localInfo[0].devIndex, 0);
  EXPECT_NE(localInfo[0].qpn, 0u);

  ASSERT_EQ(ncclIbPortRecoverySenderQpsToRts(&ctx, &remInfo, 1), ncclSuccess);
  EXPECT_EQ(ctx.portRecoveryQps[0].remDevIdx, 0);
  EXPECT_EQ(ctx.portRecoveryQps[0].rtrAttr.remoteQpNum, 777u);

  bool success = true;
  ncclIbPortRecoveryContext recovery = {};
  recovery.resCtx = &ctx;
  recovery.devIndex = 0;
  ASSERT_EQ(ncclIbPortRecoveryQpsRestore(&recovery, &success), ncclSuccess);
  EXPECT_TRUE(success);

  ASSERT_EQ(ncclIbPortRecoveryQpsDestroy(&ctx, 1), ncclSuccess);

  ResetRecoveryFakes();
  ctx.baseComm = &recvComm.base;
  ctx.portRecoveryQps[0].qp = nullptr;
  ASSERT_EQ(ncclIbPortRecoveryReceiverQpsCreateToRts(&ctx, &remInfo, localInfo, 1), ncclSuccess);
  EXPECT_EQ(localInfo[0].devIndex, 0);
  EXPECT_NE(localInfo[0].qpn, 0u);
}

TEST_F(RecoveryFixture, PostAliveAckAndProgressHelpersDriveStateTransitions) {
  ncclIbPortRecoveryContext recovery = {};
  recovery.resCtx = &ctx;
  recovery.devIndex = 0;
  recovery.recoveryCq = &fakeCq;
  recovery.state = ncclIbPortRecoveryStateAliveMessages;
  recovery.timeLastMsg = 0;
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryPostAliveMessages(&recovery, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_TRUE(recovery.send.aliveMsgPosted);
  EXPECT_EQ(fakePostSendCalls, 1);

  ASSERT_EQ(ncclIbPortRecoveryPostAck(&recovery, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_TRUE(recovery.ackPosted);
  EXPECT_EQ(fakePostSendCalls, 2);

  ncclIbPortRecoveryStateProgressResult result = ncclIbPortRecoveryStateProgressResultFailed;
  recovery.send.aliveMsgPosted = true;
  recovery.send.aliveMsgCompleted = true;
  ASSERT_EQ(ncclIbPortRecoveryProgressAliveMessagesSender(&recovery, &result), ncclSuccess);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultGoToNextState);

  recovery.state = ncclIbPortRecoveryStateAliveMessages;
  ASSERT_EQ(ncclIbPortRecoveryProgressAliveMessages(&recovery), ncclSuccess);
  EXPECT_EQ(recovery.state, ncclIbPortRecoveryStateAck);

  recovery.ackReceived = true;
  recovery.ackPosted = true;
  recovery.ackCompleted = false;
  fakePollCompletion = {};
  fakePollCompletion.status = IBV_WC_SUCCESS;
  fakePollCompletion.opcode = IBV_WC_SEND;
  fakePollCompletions = 1;
  ASSERT_EQ(ncclIbPortRecoveryProgressAckSender(&recovery, &result), ncclSuccess);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultGoToNextState);

  ctx.baseComm = &recvComm.base;
  recovery.resCtx = &ctx;
  recovery.state = ncclIbPortRecoveryStateAck;
  recovery.ackReceived = true;
  ASSERT_EQ(ncclIbPortRecoveryProgressAck(&recovery), ncclSuccess);
  EXPECT_EQ(recovery.state, ncclIbPortRecoveryStateSuccess);
}

TEST_F(RecoveryFixture, ContextProgressCloseRequestsAndInitCloseNoopPaths) {
  ncclIbPortRecoveryContext recovery = {};
  recovery.resCtx = &ctx;
  recovery.devIndex = 0;
  recovery.state = ncclIbPortRecoveryStateInit;
  recovery.timeInit = 0;
  bool done = false;

  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&recovery, &done), ncclSuccess);
  EXPECT_FALSE(done);
  recovery.state = ncclIbPortRecoveryStateSuccess;
  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&recovery, &done), ncclSuccess);
  EXPECT_TRUE(done);
  EXPECT_EQ(ctx.devs[0].state.load(std::memory_order_acquire), ncclIbResiliencyDevStateRecovered);

  std::vector<ncclIbPortRecoveryCloseRequest*> closeRequests;
  std::list<ncclIbPortRecoveryContext*> queue;
  ncclIbPortRecoveryCloseRequest closeReq = {&ctx, false};
  ncclIbPortRecoveryContext* queued = static_cast<ncclIbPortRecoveryContext*>(std::calloc(1, sizeof(*queued)));
  queued->resCtx = &ctx;
  queued->devIndex = 0;
  queue.push_back(queued);
  closeRequests.push_back(&closeReq);
  ncclIbPortRecoveryHandleCloseRequests(closeRequests, queue);
  EXPECT_TRUE(queue.empty());

  ctx.recoveryEnabled = false;
  EXPECT_EQ(ncclIbPortRecoveryInit(&ctx), ncclSuccess);
  EXPECT_FALSE(ctx.recoveryEnabled);
  EXPECT_EQ(ncclIbPortRecoveryClose(&ctx), ncclSuccess);
}

TEST(NetIbRecoveryTest, DevDestroyClearsRecoveryCq) {
  ncclIbResiliency ctx = {};
  ctx.ndevs = 1;
  ctx.devs[0].portRecoveryCq = reinterpret_cast<ibv_cq*>(0x1);

  ASSERT_EQ(ncclIbPortRecoveryDevDestroy(&ctx, 0), ncclSuccess);

  EXPECT_EQ(ctx.devs[0].portRecoveryCq, reinterpret_cast<ibv_cq*>(0x1));
}
