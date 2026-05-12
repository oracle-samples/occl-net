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

#include "common.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <memory>

struct ncclIbResiliency;
struct ncclIbNetCommBase;

int ncclParamIbOooRq() {
  return 0;
}

int ncclParamIbResiliencyPortFailover() {
  return 0;
}

int ncclParamIbReceiverSideMatchingScheme() {
  return -2;
}

ncclResult_t ncclIbResiliencyInit(struct ncclIbNetCommBase*, struct ncclIbResiliency** resCtx) {
  *resCtx = nullptr;
  return ncclSuccess;
}

#include "../common.cc"  // NOLINT(build/include)

namespace {

std::unique_ptr<ncclIbNetCommBase> MakeBaseWithActiveQps(int devices, int qps) {
  auto base = std::make_unique<ncclIbNetCommBase>();
  base->vProps.ndevs = devices;
  base->nqps = qps;
  for (int i = 0; i < qps; i++) {
    base->activeQps[i] = &base->qps[i];
  }
  return base;
}

}  // namespace

TEST(NetIbCommonTest, CompletionRecordLayoutCoversSizesAndPhysicalCompletions) {
  ncclIbRequestCompletionRecord record{};
  constexpr int expectedRecvs = 8;
  constexpr int expectedQps = 128;

  EXPECT_EQ(sizeof(record.sizes) / sizeof(record.sizes[0]), expectedRecvs);
  EXPECT_EQ(sizeof(record.completions) / sizeof(record.completions[0]), expectedQps);

  for (int i = 0; i < expectedRecvs; i++) {
    EXPECT_EQ(record.sizes[i], 0);
  }
  for (int i = 0; i < expectedQps; i++) {
    EXPECT_EQ(record.completions[i], 0);
  }
}

TEST(NetIbCommonTest, QpByDeviceIndexUsesDeviceMajorLayout) {
  auto base = MakeBaseWithActiveQps(2, 8);
  ncclIbQp* qp = nullptr;

  EXPECT_EQ(ncclIbCommBaseGetQpByIndex(base.get(), 1, 2, &qp), ncclSuccess);

  EXPECT_EQ(qp, &base->qps[5]);
}

TEST(NetIbCommonTest, SelectsQpForRequestByRequestIdAndOffset) {
  auto base = MakeBaseWithActiveQps(2, 8);
  ncclIbQp* qp = nullptr;
  int selected = -1;

  EXPECT_EQ(ncclIbCommBaseGetQpForRequest(base.get(), 6, 3, &qp, &selected), ncclSuccess);

  EXPECT_EQ(selected, 1);
  EXPECT_EQ(qp, &base->qps[1]);
}

TEST(NetIbCommonTest, QpsPerRequestHonorsSplitPolicy) {
  auto base = MakeBaseWithActiveQps(2, 8);
  base->nDataQps = 2;

  base->splitDataOnQps = 0;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(base.get()), 2);

  base->splitDataOnQps = 1;
  EXPECT_EQ(ncclIbCommBaseGetNqpsPerRequest(base.get()), 8);
}

TEST(NetIbCommonTest, QpLookupByNumberFindsDeviceScopedQp) {
  auto base = MakeBaseWithActiveQps(2, 4);
  ibv_qp verbsQps[4] = {};
  for (int i = 0; i < 4; i++) {
    verbsQps[i].qp_num = 100 + i;
    base->qps[i].qp = &verbsQps[i];
  }

  ncclIbQp* qp = nullptr;
  int qpIndex = -1;
  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(base.get(), 1, 103, &qp, &qpIndex), ncclSuccess);
  EXPECT_EQ(qp, &base->qps[3]);
  EXPECT_EQ(qpIndex, 3);

  EXPECT_EQ(ncclIbCommBaseGetQpByQpNum(base.get(), 0, 103, &qp, nullptr), ncclInternalError);
  EXPECT_EQ(qp, nullptr);
}

TEST(NetIbCommonTest, SendRequestTrackingCanUseAllOnesAsNoAttemptSentinel) {
  ncclIbRequest request{};
  constexpr int expectedQps = 128;
  std::memset(request.send.pendingData, 0, sizeof(request.send.pendingData));
  std::memset(request.send.lastAttemptQpIndex, 0xff, sizeof(request.send.lastAttemptQpIndex));
  std::memset(request.send.lastAttemptOrder, 0, sizeof(request.send.lastAttemptOrder));
  std::memset(request.send.ackedCompletions, 0, sizeof(request.send.ackedCompletions));

  for (int i = 0; i < expectedQps; i++) {
    EXPECT_FALSE(request.send.pendingData[i]);
    EXPECT_EQ(request.send.lastAttemptQpIndex[i], -1);
    EXPECT_EQ(request.send.lastAttemptOrder[i], 0);
    EXPECT_EQ(request.send.ackedCompletions[i], 0);
  }
}

TEST(NetIbCommonTest, SendAndRecvCommKeepThirtyTwoByteAlignmentContracts) {
  EXPECT_EQ(sizeof(ncclIbNetCommBase) % 32, 0u);
  EXPECT_EQ(offsetof(ncclIbSendComm, ctsFifo) % 32, 0u);
  EXPECT_EQ(sizeof(ncclIbSendFifo) % 32, 0u);
  EXPECT_EQ(offsetof(ncclIbSendComm, sges) % 32, 0u);
  EXPECT_EQ(offsetof(ncclIbSendComm, wrs) % 32, 0u);
  EXPECT_EQ(offsetof(ncclIbRecvComm, remCtsFifo) % 32, 0u);
}

TEST(NetIbCommonTest, SendCommInitPopulatesBaseDefaults) {
  ncclIbSendComm comm = {};

  ASSERT_EQ(ncclIbSendCommInit(&comm), ncclSuccess);

  EXPECT_TRUE(comm.base.isSend);
  EXPECT_EQ(comm.base.ready, 0);
  EXPECT_EQ(comm.base.nqps, -1);
  EXPECT_EQ(comm.base.nDataQps, -1);
  EXPECT_EQ(comm.sendWrIdCounter, 0u);
}

TEST(NetIbCommonTest, RecvCommInitPopulatesReceiveDefaults) {
  ncclIbRecvComm comm = {};

  ASSERT_EQ(ncclIbRecvCommInit(&comm), ncclSuccess);

  EXPECT_FALSE(comm.base.isSend);
  EXPECT_EQ(comm.ibRecvWorkRequest.wr_id, NCCL_IB_RECV_WR_ID_DUMMY);
  EXPECT_EQ(comm.ibRecvWorkRequest.next, nullptr);
  EXPECT_EQ(comm.ibRecvWorkRequest.sg_list, nullptr);
  EXPECT_EQ(comm.ibRecvWorkRequest.num_sge, 0);
}

TEST(NetIbCommonTest, FatalCountCheckHonorsAsyncEventParam) {
  ncclIbStats stats = {};

  EXPECT_EQ(ncclIbStatsCheckFatalCount(&stats, "test"), ncclSuccess);

  stats.fatalErrorCount = 1;
  EXPECT_EQ(ncclIbStatsCheckFatalCount(&stats, "test"), ncclSystemError);
}

TEST(NetIbCommonTest, FatalErrorHelpersIncrementSharedCounters) {
  ncclIbStats stats = {};
  ibv_qp qp = {};
  ibv_cq cq = {};
  ncclIbDev dev = {};
  qp.qp_context = &stats;
  cq.cq_context = &stats;

  ASSERT_EQ(ncclIbStatsInit(&stats), ncclSuccess);
  ncclIbStatsFatalError(&stats);
  EXPECT_EQ(stats.fatalErrorCount, 1);

  ncclIbQpFatalError(&qp);
  EXPECT_EQ(stats.fatalErrorCount, 2);

  ncclIbCqFatalError(&cq);
  EXPECT_EQ(stats.fatalErrorCount, 3);

  ncclIbDevFatalError(&dev);
  EXPECT_EQ(dev.stats.fatalErrorCount, 1);
}
