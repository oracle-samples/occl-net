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

namespace {

ncclIbSendComm MakeSendComm() {
  ncclIbSendComm comm = {};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  return comm;
}

}  // namespace

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  auto* comm = reinterpret_cast<ncclIbSendComm*>(base);
  return &comm->devs[devIndex].base;
}

ncclResult_t ncclIbResiliencyProgress(struct ncclIbResiliency*) {
  return ncclSuccess;
}

ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats*, const char*) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyRequestIsComplete(struct ncclIbRequest* request, bool* isComplete) {
  *isComplete = true;
  for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
    request->events[i] = 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencySendRequestComplete(struct ncclIbResiliency*, struct ncclIbRequest*) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyHandleCompletionError(struct ncclIbResiliency*, struct ibv_wc*, int) {
  return ncclSuccess;
}

#include "../p2p.cc"  // NOLINT(build/include)

TEST(NetIbP2pTest, RequestAllocationSkipsUsedSlots) {
  auto comm = MakeSendComm();
  comm.base.reqs[0].type = NCCL_NET_IB_REQ_SEND;
  comm.base.reqs[1].type = NCCL_NET_IB_REQ_RECV;
  ncclIbRequest* request = nullptr;

  ASSERT_EQ(ncclIbGetRequest(&comm.base, &request), ncclSuccess);

  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request, &comm.base.reqs[2]);
  EXPECT_EQ(request->base, &comm.base);
  EXPECT_EQ(request->sock, nullptr);
}

TEST(NetIbP2pTest, RequestAllocationFailsWhenPoolIsFull) {
  auto comm = MakeSendComm();
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    comm.base.reqs[i].type = NCCL_NET_IB_REQ_SEND;
  }
  ncclIbRequest* request = reinterpret_cast<ncclIbRequest*>(0x1);

  EXPECT_EQ(ncclIbGetRequest(&comm.base, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
}

TEST(NetIbP2pTest, AddEventTracksDeviceBase) {
  auto comm = MakeSendComm();
  ncclIbRequest request = {};
  request.base = &comm.base;

  ncclIbAddEvent(&request, 0);
  ncclIbAddEvent(&request, 0);

  EXPECT_EQ(request.events[0], 2);
  EXPECT_EQ(request.devBases[0], &comm.devs[0].base);
}

TEST(NetIbP2pTest, RefreshSendTokensAppliesSameTokenToGroupedRequests) {
  auto comm = MakeSendComm();
  ncclIbRequest req0 = {};
  ncclIbRequest req1 = {};
  comm.sendReqs[3][0] = &req0;
  comm.sendReqs[3][1] = &req1;

  ncclIbSendReqsRefreshWrIdToken(&comm, 3, 2);

  EXPECT_NE(req0.send.wrIdToken, 0u);
  EXPECT_EQ(req0.send.wrIdToken, req1.send.wrIdToken);
  EXPECT_EQ(comm.sendWrIdCounter, 1u);
}

TEST(NetIbP2pTest, FreeRequestMarksSlotUnused) {
  ncclIbRequest request = {};
  request.type = NCCL_NET_IB_REQ_FLUSH;

  EXPECT_EQ(ncclIbFreeRequest(&request), ncclSuccess);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
}
