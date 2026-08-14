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

static struct ibv_qp testQp = {};
static struct ibv_qp_attr lastRtrAttr = {};
static int lastRtrAttrMask = 0;
static int rtrModifyCalls = 0;

int64_t ncclParamIbOooRq() {
  return 0;
}

ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm*) {
  return ncclSuccess;
}

ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm*) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDeviceNumSet(struct ncclIbResiliency*, int, int) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDataCqSizeGet(struct ncclIbResiliency*, uint, int* cqSize) {
  *cqSize = 1;
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDevInit(struct ncclIbResiliency*, uint, ncclIbDev*) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencySenderCreateQps(struct ncclIbResiliency*, struct ncclIbResiliencyInfo*) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencySenderQpsToRts(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyReceiverQpsCreateToRts(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*, struct ncclIbResiliencyInfo*) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDataRqSizeGet(struct ncclIbResiliency*, uint, uint32_t* rqSize) {
  *rqSize = 1;
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyClose(struct ncclIbResiliency*) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDevDestroy(struct ncclIbResiliency*, uint) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDestroy(struct ncclIbResiliency**) {
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyRemoteCompletionRecordsSet(struct ncclIbResiliency*, uint32_t, uint64_t, uint) {
  return ncclSuccess;
}

ncclResult_t ncclIbGdrSupport() {
  return ncclSystemError;
}

ncclResult_t ncclIbDmaBufSupport(int) {
  return ncclSystemError;
}

ncclResult_t ncclIbPeerMemSupport() {
  return ncclSystemError;
}

ncclResult_t ncclIbRegMr(void*, void*, size_t, int, void**) {
  return ncclSuccess;
}

ncclResult_t ncclIbDeregMr(void*, void*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd** ret, struct ibv_context*) {
  *ret = reinterpret_cast<ibv_pd*>(0x1);
  return ncclSuccess;
}

ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_create_cq(struct ibv_cq** ret, struct ibv_context*, int, void*, struct ibv_comp_channel*, int) {
  *ret = reinterpret_cast<ibv_cq*>(0x1);
  return ncclSuccess;
}

ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_create_qp(struct ibv_qp** ret, struct ibv_pd*, struct ibv_qp_init_attr*) {
  testQp = {};
  testQp.qp_num = 17;
  *ret = &testQp;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_modify_qp(struct ibv_qp*, struct ibv_qp_attr* attr, int attrMask) {
  if (attrMask & IBV_QP_PATH_MTU) {
    lastRtrAttr = *attr;
    lastRtrAttrMask = attrMask;
    rtrModifyCalls++;
  }
  return ncclSuccess;
}

ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp*) {
  return ncclSuccess;
}

ncclResult_t wrap_ibv_query_gid(struct ibv_context*, uint8_t, int, union ibv_gid*) {
  return ncclSystemError;
}

ncclResult_t wrap_ibv_query_ece(struct ibv_qp*, struct ibv_ece*, int* supported) {
  *supported = 0;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_set_ece(struct ibv_qp*, struct ibv_ece*, int*) {
  return ncclSuccess;
}

struct ibv_qp* wrap_mlx5dv_create_qp(struct ibv_context*, struct ibv_qp_init_attr_ex*, struct mlx5dv_qp_init_attr*) {
  return nullptr;
}

#include "../connect.cc"  // NOLINT(build/include)

TEST(NetIbConnectTest, VirtualDeviceCompatibilityAllowsIntersections) {
  ncclNetVDeviceProps_t local = {};
  local.ndevs = 2;
  local.devs[0] = 1;
  local.devs[1] = 3;

  ncclNetVDeviceProps_t remote = {};
  remote.ndevs = 2;
  remote.devs[0] = 3;
  remote.devs[1] = 4;

  EXPECT_EQ(ncclIbCheckVProps(&local, &remote), ncclSuccess);
}

TEST(NetIbConnectTest, ReceiverQpUsesMinimumEndpointMtuWithoutMutatingMetadata) {
  auto verifyMtu = [](enum ibv_mtu localMtu, enum ibv_mtu remoteMtu) {
    SCOPED_TRACE(::testing::Message() << "localMtu=" << localMtu << ", remoteMtu=" << remoteMtu);

    struct ncclIbRecvComm comm = {};
    struct ncclIbConnectionMetadata remMeta = {};
    struct ncclIbConnectionMetadata localMeta = {};

    comm.base.nqps = 1;
    comm.base.vProps.ndevs = 1;
    comm.base.qps[0].coldIdx = 0;
    comm.devs[0].base.ibDevN = 0;
    comm.devs[0].base.cq = reinterpret_cast<ibv_cq*>(0x1);
    comm.devs[0].base.pd = reinterpret_cast<ibv_pd*>(0x1);

    ncclIbDevs[0].portNum = 1;
    ncclIbDevs[0].portAttr.active_mtu = localMtu;
    ncclIbDevs[0].portAttr.link_layer = IBV_LINK_LAYER_INFINIBAND;

    constexpr uint32_t remoteQpNum = 23;
    remMeta.qpInfo[0].qpn = remoteQpNum;
    remMeta.qpInfo[0].devIndex = 0;
    remMeta.devs[0].mtu = remoteMtu;
    remMeta.devs[0].link_layer = IBV_LINK_LAYER_INFINIBAND;
    remMeta.devs[0].ib_port = 1;
    remMeta.devs[0].lid = 1;

    lastRtrAttr = {};
    lastRtrAttrMask = 0;
    rtrModifyCalls = 0;

    ASSERT_EQ(ncclIbReceiverQpsCreateToRts(&comm, &remMeta, &localMeta), ncclSuccess);
    ASSERT_EQ(rtrModifyCalls, 1);
    EXPECT_NE(lastRtrAttrMask & IBV_QP_PATH_MTU, 0);
    EXPECT_EQ(lastRtrAttr.path_mtu, std::min(localMtu, remoteMtu));
    EXPECT_EQ(lastRtrAttr.dest_qp_num, remoteQpNum);
    EXPECT_EQ(ncclIbDevs[0].portAttr.active_mtu, localMtu);
    EXPECT_EQ(remMeta.devs[0].mtu, remoteMtu);
  };

  verifyMtu(IBV_MTU_1024, IBV_MTU_4096);
  verifyMtu(IBV_MTU_4096, IBV_MTU_1024);
  verifyMtu(IBV_MTU_2048, IBV_MTU_2048);
}
