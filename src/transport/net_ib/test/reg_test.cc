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

#include <cstdlib>

namespace {

int regCalls = 0;
int deregCalls = 0;

ibv_mr* AllocateMr(ibv_pd* pd, void* addr, size_t length) {
  auto* mr = static_cast<ibv_mr*>(std::calloc(1, sizeof(ibv_mr)));
  mr->pd = pd;
  mr->addr = addr;
  mr->length = length;
  mr->lkey = 0xabc000 + regCalls;
  mr->rkey = 0xdef000 + regCalls;
  return mr;
}

void ResetRegState() {
  regCalls = 0;
  deregCalls = 0;
  for (auto& dev : ncclIbDevs) {
    dev.mrCache = {};
  }
  ncclIbRelaxedOrderingEnabled = 0;
}

ncclIbSendComm MakeSendComm(ibv_pd* pd) {
  ncclIbSendComm comm = {};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base.ibDevN = 0;
  comm.devs[0].base.pd = pd;
  return comm;
}

}  // namespace

ncclResult_t wrap_ibv_reg_mr(struct ibv_mr** ret, struct ibv_pd* pd, void* addr, size_t length, int) {
  regCalls++;
  *ret = AllocateMr(pd, addr, length);
  return ncclSuccess;
}

ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr** ret, struct ibv_pd* pd, void* addr, size_t length, uint64_t, int) {
  regCalls++;
  *ret = AllocateMr(pd, addr, length);
  return ncclSuccess;
}

ncclResult_t wrap_ibv_reg_dmabuf_mr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t, size_t length, uint64_t iova, int, int) {
  regCalls++;
  *ret = AllocateMr(pd, reinterpret_cast<void*>(iova), length);
  return ncclSuccess;
}

ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t, size_t length, uint64_t iova, int, int, int) {
  regCalls++;
  *ret = AllocateMr(pd, reinterpret_cast<void*>(iova), length);
  return ncclSuccess;
}

ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr* mr) {
  deregCalls++;
  std::free(mr);
  return ncclSuccess;
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  auto* comm = reinterpret_cast<ncclIbSendComm*>(base);
  return &comm->devs[devIndex].base;
}

#include "../reg.cc"  // NOLINT(build/include)

TEST(NetIbRegTest, OverlappingHostRegistrationsReuseCachedMemoryRegion) {
  ResetRegState();
  ibv_pd pd = {};
  auto comm = MakeSendComm(&pd);
  alignas(4096) unsigned char buffer[8192] = {};
  void* h1 = nullptr;
  void* h2 = nullptr;

  ASSERT_EQ(ncclIbRegMr(&comm, buffer + 512, 2048, NCCL_PTR_HOST, &h1), ncclSuccess);
  ASSERT_EQ(ncclIbRegMr(&comm, buffer + 1024, 512, NCCL_PTR_HOST, &h2), ncclSuccess);

  auto* mh1 = static_cast<ncclIbMrHandle*>(h1);
  auto* mh2 = static_cast<ncclIbMrHandle*>(h2);
  EXPECT_EQ(mh1->mrs[0], mh2->mrs[0]);
  EXPECT_EQ(regCalls, 1);
  EXPECT_EQ(ncclIbDevs[0].mrCache.population, 1);
  EXPECT_EQ(ncclIbDevs[0].mrCache.slots[0].refs, 2);

  EXPECT_EQ(ncclIbDeregMr(&comm, h2), ncclSuccess);
  EXPECT_EQ(ncclIbDevs[0].mrCache.population, 1);
  EXPECT_EQ(ncclIbDevs[0].mrCache.slots[0].refs, 1);

  EXPECT_EQ(ncclIbDeregMr(&comm, h1), ncclSuccess);
  EXPECT_EQ(ncclIbDevs[0].mrCache.population, 0);
  EXPECT_EQ(ncclIbDevs[0].mrCache.slots, nullptr);
  EXPECT_EQ(deregCalls, 1);
}

TEST(NetIbRegTest, UnknownMemoryRegionReturnsInternalError) {
  ResetRegState();
  ibv_pd pd = {};
  ncclIbNetCommDevBase base = {};
  base.ibDevN = 0;
  base.pd = &pd;
  ibv_mr unknown = {};

  EXPECT_EQ(ncclIbDeregMrInternal(&base, &unknown), ncclInternalError);
}

TEST(NetIbRegTest, NullDeregistrationIsSuccess) {
  ResetRegState();
  ibv_pd pd = {};
  auto comm = MakeSendComm(&pd);

  EXPECT_EQ(ncclIbDeregMr(&comm, nullptr), ncclSuccess);
}
