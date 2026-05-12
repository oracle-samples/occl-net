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

ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd** ret, struct ibv_context*) {
  *ret = reinterpret_cast<ibv_pd*>(0x1);
  return ncclSuccess;
}

ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd*) {
  return ncclSuccess;
}

struct ibv_mr* wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
  errno = EOPNOTSUPP;
  return nullptr;
}

#include "../gdr.cc"  // NOLINT(build/include)

TEST(NetIbGdrTest, ModuleProbeReturnsStableResultCode) {
  ncclResult_t result = ncclIbGdrSupport();

  EXPECT_TRUE(result == ncclSuccess || result == ncclSystemError);
}

TEST(NetIbGdrTest, PeerMemoryProbeReturnsStableResultCode) {
  ncclResult_t result = ncclIbPeerMemSupport();

  EXPECT_TRUE(result == ncclSuccess || result == ncclSystemError);
}

TEST(NetIbGdrTest, DmaBufUnsupportedWhenKernelReportsOperationNotSupported) {
  ncclNMergedIbDevs = 1;
  ncclIbMergedDevs[0].vProps.ndevs = 1;
  ncclIbMergedDevs[0].vProps.devs[0] = 0;
  ncclIbDevs[0].dmaBufSupported = 0;

  EXPECT_EQ(ncclIbDmaBufSupport(0), ncclSystemError);
  EXPECT_EQ(ncclIbDevs[0].dmaBufSupported, -1);
}
