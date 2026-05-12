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

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>

ncclNet_t ncclNetIb = {};

#if CUDART_VERSION >= 11030
static CUresult TestCuDeviceGetAttribute(int* value, CUdevice_attribute, CUdevice) {
  *value = 1;
  return CUDA_SUCCESS;
}

static CUresult TestCuGetErrorString(CUresult, const char** errorString) {
  *errorString = "test-cuda";
  return CUDA_SUCCESS;
}

PFN_cuDeviceGetAttribute_v2000 pfn_cuDeviceGetAttribute = TestCuDeviceGetAttribute;
PFN_cuGetErrorString_v6000 pfn_cuGetErrorString = TestCuGetErrorString;
#endif

const char* ncclIbReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut", "IGet"};

namespace {

ncclResult_t peerMemResult = ncclSystemError;
ncclResult_t gdrResult = ncclSystemError;
ncclResult_t dmaResult = ncclSystemError;
int initDevicesResult = ncclSuccess;
int finalizeDevicesCalls = 0;
int closeSendCalls = 0;
int closeRecvCalls = 0;
int closeListenCalls = 0;
int testRegMrCalls = 0;
int testDeregMrCalls = 0;
int testIsendCalls = 0;
int testIrecvCalls = 0;
int testPostSendCalls = 0;
ibv_wr_opcode lastPostSendOpcode = IBV_WR_SEND;

void SetGinSupport(ncclResult_t peerMem, ncclResult_t gdr, ncclResult_t dma) {
  peerMemResult = peerMem;
  gdrResult = gdr;
  dmaResult = dma;
}

void ResetGinGlobals() {
  ncclNetIb = {};
  peerMemResult = ncclSystemError;
  gdrResult = ncclSystemError;
  dmaResult = ncclSystemError;
  initDevicesResult = ncclSuccess;
  finalizeDevicesCalls = 0;
  closeSendCalls = 0;
  closeRecvCalls = 0;
  closeListenCalls = 0;
  testRegMrCalls = 0;
  testDeregMrCalls = 0;
  testIsendCalls = 0;
  testIrecvCalls = 0;
  testPostSendCalls = 0;
  lastPostSendOpcode = IBV_WR_SEND;
  ncclNIbDevs = 0;
  std::memset(ncclIbDevs, 0, sizeof(ncclIbDevs));
}

}  // namespace

ncclResult_t ncclIbPeerMemSupport() {
  return peerMemResult;
}

ncclResult_t ncclIbGdrSupport() {
  return gdrResult;
}

ncclResult_t ncclIbDmaBufSupport(int) {
  return dmaResult;
}

ncclResult_t ncclIbInitDevices(ncclDebugLogger_t, ncclProfilerCallback_t) {
  return static_cast<ncclResult_t>(initDevicesResult);
}

ncclResult_t ncclIbFinalizeDevices() {
  finalizeDevicesCalls++;
  return ncclSuccess;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
  props->name = "test-phys";
  props->port = dev;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = "test";
  props->port = dev;
  return ncclSuccess;
}

void ncclIbSetTrafficClass(void*, int) {
}

ncclResult_t ncclGinGdakiCreateContext(void*, int, int, int, int, int, void**, ncclNetDeviceHandle_t**) {
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiRegMrSym(void*, void*, size_t, int, uint64_t, void**, void**) {
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiDeregMrSym(void*, void*) {
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiDestroyContext(void*) {
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiProgress(void*) {
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiQueryLastError(void*, bool*) {
  return ncclSuccess;
}

ncclResult_t ncclIbListen(void*, int, void*, void**) {
  return ncclSuccess;
}

ncclResult_t ncclIbConnect(void*, int, void*, void**, ncclNetDeviceHandle_t**) {
  return ncclSuccess;
}

ncclResult_t ncclIbAccept(void*, void**, ncclNetDeviceHandle_t**) {
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void*) {
  return ncclSuccess;
}

ncclResult_t ncclIbCloseSend(void*) {
  closeSendCalls++;
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void*) {
  closeRecvCalls++;
  return ncclSuccess;
}

ncclResult_t ncclIbRegMrDmaBufInternal(void*, void*, size_t, int, uint64_t, int, uint64_t, void** mhandle) {
  static ibv_mr mr;
  static ncclIbMrHandle handle;
  mr.lkey = 12;
  mr.rkey = 34;
  handle.mrs[0] = &mr;
  *mhandle = &handle;
  return ncclSuccess;
}

ncclResult_t ncclIbDeregMr(void*, void*) {
  return ncclSuccess;
}

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  *req = &base->reqs[0];
  (*req)->base = base;
  return ncclSuccess;
}

ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* request) {
  request->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

void ncclIbAddEvent(struct ncclIbRequest*, int) {
}

ncclResult_t ncclIbTest(void*, int*, int*) {
  return ncclSuccess;
}

ncclResult_t ncclIbGetGidIndex(struct ibv_context*, uint8_t, struct ibv_port_attr*, int* gidIndex) {
  *gidIndex = 0;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_symbols() {
  return ncclSuccess;
}

const char* ncclSocketToString(const union ncclSocketAddress*, char* line, const int) {
  std::strcpy(line, "test-socket");
  return line;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket*, union ncclSocketAddress* addr) {
  std::memset(addr, 0, sizeof(*addr));
  return ncclSuccess;
}

#include "../gin.cc"  // NOLINT(build/include)

namespace {

struct TestGinFixture : public ::testing::Test {
  void SetUp() override {
    ResetGinGlobals();
    ncclGinIbGdakiNDevs = -1;
    std::memset(ncclGinIbGdakiDevIndexes, 0, sizeof(ncclGinIbGdakiDevIndexes));
    ncclNetIb.devices = ncclIbDevices;
    ncclNetIb.getProperties = ncclIbGetProperties;
    ncclNetIb.listen = TestListen;
    ncclNetIb.connect = TestConnect;
    ncclNetIb.accept = TestAccept;
    ncclNetIb.regMr = TestRegMr;
    ncclNetIb.deregMr = TestDeregMr;
    ncclNetIb.isend = TestIsend;
    ncclNetIb.irecv = TestIrecv;
    ncclNetIb.test = TestNetTest;
    ncclNetIb.closeSend = TestCloseSend;
    ncclNetIb.closeRecv = TestCloseRecv;
    ncclNetIb.closeListen = TestCloseListen;
  }

  static ncclResult_t TestListen(void*, int, void*, void** listenComm) {
    *listenComm = reinterpret_cast<void*>(0x11);
    return ncclSuccess;
  }

  static ncclResult_t TestConnect(void*, int, void*, void** sendComm, ncclNetDeviceHandle_t**) {
    *sendComm = reinterpret_cast<void*>(0x22);
    return ncclSuccess;
  }

  static ncclResult_t TestAccept(void*, void** recvComm, ncclNetDeviceHandle_t**) {
    *recvComm = reinterpret_cast<void*>(0x33);
    return ncclSuccess;
  }

  static ncclResult_t TestRegMr(void*, void*, size_t, int, void** mhandle) {
    testRegMrCalls++;
    *mhandle = reinterpret_cast<void*>(0x44);
    return ncclSuccess;
  }

  static ncclResult_t TestDeregMr(void*, void*) {
    testDeregMrCalls++;
    return ncclSuccess;
  }

  static ncclResult_t TestIsend(void*, void*, size_t, int, void*, void*, void** request) {
    testIsendCalls++;
    *request = reinterpret_cast<void*>(0x55);
    return ncclSuccess;
  }

  static ncclResult_t TestIrecv(void*, int, void**, size_t*, int*, void**, void**, void** request) {
    testIrecvCalls++;
    *request = reinterpret_cast<void*>(0x66);
    return ncclSuccess;
  }

  static ncclResult_t TestNetTest(void*, int* done, int*) {
    *done = 1;
    return ncclSuccess;
  }

  static ncclResult_t TestCloseSend(void* sendComm) {
    return ncclIbCloseSend(sendComm);
  }

  static ncclResult_t TestCloseRecv(void* recvComm) {
    return ncclIbCloseRecv(recvComm);
  }

  static ncclResult_t TestCloseListen(void*) {
    closeListenCalls++;
    return ncclSuccess;
  }
};

struct FakePostOps {
  ibv_context context = {};
  ibv_qp qp = {};

  FakePostOps() {
    context.ops.post_send = PostSend;
    qp.context = &context;
    qp.qp_num = 0xabc;
  }

  static int PostSend(ibv_qp*, ibv_send_wr* wr, ibv_send_wr**) {
    testPostSendCalls++;
    lastPostSendOpcode = wr->opcode;
    return 0;
  }
};

ncclIbMrHandle MakeMrHandle(uint32_t lkey) {
  static ibv_mr mr;
  mr.lkey = lkey;
  mr.rkey = lkey + 100;
  ncclIbMrHandle handle = {};
  handle.mrs[0] = &mr;
  return handle;
}

ncclIbGinProxyMrHandle MakeGinProxyMr(uintptr_t base, uint32_t lkey, uint32_t rkey) {
  static ncclIbMrHandle handles[8];
  static int nextHandle = 0;
  ncclIbGinProxyMrHandle ginMr = {};
  handles[nextHandle] = MakeMrHandle(lkey);
  ginMr.mrHandle = &handles[nextHandle];
  nextHandle = (nextHandle + 1) % 8;
  ginMr.base_vas = static_cast<uintptr_t*>(std::calloc(1, sizeof(uintptr_t)));
  ginMr.rkeys = static_cast<uint32_t*>(std::calloc(1, sizeof(uint32_t)));
  ginMr.base_vas[0] = base;
  ginMr.rkeys[0] = rkey;
  return ginMr;
}

void FreeGinProxyMr(ncclIbGinProxyMrHandle* ginMr) {
  std::free(ginMr->base_vas);
  std::free(ginMr->rkeys);
  ginMr->base_vas = nullptr;
  ginMr->rkeys = nullptr;
}

}  // namespace

TEST(NetIbGinTest, GdrSupportAllowsGenericPeerMemory) {
  bool supported = false;
  SetGinSupport(ncclSystemError, ncclSuccess, ncclSystemError);

  EXPECT_EQ(ncclGinIbGdrSupport(&supported, false), ncclSuccess);

  EXPECT_TRUE(supported);
}

TEST(NetIbGinTest, GdrSupportAllowsGdakiPeerMemory) {
  bool supported = false;
  SetGinSupport(ncclSuccess, ncclSystemError, ncclSystemError);

  EXPECT_EQ(ncclGinIbGdrSupport(&supported, true), ncclSuccess);

  EXPECT_TRUE(supported);
}

TEST(NetIbGinTest, GdrSupportFallsBackToDmaBuf) {
  bool supported = false;
  SetGinSupport(ncclSystemError, ncclSystemError, ncclSuccess);

  EXPECT_EQ(ncclGinIbGdrSupport(&supported, false), ncclSuccess);

  EXPECT_TRUE(supported);
}

TEST(NetIbGinTest, GdrSupportReportsUnsupportedWhenNoBackendWorks) {
  bool supported = true;
  SetGinSupport(ncclSystemError, ncclSystemError, ncclSystemError);

  EXPECT_EQ(ncclGinIbGdrSupport(&supported, false), ncclSuccess);

  EXPECT_FALSE(supported);
}

TEST_F(TestGinFixture, GdakiDeviceDiscoveryAndPropertiesUseMlx5Devices) {
  ncclNIbDevs = 3;
  ncclIbDevs[0].ibProvider = IB_PROVIDER_NONE;
  ncclIbDevs[1].ibProvider = IB_PROVIDER_MLX5;
  ncclIbDevs[2].ibProvider = IB_PROVIDER_MLX5;

  ASSERT_EQ(ncclGinIbGdakiInit(), ncclSuccess);
  int ndevs = -1;
  EXPECT_EQ(ncclGinIbGdakiDevices(&ndevs), ncclSuccess);
  EXPECT_EQ(ndevs, 2);

  ncclNetProperties_t props = {};
  EXPECT_EQ(ncclGinIbGdakiGetProperties(1, &props), ncclSuccess);
  EXPECT_EQ(props.port, 2);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_GIN_GDAKI);
  EXPECT_EQ(props.vProps.ndevs, 1);
  EXPECT_EQ(props.vProps.devs[0], 1);

  EXPECT_EQ(ncclGinIbGdakiGetProperties(2, &props), ncclInvalidUsage);
}

TEST_F(TestGinFixture, InitTypeHandlesNoDevicesInvalidTypeAndProxySuccess) {
  void* ctx = reinterpret_cast<void*>(0x1);
  ncclNIbDevs = 0;
  EXPECT_EQ(ncclGinIbInitType(&ctx, 0, nullptr, NCCL_GIN_TYPE_PROXY, nullptr), ncclInternalError);

  ncclNIbDevs = 1;
  SetGinSupport(ncclSystemError, ncclSuccess, ncclSystemError);
  ctx = nullptr;
  EXPECT_EQ(ncclGinIbInitType(&ctx, 0, nullptr, 99, nullptr), ncclInternalError);
  EXPECT_EQ(ctx, nullptr);

  EXPECT_EQ(ncclGinIbProxyInit(&ctx, 0, nullptr), ncclSuccess);
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ncclGinIbFinalize(ctx), ncclSuccess);
  EXPECT_EQ(finalizeDevicesCalls, 1);
}

TEST_F(TestGinFixture, InitTypeFallsBackFromUnavailableGdakiToProxy) {
  void* ctx = nullptr;
  ncclGin_t selected = {};
  ncclNIbDevs = 1;
  ncclIbDevs[0].ibProvider = IB_PROVIDER_NONE;
  SetGinSupport(ncclSystemError, ncclSuccess, ncclSystemError);

  ASSERT_EQ(ncclGinIbInitType(&ctx, 0, nullptr, -1, &selected), ncclSuccess);

  ASSERT_NE(ctx, nullptr);
  EXPECT_STREQ(selected.name, "GIN_IB_PROXY");
  std::free(ctx);
}

TEST_F(TestGinFixture, SingleRankConnectCloseAndCollectivesUseNetCallbacks) {
  ncclIbListenComm listen = {};
  listen.dev = 0;
  void* handles[] = {nullptr};
  void* coll = nullptr;

  ASSERT_EQ(ncclGinIbConnect(nullptr, handles, 1, 0, &listen, &coll), ncclSuccess);
  ASSERT_NE(coll, nullptr);

  auto* comm = static_cast<ncclGinIbCollComm*>(coll);
  int source[1] = {7};
  int gathered[1] = {0};
  int allToAllOut[1] = {0};

  EXPECT_EQ(ncclGinIbAllGather(comm, source, gathered, sizeof(int)), ncclSuccess);
  EXPECT_EQ(gathered[0], 7);
  EXPECT_EQ(testRegMrCalls, 2);
  EXPECT_EQ(testDeregMrCalls, 2);

  EXPECT_EQ(ncclGinIbAllToAll(comm, source, allToAllOut, sizeof(int)), ncclSuccess);
  EXPECT_EQ(allToAllOut[0], 7);

  EXPECT_EQ(ncclGinIbP2PBarrier(comm), ncclSuccess);
  EXPECT_EQ(ncclGinIbCloseColl(coll), ncclSuccess);
  EXPECT_EQ(closeSendCalls, 1);
  EXPECT_EQ(closeRecvCalls, 1);
}

TEST_F(TestGinFixture, GdakiThinWrappersForwardToBackends) {
  ncclGinIbCollComm comm = {};
  ncclGinConfig_t config = {};
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_t* devHandle = nullptr;
  void* mhandle = nullptr;
  void* ginHandle = nullptr;
  bool hasError = true;

  EXPECT_EQ(ncclGinIbGdakiCreateContext(&comm, &config, &ginCtx, &devHandle), ncclSuccess);
  EXPECT_EQ(ncclGinIbGdakiRegMrSym(&comm, &comm, 16, NCCL_PTR_HOST, 0, &mhandle, &ginHandle), ncclSuccess);
  EXPECT_EQ(ncclGinIbGdakiDeregMrSym(&comm, mhandle), ncclSuccess);
  EXPECT_EQ(ncclGinIbGdakiDestroyContext(ginCtx), ncclSuccess);
  EXPECT_EQ(ncclGinIbGdakiProgress(&comm), ncclSuccess);
  EXPECT_EQ(ncclGinIbGdakiQueryLastError(ginCtx, &hasError), ncclSuccess);

  ncclNIbDevs = 1;
  ncclIbDevs[0].ibProvider = IB_PROVIDER_MLX5;
  ASSERT_EQ(ncclGinIbGdakiInit(), ncclSuccess);
  void* listenComm = nullptr;
  EXPECT_EQ(ncclGinIbGdakiListen(nullptr, 0, nullptr, &listenComm), ncclSuccess);
  EXPECT_EQ(listenComm, reinterpret_cast<void*>(0x11));
}

TEST_F(TestGinFixture, ProxyPropertiesConnectAndCreateContextValidation) {
  ncclNetProperties_t props = {};
  EXPECT_EQ(ncclGinIbProxyGetProperties(4, &props), ncclSuccess);
  EXPECT_EQ(props.port, 4);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_GIN_PROXY);

  ncclIbListenComm listen = {};
  listen.dev = 0;
  void* handles[] = {nullptr};
  void* coll = nullptr;
  SetGinSupport(ncclSystemError, ncclSuccess, ncclSystemError);
  EXPECT_EQ(ncclGinIbProxyConnect(nullptr, handles, 1, 0, &listen, &coll), ncclSuccess);
  ASSERT_NE(coll, nullptr);

  ncclGinConfig_t config = {};
  config.queueDepth = 1;
  config.nContexts = 1;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  EXPECT_EQ(ncclGinIbProxyCreateContext(coll, &config, &ginCtx, &devHandle), ncclInvalidUsage);
  EXPECT_EQ(ginCtx, nullptr);

  EXPECT_EQ(ncclGinIbCloseColl(coll), ncclSuccess);
}

TEST_F(TestGinFixture, ProxyMemoryRegistrationAndDeregistrationOwnSymmetricHandles) {
  ncclGinIbCollComm comm = {};
  comm.nranks = 1;
  comm.rank = 0;
  comm.recvComm = reinterpret_cast<void*>(0x33);
  comm.allGather = ncclGinIbAllGather;
  int data = 0;
  void* mhandle = nullptr;
  void* ginHandle = nullptr;

  ASSERT_EQ(ncclGinIbProxyRegMrSymDmaBuf(&comm, &data, sizeof(data), NCCL_PTR_HOST, 0, -1, 0, &mhandle, &ginHandle), ncclSuccess);
  ASSERT_NE(mhandle, nullptr);
  EXPECT_EQ(mhandle, ginHandle);

  int deregCallsBefore = testDeregMrCalls;
  EXPECT_EQ(ncclGinIbProxyDeregMrSym(&comm, mhandle), ncclSuccess);
  EXPECT_EQ(testDeregMrCalls, deregCallsBefore + 1);
}

TEST_F(TestGinFixture, ProxyRequestHelpersPopulateRequestsAndWorkRequests) {
  FakePostOps ops;
  ncclIbSendComm send = {};
  ncclIbRecvComm recv = {};
  ncclIbRequest sendReq = {};
  ncclIbRequest recvReq = {};
  ncclGinIbProxyCtx proxyCtx = {};
  void* sendComms[] = {&send};
  void* recvComms[] = {&recv};
  proxyCtx.fullSendComm = sendComms;
  proxyCtx.fullRecvComm = recvComms;
  proxyCtx.rank = 0;
  proxyCtx.nranks = 1;

  send.base.vProps.ndevs = 1;
  send.base.reqs[0] = sendReq;
  send.base.qps[0].qp = &ops.qp;
  send.base.qps[0].devIndex = 0;
  send.devs[0].putSignalScratchpadMr = new ibv_mr{};
  send.devs[0].putSignalScratchpadMr->lkey = 91;
  recv.base.vProps.ndevs = 1;
  recv.base.reqs[0] = recvReq;
  recv.devs[0].gpuFlush.qp.qp = &ops.qp;
  recv.devs[0].gpuFlush.qp.devIndex = 0;

  ncclIbGinProxyMrHandle src = MakeGinProxyMr(0x1000, 1, 11);
  ncclIbGinProxyMrHandle dst = MakeGinProxyMr(0x2000, 2, 22);
  ncclIbGinProxyMrHandle signal = MakeGinProxyMr(0x3000, 3, 33);
  void* request = nullptr;

  EXPECT_EQ(ncclGinIbProxyIPut(&proxyCtx, 0, 4, &src, 8, 16, &dst, 0, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(static_cast<ncclIbRequest*>(request)->type, NCCL_NET_IB_REQ_GIN_IPUT);
  EXPECT_EQ(lastPostSendOpcode, IBV_WR_RDMA_WRITE);

  request = nullptr;
  EXPECT_EQ(ncclGinIbProxyIGet(&proxyCtx, 0, 4, &src, 8, 16, &dst, 0, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(static_cast<ncclIbRequest*>(request)->type, NCCL_NET_IB_REQ_GIN_IGET);
  EXPECT_EQ(lastPostSendOpcode, IBV_WR_RDMA_READ);

  EXPECT_EQ(ncclGinIbProxyIPutSignal(&proxyCtx, 0, 0, &src, 1, 0, &dst, 0, 0, &signal, 1, 999, &request), ncclInvalidArgument);
  EXPECT_EQ(ncclGinIbProxyIPutSignal(&proxyCtx, 0, 0, &src, 0, 0, nullptr, 0, 0, &signal, 5, NCCL_NET_SIGNAL_OP_ADD, &request), ncclSuccess);
  EXPECT_EQ(lastPostSendOpcode, IBV_WR_ATOMIC_FETCH_AND_ADD);

  request = &recv.base.reqs[0];
  recv.base.reqs[0].events[0] = 0;
  int done = 0;
  EXPECT_EQ(ncclGinIbProxyTest(nullptr, request, &done), ncclSuccess);
  EXPECT_EQ(done, 1);

  delete send.devs[0].putSignalScratchpadMr;
  FreeGinProxyMr(&src);
  FreeGinProxyMr(&dst);
  FreeGinProxyMr(&signal);
  EXPECT_GE(testPostSendCalls, 3);
}
