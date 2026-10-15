/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "common.h"

#include <gtest/gtest.h>

namespace {

static_assert(sizeof(ncclNetVDeviceProps_v9_t) == 5 * sizeof(int));
static_assert(sizeof(ncclNetVDeviceProps_v10_t) == 5 * sizeof(int));
static_assert(sizeof(ncclNetVDeviceProps_v11_t) == 5 * sizeof(int));
static_assert(sizeof(ncclNetVDeviceProps_v12_t) == 9 * sizeof(int));
static_assert(sizeof(ncclNet_v11_t) == sizeof(ncclNet_v12_t));
static_assert(sizeof(ncclNetCommConfig_v11_t) == sizeof(ncclNetCommConfig_v12_t));
static_assert(sizeof(ncclNetAttr_v11_t) == sizeof(ncclNetAttr_v12_t));

int occlMakeCallCount = 0;
int occlPropertyCallCount = 0;
int occlInitCallCount = 0;
int occlSetAttrCallCount = 0;
int occlLastPropertyDevice = -1;
int occlReturnedDevice = 73;
int occlCapturedTrafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
bool occlCapturedNullConfig = false;
bool occlCapturedNullAttr = false;
ncclResult_t occlMakeResult = ncclSuccess;
ncclResult_t occlPropertyResult = ncclSuccess;
ncclNetVDeviceProps_t occlCapturedVProps = {};
ncclNetProperties_t occlSourceProperties = {};
ncclNetAttr_t occlCapturedAttr = {};
char occlSourceName[] = "mlx5_anchor";
char occlSourcePciPath[] = "/sys/devices/pci0000:00/0000:00:01.0";

void occlResetAdapterState() {
  occlMakeCallCount = 0;
  occlPropertyCallCount = 0;
  occlInitCallCount = 0;
  occlSetAttrCallCount = 0;
  occlLastPropertyDevice = -1;
  occlReturnedDevice = 73;
  occlCapturedTrafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  occlCapturedNullConfig = false;
  occlCapturedNullAttr = false;
  occlMakeResult = ncclSuccess;
  occlPropertyResult = ncclSuccess;
  occlCapturedVProps = {};
  occlSourceProperties = {};
  occlCapturedAttr = {};
  occlSourceProperties.name = occlSourceName;
  occlSourceProperties.pciPath = occlSourcePciPath;
  occlSourceProperties.guid = 0x123456789abcdef0ULL;
  occlSourceProperties.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF;
  occlSourceProperties.regIsGlobal = 1;
  occlSourceProperties.forceFlush = 1;
  occlSourceProperties.speed = 400000;
  occlSourceProperties.port = 17;
  occlSourceProperties.latency = 2.5f;
  occlSourceProperties.maxComms = 2048;
  occlSourceProperties.maxRecvs = 8;
  occlSourceProperties.netDeviceType = NCCL_NET_DEVICE_HOST;
  occlSourceProperties.netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  occlSourceProperties.maxP2pBytes = 1ULL << 30;
  occlSourceProperties.maxCollBytes = 1ULL << 29;
  occlSourceProperties.maxMultiRequestSize = 4;
  occlSourceProperties.railId = 7;
  occlSourceProperties.planeId = 8;
}

void occlSetSourceMembers(int count) {
  occlSourceProperties.vProps = {};
  occlSourceProperties.vProps.ndevs = count;
  for (int i = 0; i < count; i++) occlSourceProperties.vProps.devs[i] = count - i - 1;
}

class NetIbPluginAdapterTest : public ::testing::Test {
 protected:
  void SetUp() override { occlResetAdapterState(); }
};

}  // namespace

ncclResult_t ncclIbInit(void**, uint64_t, ncclNetCommConfig_t* config, ncclDebugLogger_t, ncclProfilerCallback_t) {
  occlInitCallCount++;
  occlCapturedNullConfig = config == nullptr;
  if (config == nullptr) return ncclInvalidArgument;
  occlCapturedTrafficClass = config->trafficClass;
  return ncclSuccess;
}

ncclResult_t ncclIbFinalize(void*) {
  return ncclSuccess;
}

ncclResult_t ncclIbDevices(int* ndev) {
  if (ndev == nullptr) return ncclInvalidArgument;
  *ndev = 8;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  occlPropertyCallCount++;
  occlLastPropertyDevice = dev;
  if (occlPropertyResult != ncclSuccess) return occlPropertyResult;
  if (props == nullptr) return ncclInvalidArgument;
  *props = occlSourceProperties;
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

ncclResult_t ncclIbRegMr(void*, void*, size_t, int, void**) {
  return ncclSuccess;
}

ncclResult_t ncclIbRegMrDmaBuf(void*, void*, size_t, int, uint64_t, int, void**) {
  return ncclSuccess;
}

ncclResult_t ncclIbDeregMr(void*, void*) {
  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void*, void*, size_t, int, void*, void*, void**) {
  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void*, int, void**, size_t*, int*, void**, void**, void**) {
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void*, int, void**, int*, void**, void**) {
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void*, int*, int*) {
  return ncclSuccess;
}

ncclResult_t ncclIbCloseSend(void*) {
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void*) {
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void*) {
  return ncclSuccess;
}

ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  occlMakeCallCount++;
  if (occlMakeResult != ncclSuccess) return occlMakeResult;
  if (d == nullptr || props == nullptr) return ncclInvalidArgument;
  occlCapturedVProps = *props;
  *d = occlReturnedDevice;
  return ncclSuccess;
}

ncclResult_t ncclIbSetNetAttr(void*, ncclNetAttr_t* attr) {
  occlSetAttrCallCount++;
  occlCapturedNullAttr = attr == nullptr;
  if (attr != nullptr) occlCapturedAttr = *attr;
  return ncclSuccess;
}

#include "plugin/net/net_ib_plugin_v11.cc"  // NOLINT(build/include)

TEST_F(NetIbPluginAdapterTest, ExportedTablesExposeExpectedCallbacks) {
  EXPECT_NE(ncclNetPlugin_v12.getProperties, nullptr);
  EXPECT_NE(ncclNetPlugin_v12.makeVDevice, nullptr);
  EXPECT_NE(ncclNetPlugin_v11.getProperties, nullptr);
  EXPECT_NE(ncclNetPlugin_v11.makeVDevice, nullptr);
  EXPECT_NE(ncclNetPlugin_v10.makeVDevice, nullptr);
  EXPECT_NE(ncclNetPlugin_v9.makeVDevice, nullptr);
}

TEST_F(NetIbPluginAdapterTest, V12MakeCallbackPreservesEightMembers) {
  ncclNetVDeviceProps_v12_t props = {};
  props.ndevs = NCCL_NET_MAX_DEVS_PER_NIC_V12;
  for (int i = 0; i < props.ndevs; i++) props.devs[i] = props.ndevs - i - 1;
  int device = -1;

  ASSERT_EQ(ncclNetPlugin_v12.makeVDevice(&device, &props), ncclSuccess);
  EXPECT_EQ(device, occlReturnedDevice);
  ASSERT_EQ(occlMakeCallCount, 1);
  ASSERT_EQ(occlCapturedVProps.ndevs, props.ndevs);
  for (int i = 0; i < props.ndevs; i++) EXPECT_EQ(occlCapturedVProps.devs[i], props.devs[i]);
}

TEST_F(NetIbPluginAdapterTest, LegacyMakeCallbacksPreserveFourMembers) {
  ncclNetVDeviceProps_v11_t propsV11 = {};
  ncclNetVDeviceProps_v10_t propsV10 = {};
  ncclNetVDeviceProps_v9_t propsV9 = {};
  propsV11.ndevs = propsV10.ndevs = propsV9.ndevs = NCCL_NET_MAX_DEVS_PER_NIC_V11;
  for (int i = 0; i < NCCL_NET_MAX_DEVS_PER_NIC_V11; i++) {
    propsV11.devs[i] = propsV10.devs[i] = propsV9.devs[i] = NCCL_NET_MAX_DEVS_PER_NIC_V11 - i - 1;
  }
  int device = -1;

  ASSERT_EQ(ncclNetPlugin_v11.makeVDevice(&device, &propsV11), ncclSuccess);
  ASSERT_EQ(ncclNetPlugin_v10.makeVDevice(&device, &propsV10), ncclSuccess);
  ASSERT_EQ(ncclNetPlugin_v9.makeVDevice(&device, &propsV9), ncclSuccess);
  EXPECT_EQ(occlMakeCallCount, 3);
  ASSERT_EQ(occlCapturedVProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC_V11);
  for (int i = 0; i < occlCapturedVProps.ndevs; i++) EXPECT_EQ(occlCapturedVProps.devs[i], propsV11.devs[i]);
}

TEST_F(NetIbPluginAdapterTest, LegacyMakeCallbacksRejectOversizeCountsBeforeTransportCall) {
  ncclNetVDeviceProps_v11_t propsV11 = {};
  ncclNetVDeviceProps_v10_t propsV10 = {};
  ncclNetVDeviceProps_v9_t propsV9 = {};
  propsV11.ndevs = propsV10.ndevs = propsV9.ndevs = NCCL_NET_MAX_DEVS_PER_NIC_V11 + 1;
  int device = -1;

  EXPECT_EQ(ncclNetPlugin_v11.makeVDevice(&device, &propsV11), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v10.makeVDevice(&device, &propsV10), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v9.makeVDevice(&device, &propsV9), ncclInvalidArgument);
  EXPECT_EQ(occlMakeCallCount, 0);
}

TEST_F(NetIbPluginAdapterTest, VersionedCallbacksRejectNullArgumentsBeforeTransportCall) {
  ncclNetVDeviceProps_v12_t propsV12 = {};
  ncclNetVDeviceProps_v11_t propsV11 = {};
  ncclNetVDeviceProps_v10_t propsV10 = {};
  ncclNetVDeviceProps_v9_t propsV9 = {};
  propsV12.ndevs = propsV11.ndevs = propsV10.ndevs = propsV9.ndevs = 1;
  int device = -1;

  EXPECT_EQ(ncclNetPlugin_v12.makeVDevice(nullptr, &propsV12), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v11.makeVDevice(&device, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v10.makeVDevice(nullptr, &propsV10), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v9.makeVDevice(&device, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v12.getProperties(12, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v11.getProperties(11, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v10.getProperties(10, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclNetPlugin_v9.getProperties(9, nullptr), ncclInvalidArgument);
  EXPECT_EQ(occlMakeCallCount, 0);
  EXPECT_EQ(occlPropertyCallCount, 0);
}

TEST_F(NetIbPluginAdapterTest, V12PropertiesPreserveEightMembersAndDefaultIds) {
  occlSetSourceMembers(NCCL_NET_MAX_DEVS_PER_NIC_V12);
  ncclNetProperties_v12_t props = {};

  ASSERT_EQ(ncclNetPlugin_v12.getProperties(12, &props), ncclSuccess);
  EXPECT_EQ(occlPropertyCallCount, 1);
  EXPECT_EQ(occlLastPropertyDevice, 12);
  ASSERT_EQ(props.vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC_V12);
  for (int i = 0; i < props.vProps.ndevs; i++) EXPECT_EQ(props.vProps.devs[i], occlSourceProperties.vProps.devs[i]);
  EXPECT_EQ(props.railId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(props.planeId, NCCL_NET_ID_UNDEF);
}

TEST_F(NetIbPluginAdapterTest, LegacyPropertiesPreserveFourMembers) {
  occlSetSourceMembers(NCCL_NET_MAX_DEVS_PER_NIC_V11);
  ncclNetProperties_v11_t propsV11 = {};
  ncclNetProperties_v10_t propsV10 = {};
  ncclNetProperties_v9_t propsV9 = {};

  ASSERT_EQ(ncclNetPlugin_v11.getProperties(11, &propsV11), ncclSuccess);
  ASSERT_EQ(ncclNetPlugin_v10.getProperties(10, &propsV10), ncclSuccess);
  ASSERT_EQ(ncclNetPlugin_v9.getProperties(9, &propsV9), ncclSuccess);
  EXPECT_STREQ(propsV11.name, occlSourceName);
  EXPECT_STREQ(propsV10.name, occlSourceName);
  EXPECT_STREQ(propsV9.name, occlSourceName);
  EXPECT_EQ(propsV11.speed, occlSourceProperties.speed);
  EXPECT_EQ(propsV10.speed, occlSourceProperties.speed);
  EXPECT_EQ(propsV9.speed, occlSourceProperties.speed);
  EXPECT_EQ(propsV11.maxMultiRequestSize, occlSourceProperties.maxMultiRequestSize);
  ASSERT_EQ(propsV11.vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC_V11);
  ASSERT_EQ(propsV10.vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC_V11);
  ASSERT_EQ(propsV9.vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC_V11);
  for (int i = 0; i < propsV11.vProps.ndevs; i++) {
    EXPECT_EQ(propsV11.vProps.devs[i], occlSourceProperties.vProps.devs[i]);
    EXPECT_EQ(propsV10.vProps.devs[i], occlSourceProperties.vProps.devs[i]);
    EXPECT_EQ(propsV9.vProps.devs[i], occlSourceProperties.vProps.devs[i]);
  }
}

TEST_F(NetIbPluginAdapterTest, LegacyPropertiesRejectEightMembersWithoutWritingOutput) {
  occlSetSourceMembers(NCCL_NET_MAX_DEVS_PER_NIC_V12);
  ncclNetProperties_v11_t propsV11 = {};
  ncclNetProperties_v10_t propsV10 = {};
  ncclNetProperties_v9_t propsV9 = {};
  propsV11.speed = -11;
  propsV10.speed = -10;
  propsV9.speed = -9;

  EXPECT_EQ(ncclNetPlugin_v11.getProperties(11, &propsV11), ncclInvalidUsage);
  EXPECT_EQ(ncclNetPlugin_v10.getProperties(10, &propsV10), ncclInvalidUsage);
  EXPECT_EQ(ncclNetPlugin_v9.getProperties(9, &propsV9), ncclInvalidUsage);
  EXPECT_EQ(propsV11.speed, -11);
  EXPECT_EQ(propsV10.speed, -10);
  EXPECT_EQ(propsV9.speed, -9);
}

TEST_F(NetIbPluginAdapterTest, V11InitConvertsCommunicationConfig) {
  ncclNetCommConfig_v11_t config = {};
  config.trafficClass = 106;
  void* context = nullptr;

  EXPECT_EQ(ncclNetPlugin_v11.init(&context, 42, &config, nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(occlInitCallCount, 1);
  EXPECT_EQ(occlCapturedTrafficClass, 106);
}

TEST_F(NetIbPluginAdapterTest, V11InitPassesNullConfigThrough) {
  void* context = nullptr;

  EXPECT_EQ(ncclNetPlugin_v11.init(&context, 42, nullptr, nullptr, nullptr), ncclInvalidArgument);
  EXPECT_EQ(occlInitCallCount, 1);
  EXPECT_TRUE(occlCapturedNullConfig);
}

TEST_F(NetIbPluginAdapterTest, V11SetNetAttrCopiesFieldsWithoutTypePunning) {
  ncclNetAttr_v11_t attr = {};
  attr.sendCommAttr.maxConcurrentPeers = 1;
  attr.sendCommAttr.minConcurrentPeers = 2;
  attr.sendCommAttr.maxFlowsPerPeer = 3;
  attr.sendCommAttr.minFlowsPerPeer = 4;
  attr.recvCommAttr.maxConcurrentPeers = 5;
  attr.recvCommAttr.minConcurrentPeers = 6;
  attr.recvCommAttr.maxFlowsPerPeer = 7;
  attr.recvCommAttr.minFlowsPerPeer = 8;
  attr.op = 9;
  attr.algo = 10;
  attr.proto = 11;

  ASSERT_EQ(ncclNetPlugin_v11.setNetAttr(nullptr, &attr), ncclSuccess);
  ASSERT_EQ(occlSetAttrCallCount, 1);
  EXPECT_FALSE(occlCapturedNullAttr);
  EXPECT_EQ(occlCapturedAttr.sendCommAttr.maxConcurrentPeers, 1);
  EXPECT_EQ(occlCapturedAttr.sendCommAttr.minConcurrentPeers, 2);
  EXPECT_EQ(occlCapturedAttr.sendCommAttr.maxFlowsPerPeer, 3);
  EXPECT_EQ(occlCapturedAttr.sendCommAttr.minFlowsPerPeer, 4);
  EXPECT_EQ(occlCapturedAttr.recvCommAttr.maxConcurrentPeers, 5);
  EXPECT_EQ(occlCapturedAttr.recvCommAttr.minConcurrentPeers, 6);
  EXPECT_EQ(occlCapturedAttr.recvCommAttr.maxFlowsPerPeer, 7);
  EXPECT_EQ(occlCapturedAttr.recvCommAttr.minFlowsPerPeer, 8);
  EXPECT_EQ(occlCapturedAttr.op, 9u);
  EXPECT_EQ(occlCapturedAttr.algo, 10u);
  EXPECT_EQ(occlCapturedAttr.proto, 11u);

  ASSERT_EQ(ncclNetPlugin_v11.setNetAttr(nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(occlSetAttrCallCount, 2);
  EXPECT_TRUE(occlCapturedNullAttr);
}
