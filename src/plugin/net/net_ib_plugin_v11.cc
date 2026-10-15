/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * Modifications by Oracle and/or its affiliates are licensed under the
 * Universal Permissive License (UPL), Version 1.0. See LICENSE.txt.
 * Modified by Oracle and/or its affiliates on 2026-03-06.
 *************************************************************************/

/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See third_party/licenses/Apache-2.0.txt for more license information
 *************************************************************************/

#include "plugin/nccl_net.h"
#include "transport/net_ib/common.h"

static void* gLegacyCtx = nullptr;
static int gLegacyRefCount = 0;

static ncclResult_t makeLegacyConfig(ncclNetCommConfig_t* config) {
  if (config == nullptr) return ncclInvalidArgument;
  *config = {};
  config->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  return ncclSuccess;
}

static ncclResult_t pluginInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  return ncclIbInit(ctx, commId, config, logFunction, profFunction);
}

static ncclResult_t occlPluginInitV11(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* configV11, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  if (configV11 == nullptr) return ncclIbInit(ctx, commId, nullptr, logFunction, profFunction);
  ncclNetCommConfig_t occlConfig = {};
  occlConfig.trafficClass = configV11->trafficClass;
  return ncclIbInit(ctx, commId, &occlConfig, logFunction, profFunction);
}

static ncclResult_t pluginFinalize(void* ctx) {
  return ncclIbFinalize(ctx);
}

static ncclResult_t pluginInitV10(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  if (gLegacyRefCount > 0) {
    gLegacyRefCount++;
    return ncclSuccess;
  }
  ncclNetCommConfig_t config;
  NCCLCHECK(makeLegacyConfig(&config));
  ncclResult_t ret = ncclIbInit(&gLegacyCtx, 0, &config, logFunction, profFunction);
  if (ret == ncclSuccess) gLegacyRefCount = 1;
  return ret;
}

static ncclResult_t pluginInitV9(ncclDebugLogger_t logFunction) {
  return pluginInitV10(logFunction, nullptr);
}

static ncclResult_t pluginDevices(int* ndev) {
  return ncclIbDevices(ndev);
}

static ncclResult_t occlPluginGetPropertiesV12(int dev, ncclNetProperties_t* props) {
  if (props == nullptr) return ncclInvalidArgument;
  NCCLCHECK(ncclIbGetProperties(dev, props));
  if (props->vProps.ndevs <= 0 || props->vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC_V12) return ncclInvalidUsage;
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
  return ncclSuccess;
}

static ncclResult_t occlPluginGetPropertiesV11(int dev, ncclNetProperties_v11_t* propsV11) {
  if (propsV11 == nullptr) return ncclInvalidArgument;
  ncclNetProperties_t occlProps = {};
  NCCLCHECK(occlPluginGetPropertiesV12(dev, &occlProps));
  if (occlProps.vProps.ndevs <= 0 || occlProps.vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC_V11) return ncclInvalidUsage;

  ncclNetProperties_v11_t occlPropsV11 = {};
  occlPropsV11.name = occlProps.name;
  occlPropsV11.pciPath = occlProps.pciPath;
  occlPropsV11.guid = occlProps.guid;
  occlPropsV11.ptrSupport = occlProps.ptrSupport;
  occlPropsV11.regIsGlobal = occlProps.regIsGlobal;
  occlPropsV11.forceFlush = occlProps.forceFlush;
  occlPropsV11.speed = occlProps.speed;
  occlPropsV11.port = occlProps.port;
  occlPropsV11.latency = occlProps.latency;
  occlPropsV11.maxComms = occlProps.maxComms;
  occlPropsV11.maxRecvs = occlProps.maxRecvs;
  occlPropsV11.netDeviceType = occlProps.netDeviceType;
  occlPropsV11.netDeviceVersion = occlProps.netDeviceVersion;
  occlPropsV11.vProps.ndevs = occlProps.vProps.ndevs;
  for (int i = 0; i < occlProps.vProps.ndevs; i++) occlPropsV11.vProps.devs[i] = occlProps.vProps.devs[i];
  occlPropsV11.maxP2pBytes = occlProps.maxP2pBytes;
  occlPropsV11.maxCollBytes = occlProps.maxCollBytes;
  occlPropsV11.maxMultiRequestSize = occlProps.maxMultiRequestSize;
  *propsV11 = occlPropsV11;
  return ncclSuccess;
}

static ncclResult_t pluginGetPropertiesV10(int dev, ncclNetProperties_v10_t* propsV10) {
  if (propsV10 == nullptr) return ncclInvalidArgument;
  ncclNetProperties_v11_t occlPropsV11 = {};
  NCCLCHECK(occlPluginGetPropertiesV11(dev, &occlPropsV11));
  ncclNetProperties_v10_t occlPropsV10 = {};
  occlPropsV10.name = occlPropsV11.name;
  occlPropsV10.pciPath = occlPropsV11.pciPath;
  occlPropsV10.guid = occlPropsV11.guid;
  occlPropsV10.ptrSupport = occlPropsV11.ptrSupport;
  occlPropsV10.regIsGlobal = occlPropsV11.regIsGlobal;
  occlPropsV10.forceFlush = occlPropsV11.forceFlush;
  occlPropsV10.speed = occlPropsV11.speed;
  occlPropsV10.port = occlPropsV11.port;
  occlPropsV10.latency = occlPropsV11.latency;
  occlPropsV10.maxComms = occlPropsV11.maxComms;
  occlPropsV10.maxRecvs = occlPropsV11.maxRecvs;
  occlPropsV10.netDeviceType = occlPropsV11.netDeviceType;
  occlPropsV10.netDeviceVersion = occlPropsV11.netDeviceVersion;
  occlPropsV10.vProps.ndevs = occlPropsV11.vProps.ndevs;
  for (int i = 0; i < occlPropsV11.vProps.ndevs; i++) occlPropsV10.vProps.devs[i] = occlPropsV11.vProps.devs[i];
  occlPropsV10.maxP2pBytes = occlPropsV11.maxP2pBytes;
  occlPropsV10.maxCollBytes = occlPropsV11.maxCollBytes;
  *propsV10 = occlPropsV10;
  return ncclSuccess;
}

static ncclResult_t pluginGetPropertiesV9(int dev, ncclNetProperties_v9_t* propsV9) {
  if (propsV9 == nullptr) return ncclInvalidArgument;
  ncclNetProperties_v10_t propsV10;
  NCCLCHECK(pluginGetPropertiesV10(dev, &propsV10));
  ncclNetProperties_v9_t occlPropsV9 = {};
  occlPropsV9.name = propsV10.name;
  occlPropsV9.pciPath = propsV10.pciPath;
  occlPropsV9.guid = propsV10.guid;
  occlPropsV9.ptrSupport = propsV10.ptrSupport;
  occlPropsV9.regIsGlobal = propsV10.regIsGlobal;
  occlPropsV9.forceFlush = propsV10.forceFlush;
  occlPropsV9.speed = propsV10.speed;
  occlPropsV9.port = propsV10.port;
  occlPropsV9.latency = propsV10.latency;
  occlPropsV9.maxComms = propsV10.maxComms;
  occlPropsV9.maxRecvs = propsV10.maxRecvs;
  occlPropsV9.netDeviceType = propsV10.netDeviceType;
  occlPropsV9.netDeviceVersion = propsV10.netDeviceVersion;
  occlPropsV9.vProps.ndevs = propsV10.vProps.ndevs;
  for (int i = 0; i < propsV10.vProps.ndevs; i++) occlPropsV9.vProps.devs[i] = propsV10.vProps.devs[i];
  occlPropsV9.maxP2pBytes = propsV10.maxP2pBytes;
  occlPropsV9.maxCollBytes = propsV10.maxCollBytes;
  *propsV9 = occlPropsV9;
  return ncclSuccess;
}

static ncclResult_t pluginGetPropertiesV8(int dev, ncclNetProperties_v8_t* propsV8) {
  ncclNetProperties_t props;
  NCCLCHECK(occlPluginGetPropertiesV12(dev, &props));
  propsV8->name = props.name;
  propsV8->pciPath = props.pciPath;
  propsV8->guid = props.guid;
  propsV8->ptrSupport = props.ptrSupport;
  propsV8->regIsGlobal = props.regIsGlobal;
  propsV8->speed = props.speed;
  propsV8->port = props.port;
  propsV8->latency = props.latency;
  propsV8->maxComms = props.maxComms;
  propsV8->maxRecvs = props.maxRecvs;
  propsV8->netDeviceType = props.netDeviceType;
  propsV8->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

static ncclResult_t pluginGetPropertiesV7(int dev, ncclNetProperties_v7_t* propsV7) {
  ncclNetProperties_t props;
  NCCLCHECK(occlPluginGetPropertiesV12(dev, &props));
  propsV7->name = props.name;
  propsV7->pciPath = props.pciPath;
  propsV7->guid = props.guid;
  propsV7->ptrSupport = props.ptrSupport;
  propsV7->speed = props.speed;
  propsV7->port = props.port;
  propsV7->latency = props.latency;
  propsV7->maxComms = props.maxComms;
  propsV7->maxRecvs = props.maxRecvs;
  propsV7->netDeviceType = props.netDeviceType;
  propsV7->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

static ncclResult_t pluginGetPropertiesV6(int dev, ncclNetProperties_v6_t* propsV6) {
  ncclNetProperties_t props;
  NCCLCHECK(occlPluginGetPropertiesV12(dev, &props));
  propsV6->name = props.name;
  propsV6->pciPath = props.pciPath;
  propsV6->guid = props.guid;
  propsV6->ptrSupport = props.ptrSupport;
  propsV6->speed = props.speed;
  propsV6->port = props.port;
  propsV6->latency = props.latency;
  propsV6->maxComms = props.maxComms;
  propsV6->maxRecvs = props.maxRecvs;
  return ncclSuccess;
}

static ncclResult_t pluginListen(void* ctx, int dev, void* handle, void** listenComm) {
  return ncclIbListen(ctx, dev, handle, listenComm);
}

static ncclResult_t pluginListenLegacy(int dev, void* handle, void** listenComm) {
  return ncclIbListen(gLegacyCtx, dev, handle, listenComm);
}

static ncclResult_t pluginConnect(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclIbConnect(ctx, dev, handle, sendComm, sendDevComm);
}

static ncclResult_t pluginConnectV10(int dev, ncclNetCommConfig_v10_t* configV10, void* handle, void** sendComm, ncclNetDeviceHandle_v10_t** sendDevComm) {
  (void)configV10;
  return ncclIbConnect(gLegacyCtx, dev, handle, sendComm, reinterpret_cast<ncclNetDeviceHandle_t**>(sendDevComm));
}

static ncclResult_t pluginConnectV9(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v9_t** sendDevComm) {
  return ncclIbConnect(gLegacyCtx, dev, handle, sendComm, reinterpret_cast<ncclNetDeviceHandle_t**>(sendDevComm));
}

static ncclResult_t pluginConnectV8(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v8_t** sendDevComm) {
  return ncclIbConnect(gLegacyCtx, dev, handle, sendComm, reinterpret_cast<ncclNetDeviceHandle_t**>(sendDevComm));
}

static ncclResult_t pluginConnectV7(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v7_t** sendDevComm) {
  return ncclIbConnect(gLegacyCtx, dev, handle, sendComm, reinterpret_cast<ncclNetDeviceHandle_t**>(sendDevComm));
}

static ncclResult_t pluginConnectV6(int dev, void* handle, void** sendComm) {
  ncclNetDeviceHandle_t* ignored = nullptr;
  return ncclIbConnect(gLegacyCtx, dev, handle, sendComm, &ignored);
}

static ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  return ncclIbAccept(listenComm, recvComm, recvDevComm);
}

static ncclResult_t pluginAcceptV10(void* listenComm, void** recvComm, ncclNetDeviceHandle_v10_t** recvDevComm) {
  return ncclIbAccept(listenComm, recvComm, reinterpret_cast<ncclNetDeviceHandle_t**>(recvDevComm));
}

static ncclResult_t pluginAcceptV9(void* listenComm, void** recvComm, ncclNetDeviceHandle_v9_t** recvDevComm) {
  return ncclIbAccept(listenComm, recvComm, reinterpret_cast<ncclNetDeviceHandle_t**>(recvDevComm));
}

static ncclResult_t pluginAcceptV8(void* listenComm, void** recvComm, ncclNetDeviceHandle_v8_t** recvDevComm) {
  return ncclIbAccept(listenComm, recvComm, reinterpret_cast<ncclNetDeviceHandle_t**>(recvDevComm));
}

static ncclResult_t pluginAcceptV7(void* listenComm, void** recvComm, ncclNetDeviceHandle_v7_t** recvDevComm) {
  return ncclIbAccept(listenComm, recvComm, reinterpret_cast<ncclNetDeviceHandle_t**>(recvDevComm));
}

static ncclResult_t pluginAcceptV6(void* listenComm, void** recvComm) {
  ncclNetDeviceHandle_t* ignored = nullptr;
  return ncclIbAccept(listenComm, recvComm, &ignored);
}

static ncclResult_t pluginRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclIbRegMr(comm, data, size, type, mhandle);
}

static ncclResult_t pluginRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return ncclIbRegMrDmaBuf(comm, data, size, type, offset, fd, mhandle);
}

static ncclResult_t pluginRegMrV7(void* comm, void* data, int size, int type, void** mhandle) {
  return ncclIbRegMr(comm, data, size, type, mhandle);
}

static ncclResult_t pluginDeregMr(void* comm, void* mhandle) {
  return ncclIbDeregMr(comm, mhandle);
}

static ncclResult_t pluginIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  return ncclIbIsend(sendComm, data, size, tag, mhandle, phandle, request);
}

static ncclResult_t pluginIsendV9(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  return ncclIbIsend(sendComm, data, size, tag, mhandle, nullptr, request);
}

static ncclResult_t pluginIsendV8(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  return ncclIbIsend(sendComm, data, size, tag, mhandle, nullptr, request);
}

static ncclResult_t pluginIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  return ncclIbIrecv(recvComm, n, data, sizes, tags, mhandles, phandles, request);
}

static ncclResult_t pluginIrecvV9(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  return ncclIbIrecv(recvComm, n, data, sizes, tags, mhandles, nullptr, request);
}

static ncclResult_t pluginIrecvV8(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  if (n < 0 || n > NCCL_NET_MAX_REQUESTS) return ncclInvalidArgument;
  size_t converted[NCCL_NET_MAX_REQUESTS];
  for (int i = 0; i < n; i++) converted[i] = sizes[i];
  return ncclIbIrecv(recvComm, n, data, converted, tags, mhandles, nullptr, request);
}

static ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  return ncclIbIflush(recvComm, n, data, sizes, mhandles, request);
}

static ncclResult_t pluginTest(void* request, int* done, int* sizes) {
  return ncclIbTest(request, done, sizes);
}

static ncclResult_t pluginCloseSend(void* sendComm) {
  return ncclIbCloseSend(sendComm);
}

static ncclResult_t pluginCloseRecv(void* recvComm) {
  return ncclIbCloseRecv(recvComm);
}

static ncclResult_t pluginCloseListen(void* listenComm) {
  return ncclIbCloseListen(listenComm);
}

static ncclResult_t occlPluginMakeVDeviceV12(int* d, ncclNetVDeviceProps_t* props) {
  if (d == nullptr || props == nullptr) return ncclInvalidArgument;
  if (props->ndevs <= 0 || props->ndevs > NCCL_NET_MAX_DEVS_PER_NIC_V12) return ncclInvalidArgument;
  return ncclIbMakeVDevice(d, props);
}

template <typename LegacyVProps>
static ncclResult_t occlPluginMakeLegacyVDevice(int* d, LegacyVProps* legacyProps) {
  if (d == nullptr || legacyProps == nullptr) return ncclInvalidArgument;
  constexpr int occlLegacyMaxDevs = sizeof(legacyProps->devs) / sizeof(legacyProps->devs[0]);
  if (legacyProps->ndevs <= 0 || legacyProps->ndevs > occlLegacyMaxDevs) return ncclInvalidArgument;

  ncclNetVDeviceProps_t occlProps = {};
  occlProps.ndevs = legacyProps->ndevs;
  for (int i = 0; i < occlProps.ndevs; i++) occlProps.devs[i] = legacyProps->devs[i];
  return ncclIbMakeVDevice(d, &occlProps);
}

static ncclResult_t occlPluginMakeVDeviceV11(int* d, ncclNetVDeviceProps_v11_t* propsV11) {
  return occlPluginMakeLegacyVDevice(d, propsV11);
}

static ncclResult_t pluginMakeVDeviceV10(int* d, ncclNetVDeviceProps_v10_t* propsV10) {
  return occlPluginMakeLegacyVDevice(d, propsV10);
}

static ncclResult_t pluginMakeVDeviceV9(int* d, ncclNetVDeviceProps_v9_t* propsV9) {
  return occlPluginMakeLegacyVDevice(d, propsV9);
}

static ncclResult_t occlPluginSetNetAttrV11(void* ctx, ncclNetAttr_v11_t* netAttrV11) {
  if (netAttrV11 == nullptr) return ncclIbSetNetAttr(ctx, nullptr);

  ncclNetAttr_t occlAttr = {};
  occlAttr.sendCommAttr.maxConcurrentPeers = netAttrV11->sendCommAttr.maxConcurrentPeers;
  occlAttr.sendCommAttr.minConcurrentPeers = netAttrV11->sendCommAttr.minConcurrentPeers;
  occlAttr.sendCommAttr.maxFlowsPerPeer = netAttrV11->sendCommAttr.maxFlowsPerPeer;
  occlAttr.sendCommAttr.minFlowsPerPeer = netAttrV11->sendCommAttr.minFlowsPerPeer;
  occlAttr.recvCommAttr.maxConcurrentPeers = netAttrV11->recvCommAttr.maxConcurrentPeers;
  occlAttr.recvCommAttr.minConcurrentPeers = netAttrV11->recvCommAttr.minConcurrentPeers;
  occlAttr.recvCommAttr.maxFlowsPerPeer = netAttrV11->recvCommAttr.maxFlowsPerPeer;
  occlAttr.recvCommAttr.minFlowsPerPeer = netAttrV11->recvCommAttr.minFlowsPerPeer;
  occlAttr.op = netAttrV11->op;
  occlAttr.algo = netAttrV11->algo;
  occlAttr.proto = netAttrV11->proto;
  return ncclIbSetNetAttr(ctx, &occlAttr);
}

extern "C" {

__attribute__((visibility("default"))) ncclNet_v12_t ncclNetPlugin_v12 = {
  "IB",
  pluginInit,
  pluginDevices,
  occlPluginGetPropertiesV12,
  pluginListen,
  pluginConnect,
  pluginAccept,
  pluginRegMr,
  pluginRegMrDmaBuf,
  pluginDeregMr,
  pluginIsend,
  pluginIrecv,
  pluginIflush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen,
  nullptr,
  nullptr,
  occlPluginMakeVDeviceV12,
  pluginFinalize,
  ncclIbSetNetAttr
};

__attribute__((visibility("default"))) ncclNet_v11_t ncclNetPlugin_v11 = {
  "IB",
  occlPluginInitV11,
  pluginDevices,
  occlPluginGetPropertiesV11,
  pluginListen,
  pluginConnect,
  pluginAccept,
  pluginRegMr,
  pluginRegMrDmaBuf,
  pluginDeregMr,
  pluginIsend,
  pluginIrecv,
  pluginIflush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen,
  nullptr,
  nullptr,
  occlPluginMakeVDeviceV11,
  pluginFinalize,
  occlPluginSetNetAttrV11
};

__attribute__((visibility("default"))) ncclNet_v10_t ncclNetPlugin_v10 = {
  "IB",
  pluginInitV10,
  pluginDevices,
  pluginGetPropertiesV10,
  pluginListenLegacy,
  pluginConnectV10,
  pluginAcceptV10,
  pluginRegMr,
  pluginRegMrDmaBuf,
  pluginDeregMr,
  pluginIsend,
  pluginIrecv,
  pluginIflush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen,
  nullptr,
  nullptr,
  pluginMakeVDeviceV10
};

__attribute__((visibility("default"))) ncclNet_v9_t ncclNetPlugin_v9 = {
  "IB",
  pluginInitV9,
  pluginDevices,
  pluginGetPropertiesV9,
  pluginListenLegacy,
  pluginConnectV9,
  pluginAcceptV9,
  pluginRegMr,
  pluginRegMrDmaBuf,
  pluginDeregMr,
  pluginIsendV9,
  pluginIrecvV9,
  pluginIflush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen,
  nullptr,
  nullptr,
  pluginMakeVDeviceV9
};

__attribute__((visibility("default"))) ncclNet_v8_t ncclNetPlugin_v8 = {
  "IB",
  pluginInitV9,
  pluginDevices,
  pluginGetPropertiesV8,
  pluginListenLegacy,
  pluginConnectV8,
  pluginAcceptV8,
  pluginRegMr,
  pluginRegMrDmaBuf,
  pluginDeregMr,
  pluginIsendV8,
  pluginIrecvV8,
  pluginIflush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen,
  nullptr,
  nullptr
};

__attribute__((visibility("default"))) ncclNet_v7_t ncclNetPlugin_v7 = {
  "IB",
  pluginInitV9,
  pluginDevices,
  pluginGetPropertiesV7,
  pluginListenLegacy,
  pluginConnectV7,
  pluginAcceptV7,
  pluginRegMrV7,
  pluginRegMrDmaBuf,
  pluginDeregMr,
  pluginIsendV8,
  pluginIrecvV8,
  pluginIflush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen,
  nullptr,
  nullptr
};

__attribute__((visibility("default"))) ncclNet_v6_t ncclNetPlugin_v6 = {
  "IB",
  pluginInitV9,
  pluginDevices,
  pluginGetPropertiesV6,
  pluginListenLegacy,
  pluginConnectV6,
  pluginAcceptV6,
  pluginRegMrV7,
  pluginRegMrDmaBuf,
  pluginDeregMr,
  pluginIsendV8,
  pluginIrecvV8,
  pluginIflush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen
};

}
