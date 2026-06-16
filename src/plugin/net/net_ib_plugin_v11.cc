/*************************************************************************
 * Copyright (c) 2026 OCCL-net.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Modified by OCCL-net on 2026-03-06.
 *************************************************************************/

/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
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

static ncclResult_t pluginGetProperties(int dev, ncclNetProperties_t* props) {
  return ncclIbGetProperties(dev, props);
}

static ncclResult_t pluginGetPropertiesV10(int dev, ncclNetProperties_v10_t* propsV10) {
  ncclNetProperties_t props;
  NCCLCHECK(pluginGetProperties(dev, &props));
  propsV10->name = props.name;
  propsV10->pciPath = props.pciPath;
  propsV10->guid = props.guid;
  propsV10->ptrSupport = props.ptrSupport;
  propsV10->regIsGlobal = props.regIsGlobal;
  propsV10->forceFlush = props.forceFlush;
  propsV10->speed = props.speed;
  propsV10->port = props.port;
  propsV10->latency = props.latency;
  propsV10->maxComms = props.maxComms;
  propsV10->maxRecvs = props.maxRecvs;
  propsV10->netDeviceType = props.netDeviceType;
  propsV10->netDeviceVersion = props.netDeviceVersion;
  propsV10->vProps.ndevs = props.vProps.ndevs;
  for (int i = 0; i < props.vProps.ndevs; i++) propsV10->vProps.devs[i] = props.vProps.devs[i];
  propsV10->maxP2pBytes = props.maxP2pBytes;
  propsV10->maxCollBytes = props.maxCollBytes;
  return ncclSuccess;
}

static ncclResult_t pluginGetPropertiesV9(int dev, ncclNetProperties_v9_t* propsV9) {
  ncclNetProperties_v10_t propsV10;
  NCCLCHECK(pluginGetPropertiesV10(dev, &propsV10));
  propsV9->name = propsV10.name;
  propsV9->pciPath = propsV10.pciPath;
  propsV9->guid = propsV10.guid;
  propsV9->ptrSupport = propsV10.ptrSupport;
  propsV9->regIsGlobal = propsV10.regIsGlobal;
  propsV9->forceFlush = propsV10.forceFlush;
  propsV9->speed = propsV10.speed;
  propsV9->port = propsV10.port;
  propsV9->latency = propsV10.latency;
  propsV9->maxComms = propsV10.maxComms;
  propsV9->maxRecvs = propsV10.maxRecvs;
  propsV9->netDeviceType = propsV10.netDeviceType;
  propsV9->netDeviceVersion = propsV10.netDeviceVersion;
  propsV9->vProps.ndevs = propsV10.vProps.ndevs;
  for (int i = 0; i < propsV10.vProps.ndevs; i++) propsV9->vProps.devs[i] = propsV10.vProps.devs[i];
  propsV9->maxP2pBytes = propsV10.maxP2pBytes;
  propsV9->maxCollBytes = propsV10.maxCollBytes;
  return ncclSuccess;
}

static ncclResult_t pluginGetPropertiesV8(int dev, ncclNetProperties_v8_t* propsV8) {
  ncclNetProperties_t props;
  NCCLCHECK(pluginGetProperties(dev, &props));
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
  NCCLCHECK(pluginGetProperties(dev, &props));
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
  NCCLCHECK(pluginGetProperties(dev, &props));
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

static ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  return ncclIbMakeVDevice(d, props);
}

static ncclResult_t pluginMakeVDeviceV10(int* d, ncclNetVDeviceProps_v10_t* propsV10) {
  ncclNetVDeviceProps_t props = {};
  props.ndevs = propsV10->ndevs;
  for (int i = 0; i < props.ndevs; i++) props.devs[i] = propsV10->devs[i];
  return ncclIbMakeVDevice(d, &props);
}

static ncclResult_t pluginMakeVDeviceV9(int* d, ncclNetVDeviceProps_v9_t* propsV9) {
  ncclNetVDeviceProps_t props = {};
  props.ndevs = propsV9->ndevs;
  for (int i = 0; i < props.ndevs; i++) props.devs[i] = propsV9->devs[i];
  return ncclIbMakeVDevice(d, &props);
}

extern "C" {

__attribute__((visibility("default"))) ncclNet_v11_t ncclNetPlugin_v11 = {
  "IB",
  pluginInit,
  pluginDevices,
  pluginGetProperties,
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
  pluginMakeVDevice,
  pluginFinalize,
  ncclIbSetNetAttr
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
