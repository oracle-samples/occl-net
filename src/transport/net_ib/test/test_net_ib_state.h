/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#ifndef OCCL_NET_IB_TEST_NET_IB_STATE_H_
#define OCCL_NET_IB_TEST_NET_IB_STATE_H_

#include "common.h"

#include <thread>

char ncclIbIfName[MAX_IF_NAME_SIZE + 1] = {};
union ncclSocketAddress ncclIbIfAddr = {};

int ncclNMergedIbDevs = -1;
int ncclNIbDevs = -1;
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS] = {};
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS] = {};
int ncclIbRelaxedOrderingEnabled = 0;

ncclProfilerCallback_t ncclProfilerFunction = nullptr;
std::thread ncclIbAsyncThread;

#endif
