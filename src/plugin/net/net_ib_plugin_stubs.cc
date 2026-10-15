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

#include <cstdlib>

#include "env.h"

#ifndef OCCL_NET_IB_PLUGIN_VERSION
#define OCCL_NET_IB_PLUGIN_VERSION "0.1.0"
#endif

namespace {
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((used, section(".occl_net_ib_version")))
#endif
const char occlNetIbPluginVersionSection[] =
    "occl-net-ib version " OCCL_NET_IB_PLUGIN_VERSION;
}  // namespace

// Standalone net_ib plugin build does not link the full core environment
// initialization objects. Provide the minimal symbols required by shared
// utility code.

ncclResult_t ncclInitEnv(void) {
  return ncclSuccess;
}

const char* ncclEnvPluginGetEnv(const char* name) {
  return std::getenv(name);
}

bool ncclEnvPluginInitialized(void) {
  return false;
}

int ncclParamSetCpuStackSize(void) {
  return 1;
}
