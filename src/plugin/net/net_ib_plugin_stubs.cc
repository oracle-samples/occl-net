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

#include <cstdlib>

#include "env.h"

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
