/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * Modifications by Oracle and/or its affiliates are licensed under the
 * Universal Permissive License (UPL), Version 1.0. See LICENSE.txt.
 * Modified by Oracle and/or its affiliates on 2026-03-06.
 *************************************************************************/

/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See third_party/licenses/Apache-2.0.txt for more license information
 *************************************************************************/

#include "common.h"

#include <algorithm>
#include <limits>

NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbDataDirect,"IB_DATA_DIRECT",1);
NCCL_PARAM(OciFrmEnable, "OCI_FRM_ENABLE", 1);

double g_inv_cyc_per_us = 0.0;
bool g_ncclIbUseCim = false;

static std::mutex ncclIbMutex;

// With ncclNet_v11_t the NCCL core initializes the network plugin per-communicator
// rather than once for all communicators. However, the internal plugin implementation
// still assumes the plugin is initialized only once across all communicators. The ref
// counter makes sure the plugin internally initializes only once. When per communicator
// context support is added to the plugin the ref counter can be removed.
static int netRefCount;

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);
NCCL_PARAM(IbDevicePciOrder, "IB_DEVICE_PCI_ORDER", 1);

// Returns 0 if this is the path of two VFs of the same physical device
static int ncclIbMatchVfPath(char* path1, char* path2) {
  // Merge multi-port NICs into the same PCI device
  if (ncclParamIbMergeVfs()) {
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}

static int ncclIbCompareDevs(const void* dev1, const void* dev2) {
  // Compare devices using the last component of the PCI path.
  // Note: fullPciPath is never NULL but empty if not found by realpath.
  char* path1 = ((struct ncclIbDev*)dev1)->fullPciPath;
  char* path2 = ((struct ncclIbDev*)dev2)->fullPciPath;

  // if a path is empty, order the devices to the back of the list
  if (strlen(path1) == 0 || strlen(path2) == 0) return strlen(path2) - strlen(path1);

  int64_t id1, id2;
  pciPathToInt64(path1, &id1);
  pciPathToInt64(path2, &id2);

  return (id1 < id2) ? -1 : ((id1 == id2) ? 0 : 1);
}

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, char* fullPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  // set fullPath to empty if realpath returned NULL
  snprintf(fullPath, PATH_MAX, "%s", p ? p : "");
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (ncclParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
  }
  if (path) *path = p;
  return ncclSuccess;
}

static ncclResult_t ncclIbGetRealPort(char* pciPath, int* realPort, int devIdx) {
  *realPort = 0;
  if (pciPath == NULL) return ncclSuccess;
  // Keep the real port aside (the ibv port is always 1 on recent cards)
  // Count only devices before the current device index to assign unique port numbers
  for (int d = 0; d < devIdx; d++) {
    if (ncclIbMatchVfPath(pciPath, ncclIbDevs[d].pciPath)) (*realPort)++;
  }
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000, /* NDR */
  200000  /* XDR */
};

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

static bool ncclMlx5dvDmaBufCapable(ibv_context *context){
  ncclResult_t res;
  int dev_fail = 0;

  struct ibv_pd* pd;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, context), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  (void)wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/, 0 /* mlx5 flags*/);
  // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  return true;
failure:
  return false;
}

static bool occlIbForceMergeAliasesEnabled() {
  const char* value = ncclGetEnv("NCCL_NET_FORCE_MERGE");
  return value != NULL && value[0] != '\0';
}

static bool occlIbSameMemberSet(const ncclNetVDeviceProps_t* left, const ncclNetVDeviceProps_t* right) {
  if (left == NULL || right == NULL || left->ndevs != right->ndevs) return false;
  for (int i = 0; i < left->ndevs; i++) {
    bool found = false;
    for (int j = 0; j < right->ndevs; j++) {
      if (left->devs[i] == right->devs[j]) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

static bool occlIbMemberSetsOverlap(const ncclNetVDeviceProps_t* left, const ncclNetVDeviceProps_t* right) {
  if (left == NULL || right == NULL) return false;
  for (int i = 0; i < left->ndevs; i++) {
    for (int j = 0; j < right->ndevs; j++) {
      if (left->devs[i] == right->devs[j]) return true;
    }
  }
  return false;
}

static ncclResult_t occlIbValidateVDeviceMembers(const ncclNetVDeviceProps_t* props) {
  if (props == NULL || props->ndevs <= 0 || props->ndevs > NCCL_IB_MAX_DEVS_PER_NIC) {
    WARN("NET/IB : Invalid virtual device member count %d", props == NULL ? -1 : props->ndevs);
    return ncclInvalidUsage;
  }
  for (int i = 0; i < props->ndevs; i++) {
    if (props->devs[i] < 0 || props->devs[i] >= ncclNIbDevs) {
      WARN("NET/IB : Cannot use physical device %d, physical count %d", props->devs[i], ncclNIbDevs);
      return ncclInvalidUsage;
    }
    for (int j = 0; j < i; j++) {
      if (props->devs[j] == props->devs[i]) {
        WARN("NET/IB : Cannot create a virtual device with duplicate physical device %d", props->devs[i]);
        return ncclInvalidUsage;
      }
    }
  }
  ncclIbDev* first = ncclIbDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    ncclIbDev* member = ncclIbDevs + props->devs[i];
    if (member->link != first->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           props->devs[0], first->devName, first->portNum, NCCL_IB_LLSTR(first->link),
           props->devs[i], member->devName, member->portNum, NCCL_IB_LLSTR(member->link));
      return ncclInvalidUsage;
    }
  }
  return ncclSuccess;
}

static ncclResult_t occlIbMakeLegacyVDevice(int* d, const ncclNetVDeviceProps_t* props) {
  if (ncclNMergedIbDevs >= MAX_IB_VDEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_VDEVS);
    return ncclInvalidUsage;
  }

  ncclIbMergedDev staged = {};
  staged.vProps = *props;
  staged.publicVProps = *props;
  staged.anchorDev = props->devs[0];
  staged.firstAlias = ncclNMergedIbDevs;
  staged.aliasCount = 1;
  for (int i = 0; i < props->ndevs; i++) {
    ncclIbDev* member = ncclIbDevs + props->devs[i];
    staged.speed += member->speed;
    size_t used = strlen(staged.devName);
    int written = snprintf(staged.devName + used, sizeof(staged.devName) - used,
                           "%s%s", i == 0 ? "" : "+", member->devName);
    if (written < 0 || (size_t)written >= sizeof(staged.devName) - used) {
      WARN("NET/IB : Virtual device name is too long");
      return ncclInvalidUsage;
    }
  }

  *d = ncclNMergedIbDevs;
  ncclIbMergedDevs[ncclNMergedIbDevs++] = staged;
  INFO(NCCL_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, staged.devName, staged.speed, staged.vProps.ndevs);
  return ncclSuccess;
}

static ncclResult_t occlIbMakeForcedAliases(int* d, const ncclNetVDeviceProps_t* props) {
  for (int virtualDev = 0; virtualDev < ncclNMergedIbDevs; virtualDev++) {
    const ncclIbMergedDev* existing = ncclIbMergedDevs + virtualDev;
    if (!existing->isForcedAlias || existing->firstAlias != virtualDev) continue;
    if (occlIbSameMemberSet(&existing->vProps, props)) {
      *d = existing->firstAlias;
      return ncclSuccess;
    }
    if (occlIbMemberSetsOverlap(&existing->vProps, props)) {
      WARN("NET/IB : Forced merge overlaps existing alias group beginning at virtual device %d", existing->firstAlias);
      return ncclInvalidUsage;
    }
  }

  // pciPath already applies the multi-port/VF normalization used by the
  // physical properties. Preserve up to two ports at that locality as the
  // active unit; missing topology information must not join unrelated NICs.
  ncclNetVDeviceProps_t activeGroups[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  bool assigned[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  int orderedMembers[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  int groupOffsets[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  int groupCount = 0;
  int memberCount = 0;
  for (int i = 0; i < props->ndevs; i++) {
    if (assigned[i]) continue;
    ncclNetVDeviceProps_t* active = &activeGroups[groupCount];
    const int anchor = props->devs[i];
    active->devs[active->ndevs++] = anchor;
    assigned[i] = true;
    const char* anchorPath = ncclIbDevs[anchor].pciPath;
    if (anchorPath != NULL && anchorPath[0] != '\0') {
      for (int j = i + 1; j < props->ndevs; j++) {
        if (assigned[j]) continue;
        const char* memberPath = ncclIbDevs[props->devs[j]].pciPath;
        if (memberPath != NULL && memberPath[0] != '\0' && strcmp(anchorPath, memberPath) == 0) {
          active->devs[active->ndevs++] = props->devs[j];
          assigned[j] = true;
          break;
        }
      }
    }
    groupOffsets[groupCount++] = memberCount;
    for (int member = 0; member < active->ndevs; member++) {
      orderedMembers[memberCount++] = active->devs[member];
    }
  }

  if (ncclNMergedIbDevs > MAX_IB_VDEVS - groupCount) {
    WARN("NET/IB : Cannot allocate %d forced-merge aliases, only %d virtual slots remain",
         groupCount, MAX_IB_VDEVS - ncclNMergedIbDevs);
    return ncclInvalidUsage;
  }

  const int firstAlias = ncclNMergedIbDevs;
  ncclIbMergedDev staged[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  for (int alias = 0; alias < groupCount; alias++) {
    const ncclNetVDeviceProps_t* active = &activeGroups[alias];
    staged[alias].vProps.ndevs = props->ndevs;
    for (int member = 0; member < props->ndevs; member++) {
      staged[alias].vProps.devs[member] = orderedMembers[(groupOffsets[alias] + member) % memberCount];
    }
    staged[alias].publicVProps = *active;
    staged[alias].anchorDev = active->devs[0];
    staged[alias].firstAlias = firstAlias;
    staged[alias].aliasCount = groupCount;
    staged[alias].isForcedAlias = 1;
    int64_t activeSpeed = 0;
    for (int member = 0; member < active->ndevs; member++) {
      const ncclIbDev* physical = &ncclIbDevs[active->devs[member]];
      activeSpeed += physical->speed;
      const size_t used = strlen(staged[alias].devName);
      const int written = snprintf(staged[alias].devName + used, sizeof(staged[alias].devName) - used,
                                   "%s%s", member == 0 ? "" : "+", physical->devName);
      if (written < 0 || (size_t)written >= sizeof(staged[alias].devName) - used) {
        WARN("NET/IB : Forced-merge alias name is too long");
        return ncclInvalidUsage;
      }
    }
    if (activeSpeed > std::numeric_limits<int>::max() || activeSpeed < std::numeric_limits<int>::min()) {
      WARN("NET/IB : Forced-merge active speed exceeds the property limit");
      return ncclInvalidUsage;
    }
    staged[alias].speed = (int)activeSpeed;
  }

  for (int alias = 0; alias < groupCount; alias++) {
    ncclIbMergedDevs[firstAlias + alias] = staged[alias];
  }
  ncclNMergedIbDevs += groupCount;
  *d = firstAlias;
  INFO(NCCL_NET, "NET/IB : Made %d GPU-local aliases for forced merge beginning at virtual device %d", groupCount, firstAlias);
  return ncclSuccess;
}

ncclResult_t ncclIbMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props) {
  if (d == NULL || props == NULL) return ncclInvalidArgument;
  if (ncclParamIbMergeNics() == 0 && props->ndevs > 1) {
    INFO(NCCL_NET, "NET/IB : Skipping makeVDevice, NCCL_IB_MERGE_NICS=0");
    return ncclInvalidUsage;
  }
  NCCLCHECK(occlIbValidateVDeviceMembers(props));
  if (props->ndevs > 1 && occlIbForceMergeAliasesEnabled()) {
    return occlIbMakeForcedAliases(d, props);
  }
  return occlIbMakeLegacyVDevice(d, props);
}

ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  std::lock_guard<std::mutex> lock(ncclIbMutex);
  ncclResult_t res = ncclIbMakeVDeviceInternal(d, props);
  return res;

}

ncclResult_t ncclIbSetNetAttr(void *ctx, ncclNetAttr_t *netAttr) {
  (void)ctx;
  (void)netAttr;
  return ncclSuccess;
}

const char* ibProviderName[] = {
  "None",
  "Mlx5",
};

ncclResult_t ncclIbFinalizeDevices(void) {
  netRefCount--;
  return ncclSuccess;
}

ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  if (netRefCount++) return ret;
  ncclProfilerFunction = profFunction;
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }
  if(wrap_mlx5dv_symbols() != ncclSuccess) { INFO(NCCL_NET, "NET/IB : Failed to open mlx5dv symbols. Advance features like CX-8 Direct-NIC will be disabled."); }

  if (ncclNIbDevs == -1) {
    std::lock_guard<std::mutex> lock(ncclIbMutex);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      int nIpIfs = 0;
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      NCCLCHECK(ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = ncclInternalError; goto fail; }

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context = NULL;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        char dataDirectDevicePath[PATH_MAX] = "/sys";
        int devCount = /*undefined*/-1, devOffset = 0;
        enum ncclIbProvider ibProvider = wrap_mlx5dv_is_supported(devices[d]) ? IB_PROVIDER_MLX5 : IB_PROVIDER_NONE;

        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
            struct ibv_port_attr portAttr;
            if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
              WARN("NET/IB : Unable to query port_num %d", port_num);
              continue;
            }
            if (portAttr.state != IBV_PORT_ACTIVE) continue;
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

            // check against user specified HCAs/ports
            if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
              continue;
            }

            // check for mlx5 data direct support only once for a each device
            if (devCount == -1) {
              devCount = 1;
              devOffset = 0;
              if (ncclParamIbDataDirect() > 0 && ibProvider == IB_PROVIDER_MLX5 && ncclMlx5dvDmaBufCapable(context)) {
                int pathLen = strlen(dataDirectDevicePath);
                ncclResult_t res = wrap_mlx5dv_get_data_direct_sysfs_path(context, dataDirectDevicePath + pathLen, sizeof(dataDirectDevicePath) - pathLen);
                if (res == ncclSuccess) {
                  // data direct devices are exposed twice: with the C2C + PCIe link and with the data direct link
                  devCount = 2;
                  // by default only expose the data direct NIC (devOffset = 1), unless set to 2 by the user
                  devOffset = (ncclParamIbDataDirect() == 2) ? 0 : 1;
                  INFO(NCCL_INIT | NCCL_NET, "NET/IB: Data Direct DMA Interface is detected for device %s", devices[d]->name);
                } else if (res == ncclInvalidArgument) {
                  TRACE(NCCL_NET, "NET/IB: Device %s does not support Data Direct DMA.", devices[d]->name);
                } else {
                  WARN("NET/IB: Error in mlx5dv_get_data_direct_sysfs_path with device %s", devices[d]->name);
                  return res;
                }
              }
            }
            for (int dev = devOffset; dev < devCount; ++dev) {
              ncclIbDevs[ncclNIbDevs].device = d;
              ncclIbDevs[ncclNIbDevs].ibProvider = ibProvider;
              ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
              ncclIbDevs[ncclNIbDevs].portAttr = portAttr;
              ncclIbDevs[ncclNIbDevs].portNum = port_num;
              ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
              if (portAttr.active_speed_ex) {
                // A non-zero active_speed_ex indicates XDR rate (0x100) or higher
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed_ex) * ncclIbWidth(portAttr.active_width);
              } else {
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
              }
              ncclIbDevs[ncclNIbDevs].context = context;
              ncclIbDevs[ncclNIbDevs].pdRefs = 0;
              ncclIbDevs[ncclNIbDevs].pd = NULL;
              // for dev==1 (data direct device), pciPath is given by mlx5
              strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
              NCCLCHECKGOTO(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, (dev == 1) ? NULL : &ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].fullPciPath), ret, fail);
              if (dev == 1) {
                snprintf(ncclIbDevs[ncclNIbDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
                NCCLCHECK(ncclCalloc(&ncclIbDevs[ncclNIbDevs].pciPath, PATH_MAX));
                strncpy(ncclIbDevs[ncclNIbDevs].pciPath, dataDirectDevicePath, PATH_MAX);
                ncclIbDevs[ncclNIbDevs].capsProvider.mlx5.dataDirect = 1;
              }

              ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
              ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
              NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

              // Enable ADAPTIVE_ROUTING by default on IB networks
              // But allow it to be overloaded by an env parameter
              ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
              if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();

              INFO(NCCL_NET, "NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name,
                   ncclIbDevs[ncclNIbDevs].portNum, NCCL_IB_LLSTR(portAttr.link_layer), ibProviderName[ncclIbDevs[ncclNIbDevs].ibProvider], ncclIbDevs[ncclNIbDevs].speed, context,
                   ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar);

              ncclIbAsyncThread = std::thread(ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs);
              ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
              ncclIbAsyncThread.detach();

              ncclNIbDevs++;
              nPorts++;
            }
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
      }

      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; }
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }
    // Determine whether RELAXED_ORDERING is enabled and possible
    ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
    // sort devices to ensure a consistent order across nodes
    if (ncclParamIbDevicePciOrder()) qsort(ncclIbDevs, ncclNIbDevs, sizeof(struct ncclIbDev), ncclIbCompareDevs);
    // Once sorted, get the realPort ID and create the virtual devices.
    // Doing it after sorting ensures that devices will have consistent realPort ids across nodes.
    char line[2048] = "";
    for (int d = 0; d < ncclNIbDevs; d++) {
      NCCLCHECKGOTO(ncclIbGetRealPort(ncclIbDevs[d].pciPath, &ncclIbDevs[d].realPort, d), ret, fail);
      snprintf(line + strlen(line), sizeof(line) - strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName, ncclIbDevs[d].portNum, NCCL_IB_LLSTR(ncclIbDevs[d].link));

      // Add this plain physical device to the list of virtual devices (after sorting)
      int vDev;
      ncclNetVDeviceProps_t vProps = {0};
      vProps.ndevs = 1;
      vProps.devs[0] = d;
      NCCLCHECK(ncclIbMakeVDeviceInternal(&vDev, &vProps));
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT | NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "", ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  ncclNetCommConfig_t* netCommConfig = nullptr;
  g_ncclIbUseCim = !!ncclParamOciFrmEnable();
  NCCLCHECK(ncclIbInitDevices(logFunction, profFunction));
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = config->trafficClass;
  *ctx = (void *)netCommConfig;
  (void)gettime();
  double f = timer_cycles_per_us();
  g_inv_cyc_per_us = (f > 0.0) ? (1.0 / f) : 0.0;
  return ret;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  bool dmaBufSupported = ncclIbDmaBufSupport(dev) == ncclSuccess;
  if (ncclIbGdrSupport() == ncclSuccess || dmaBufSupported) {
    // The plugin ABI keeps the GPU pointer token name even when the underlying
    // device runtime is ROCm. DMA-BUF therefore still needs NCCL_PTR_CUDA set.
    props->ptrSupport |= NCCL_PTR_CUDA;
  }
  props->regIsGlobal = 1;
  if (dmaBufSupported) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->forceFlush = 0;
  if (ibDev->capsProvider.mlx5.dataDirect) {
    props->forceFlush = 1;
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  if (props == NULL) return ncclInvalidArgument;
  if (dev < 0 || dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, ncclNMergedIbDevs);
    return ncclInvalidUsage;
  }
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + dev;
  NCCLCHECK(ncclIbGetPhysProperties(mergedDev->anchorDev, props));
  const bool pairedAlias = mergedDev->isForcedAlias && mergedDev->publicVProps.ndevs > 1;
  if (!mergedDev->isForcedAlias || pairedAlias) {
    props->name = mergedDev->devName;
    props->speed = mergedDev->speed;
  }
  if (pairedAlias) {
    // Registration covers private standby members too. Advertise only modes
    // and limits that remain usable on every member, while topology/speed
    // describe the active local pair alone.
    for (int member = 0; member < mergedDev->vProps.ndevs; member++) {
      if (mergedDev->vProps.devs[member] == mergedDev->anchorDev) continue;
      ncclNetProperties_t memberProps = {};
      NCCLCHECK(ncclIbGetPhysProperties(mergedDev->vProps.devs[member], &memberProps));
      props->ptrSupport &= memberProps.ptrSupport;
      props->regIsGlobal &= memberProps.regIsGlobal;
      props->forceFlush |= memberProps.forceFlush;
      props->maxComms = std::min(props->maxComms, memberProps.maxComms);
      props->maxRecvs = std::min(props->maxRecvs, memberProps.maxRecvs);
      props->maxP2pBytes = std::min(props->maxP2pBytes, memberProps.maxP2pBytes);
      props->maxCollBytes = std::min(props->maxCollBytes, memberProps.maxCollBytes);
      props->maxMultiRequestSize = std::min(props->maxMultiRequestSize, memberProps.maxMultiRequestSize);
    }
  }
  memcpy(&props->vProps, &mergedDev->publicVProps, sizeof(ncclNetVDeviceProps_t));
  return ncclSuccess;
}

ncclResult_t ncclIbFinalize(void* ctx) {
  free(ctx);
  return ncclIbFinalizeDevices();
}
