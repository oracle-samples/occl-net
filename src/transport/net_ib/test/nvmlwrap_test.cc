/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "nvmlwrap.h"

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::array<const char*, 19> kNvmlSymbolNames = {{
  "nvmlInit",
  "nvmlInit_v2",
  "nvmlShutdown",
  "nvmlDeviceGetCount",
  "nvmlDeviceGetCount_v2",
  "nvmlDeviceGetHandleByPciBusId",
  "nvmlDeviceGetHandleByIndex",
  "nvmlDeviceGetIndex",
  "nvmlErrorString",
  "nvmlDeviceGetNvLinkState",
  "nvmlDeviceGetNvLinkRemotePciInfo",
  "nvmlDeviceGetNvLinkCapability",
  "nvmlDeviceGetCudaComputeCapability",
  "nvmlDeviceGetP2PStatus",
  "nvmlDeviceGetFieldValues",
  "nvmlDeviceGetGpuFabricInfoV",
  "nvmlDeviceGetPlatformInfo",
  "nvmlSystemGetConfComputeState",
  "nvmlSystemGetConfComputeSettings",
}};

/** @brief Selects one fake NVML provider failure point. */
enum class NvmlFailure {
  None,
  Init,
  Count,
  Handle,
  ComputeCapability,
  P2pRead,
  P2pWrite,
  P2pOther,
  PciHandle,
  LinkState,
  RemotePci,
  LinkCapability,
  FieldValues,
  Fabric,
  Platform,
  CcModern,
  CcLegacy,
};

/** @brief Records one directional peer-capability query. */
struct P2pCall {
  nvmlDevice_t first;
  nvmlDevice_t second;
  nvmlGpuP2PCapsIndex_t capability;
};

/**
 * @brief Models NVML loading and failures while recording inventory, topology, and
 * optional-query interactions.
 */
struct NvmlProviderState {
  bool libraryAvailable = true;
  std::vector<std::string> missingSymbols;
  std::vector<std::pair<std::string, int>> openCalls;
  std::vector<std::pair<void*, std::string>> resolveCalls;
  unsigned char libraryHandle = 0;
  NvmlFailure failure = NvmlFailure::None;
  unsigned int deviceCount = 2;
  int legacyInitCalls = 0;
  int modernInitCalls = 0;
  int legacyCountCalls = 0;
  int modernCountCalls = 0;
  int shutdownCalls = 0;
  int handleCalls = 0;
  int computeCalls = 0;
  int providerIndexCalls = 0;
  std::vector<P2pCall> p2pCalls;
  int pciHandleCalls = 0;
  std::string pciBusId;
  int linkStateCalls = 0;
  nvmlDevice_t linkStateDevice = nullptr;
  unsigned int linkStateLink = 0;
  int remotePciCalls = 0;
  nvmlDevice_t remotePciDevice = nullptr;
  unsigned int remotePciLink = 0;
  int linkCapabilityCalls = 0;
  nvmlDevice_t linkCapabilityDevice = nullptr;
  unsigned int linkCapabilityLink = 0;
  nvmlNvLinkCapability_t linkCapability = NVML_NVLINK_CAP_COUNT;
  int fieldCalls = 0;
  nvmlDevice_t fieldDevice = nullptr;
  int fieldCount = 0;
  std::vector<std::pair<unsigned int, unsigned int>> fieldInputs;
  int fabricCalls = 0;
  nvmlDevice_t fabricDevice = nullptr;
  unsigned int fabricInputVersion = 0;
  std::array<unsigned char, NVML_GPU_FABRIC_UUID_LEN> fabricInputUuid = {};
  nvmlReturn_t fabricInputStatus = NVML_SUCCESS;
  unsigned int fabricInputClique = 0;
  nvmlGpuFabricState_t fabricInputState = 0;
  unsigned int fabricInputHealth = 0;
  int platformCalls = 0;
  nvmlDevice_t platformDevice = nullptr;
  unsigned int platformInputVersion = 0;
  std::array<unsigned char, 16> platformInputGuid = {};
  std::array<unsigned char, 16> platformInputSerial = {};
  unsigned char platformInputSlot = 0;
  unsigned char platformInputTray = 0;
  unsigned char platformInputHost = 0;
  unsigned char platformInputPeer = 0;
  unsigned char platformInputModule = 0;
  int modernCcCalls = 0;
  int legacyCcCalls = 0;
  unsigned int modernCcInputVersion = 0;
  unsigned int ccFeature = NVML_CC_SYSTEM_FEATURE_DISABLED;
  unsigned int multiGpuMode = NVML_CC_SYSTEM_MULTIGPU_NONE;
};

NvmlProviderState provider;
std::array<int, ncclNvmlMaxDevices> deviceTokens = {};
int unknownDeviceToken = 0;

/**
 * @brief Reports whether one fake dynamic symbol is unavailable.
 * @param[in] name Valid non-NULL null-terminated symbol name.
 * @return true when name is configured as missing; otherwise false.
 * @note This function has no side effects.
 */
bool SymbolIsMissingForTest(const char* name) {
  return std::find(provider.missingSymbols.begin(), provider.missingSymbols.end(), name) !=
         provider.missingSymbols.end();
}

/**
 * @brief Converts a provider function pointer to its POSIX dynamic-loader representation.
 * @tparam Symbol Provider function-pointer type with POSIX-compatible representation.
 * @param[in] symbol Non-NULL provider function pointer.
 * @return The corresponding non-NULL object-pointer representation.
 * @note This function has no side effects.
 */
template <typename Symbol>
void* SymbolAddressForTest(Symbol symbol) {
  static_assert(sizeof(symbol) == sizeof(void*), "The test requires POSIX-compatible pointer representations.");
  void* address = nullptr;
  std::memcpy(&address, &symbol, sizeof(address));
  return address;
}

/**
 * @brief Returns the stable fake device handle for one supported table index.
 * @param[in] index Device index in [0, ncclNvmlMaxDevices).
 * @return A distinct non-NULL device handle.
 * @note This function has no side effects.
 */
nvmlDevice_t DeviceHandleForTest(unsigned int index) {
  return reinterpret_cast<nvmlDevice_t>(&deviceTokens[index]);
}

/**
 * @brief Finds the table index represented by one fake device handle.
 * @param[in] device Any device handle, including NULL or an unknown handle.
 * @return The index in [0, ncclNvmlMaxDevices), or -1 when device is unknown.
 * @note This function has no side effects.
 */
int DeviceIndexForTest(nvmlDevice_t device) {
  for (int index = 0; index < ncclNvmlMaxDevices; ++index) {
    if (device == DeviceHandleForTest(index)) return index;
  }
  return -1;
}

/**
 * @brief Returns a provider result for one configured failure point.
 * @param[in] point Failure point being executed.
 * @return NVML_ERROR_UNKNOWN when point is selected; otherwise NVML_SUCCESS.
 * @note This function has no side effects.
 */
nvmlReturn_t ProviderResultForTest(NvmlFailure point) {
  return provider.failure == point ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
}

/**
 * @brief Implements deterministic legacy NVML initialization.
 * @return The configured initialization result.
 * @note Side effects: Increments legacyInitCalls.
 */
nvmlReturn_t LegacyInitForTest() {
  ++provider.legacyInitCalls;
  return ProviderResultForTest(NvmlFailure::Init);
}

/**
 * @brief Implements deterministic modern NVML initialization.
 * @return The configured initialization result.
 * @note Side effects: Increments modernInitCalls.
 */
nvmlReturn_t ModernInitForTest() {
  ++provider.modernInitCalls;
  return ProviderResultForTest(NvmlFailure::Init);
}

/**
 * @brief Implements a harmless NVML shutdown symbol for loader completeness.
 * @return NVML_SUCCESS.
 * @note Side effects: Increments shutdownCalls.
 */
nvmlReturn_t ShutdownForTest() {
  ++provider.shutdownCalls;
  return NVML_SUCCESS;
}

/**
 * @brief Returns the configured device count through the legacy API.
 * @param[out] count Valid non-NULL output receiving deviceCount on success.
 * @return The configured count result.
 * @note Side effects: Increments legacyCountCalls and writes count on success.
 */
nvmlReturn_t LegacyDeviceCountForTest(unsigned int* count) {
  ++provider.legacyCountCalls;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::Count);
  if (result == NVML_SUCCESS) *count = provider.deviceCount;
  return result;
}

/**
 * @brief Returns the configured device count through the modern API.
 * @param[out] count Valid non-NULL output receiving deviceCount on success.
 * @return The configured count result.
 * @note Side effects: Increments modernCountCalls and writes count on success.
 */
nvmlReturn_t ModernDeviceCountForTest(unsigned int* count) {
  ++provider.modernCountCalls;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::Count);
  if (result == NVML_SUCCESS) *count = provider.deviceCount;
  return result;
}

/**
 * @brief Returns one deterministic device handle by index.
 * @param[in] index Device index in [0, ncclNvmlMaxDevices).
 * @param[out] device Valid non-NULL output receiving the indexed handle on success.
 * @return The configured handle result.
 * @note Side effects: Increments handleCalls and writes device on success.
 */
nvmlReturn_t HandleByIndexForTest(unsigned int index, nvmlDevice_t* device) {
  ++provider.handleCalls;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::Handle);
  if (result == NVML_SUCCESS) *device = DeviceHandleForTest(index);
  return result;
}

/**
 * @brief Returns one deterministic device handle by PCI bus ID.
 * @param[in] busId Valid non-NULL null-terminated PCI bus ID recorded exactly.
 * @param[out] device Valid non-NULL output receiving device 1 on success.
 * @return The configured PCI lookup result.
 * @note Side effects: Records busId, increments pciHandleCalls, and writes device on success.
 */
nvmlReturn_t HandleByPciBusIdForTest(const char* busId, nvmlDevice_t* device) {
  ++provider.pciHandleCalls;
  provider.pciBusId = busId;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::PciHandle);
  if (result == NVML_SUCCESS) *device = DeviceHandleForTest(1);
  return result;
}

/**
 * @brief Supplies an unused provider-side device-index lookup implementation.
 * @param[in] device Any device handle.
 * @param[out] index Valid non-NULL output receiving the matching fake index on success.
 * @return NVML_SUCCESS for known handles or NVML_ERROR_INVALID_ARGUMENT otherwise.
 * @note Side effects: Increments providerIndexCalls and writes index for known handles.
 */
nvmlReturn_t DeviceIndexProviderForTest(nvmlDevice_t device, unsigned int* index) {
  ++provider.providerIndexCalls;
  const int found = DeviceIndexForTest(device);
  if (found < 0) return NVML_ERROR_INVALID_ARGUMENT;
  *index = static_cast<unsigned int>(found);
  return NVML_SUCCESS;
}

/**
 * @brief Supplies deterministic compute capability for one device.
 * @param[in] device Valid fake device handle.
 * @param[out] major Valid non-NULL output receiving 8 plus the device index.
 * @param[out] minor Valid non-NULL output receiving the device index.
 * @return The configured compute-capability result.
 * @note Side effects: Increments computeCalls and writes both outputs on success.
 */
nvmlReturn_t ComputeCapabilityForTest(nvmlDevice_t device, int* major, int* minor) {
  ++provider.computeCalls;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::ComputeCapability);
  if (result != NVML_SUCCESS) return result;
  const int index = DeviceIndexForTest(device);
  *major = 8 + index;
  *minor = index;
  return NVML_SUCCESS;
}

/**
 * @brief Returns a deterministic asymmetric P2P status.
 * @param[in] first Valid fake source device handle.
 * @param[in] second Valid fake destination device handle.
 * @param[in] capability Any P2P capability index.
 * @param[out] status Valid non-NULL output receiving a status on success.
 * @return The configured READ or WRITE failure, or NVML_SUCCESS.
 * @note Side effects: Appends one call record and writes status on success.
 */
nvmlReturn_t P2pStatusForTest(nvmlDevice_t first, nvmlDevice_t second,
                              nvmlGpuP2PCapsIndex_t capability, nvmlGpuP2PStatus_t* status) {
  provider.p2pCalls.push_back({first, second, capability});
  if (capability == NVML_P2P_CAPS_INDEX_READ && provider.failure == NvmlFailure::P2pRead) {
    return NVML_ERROR_UNKNOWN;
  }
  if (capability == NVML_P2P_CAPS_INDEX_WRITE && provider.failure == NvmlFailure::P2pWrite) {
    return NVML_ERROR_UNKNOWN;
  }
  if (capability != NVML_P2P_CAPS_INDEX_READ && capability != NVML_P2P_CAPS_INDEX_WRITE &&
      provider.failure == NvmlFailure::P2pOther) {
    return NVML_ERROR_UNKNOWN;
  }

  const int firstIndex = DeviceIndexForTest(first);
  const int secondIndex = DeviceIndexForTest(second);
  if (firstIndex < 2 && secondIndex < 2 && capability == NVML_P2P_CAPS_INDEX_READ) {
    const nvmlGpuP2PStatus_t values[2][2] = {
      {NVML_P2P_STATUS_OK, NVML_P2P_STATUS_GPU_NOT_SUPPORTED},
      {NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED, NVML_P2P_STATUS_DISABLED_BY_REGKEY},
    };
    *status = values[firstIndex][secondIndex];
  } else if (firstIndex < 2 && secondIndex < 2 && capability == NVML_P2P_CAPS_INDEX_WRITE) {
    const nvmlGpuP2PStatus_t values[2][2] = {
      {NVML_P2P_STATUS_NOT_SUPPORTED, NVML_P2P_STATUS_UNKNOWN},
      {NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED, NVML_P2P_STATUS_OK},
    };
    *status = values[firstIndex][secondIndex];
  } else {
    const int discriminator = firstIndex * 3 + secondIndex * 2 + static_cast<int>(capability);
    *status = static_cast<nvmlGpuP2PStatus_t>(discriminator % (NVML_P2P_STATUS_UNKNOWN + 1));
  }
  return NVML_SUCCESS;
}

/**
 * @brief Returns a deterministic NVLink active state.
 * @param[in] device Any device handle recorded exactly.
 * @param[in] link Link index in [0, UINT_MAX] recorded exactly.
 * @param[out] active Valid non-NULL output receiving NVML_FEATURE_ENABLED on success.
 * @return The configured link-state result.
 * @note Side effects: Records inputs, increments linkStateCalls, and writes active on success.
 */
nvmlReturn_t LinkStateForTest(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* active) {
  ++provider.linkStateCalls;
  provider.linkStateDevice = device;
  provider.linkStateLink = link;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::LinkState);
  if (result == NVML_SUCCESS) *active = NVML_FEATURE_ENABLED;
  return result;
}

/**
 * @brief Returns deterministic remote PCI information for one NVLink.
 * @param[in] device Any device handle recorded exactly.
 * @param[in] link Link index in [0, UINT_MAX] recorded exactly.
 * @param[out] pci Valid non-NULL output receiving complete PCI identity on success.
 * @return The configured remote-PCI result.
 * @note Side effects: Records inputs, increments remotePciCalls, and writes pci on success.
 */
nvmlReturn_t RemotePciForTest(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) {
  ++provider.remotePciCalls;
  provider.remotePciDevice = device;
  provider.remotePciLink = link;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::RemotePci);
  if (result == NVML_SUCCESS) {
    std::snprintf(pci->busId, sizeof(pci->busId), "0042:17:03.1");
    pci->domain = 0x42;
    pci->bus = 0x17;
    pci->device = 3;
    pci->pciDeviceId = 0x123410de;
    pci->pciSubSystemId = 0x567810de;
  }
  return result;
}

/**
 * @brief Returns a deterministic NVLink capability value.
 * @param[in] device Any device handle recorded exactly.
 * @param[in] link Link index in [0, UINT_MAX] recorded exactly.
 * @param[in] capability Capability selector in [0, NVML_NVLINK_CAP_COUNT).
 * @param[out] resultOutput Valid non-NULL output receiving 17 on success.
 * @return The configured link-capability result.
 * @note Side effects: Records inputs, increments linkCapabilityCalls, and writes resultOutput
 * on success.
 */
nvmlReturn_t LinkCapabilityForTest(nvmlDevice_t device, unsigned int link,
                                   nvmlNvLinkCapability_t capability, unsigned int* resultOutput) {
  ++provider.linkCapabilityCalls;
  provider.linkCapabilityDevice = device;
  provider.linkCapabilityLink = link;
  provider.linkCapability = capability;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::LinkCapability);
  if (result == NVML_SUCCESS) *resultOutput = 17;
  return result;
}

/**
 * @brief Returns deterministic values for caller-selected NVML fields.
 * @param[in] device Any device handle recorded exactly.
 * @param[in] count Number of entries in [0, INT_MAX] recorded exactly.
 * @param[in,out] values Valid non-NULL array whose request IDs are recorded and whose results are
 * written on success.
 * @return The configured field-query result.
 * @note Side effects: Records request fields, increments fieldCalls, and writes result fields on
 * success.
 */
nvmlReturn_t FieldValuesForTest(nvmlDevice_t device, int count, nvmlFieldValue_t* values) {
  ++provider.fieldCalls;
  provider.fieldDevice = device;
  provider.fieldCount = count;
  provider.fieldInputs.clear();
  for (int index = 0; index < count; ++index) {
    provider.fieldInputs.push_back({values[index].fieldId, values[index].scopeId});
  }
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::FieldValues);
  if (result == NVML_SUCCESS) {
    for (int index = 0; index < count; ++index) {
      values[index].timestamp = 1000 + index;
      values[index].latencyUsec = 20 + index;
      values[index].valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
      values[index].nvmlReturn = NVML_SUCCESS;
      values[index].value.uiVal = 700 + index;
    }
  }
  return result;
}

/**
 * @brief Returns deterministic GPU fabric information.
 * @param[in] device Any device handle recorded exactly.
 * @param[in,out] fabric Valid non-NULL versioned structure written on success.
 * @return The configured fabric-query result.
 * @note Side effects: Records input fields, increments fabricCalls, and writes fabric on success.
 */
nvmlReturn_t FabricInfoForTest(nvmlDevice_t device, nvmlGpuFabricInfoV_t* fabric) {
  ++provider.fabricCalls;
  provider.fabricDevice = device;
  provider.fabricInputVersion = fabric->version;
  std::copy(std::begin(fabric->clusterUuid), std::end(fabric->clusterUuid), provider.fabricInputUuid.begin());
  provider.fabricInputStatus = fabric->status;
  provider.fabricInputClique = fabric->cliqueId;
  provider.fabricInputState = fabric->state;
  provider.fabricInputHealth = fabric->healthMask;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::Fabric);
  if (result == NVML_SUCCESS) {
    fabric->clusterUuid[0] = 0xa5;
    fabric->status = NVML_SUCCESS;
    fabric->cliqueId = 41;
    fabric->state = NVML_GPU_FABRIC_STATE_COMPLETED;
    fabric->healthMask = NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_FALSE;
  }
  return result;
}

/**
 * @brief Returns deterministic platform information.
 * @param[in] device Any device handle recorded exactly.
 * @param[in,out] platform Valid non-NULL versioned structure written on success.
 * @return The configured platform-query result.
 * @note Side effects: Records input fields, increments platformCalls, and writes platform on
 * success.
 */
nvmlReturn_t PlatformInfoForTest(nvmlDevice_t device, nvmlPlatformInfo_t* platform) {
  ++provider.platformCalls;
  provider.platformDevice = device;
  provider.platformInputVersion = platform->version;
  std::copy(std::begin(platform->ibGuid), std::end(platform->ibGuid), provider.platformInputGuid.begin());
  std::copy(std::begin(platform->chassisSerialNumber), std::end(platform->chassisSerialNumber),
            provider.platformInputSerial.begin());
  provider.platformInputSlot = platform->slotNumber;
  provider.platformInputTray = platform->trayIndex;
  provider.platformInputHost = platform->hostId;
  provider.platformInputPeer = platform->peerType;
  provider.platformInputModule = platform->moduleId;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::Platform);
  if (result == NVML_SUCCESS) {
    platform->ibGuid[0] = 0xb6;
    platform->chassisSerialNumber[0] = 0xc7;
    platform->slotNumber = 5;
    platform->trayIndex = 6;
    platform->hostId = 7;
    platform->peerType = 8;
    platform->moduleId = 9;
  }
  return result;
}

/**
 * @brief Returns deterministic legacy confidential-compute state.
 * @param[out] state Valid non-NULL output receiving ccFeature on success.
 * @return The configured legacy confidential-compute result.
 * @note Side effects: Increments legacyCcCalls and writes state on success.
 */
nvmlReturn_t LegacyCcStateForTest(nvmlConfComputeSystemState_t* state) {
  ++provider.legacyCcCalls;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::CcLegacy);
  if (result == NVML_SUCCESS) state->ccFeature = provider.ccFeature;
  return result;
}

/**
 * @brief Returns deterministic modern confidential-compute settings.
 * @param[in,out] settings Valid non-NULL versioned output receiving configured settings on success.
 * @return The configured modern confidential-compute result.
 * @note Side effects: Records version, increments modernCcCalls, and writes settings on success.
 */
nvmlReturn_t ModernCcSettingsForTest(nvmlSystemConfComputeSettings_t* settings) {
  ++provider.modernCcCalls;
  provider.modernCcInputVersion = settings->version;
  const nvmlReturn_t result = ProviderResultForTest(NvmlFailure::CcModern);
  if (result == NVML_SUCCESS) {
    settings->ccFeature = provider.ccFeature;
    settings->multiGpuMode = provider.multiGpuMode;
  }
  return result;
}

/**
 * @brief Returns a stable description for fake provider failures.
 * @param[in] result Any NVML status value.
 * @return A valid non-NULL process-lifetime string.
 * @note This function has no side effects.
 */
const char* ErrorStringForTest(nvmlReturn_t result) {
  (void)result;
  return "controlled NVML failure";
}

/**
 * @brief Implements deterministic loading of the NVML soname.
 * @param[in] name Valid non-NULL null-terminated library name recorded exactly.
 * @param[in] flags Dynamic-loader flags recorded exactly.
 * @return A stable fake handle when available; otherwise NULL.
 * @note Side effects: Appends one entry to openCalls.
 */
void* OpenNvmlLibraryForTest(const char* name, int flags) {
  provider.openCalls.push_back({name, flags});
  return provider.libraryAvailable ? &provider.libraryHandle : nullptr;
}

/**
 * @brief Resolves one configured NVML provider symbol by exact name.
 * @param[in] handle Any dynamic-library handle recorded exactly.
 * @param[in] name Valid non-NULL null-terminated symbol name.
 * @return The matching provider address, or NULL for missing or unknown symbols.
 * @note Side effects: Appends one entry to resolveCalls.
 */
void* ResolveNvmlSymbolForTest(void* handle, const char* name) {
  provider.resolveCalls.push_back({handle, name});
  if (SymbolIsMissingForTest(name)) return nullptr;
#define RETURN_NVML_SYMBOL(symbolName, implementation) \
  if (std::strcmp(name, symbolName) == 0) return SymbolAddressForTest(implementation)
  RETURN_NVML_SYMBOL("nvmlInit", LegacyInitForTest);
  RETURN_NVML_SYMBOL("nvmlInit_v2", ModernInitForTest);
  RETURN_NVML_SYMBOL("nvmlShutdown", ShutdownForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetCount", LegacyDeviceCountForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetCount_v2", ModernDeviceCountForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetHandleByPciBusId", HandleByPciBusIdForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetHandleByIndex", HandleByIndexForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetIndex", DeviceIndexProviderForTest);
  RETURN_NVML_SYMBOL("nvmlErrorString", ErrorStringForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetNvLinkState", LinkStateForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetNvLinkRemotePciInfo", RemotePciForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetNvLinkCapability", LinkCapabilityForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetCudaComputeCapability", ComputeCapabilityForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetP2PStatus", P2pStatusForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetFieldValues", FieldValuesForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetGpuFabricInfoV", FabricInfoForTest);
  RETURN_NVML_SYMBOL("nvmlDeviceGetPlatformInfo", PlatformInfoForTest);
  RETURN_NVML_SYMBOL("nvmlSystemGetConfComputeState", LegacyCcStateForTest);
  RETURN_NVML_SYMBOL("nvmlSystemGetConfComputeSettings", ModernCcSettingsForTest);
#undef RETURN_NVML_SYMBOL
  return nullptr;
}

}  // namespace

/**
 * @brief Compiles the production NVML wrapper against deterministic dynamic-loader seams.
 */
#define dlopen OpenNvmlLibraryForTest
#define dlsym ResolveNvmlSymbolForTest
#include "../../../misc/nvmlwrap.cc"  // NOLINT(build/include)
#undef dlsym
#undef dlopen
#undef NVMLTRY
#undef NVMLCHECK
#undef NCCL_NVML_FN

namespace {

/**
 * @brief Installs every fake provider function without exercising the dynamic loader.
 * @note Side effects: Replaces all production NVML function pointers.
 */
void InstallProviderSymbolsForTest() {
  pfn_nvmlInit = LegacyInitForTest;
  pfn_nvmlInit_v2 = ModernInitForTest;
  pfn_nvmlShutdown = ShutdownForTest;
  pfn_nvmlDeviceGetCount = LegacyDeviceCountForTest;
  pfn_nvmlDeviceGetCount_v2 = ModernDeviceCountForTest;
  pfn_nvmlDeviceGetHandleByPciBusId = HandleByPciBusIdForTest;
  pfn_nvmlDeviceGetHandleByIndex = HandleByIndexForTest;
  pfn_nvmlDeviceGetIndex = DeviceIndexProviderForTest;
  pfn_nvmlErrorString = ErrorStringForTest;
  pfn_nvmlDeviceGetNvLinkState = LinkStateForTest;
  pfn_nvmlDeviceGetNvLinkRemotePciInfo = RemotePciForTest;
  pfn_nvmlDeviceGetNvLinkCapability = LinkCapabilityForTest;
  pfn_nvmlDeviceGetCudaComputeCapability = ComputeCapabilityForTest;
  pfn_nvmlDeviceGetP2PStatus = P2pStatusForTest;
  pfn_nvmlDeviceGetFieldValues = FieldValuesForTest;
  pfn_nvmlDeviceGetGpuFabricInfoV = FabricInfoForTest;
  pfn_nvmlDeviceGetPlatformInfo = PlatformInfoForTest;
  pfn_nvmlSystemGetConfComputeState = LegacyCcStateForTest;
  pfn_nvmlSystemGetConfComputeSettings = ModernCcSettingsForTest;
}

/**
 * @brief Clears production and provider state before one independent scenario.
 * @note Side effects: Replaces all fake behavior, caches, tables, and NVML function pointers.
 */
void ResetNvmlForTest() {
  provider = NvmlProviderState{};
  ncclNvmlDeviceCount = 0;
  std::memset(ncclNvmlDevices, 0, sizeof(ncclNvmlDevices));
  std::memset(ncclNvmlDevicePairs, 0, sizeof(ncclNvmlDevicePairs));
  initialized = false;
  threadInitialized = false;
  initResult = ncclSuccess;
  pfn_nvmlInit = nullptr;
  pfn_nvmlInit_v2 = nullptr;
  pfn_nvmlShutdown = nullptr;
  pfn_nvmlDeviceGetCount = nullptr;
  pfn_nvmlDeviceGetCount_v2 = nullptr;
  pfn_nvmlDeviceGetHandleByPciBusId = nullptr;
  pfn_nvmlDeviceGetHandleByIndex = nullptr;
  pfn_nvmlDeviceGetIndex = nullptr;
  pfn_nvmlErrorString = nullptr;
  pfn_nvmlDeviceGetNvLinkState = nullptr;
  pfn_nvmlDeviceGetNvLinkRemotePciInfo = nullptr;
  pfn_nvmlDeviceGetNvLinkCapability = nullptr;
  pfn_nvmlDeviceGetCudaComputeCapability = nullptr;
  pfn_nvmlDeviceGetP2PStatus = nullptr;
  pfn_nvmlDeviceGetFieldValues = nullptr;
  pfn_nvmlDeviceGetGpuFabricInfoV = nullptr;
  pfn_nvmlDeviceGetPlatformInfo = nullptr;
  pfn_nvmlSystemGetConfComputeState = nullptr;
  pfn_nvmlSystemGetConfComputeSettings = nullptr;
}

/**
 * @brief Prepares an initialized cache without repeating initialization behavior in wrapper
 * tests.
 * @param[in] count Active device count in [0, ncclNvmlMaxDevices].
 * @note Side effects: Installs provider symbols and replaces production cache state and active
 * table entries.
 */
void PrepareReadyCacheForTest(int count = 2) {
  InstallProviderSymbolsForTest();
  initialized = true;
  threadInitialized = true;
  initResult = ncclSuccess;
  ncclNvmlDeviceCount = count;
  for (int index = 0; index < count; ++index) {
    ncclNvmlDevices[index].handle = DeviceHandleForTest(index);
    ncclNvmlDevices[index].computeCapabilityMajor = 8 + index;
    ncclNvmlDevices[index].computeCapabilityMinor = index;
  }
  if (count >= 2) {
    ncclNvmlDevicePairs[0][1].p2pStatusRead = NVML_P2P_STATUS_GPU_NOT_SUPPORTED;
    ncclNvmlDevicePairs[0][1].p2pStatusWrite = NVML_P2P_STATUS_UNKNOWN;
    ncclNvmlDevicePairs[1][0].p2pStatusRead = NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED;
    ncclNvmlDevicePairs[1][0].p2pStatusWrite = NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED;
  }
}

/**
 * @brief Checks that dynamic loading installed every symbol in its matching production
 * function-pointer slot.
 * @note Side effects: Records GoogleTest failures for missing or misassigned addresses.
 */
void ExpectInstalledSymbolsForTest() {
#define EXPECT_NVML_SYMBOL(field, implementation) \
  EXPECT_EQ(SymbolAddressForTest(field), SymbolAddressForTest(implementation)) \
    << #field " must contain the address resolved for its matching NVML symbol."
  EXPECT_NVML_SYMBOL(pfn_nvmlInit, LegacyInitForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlInit_v2, ModernInitForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlShutdown, ShutdownForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetCount, LegacyDeviceCountForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetCount_v2, ModernDeviceCountForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetHandleByPciBusId, HandleByPciBusIdForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetHandleByIndex, HandleByIndexForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetIndex, DeviceIndexProviderForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlErrorString, ErrorStringForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetNvLinkState, LinkStateForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetNvLinkRemotePciInfo, RemotePciForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetNvLinkCapability, LinkCapabilityForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetCudaComputeCapability, ComputeCapabilityForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetP2PStatus, P2pStatusForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetFieldValues, FieldValuesForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetGpuFabricInfoV, FabricInfoForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlDeviceGetPlatformInfo, PlatformInfoForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlSystemGetConfComputeState, LegacyCcStateForTest);
  EXPECT_NVML_SYMBOL(pfn_nvmlSystemGetConfComputeSettings, ModernCcSettingsForTest);
#undef EXPECT_NVML_SYMBOL
}

/**
 * @brief Counts fake provider invocations excluding dynamic-loader activity.
 * @return Total provider call count in [0, INT_MAX].
 * @note This function has no side effects.
 */
int ProviderCallCountForTest() {
  return provider.legacyInitCalls + provider.modernInitCalls + provider.legacyCountCalls +
         provider.modernCountCalls + provider.shutdownCalls + provider.handleCalls + provider.computeCalls +
         provider.providerIndexCalls + static_cast<int>(provider.p2pCalls.size()) + provider.pciHandleCalls +
         provider.linkStateCalls + provider.remotePciCalls + provider.linkCapabilityCalls + provider.fieldCalls +
         provider.fabricCalls + provider.platformCalls + provider.modernCcCalls + provider.legacyCcCalls;
}

/**
 * @brief Verifies one reachable initialization failure and its short-circuit and caching behavior.
 * @param[in] point Provider failure point from Init through P2pWrite.
 * @param[in] description Valid non-NULL scenario description used in failure diagnostics.
 * @note Side effects: Resets and initializes production NVML state and records GoogleTest failures.
 */
void ExpectInitializationFailureForTest(NvmlFailure point, const char* description) {
  SCOPED_TRACE(description);
  ResetNvmlForTest();
  provider.failure = point;

  EXPECT_EQ(ncclNvmlEnsureInitialized(), ncclSystemError)
    << "A failing NVML initialization dependency must produce ncclSystemError.";
  const int callsAfterFailure = ProviderCallCountForTest();
  provider.failure = NvmlFailure::None;
  EXPECT_EQ(ncclNvmlEnsureInitialized(), ncclSystemError)
    << "The first initialization failure must remain cached for the calling thread.";
  EXPECT_EQ(ProviderCallCountForTest(), callsAfterFailure)
    << "A cached initialization failure must not retry a provider operation.";

  if (point == NvmlFailure::Init) {
    EXPECT_EQ(provider.modernCountCalls, 0) << "Device discovery must not follow a failed initialization.";
  } else if (point == NvmlFailure::Count) {
    EXPECT_EQ(provider.handleCalls, 0) << "Device lookup must not follow a failed count query.";
  } else if (point == NvmlFailure::Handle) {
    EXPECT_EQ(provider.computeCalls, 0) << "Capability lookup must not follow a failed handle query.";
  } else if (point == NvmlFailure::ComputeCapability) {
    EXPECT_TRUE(provider.p2pCalls.empty()) << "P2P discovery must not follow a failed capability query.";
  } else if (point == NvmlFailure::P2pRead) {
    ASSERT_EQ(provider.p2pCalls.size(), 1U) << "The first failed READ query must stop P2P discovery.";
    EXPECT_EQ(provider.p2pCalls[0].capability, NVML_P2P_CAPS_INDEX_READ)
      << "The failing first P2P query must be READ.";
  } else if (point == NvmlFailure::P2pWrite) {
    ASSERT_EQ(provider.p2pCalls.size(), 2U) << "The failed WRITE query must follow exactly one successful READ.";
    EXPECT_EQ(provider.p2pCalls[1].capability, NVML_P2P_CAPS_INDEX_WRITE)
      << "The second P2P query must be WRITE.";
  }
}

nvmlPciInfo_t MakePciSentinelForTest();
void ExpectPciSentinelForTest(const nvmlPciInfo_t& pci);
nvmlFieldValue_t MakeFieldSentinelForTest();
void ExpectFieldSentinelForTest(const nvmlFieldValue_t& field);
nvmlGpuFabricInfoV_t MakeFabricSentinelForTest();
void ExpectFabricSentinelForTest(const nvmlGpuFabricInfoV_t& fabric, unsigned int expectedVersion);
nvmlPlatformInfo_t MakePlatformSentinelForTest();
void ExpectPlatformSentinelForTest(const nvmlPlatformInfo_t& platform, unsigned int expectedVersion);

/**
 * @brief Verifies that every public wrapper propagates one common initialization failure without
 * changing outputs.
 * @param[in] expected Expected ncclSystemError or ncclInternalError result.
 * @param[in] libraryAvailable false for loader failure or true for excessive device count.
 * @note Side effects: Resets production state, invokes all public wrappers, and records GoogleTest
 * failures.
 */
void ExpectInitializationGateForTest(ncclResult_t expected, bool libraryAvailable) {
  ResetNvmlForTest();
  provider.libraryAvailable = libraryAvailable;
  provider.deviceCount = libraryAvailable ? ncclNvmlMaxDevices + 1 : 2;

  nvmlDevice_t deviceOutput = DeviceHandleForTest(7);
  unsigned int indexOutput = 77;
  nvmlEnableState_t activeOutput = NVML_FEATURE_ENABLED;
  nvmlPciInfo_t pciOutput = MakePciSentinelForTest();
  unsigned int capabilityOutput = 77;
  int majorOutput = 77;
  int minorOutput = 78;
  nvmlGpuP2PStatus_t p2pOutput = NVML_P2P_STATUS_UNKNOWN;
  nvmlFieldValue_t fieldOutput = MakeFieldSentinelForTest();
  nvmlGpuFabricInfoV_t fabricOutput = MakeFabricSentinelForTest();
  nvmlPlatformInfo_t platformOutput = MakePlatformSentinelForTest();
  ncclNvmlCCStatus ccOutput = {true, true, true};

  EXPECT_EQ(ncclNvmlDeviceGetHandleByPciBusId("0042:17:03.1", &deviceOutput), expected)
    << "PCI lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetHandleByIndex(0, &deviceOutput), expected)
    << "Indexed lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetIndex(DeviceHandleForTest(0), &indexOutput), expected)
    << "Reverse lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkState(DeviceHandleForTest(0), 1, &activeOutput), expected)
    << "Link-state lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkRemotePciInfo(DeviceHandleForTest(0), 1, &pciOutput), expected)
    << "Remote-PCI lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkCapability(
              DeviceHandleForTest(0), 1, NVML_NVLINK_CAP_VALID, &capabilityOutput), expected)
    << "Link-capability lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetCudaComputeCapability(DeviceHandleForTest(0), &majorOutput, &minorOutput), expected)
    << "Compute-capability lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              DeviceHandleForTest(0), DeviceHandleForTest(1), NVML_P2P_CAPS_INDEX_READ, &p2pOutput), expected)
    << "P2P lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetFieldValues(DeviceHandleForTest(0), 1, &fieldOutput), expected)
    << "Field lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetGpuFabricInfoV(DeviceHandleForTest(0), &fabricOutput), expected)
    << "Fabric lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlDeviceGetPlatformInfo(DeviceHandleForTest(0), &platformOutput), expected)
    << "Platform lookup must propagate the common initialization failure.";
  EXPECT_EQ(ncclNvmlGetCCStatus(&ccOutput), expected)
    << "Confidential-compute lookup must propagate the common initialization failure.";

  EXPECT_EQ(deviceOutput, DeviceHandleForTest(7)) << "Failed initialization must preserve device output.";
  EXPECT_EQ(indexOutput, 77U) << "Failed initialization must preserve index output.";
  EXPECT_EQ(activeOutput, NVML_FEATURE_ENABLED) << "Failed initialization must preserve link-state output.";
  ExpectPciSentinelForTest(pciOutput);
  EXPECT_EQ(capabilityOutput, 77U) << "Failed initialization must preserve capability output.";
  EXPECT_EQ(majorOutput, 77) << "Failed initialization must preserve compute major output.";
  EXPECT_EQ(minorOutput, 78) << "Failed initialization must preserve compute minor output.";
  EXPECT_EQ(p2pOutput, NVML_P2P_STATUS_UNKNOWN) << "Failed initialization must preserve P2P output.";
  ExpectFieldSentinelForTest(fieldOutput);
  ExpectFabricSentinelForTest(fabricOutput, 77);
  ExpectPlatformSentinelForTest(platformOutput, 77);
  EXPECT_TRUE(ccOutput.CCEnabled) << "Failed initialization must preserve confidential-compute status.";
  EXPECT_TRUE(ccOutput.multiGpuProtectedPCIE) << "Failed initialization must preserve protected-PCIe status.";
  EXPECT_TRUE(ccOutput.multiGpuNVLE) << "Failed initialization must preserve NVLE status.";
  EXPECT_EQ(provider.pciHandleCalls + provider.linkStateCalls + provider.remotePciCalls +
              provider.linkCapabilityCalls + provider.fieldCalls + provider.fabricCalls +
              provider.platformCalls + provider.modernCcCalls + provider.legacyCcCalls, 0)
    << "No wrapper-specific provider must run after common initialization fails.";
}

/**
 * @brief Checks exact semantic fields of one PCI output sentinel.
 * @param[in] pci PCI structure expected to retain its sentinel fields.
 * @note Side effects: Records GoogleTest failures for changed fields.
 */
void ExpectPciSentinelForTest(const nvmlPciInfo_t& pci) {
  EXPECT_STREQ(pci.busId, "sentinel") << "An unavailable provider must preserve the PCI bus ID.";
  EXPECT_EQ(pci.domain, 91U) << "An unavailable provider must preserve the PCI domain.";
  EXPECT_EQ(pci.bus, 92U) << "An unavailable provider must preserve the PCI bus.";
  EXPECT_EQ(pci.device, 93U) << "An unavailable provider must preserve the PCI device.";
  EXPECT_EQ(pci.pciDeviceId, 94U) << "An unavailable provider must preserve the PCI device ID.";
  EXPECT_EQ(pci.pciSubSystemId, 95U) << "An unavailable provider must preserve the PCI subsystem ID.";
  EXPECT_EQ(pci.reserved0, 96U) << "An unavailable provider must preserve PCI reserved field 0.";
  EXPECT_EQ(pci.reserved1, 97U) << "An unavailable provider must preserve PCI reserved field 1.";
  EXPECT_EQ(pci.reserved2, 98U) << "An unavailable provider must preserve PCI reserved field 2.";
  EXPECT_EQ(pci.reserved3, 99U) << "An unavailable provider must preserve PCI reserved field 3.";
}

/**
 * @brief Creates a fully initialized remote-PCI output sentinel.
 * @return A value whose semantic fields differ from provider success output.
 * @note This function has no side effects.
 */
nvmlPciInfo_t MakePciSentinelForTest() {
  nvmlPciInfo_t pci = {};
  std::snprintf(pci.busId, sizeof(pci.busId), "sentinel");
  pci.domain = 91;
  pci.bus = 92;
  pci.device = 93;
  pci.pciDeviceId = 94;
  pci.pciSubSystemId = 95;
  pci.reserved0 = 96;
  pci.reserved1 = 97;
  pci.reserved2 = 98;
  pci.reserved3 = 99;
  return pci;
}

/**
 * @brief Creates a fully initialized field-value sentinel.
 * @return A value whose semantic request and result fields are all non-default.
 * @note This function has no side effects.
 */
nvmlFieldValue_t MakeFieldSentinelForTest() {
  nvmlFieldValue_t field = {};
  field.fieldId = 77;
  field.scopeId = 78;
  field.timestamp = 79;
  field.latencyUsec = 80;
  field.valueType = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
  field.nvmlReturn = NVML_ERROR_NOT_FOUND;
  field.value.ullVal = 81;
  return field;
}

/**
 * @brief Checks every semantic field of one field-value sentinel.
 * @param[in] field Field value expected to remain unchanged.
 * @note Side effects: Records GoogleTest failures for changed fields.
 */
void ExpectFieldSentinelForTest(const nvmlFieldValue_t& field) {
  EXPECT_EQ(field.fieldId, 77U) << "An unavailable provider must preserve the field ID.";
  EXPECT_EQ(field.scopeId, 78U) << "An unavailable provider must preserve the field scope.";
  EXPECT_EQ(field.timestamp, 79) << "An unavailable provider must preserve the field timestamp.";
  EXPECT_EQ(field.latencyUsec, 80) << "An unavailable provider must preserve the field latency.";
  EXPECT_EQ(field.valueType, NVML_VALUE_TYPE_UNSIGNED_LONG_LONG)
    << "An unavailable provider must preserve the field value type.";
  EXPECT_EQ(field.nvmlReturn, NVML_ERROR_NOT_FOUND)
    << "An unavailable provider must preserve the field result status.";
  EXPECT_EQ(field.value.ullVal, 81U) << "An unavailable provider must preserve the field value.";
}

/**
 * @brief Creates a fully initialized GPU-fabric sentinel.
 * @return A value whose semantic fields and UUID bytes are all non-default.
 * @note This function has no side effects.
 */
nvmlGpuFabricInfoV_t MakeFabricSentinelForTest() {
  nvmlGpuFabricInfoV_t fabric = {};
  fabric.version = 77;
  for (size_t index = 0; index < sizeof(fabric.clusterUuid); ++index) {
    fabric.clusterUuid[index] = static_cast<unsigned char>(0xd0 + index);
  }
  fabric.status = NVML_ERROR_NOT_FOUND;
  fabric.cliqueId = 78;
  fabric.state = NVML_GPU_FABRIC_STATE_IN_PROGRESS;
  fabric.healthMask = 79;
  return fabric;
}

/**
 * @brief Checks every semantic field of a GPU-fabric sentinel except its replaceable version.
 * @param[in] fabric Fabric value expected to retain caller data.
 * @param[in] expectedVersion Exact version expected after the wrapper path.
 * @note Side effects: Records GoogleTest failures for changed fields.
 */
void ExpectFabricSentinelForTest(const nvmlGpuFabricInfoV_t& fabric, unsigned int expectedVersion) {
  EXPECT_EQ(fabric.version, expectedVersion) << "Fabric version must match the selected wrapper path.";
  for (size_t index = 0; index < sizeof(fabric.clusterUuid); ++index) {
    SCOPED_TRACE("fabric UUID byte: " + std::to_string(index));
    EXPECT_EQ(fabric.clusterUuid[index], static_cast<unsigned char>(0xd0 + index))
      << "An unavailable provider must preserve every fabric UUID byte.";
  }
  EXPECT_EQ(fabric.status, NVML_ERROR_NOT_FOUND) << "An unavailable provider must preserve fabric status.";
  EXPECT_EQ(fabric.cliqueId, 78U) << "An unavailable provider must preserve the fabric clique ID.";
  EXPECT_EQ(fabric.state, NVML_GPU_FABRIC_STATE_IN_PROGRESS)
    << "An unavailable provider must preserve fabric state.";
  EXPECT_EQ(fabric.healthMask, 79U) << "An unavailable provider must preserve the fabric health mask.";
}

/**
 * @brief Creates a fully initialized platform-information sentinel.
 * @return A value whose semantic scalar and identity-array fields are all non-default.
 * @note This function has no side effects.
 */
nvmlPlatformInfo_t MakePlatformSentinelForTest() {
  nvmlPlatformInfo_t platform = {};
  platform.version = 77;
  for (size_t index = 0; index < sizeof(platform.ibGuid); ++index) {
    platform.ibGuid[index] = static_cast<unsigned char>(0xa0 + index);
    platform.chassisSerialNumber[index] = static_cast<unsigned char>(0xb0 + index);
  }
  platform.slotNumber = 78;
  platform.trayIndex = 79;
  platform.hostId = 80;
  platform.peerType = 81;
  platform.moduleId = 82;
  return platform;
}

/**
 * @brief Checks every semantic field of a platform sentinel except its replaceable version.
 * @param[in] platform Platform value expected to retain caller data.
 * @param[in] expectedVersion Exact version expected after the wrapper path.
 * @note Side effects: Records GoogleTest failures for changed fields.
 */
void ExpectPlatformSentinelForTest(const nvmlPlatformInfo_t& platform, unsigned int expectedVersion) {
  EXPECT_EQ(platform.version, expectedVersion) << "Platform version must match the selected wrapper path.";
  for (size_t index = 0; index < sizeof(platform.ibGuid); ++index) {
    SCOPED_TRACE("platform identity byte: " + std::to_string(index));
    EXPECT_EQ(platform.ibGuid[index], static_cast<unsigned char>(0xa0 + index))
      << "An unavailable provider must preserve every platform IB GUID byte.";
    EXPECT_EQ(platform.chassisSerialNumber[index], static_cast<unsigned char>(0xb0 + index))
      << "An unavailable provider must preserve every chassis serial byte.";
  }
  EXPECT_EQ(platform.slotNumber, 78U) << "An unavailable provider must preserve the platform slot.";
  EXPECT_EQ(platform.trayIndex, 79U) << "An unavailable provider must preserve the platform tray.";
  EXPECT_EQ(platform.hostId, 80U) << "An unavailable provider must preserve the platform host ID.";
  EXPECT_EQ(platform.peerType, 81U) << "An unavailable provider must preserve the platform peer type.";
  EXPECT_EQ(platform.moduleId, 82U) << "An unavailable provider must preserve the platform module ID.";
}

/** @brief Provides isolated fake-provider and cache state for NVML wrapper tests. */
class NvmlTestBase : public ::testing::Test {
 protected:
  /**
   * @brief Restores independent fake-provider and production NVML state before each test.
   * @note Side effects: Replaces all NVML test and production cache state.
   */
  void SetUp() override { ResetNvmlForTest(); }
};

/** @brief Groups NVML loader, inventory, and cache scenarios. */
class NvmlInitializationTest : public NvmlTestBase {};
/** @brief Groups cached device-lookup scenarios. */
class NvmlLookupTest : public NvmlTestBase {};
/** @brief Groups NVML topology and optional-query scenarios. */
class NvmlQueryTest : public NvmlTestBase {};
/** @brief Groups confidential-compute status scenarios. */
class NvmlCcTest : public NvmlTestBase {};

/**
 * @brief Verifies modern initialization, asymmetric cache contents, and process-wide result reuse.
 * @note Side effects: Calls the fake loader and provider and initializes production NVML cache
 * state.
 */
TEST_F(NvmlInitializationTest, LoadsAndCachesInventory) {
  SCOPED_TRACE("Modern initialization must load the complete interface and cache exact provider inventory data.");

  // A two-device inventory makes direction and READ versus WRITE observable.
  ASSERT_EQ(ncclNvmlEnsureInitialized(), ncclSuccess)
    << "A complete modern NVML provider must initialize successfully.";

  // The loader contract concerns the soname, mode, handle, and complete set, not lookup order.
  ASSERT_EQ(provider.openCalls.size(), 1U) << "Successful initialization must open NVML exactly once.";
  EXPECT_EQ(provider.openCalls[0].first, "libnvidia-ml.so.1") << "The NVML soname must be exact.";
  EXPECT_EQ(provider.openCalls[0].second, RTLD_NOW) << "NVML dependencies must be resolved immediately.";
  ASSERT_EQ(provider.resolveCalls.size(), kNvmlSymbolNames.size())
    << "Initialization must request the complete configured NVML interface.";
  for (const char* name : kNvmlSymbolNames) {
    SCOPED_TRACE(std::string("symbol: ") + name);
    const size_t matches = std::count_if(provider.resolveCalls.begin(), provider.resolveCalls.end(),
      [name](const auto& call) { return call.second == name; });
    EXPECT_EQ(matches, 1U) << "Every NVML symbol must be requested exactly once.";
    for (const auto& call : provider.resolveCalls) {
      if (call.second == name) {
        EXPECT_EQ(call.first, &provider.libraryHandle) << "Every symbol lookup must use the opened library handle.";
      }
    }
  }
  ExpectInstalledSymbolsForTest();

  // Presence of the modern entry point selects the matching modern count API.
  EXPECT_EQ(provider.modernInitCalls, 1) << "The modern initialization entry point must be preferred.";
  EXPECT_EQ(provider.legacyInitCalls, 0) << "The legacy initialization entry point must not run when v2 exists.";
  EXPECT_EQ(provider.modernCountCalls, 1) << "Modern initialization must use the modern count API.";
  EXPECT_EQ(provider.legacyCountCalls, 0) << "Modern initialization must not mix in the legacy count API.";

  // Every active device and ordered pair must preserve its provider identity and capabilities.
  ASSERT_EQ(ncclNvmlDeviceCount, 2) << "The global device count must equal the provider inventory.";
  for (int index = 0; index < 2; ++index) {
    SCOPED_TRACE("device index: " + std::to_string(index));
    EXPECT_EQ(ncclNvmlDevices[index].handle, DeviceHandleForTest(index))
      << "The indexed provider handle must be cached exactly.";
    EXPECT_EQ(ncclNvmlDevices[index].computeCapabilityMajor, 8 + index)
      << "The provider compute major must be cached exactly.";
    EXPECT_EQ(ncclNvmlDevices[index].computeCapabilityMinor, index)
      << "The provider compute minor must be cached exactly.";
  }
  const nvmlGpuP2PStatus_t expectedRead[2][2] = {
    {NVML_P2P_STATUS_OK, NVML_P2P_STATUS_GPU_NOT_SUPPORTED},
    {NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED, NVML_P2P_STATUS_DISABLED_BY_REGKEY},
  };
  const nvmlGpuP2PStatus_t expectedWrite[2][2] = {
    {NVML_P2P_STATUS_NOT_SUPPORTED, NVML_P2P_STATUS_UNKNOWN},
    {NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED, NVML_P2P_STATUS_OK},
  };
  ASSERT_EQ(provider.p2pCalls.size(), 8U)
    << "Two devices require READ and WRITE queries for all four ordered pairs.";
  for (int first = 0; first < 2; ++first) {
    for (int second = 0; second < 2; ++second) {
      SCOPED_TRACE("P2P pair: " + std::to_string(first) + " -> " + std::to_string(second));
      EXPECT_EQ(ncclNvmlDevicePairs[first][second].p2pStatusRead, expectedRead[first][second])
        << "Every cached READ status must equal its directional provider result.";
      EXPECT_EQ(ncclNvmlDevicePairs[first][second].p2pStatusWrite, expectedWrite[first][second])
        << "Every cached WRITE status must equal its directional provider result.";
      for (const nvmlGpuP2PCapsIndex_t capability :
           {NVML_P2P_CAPS_INDEX_READ, NVML_P2P_CAPS_INDEX_WRITE}) {
        const size_t matches = std::count_if(provider.p2pCalls.begin(), provider.p2pCalls.end(),
          [first, second, capability](const P2pCall& call) {
            return call.first == DeviceHandleForTest(first) &&
                   call.second == DeviceHandleForTest(second) && call.capability == capability;
          });
        EXPECT_EQ(matches, 1U)
          << "Initialization must query each capability once for every exact ordered pair.";
      }
    }
  }

  // Both the thread-local fast path and a new thread must reuse the initialized global state.
  const int callsBeforeReuse = ProviderCallCountForTest();
  EXPECT_EQ(ncclNvmlEnsureInitialized(), ncclSuccess)
    << "Repeated initialization in one thread must return the cached success.";
  ncclResult_t otherThreadResult = ncclInternalError;
  /**
   * @brief Reads the process-wide successful initialization result from another thread.
   * @note Side effects: Updates otherThreadResult and accesses the production NVML cache.
   */
  std::thread otherThread([&] { otherThreadResult = ncclNvmlEnsureInitialized(); });
  otherThread.join();
  EXPECT_EQ(otherThreadResult, ncclSuccess) << "A new thread must observe the process-wide successful result.";
  EXPECT_EQ(ProviderCallCountForTest(), callsBeforeReuse)
    << "Cached initialization in either thread must not call the provider again.";
  EXPECT_EQ(provider.openCalls.size(), 1U) << "Cached initialization must not reopen the NVML library.";
  EXPECT_EQ(provider.shutdownCalls, 0) << "The process-lifetime NVML provider must not be shut down.";
}

/**
 * @brief Verifies legacy fallback and zero, exact-maximum, and excessive device-count boundaries.
 * @note Side effects: Repeatedly resets and initializes fake and production NVML state.
 */
TEST_F(NvmlInitializationTest, HandlesLegacyAndDeviceLimits) {
  // Missing v2 entry points must select both legacy APIs without rejecting an otherwise complete
  // provider.
  provider.missingSymbols = {"nvmlInit_v2", "nvmlDeviceGetCount_v2"};
  ASSERT_EQ(ncclNvmlEnsureInitialized(), ncclSuccess)
    << "A complete legacy NVML provider must initialize successfully.";
  EXPECT_EQ(provider.legacyInitCalls, 1) << "Legacy fallback must call nvmlInit exactly once.";
  EXPECT_EQ(provider.modernInitCalls, 0) << "Legacy fallback must not call the missing modern initializer.";
  EXPECT_EQ(provider.legacyCountCalls, 1) << "Legacy fallback must use the legacy count API.";
  EXPECT_EQ(provider.modernCountCalls, 0) << "Legacy fallback must not call the missing modern count API.";

  // An empty inventory is a valid initialized state and must skip per-device discovery.
  ResetNvmlForTest();
  provider.deviceCount = 0;
  EXPECT_EQ(ncclNvmlEnsureInitialized(), ncclSuccess) << "Zero NVML devices must be a successful inventory.";
  EXPECT_EQ(ncclNvmlDeviceCount, 0) << "The empty device count must be cached exactly.";
  EXPECT_EQ(provider.handleCalls, 0) << "An empty inventory must not request any device handles.";
  EXPECT_TRUE(provider.p2pCalls.empty()) << "An empty inventory must not request any P2P status.";

  // The declared maximum must remain a supported value and initialize its last table entry.
  ResetNvmlForTest();
  provider.deviceCount = ncclNvmlMaxDevices;
  ASSERT_EQ(ncclNvmlEnsureInitialized(), ncclSuccess)
    << "The exact internal device maximum must initialize successfully.";
  EXPECT_EQ(ncclNvmlDeviceCount, ncclNvmlMaxDevices) << "The maximum device count must be cached exactly.";
  EXPECT_EQ(provider.handleCalls, ncclNvmlMaxDevices) << "Every maximum-boundary device must be discovered.";
  EXPECT_EQ(ncclNvmlDevices[ncclNvmlMaxDevices - 1].handle,
            DeviceHandleForTest(ncclNvmlMaxDevices - 1))
    << "The final supported handle must be initialized.";
  EXPECT_EQ(ncclNvmlDevices[ncclNvmlMaxDevices - 1].computeCapabilityMajor,
            8 + ncclNvmlMaxDevices - 1)
    << "The final supported compute capability must be initialized.";
  EXPECT_EQ(provider.p2pCalls.size(), 2U * ncclNvmlMaxDevices * ncclNvmlMaxDevices)
    << "The maximum inventory must query READ and WRITE for every ordered pair.";

  // One device beyond the table limit must fail before any indexed table access.
  ResetNvmlForTest();
  provider.deviceCount = ncclNvmlMaxDevices + 1;
  EXPECT_EQ(ncclNvmlEnsureInitialized(), ncclInternalError)
    << "A device count exceeding the internal table must return ncclInternalError.";
  EXPECT_EQ(provider.handleCalls, 0) << "An excessive count must be rejected before device-table writes.";
  EXPECT_TRUE(provider.p2pCalls.empty()) << "An excessive count must be rejected before pair-table writes.";
}

/**
 * @brief Verifies selected staged failures, common error propagation, output preservation,
 * and no retry.
 * @note Side effects: Repeatedly resets fake and production NVML state and invokes initialization.
 */
TEST_F(NvmlInitializationTest, ReturnsInitializationErrors) {
  // A missing library is the only defined dynamic-loader failure.
  provider.libraryAvailable = false;
  EXPECT_EQ(ncclNvmlEnsureInitialized(), ncclSystemError)
    << "An unavailable NVML library must return ncclSystemError.";
  EXPECT_EQ(provider.openCalls.size(), 1U) << "The unavailable NVML soname must be attempted exactly once.";
  EXPECT_TRUE(provider.resolveCalls.empty()) << "Symbol lookup must not run without a library handle.";
  ncclResult_t otherThreadResult = ncclSuccess;
  /**
   * @brief Reads the process-wide failed initialization result from another thread.
   * @note Side effects: Updates otherThreadResult and accesses the production NVML cache.
   */
  std::thread otherThread([&] { otherThreadResult = ncclNvmlEnsureInitialized(); });
  otherThread.join();
  EXPECT_EQ(otherThreadResult, ncclSystemError)
    << "A new thread must observe the process-wide initialization failure.";
  EXPECT_EQ(provider.openCalls.size(), 1U)
    << "A new thread must not retry a process-wide failed library initialization.";

  // Every provider stage must translate failure and stop before later work.
  for (const auto& scenario : std::array<std::pair<NvmlFailure, const char*>, 6>{{
         {NvmlFailure::Init, "initialization failure"},
         {NvmlFailure::Count, "device-count failure"},
         {NvmlFailure::Handle, "device-handle failure"},
         {NvmlFailure::ComputeCapability, "compute-capability failure"},
         {NvmlFailure::P2pRead, "P2P READ failure"},
         {NvmlFailure::P2pWrite, "P2P WRITE failure"},
       }}) {
    ExpectInitializationFailureForTest(scenario.first, scenario.second);
  }

  // Every public wrapper must honor both common initialization errors before touching its own
  // output.
  ExpectInitializationGateForTest(ncclSystemError, false);
  ExpectInitializationGateForTest(ncclInternalError, true);
}

/**
 * @brief Verifies successful and invalid cached device-handle, index, and compute-capability
 * lookups.
 * @note Side effects: Prepares fake and production NVML cache state and writes caller outputs on
 * success.
 */
TEST_F(NvmlLookupTest, ReadsCachedDeviceData) {
  PrepareReadyCacheForTest();

  // Both supported index boundaries must return exact cached handles without consulting the
  // provider.
  nvmlDevice_t device = nullptr;
  EXPECT_EQ(ncclNvmlDeviceGetHandleByIndex(0, &device), ncclSuccess)
    << "The first valid index must return ncclSuccess.";
  EXPECT_EQ(device, DeviceHandleForTest(0)) << "The first valid index must return the first cached handle.";
  EXPECT_EQ(ncclNvmlDeviceGetHandleByIndex(1, &device), ncclSuccess)
    << "The last valid index must return ncclSuccess.";
  EXPECT_EQ(device, DeviceHandleForTest(1)) << "The last valid index must return the last cached handle.";
  EXPECT_EQ(provider.providerIndexCalls, 0) << "Cached handle lookup must not call provider index lookup.";

  // Reverse lookup must find either cached handle and reject an unknown handle without changing
  // output.
  unsigned int index = 77;
  EXPECT_EQ(ncclNvmlDeviceGetIndex(DeviceHandleForTest(1), &index), ncclSuccess)
    << "A cached device handle must be found successfully.";
  EXPECT_EQ(index, 1U) << "Reverse lookup must return the matching cache index.";
  index = 77;
  EXPECT_EQ(ncclNvmlDeviceGetIndex(reinterpret_cast<nvmlDevice_t>(&unknownDeviceToken), &index),
            ncclInvalidArgument)
    << "An unknown device handle must return ncclInvalidArgument.";
  EXPECT_EQ(index, 77U) << "A failed reverse lookup must preserve the caller's index output.";

  // Cached compute capability follows the same success and invalid-handle contract.
  int major = -1;
  int minor = -2;
  EXPECT_EQ(ncclNvmlDeviceGetCudaComputeCapability(DeviceHandleForTest(0), &major, &minor), ncclSuccess)
    << "A cached device must return compute capability successfully.";
  EXPECT_EQ(major, 8) << "Cached compute major must match initialization data.";
  EXPECT_EQ(minor, 0) << "Cached compute minor must match initialization data.";
  major = 77;
  minor = 78;
  EXPECT_EQ(ncclNvmlDeviceGetCudaComputeCapability(
              reinterpret_cast<nvmlDevice_t>(&unknownDeviceToken), &major, &minor), ncclInvalidArgument)
    << "An unknown compute-capability handle must return ncclInvalidArgument.";
  EXPECT_EQ(major, 77) << "Failed compute lookup must preserve the major output.";
  EXPECT_EQ(minor, 78) << "Failed compute lookup must preserve the minor output.";
  EXPECT_EQ(provider.computeCalls, 0) << "Public compute lookup must read the cache without calling the provider.";
}

/**
 * @brief Verifies PCI forwarding, directional cached P2P results, invalid handles, and delegated
 * P2P errors.
 * @note Side effects: Prepares cache state, calls fake PCI and P2P providers, and writes caller
 * outputs.
 */
TEST_F(NvmlLookupTest, QueriesPciAndP2p) {
  PrepareReadyCacheForTest();

  // PCI lookup must forward immutable text and return the exact provider handle or provider error.
  std::array<char, 16> busId = {};
  std::snprintf(busId.data(), busId.size(), "0042:17:03.1");
  const std::array<char, 16> busIdBefore = busId;
  nvmlDevice_t device = nullptr;
  EXPECT_EQ(ncclNvmlDeviceGetHandleByPciBusId(busId.data(), &device), ncclSuccess)
    << "A successful PCI provider lookup must return ncclSuccess.";
  EXPECT_EQ(device, DeviceHandleForTest(1)) << "PCI lookup must return the provider's exact handle.";
  EXPECT_EQ(provider.pciBusId, busId.data()) << "PCI lookup must forward the complete bus ID.";
  EXPECT_EQ(busId, busIdBefore) << "PCI lookup must not modify its input string.";
  provider.failure = NvmlFailure::PciHandle;
  EXPECT_EQ(ncclNvmlDeviceGetHandleByPciBusId(busId.data(), &device), ncclSystemError)
    << "A failing PCI provider lookup must return ncclSystemError.";
  EXPECT_EQ(provider.pciHandleCalls, 2) << "Both PCI scenarios must call the provider exactly once.";
  provider.failure = NvmlFailure::None;

  // Cached READ and WRITE results must preserve pair direction and capability selection.
  nvmlGpuP2PStatus_t status = NVML_P2P_STATUS_OK;
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              DeviceHandleForTest(0), DeviceHandleForTest(1), NVML_P2P_CAPS_INDEX_READ, &status), ncclSuccess)
    << "A cached forward READ query must return ncclSuccess.";
  EXPECT_EQ(status, NVML_P2P_STATUS_GPU_NOT_SUPPORTED)
    << "A cached forward READ query must return its exact directional value.";
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              DeviceHandleForTest(1), DeviceHandleForTest(0), NVML_P2P_CAPS_INDEX_READ, &status), ncclSuccess)
    << "A cached reverse READ query must return ncclSuccess.";
  EXPECT_EQ(status, NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED)
    << "A cached reverse READ query must not reuse the forward value.";
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              DeviceHandleForTest(0), DeviceHandleForTest(1), NVML_P2P_CAPS_INDEX_WRITE, &status), ncclSuccess)
    << "A cached forward WRITE query must return ncclSuccess.";
  EXPECT_EQ(status, NVML_P2P_STATUS_UNKNOWN) << "Cached WRITE must not return the cached READ value.";
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              DeviceHandleForTest(1), DeviceHandleForTest(0), NVML_P2P_CAPS_INDEX_WRITE, &status), ncclSuccess)
    << "A cached reverse WRITE query must return ncclSuccess.";
  EXPECT_EQ(status, NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED)
    << "A cached reverse WRITE query must not reuse the forward value.";
  EXPECT_TRUE(provider.p2pCalls.empty()) << "Cached READ and WRITE queries must not call the provider.";

  // Either unknown cached endpoint must return invalid argument and preserve output.
  status = NVML_P2P_STATUS_DISABLED_BY_REGKEY;
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              reinterpret_cast<nvmlDevice_t>(&unknownDeviceToken), DeviceHandleForTest(1),
              NVML_P2P_CAPS_INDEX_READ, &status), ncclInvalidArgument)
    << "An unknown first cached P2P endpoint must return ncclInvalidArgument.";
  EXPECT_EQ(status, NVML_P2P_STATUS_DISABLED_BY_REGKEY)
    << "An invalid first cached P2P endpoint must preserve the caller's output.";
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              DeviceHandleForTest(0), reinterpret_cast<nvmlDevice_t>(&unknownDeviceToken),
              NVML_P2P_CAPS_INDEX_WRITE, &status), ncclInvalidArgument)
    << "An unknown second cached P2P endpoint must return ncclInvalidArgument.";
  EXPECT_EQ(status, NVML_P2P_STATUS_DISABLED_BY_REGKEY)
    << "An invalid second cached P2P endpoint must preserve the caller's output.";

  // Other capability indices must dispatch exact arguments and translate provider failure.
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              DeviceHandleForTest(0), DeviceHandleForTest(1), NVML_P2P_CAPS_INDEX_NVLINK, &status), ncclSuccess)
    << "A successful noncached P2P query must return ncclSuccess.";
  ASSERT_EQ(provider.p2pCalls.size(), 1U) << "A noncached P2P query must call the provider exactly once.";
  EXPECT_EQ(provider.p2pCalls[0].first, DeviceHandleForTest(0)) << "The exact source device must be forwarded.";
  EXPECT_EQ(provider.p2pCalls[0].second, DeviceHandleForTest(1))
    << "The exact destination device must be forwarded.";
  EXPECT_EQ(provider.p2pCalls[0].capability, NVML_P2P_CAPS_INDEX_NVLINK)
    << "The noncached capability selector must be forwarded exactly.";
  EXPECT_EQ(status, NVML_P2P_STATUS_DISABLED_BY_REGKEY)
    << "The noncached P2P output must equal the provider result.";
  provider.failure = NvmlFailure::P2pOther;
  EXPECT_EQ(ncclNvmlDeviceGetP2PStatus(
              DeviceHandleForTest(0), DeviceHandleForTest(1), NVML_P2P_CAPS_INDEX_NVLINK, &status),
            ncclSystemError)
    << "A failing noncached P2P provider query must return ncclSystemError.";
  EXPECT_EQ(provider.p2pCalls.size(), 2U)
    << "Each noncached P2P scenario must call the provider exactly once.";
  EXPECT_EQ(provider.p2pCalls[1].first, DeviceHandleForTest(0))
    << "The failing noncached query must forward the exact source device.";
  EXPECT_EQ(provider.p2pCalls[1].second, DeviceHandleForTest(1))
    << "The failing noncached query must forward the exact destination device.";
  EXPECT_EQ(provider.p2pCalls[1].capability, NVML_P2P_CAPS_INDEX_NVLINK)
    << "The failing noncached query must forward the exact capability selector.";
}

/**
 * @brief Verifies successful forwarding and output behavior for all optional NVML device queries.
 * @note Side effects: Prepares cache state, calls fake optional providers, and writes caller
 * outputs.
 */
TEST_F(NvmlQueryTest, RunsOptionalQueries) {
  PrepareReadyCacheForTest();
  const nvmlDevice_t device = DeviceHandleForTest(1);

  // Link state must forward both selectors and replace its scalar output.
  nvmlEnableState_t active = NVML_FEATURE_DISABLED;
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkState(device, 3, &active), ncclSuccess)
    << "A successful link-state query must return ncclSuccess.";
  EXPECT_EQ(active, NVML_FEATURE_ENABLED) << "Link-state output must equal the provider result.";
  EXPECT_EQ(provider.linkStateDevice, device) << "Link-state query must forward the exact device.";
  EXPECT_EQ(provider.linkStateLink, 3U) << "Link-state query must forward the exact link index.";

  // Remote PCI lookup must forward selectors and expose every provider identity field.
  nvmlPciInfo_t pci = MakePciSentinelForTest();
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkRemotePciInfo(device, 4, &pci), ncclSuccess)
    << "A successful remote-PCI query must return ncclSuccess.";
  EXPECT_EQ(provider.remotePciDevice, device) << "Remote-PCI query must forward the exact device.";
  EXPECT_EQ(provider.remotePciLink, 4U) << "Remote-PCI query must forward the exact link index.";
  EXPECT_STREQ(pci.busId, "0042:17:03.1") << "Remote-PCI bus ID must equal the provider output.";
  EXPECT_EQ(pci.domain, 0x42U) << "Remote-PCI domain must equal the provider output.";
  EXPECT_EQ(pci.bus, 0x17U) << "Remote-PCI bus must equal the provider output.";
  EXPECT_EQ(pci.device, 3U) << "Remote-PCI device must equal the provider output.";
  EXPECT_EQ(pci.pciDeviceId, 0x123410deU) << "Remote-PCI device ID must equal the provider output.";
  EXPECT_EQ(pci.pciSubSystemId, 0x567810deU) << "Remote-PCI subsystem ID must equal the provider output.";

  // Capability lookup must forward all three selectors and replace its output.
  unsigned int capability = 99;
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkCapability(
              device, 5, NVML_NVLINK_CAP_SYSMEM_ATOMICS, &capability), ncclSuccess)
    << "A successful link-capability query must return ncclSuccess.";
  EXPECT_EQ(capability, 17U) << "Link-capability output must equal the provider result.";
  EXPECT_EQ(provider.linkCapabilityDevice, device) << "Link-capability query must forward the exact device.";
  EXPECT_EQ(provider.linkCapabilityLink, 5U) << "Link-capability query must forward the exact link index.";
  EXPECT_EQ(provider.linkCapability, NVML_NVLINK_CAP_SYSMEM_ATOMICS)
    << "Link-capability query must forward the exact capability selector.";

  // Field query must preserve request identity until provider consumption, then expose all result
  // fields.
  std::array<nvmlFieldValue_t, 2> fields = {};
  fields[0].fieldId = NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON;
  fields[0].scopeId = 6;
  fields[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
  fields[1].scopeId = 7;
  EXPECT_EQ(ncclNvmlDeviceGetFieldValues(device, fields.size(), fields.data()), ncclSuccess)
    << "A successful field query must return ncclSuccess.";
  EXPECT_EQ(provider.fieldDevice, device) << "Field query must forward the exact device.";
  EXPECT_EQ(provider.fieldCount, 2) << "Field query must forward the exact entry count.";
  ASSERT_EQ(provider.fieldInputs.size(), 2U) << "The provider must receive both field requests.";
  EXPECT_EQ(provider.fieldInputs[0],
            std::make_pair(static_cast<unsigned int>(NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON), 6U))
    << "The first field ID and scope must reach the provider unchanged.";
  EXPECT_EQ(provider.fieldInputs[1],
            std::make_pair(static_cast<unsigned int>(NVML_FI_DEV_C2C_LINK_GET_MAX_BW), 7U))
    << "The second field ID and scope must reach the provider unchanged.";
  for (size_t index = 0; index < fields.size(); ++index) {
    SCOPED_TRACE("field index: " + std::to_string(index));
    EXPECT_EQ(fields[index].timestamp, 1000 + static_cast<long long>(index))
      << "Field timestamp must equal the provider output.";
    EXPECT_EQ(fields[index].latencyUsec, 20 + static_cast<long long>(index))
      << "Field latency must equal the provider output.";
    EXPECT_EQ(fields[index].valueType, NVML_VALUE_TYPE_UNSIGNED_INT)
      << "Field value type must equal the provider output.";
    EXPECT_EQ(fields[index].nvmlReturn, NVML_SUCCESS) << "Field status must equal the provider output.";
    EXPECT_EQ(fields[index].value.uiVal, 700U + index) << "Field value must equal the provider output.";
  }

  // Fabric query must set the required version without clearing provider-visible input or returned
  // fields.
  nvmlGpuFabricInfoV_t fabric = MakeFabricSentinelForTest();
  EXPECT_EQ(ncclNvmlDeviceGetGpuFabricInfoV(device, &fabric), ncclSuccess)
    << "A successful fabric query must return ncclSuccess.";
  EXPECT_EQ(provider.fabricDevice, device) << "Fabric query must forward the exact device.";
  EXPECT_EQ(provider.fabricInputVersion, nvmlGpuFabricInfo_v2)
    << "Fabric query must supply the required v2 structure version.";
  for (size_t index = 0; index < provider.fabricInputUuid.size(); ++index) {
    SCOPED_TRACE("fabric input UUID byte: " + std::to_string(index));
    EXPECT_EQ(provider.fabricInputUuid[index], static_cast<unsigned char>(0xd0 + index))
      << "Fabric version setup must preserve every caller UUID byte before provider dispatch.";
  }
  EXPECT_EQ(provider.fabricInputStatus, NVML_ERROR_NOT_FOUND)
    << "Fabric version setup must preserve caller status before provider dispatch.";
  EXPECT_EQ(provider.fabricInputClique, 78U)
    << "Fabric version setup must preserve caller clique before provider dispatch.";
  EXPECT_EQ(provider.fabricInputState, NVML_GPU_FABRIC_STATE_IN_PROGRESS)
    << "Fabric version setup must not clear the caller's state before provider dispatch.";
  EXPECT_EQ(provider.fabricInputHealth, 79U)
    << "Fabric version setup must not clear the caller's health mask before provider dispatch.";
  EXPECT_EQ(fabric.version, nvmlGpuFabricInfo_v2) << "Fabric output must retain the required v2 version.";
  EXPECT_EQ(fabric.clusterUuid[0], 0xa5U) << "Fabric UUID must equal the provider output.";
  for (size_t index = 1; index < sizeof(fabric.clusterUuid); ++index) {
    SCOPED_TRACE("fabric output UUID byte: " + std::to_string(index));
    EXPECT_EQ(fabric.clusterUuid[index], static_cast<unsigned char>(0xd0 + index))
      << "Fabric wrapper must preserve provider-untouched UUID bytes.";
  }
  EXPECT_EQ(fabric.status, NVML_SUCCESS) << "Fabric status must equal the provider output.";
  EXPECT_EQ(fabric.cliqueId, 41U) << "Fabric clique ID must equal the provider output.";
  EXPECT_EQ(fabric.state, NVML_GPU_FABRIC_STATE_COMPLETED) << "Fabric state must equal the provider output.";
  EXPECT_EQ(fabric.healthMask, NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_FALSE)
    << "Fabric health mask must equal the provider output.";

  // Platform query follows the same version-before-provider contract and returns every semantic
  // identity field.
  nvmlPlatformInfo_t platform = MakePlatformSentinelForTest();
  EXPECT_EQ(ncclNvmlDeviceGetPlatformInfo(device, &platform), ncclSuccess)
    << "A successful platform query must return ncclSuccess.";
  EXPECT_EQ(provider.platformDevice, device) << "Platform query must forward the exact device.";
  EXPECT_EQ(provider.platformInputVersion, nvmlPlatformInfo_v2)
    << "Platform query must supply the required v2 structure version.";
  for (size_t index = 0; index < provider.platformInputGuid.size(); ++index) {
    SCOPED_TRACE("platform input identity byte: " + std::to_string(index));
    EXPECT_EQ(provider.platformInputGuid[index], static_cast<unsigned char>(0xa0 + index))
      << "Platform version setup must preserve every caller IB GUID byte before provider dispatch.";
    EXPECT_EQ(provider.platformInputSerial[index], static_cast<unsigned char>(0xb0 + index))
      << "Platform version setup must preserve every caller serial byte before provider dispatch.";
  }
  EXPECT_EQ(provider.platformInputSlot, 78U)
    << "Platform version setup must not clear caller data before provider dispatch.";
  EXPECT_EQ(provider.platformInputTray, 79U) << "Platform version setup must preserve the caller tray.";
  EXPECT_EQ(provider.platformInputHost, 80U) << "Platform version setup must preserve the caller host ID.";
  EXPECT_EQ(provider.platformInputPeer, 81U) << "Platform version setup must preserve the caller peer type.";
  EXPECT_EQ(provider.platformInputModule, 82U) << "Platform version setup must preserve the caller module ID.";
  EXPECT_EQ(platform.version, nvmlPlatformInfo_v2) << "Platform output must retain the required v2 version.";
  EXPECT_EQ(platform.ibGuid[0], 0xb6U) << "Platform IB GUID must equal the provider output.";
  EXPECT_EQ(platform.chassisSerialNumber[0], 0xc7U)
    << "Platform chassis serial must equal the provider output.";
  for (size_t index = 1; index < sizeof(platform.ibGuid); ++index) {
    SCOPED_TRACE("platform output identity byte: " + std::to_string(index));
    EXPECT_EQ(platform.ibGuid[index], static_cast<unsigned char>(0xa0 + index))
      << "Platform wrapper must preserve provider-untouched IB GUID bytes.";
    EXPECT_EQ(platform.chassisSerialNumber[index], static_cast<unsigned char>(0xb0 + index))
      << "Platform wrapper must preserve provider-untouched serial bytes.";
  }
  EXPECT_EQ(platform.slotNumber, 5U) << "Platform slot must equal the provider output.";
  EXPECT_EQ(platform.trayIndex, 6U) << "Platform tray must equal the provider output.";
  EXPECT_EQ(platform.hostId, 7U) << "Platform host ID must equal the provider output.";
  EXPECT_EQ(platform.peerType, 8U) << "Platform peer type must equal the provider output.";
  EXPECT_EQ(platform.moduleId, 9U) << "Platform module ID must equal the provider output.";
}

/**
 * @brief Verifies missing-symbol and provider-failure results for every optional NVML device query.
 * @note Side effects: Prepares cache state, reconfigures function pointers, calls fake providers,
 * and may write caller outputs.
 */
TEST_F(NvmlQueryTest, HandlesOptionalQueryFailures) {
  PrepareReadyCacheForTest();
  const nvmlDevice_t device = DeviceHandleForTest(0);

  // Missing scalar-query symbols must return internal error before changing output or calling a
  // provider.
  pfn_nvmlDeviceGetNvLinkState = nullptr;
  nvmlEnableState_t active = NVML_FEATURE_ENABLED;
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkState(device, 1, &active), ncclInternalError)
    << "A missing link-state symbol must return ncclInternalError.";
  EXPECT_EQ(active, NVML_FEATURE_ENABLED) << "A missing link-state symbol must preserve output.";
  EXPECT_EQ(provider.linkStateCalls, 0) << "A missing link-state symbol must not call a provider.";

  pfn_nvmlDeviceGetNvLinkRemotePciInfo = nullptr;
  nvmlPciInfo_t pci = MakePciSentinelForTest();
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkRemotePciInfo(device, 2, &pci), ncclInternalError)
    << "A missing remote-PCI symbol must return ncclInternalError.";
  ExpectPciSentinelForTest(pci);
  EXPECT_EQ(provider.remotePciCalls, 0) << "A missing remote-PCI symbol must not call a provider.";

  pfn_nvmlDeviceGetNvLinkCapability = nullptr;
  unsigned int capability = 77;
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkCapability(
              device, 3, NVML_NVLINK_CAP_VALID, &capability), ncclInternalError)
    << "A missing link-capability symbol must return ncclInternalError.";
  EXPECT_EQ(capability, 77U) << "A missing link-capability symbol must preserve output.";
  EXPECT_EQ(provider.linkCapabilityCalls, 0) << "A missing link-capability symbol must not call a provider.";

  pfn_nvmlDeviceGetFieldValues = nullptr;
  nvmlFieldValue_t field = MakeFieldSentinelForTest();
  EXPECT_EQ(ncclNvmlDeviceGetFieldValues(device, 1, &field), ncclInternalError)
    << "A missing field-query symbol must return ncclInternalError.";
  ExpectFieldSentinelForTest(field);
  EXPECT_EQ(provider.fieldCalls, 0) << "A missing field-query symbol must not call a provider.";

  // Versioned wrappers must write only their required version before reporting a missing provider.
  pfn_nvmlDeviceGetGpuFabricInfoV = nullptr;
  nvmlGpuFabricInfoV_t fabric = MakeFabricSentinelForTest();
  EXPECT_EQ(ncclNvmlDeviceGetGpuFabricInfoV(device, &fabric), ncclInternalError)
    << "A missing fabric symbol must return ncclInternalError.";
  ExpectFabricSentinelForTest(fabric, nvmlGpuFabricInfo_v2);
  EXPECT_EQ(provider.fabricCalls, 0) << "A missing fabric symbol must not call a provider.";

  pfn_nvmlDeviceGetPlatformInfo = nullptr;
  nvmlPlatformInfo_t platform = MakePlatformSentinelForTest();
  EXPECT_EQ(ncclNvmlDeviceGetPlatformInfo(device, &platform), ncclInternalError)
    << "A missing platform symbol must return ncclInternalError.";
  ExpectPlatformSentinelForTest(platform, nvmlPlatformInfo_v2);
  EXPECT_EQ(provider.platformCalls, 0) << "A missing platform symbol must not call a provider.";

  // Installed providers must translate their own failure to system error, with one dispatch each.
  /** @brief Describes one optional NVML provider failure. */
  struct FailureScenario {
    const char* description;
    NvmlFailure point;
  };
  for (const FailureScenario& scenario : std::array<FailureScenario, 6>{{
         {"link state", NvmlFailure::LinkState},
         {"remote PCI", NvmlFailure::RemotePci},
         {"link capability", NvmlFailure::LinkCapability},
         {"field values", NvmlFailure::FieldValues},
         {"fabric", NvmlFailure::Fabric},
         {"platform", NvmlFailure::Platform},
       }}) {
    SCOPED_TRACE(scenario.description);
    ResetNvmlForTest();
    PrepareReadyCacheForTest();
    provider.failure = scenario.point;
    ncclResult_t result = ncclSuccess;
    if (scenario.point == NvmlFailure::LinkState) {
      result = ncclNvmlDeviceGetNvLinkState(device, 1, &active);
      EXPECT_EQ(provider.linkStateCalls, 1) << "The failing link-state provider must run exactly once.";
    } else if (scenario.point == NvmlFailure::RemotePci) {
      result = ncclNvmlDeviceGetNvLinkRemotePciInfo(device, 2, &pci);
      EXPECT_EQ(provider.remotePciCalls, 1) << "The failing remote-PCI provider must run exactly once.";
    } else if (scenario.point == NvmlFailure::LinkCapability) {
      result = ncclNvmlDeviceGetNvLinkCapability(device, 3, NVML_NVLINK_CAP_VALID, &capability);
      EXPECT_EQ(provider.linkCapabilityCalls, 1) << "The failing capability provider must run exactly once.";
    } else if (scenario.point == NvmlFailure::FieldValues) {
      result = ncclNvmlDeviceGetFieldValues(device, 1, &field);
      EXPECT_EQ(provider.fieldCalls, 1) << "The failing field provider must run exactly once.";
    } else if (scenario.point == NvmlFailure::Fabric) {
      result = ncclNvmlDeviceGetGpuFabricInfoV(device, &fabric);
      EXPECT_EQ(provider.fabricCalls, 1) << "The failing fabric provider must run exactly once.";
    } else {
      result = ncclNvmlDeviceGetPlatformInfo(device, &platform);
      EXPECT_EQ(provider.platformCalls, 1) << "The failing platform provider must run exactly once.";
    }
    EXPECT_EQ(result, ncclSystemError) << "An installed optional provider failure must return ncclSystemError.";
  }
}

/**
 * @brief Verifies modern, legacy, unavailable, and failed confidential-compute status queries.
 * @note Side effects: Prepares cache state, reconfigures CC provider symbols, invokes providers,
 * and replaces caller status outputs.
 */
TEST_F(NvmlCcTest, ReportsConfidentialComputeModes) {
  PrepareReadyCacheForTest();

  // Each modern multi-GPU mode must map independently while preserving the feature flag.
  /** @brief Describes one modern confidential-compute mode and expected flags. */
  struct ModernScenario {
    const char* description;
    unsigned int feature;
    unsigned int mode;
    bool enabled;
    bool protectedPcie;
    bool nvle;
  };
  const std::array<ModernScenario, 3> modernScenarios = {{
    {"protected PCIe", NVML_CC_SYSTEM_FEATURE_ENABLED, NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE,
      true, true, false},
    {"NVLE", NVML_CC_SYSTEM_FEATURE_ENABLED, NVML_CC_SYSTEM_MULTIGPU_NVLE, true, false, true},
    {"disabled", NVML_CC_SYSTEM_FEATURE_DISABLED, NVML_CC_SYSTEM_MULTIGPU_NONE, false, false, false},
  }};
  for (const ModernScenario& scenario : modernScenarios) {
    SCOPED_TRACE(scenario.description);
    provider.ccFeature = scenario.feature;
    provider.multiGpuMode = scenario.mode;
    ncclNvmlCCStatus status = {!scenario.enabled, !scenario.protectedPcie, !scenario.nvle};
    EXPECT_EQ(ncclNvmlGetCCStatus(&status), ncclSuccess)
      << "A successful modern confidential-compute query must return ncclSuccess.";
    EXPECT_EQ(status.CCEnabled, scenario.enabled) << "CC feature status must map exactly.";
    EXPECT_EQ(status.multiGpuProtectedPCIE, scenario.protectedPcie)
      << "Protected-PCIe status must map independently.";
    EXPECT_EQ(status.multiGpuNVLE, scenario.nvle) << "NVLE status must map independently.";
    EXPECT_EQ(provider.modernCcInputVersion, nvmlSystemConfComputeSettings_v1)
      << "Modern query must supply the required settings version.";
  }
  EXPECT_EQ(provider.modernCcCalls, 3) << "Each modern scenario must call the modern provider once.";
  EXPECT_EQ(provider.legacyCcCalls, 0) << "An available modern provider must take precedence over legacy.";

  // Legacy support reports only the feature state and must clear both modern-only mode outputs.
  pfn_nvmlSystemGetConfComputeSettings = nullptr;
  for (const unsigned int feature : {NVML_CC_SYSTEM_FEATURE_ENABLED, NVML_CC_SYSTEM_FEATURE_DISABLED}) {
    provider.ccFeature = feature;
    ncclNvmlCCStatus status = {feature == NVML_CC_SYSTEM_FEATURE_DISABLED, true, true};
    EXPECT_EQ(ncclNvmlGetCCStatus(&status), ncclSuccess)
      << "A successful legacy confidential-compute query must return ncclSuccess.";
    EXPECT_EQ(status.CCEnabled, feature == NVML_CC_SYSTEM_FEATURE_ENABLED)
      << "Legacy CC feature state must map exactly.";
    EXPECT_FALSE(status.multiGpuProtectedPCIE) << "Legacy API cannot report protected-PCIe mode.";
    EXPECT_FALSE(status.multiGpuNVLE) << "Legacy API cannot report NVLE mode.";
  }
  EXPECT_EQ(provider.legacyCcCalls, 2) << "Each legacy feature state must call the legacy provider once.";

  // Absence of both optional APIs is a successful explicit all-disabled result.
  pfn_nvmlSystemGetConfComputeState = nullptr;
  ncclNvmlCCStatus unavailable = {true, true, true};
  EXPECT_EQ(ncclNvmlGetCCStatus(&unavailable), ncclSuccess)
    << "Unavailable optional CC APIs must return ncclSuccess.";
  EXPECT_FALSE(unavailable.CCEnabled) << "Unavailable CC APIs must report the feature disabled.";
  EXPECT_FALSE(unavailable.multiGpuProtectedPCIE) << "Unavailable CC APIs must clear protected-PCIe mode.";
  EXPECT_FALSE(unavailable.multiGpuNVLE) << "Unavailable CC APIs must clear NVLE mode.";

  // Each installed API must translate its own provider failure and stop at the selected interface.
  ResetNvmlForTest();
  PrepareReadyCacheForTest();
  provider.failure = NvmlFailure::CcModern;
  ncclNvmlCCStatus failed = {true, true, true};
  EXPECT_EQ(ncclNvmlGetCCStatus(&failed), ncclSystemError)
    << "A modern confidential-compute provider failure must return ncclSystemError.";
  EXPECT_EQ(provider.modernCcCalls, 1) << "The failing modern provider must run exactly once.";
  EXPECT_EQ(provider.legacyCcCalls, 0) << "Modern provider failure must not fall back to legacy data.";

  ResetNvmlForTest();
  PrepareReadyCacheForTest();
  pfn_nvmlSystemGetConfComputeSettings = nullptr;
  provider.failure = NvmlFailure::CcLegacy;
  EXPECT_EQ(ncclNvmlGetCCStatus(&failed), ncclSystemError)
    << "A legacy confidential-compute provider failure must return ncclSystemError.";
  EXPECT_EQ(provider.legacyCcCalls, 1) << "The failing legacy provider must run exactly once.";
}

}  // namespace
