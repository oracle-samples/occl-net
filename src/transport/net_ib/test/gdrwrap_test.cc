/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "gdrwrap.h"
#include "os.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::array<const char*, 11> requiredSymbolNames = {{
  "gdr_open",
  "gdr_close",
  "gdr_pin_buffer",
  "gdr_unpin_buffer",
  "gdr_get_info",
  "gdr_map",
  "gdr_unmap",
  "gdr_runtime_get_version",
  "gdr_driver_get_version",
  "gdr_copy_to_mapping",
  "gdr_copy_from_mapping",
}};

/** @brief Records one normalized call to the fake GDR provider. */
struct ProviderCall {
  std::string operation;
  std::array<uintptr_t, 6> arguments = {};
  size_t argumentCount = 0;
};

/**
 * @brief Scripts fake loader/provider outcomes and records a normalized call trace.
 */
struct ProviderState {
  bool libraryAvailable = true;
  std::string missingSymbol;
  int libraryOpenCalls = 0;
  std::string openedLibrary;
  int openedMode = -1;
  int libraryCloseCalls = 0;
  void* closedLibrary = nullptr;
  std::vector<std::string> resolvedSymbols;
  int operationResult = 0;
  gdr_t openResult = nullptr;
  gdr_mh_t outputHandle = {0xabcdu};
  gdr_info_t outputInfo = {};
  void* outputMapping = nullptr;
  int runtimeMajor = 2;
  int runtimeMinor = 5;
  int driverMajor = 2;
  int driverMinor = 7;
  std::vector<ProviderCall> calls;
};

ProviderState providerState;
int libraryToken = 0x45;
int handleToken = 0x67;
std::array<unsigned char, 64> mappedStorage = {};

/** @brief Signals that the blocking copy provider has entered. */
std::atomic<bool> blockingProviderEntered{false};
/** @brief Releases the blocking copy provider. */
std::atomic<bool> releaseBlockingProvider{false};
/** @brief Counts blocking copy-provider calls. */
std::atomic<int> blockingProviderCalls{0};

/**
 * @brief Converts any pointer value to its pointer-sized integer representation.
 * @param[in] pointer Any pointer value, including NULL.
 * @return The pointer representation as uintptr_t.
 * @note Side effects: None.
 */
uintptr_t PointerValueForTest(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer);
}

/**
 * @brief Converts a provider function pointer to the POSIX dynamic-loader representation.
 * @tparam Symbol Provider function-pointer type.
 * @param[in] symbol Valid non-NULL provider function pointer.
 * @return The corresponding non-NULL object pointer.
 * @note Side effects: None.
 */
template <typename Symbol>
void* SymbolAddressForTest(Symbol symbol) {
  static_assert(sizeof(symbol) == sizeof(void*), "The test requires POSIX-compatible function pointers.");
  void* address = nullptr;
  std::memcpy(&address, &symbol, sizeof(address));
  return address;
}

/**
 * @brief Records one fake-provider call and its ordered arguments.
 * @param[in] operation Valid non-NULL operation name.
 * @param[in] arguments Zero to six pointer-sized argument values.
 * @note Side effects: Appends one entry to providerState.calls.
 */
void RecordProviderCallForTest(const char* operation, std::initializer_list<uintptr_t> arguments) {
  ProviderCall call;
  call.operation = operation;
  call.argumentCount = arguments.size();
  std::copy(arguments.begin(), arguments.end(), call.arguments.begin());
  providerState.calls.push_back(call);
}

/**
 * @brief Returns the configured fake GDR context.
 * @return providerState.openResult, including NULL.
 * @note Side effects: Records one open operation.
 */
gdr_t OpenProviderForTest() {
  RecordProviderCallForTest("open", {});
  return providerState.openResult;
}

/**
 * @brief Implements deterministic fake GDR context closure.
 * @param[in] context Any GDR context recorded exactly.
 * @return providerState.operationResult.
 * @note Side effects: Records context.
 */
int CloseProviderForTest(gdr_t context) {
  RecordProviderCallForTest("close", {PointerValueForTest(context)});
  return providerState.operationResult;
}

/**
 * @brief Implements deterministic legacy buffer pinning.
 * @param[in] context Any GDR context recorded exactly.
 * @param[in] address Device address in [0, ULONG_MAX] recorded exactly.
 * @param[in] size Buffer size in [0, SIZE_MAX] recorded exactly.
 * @param[in] token Peer token in [0, UINT64_MAX] recorded exactly.
 * @param[in] addressSpace VA-space identifier in [0, UINT32_MAX] recorded exactly.
 * @param[out] handle Valid non-NULL output receiving outputHandle on success.
 * @return providerState.operationResult.
 * @note Side effects: Records all arguments and writes handle only on success.
 */
int PinProviderForTest(gdr_t context, unsigned long address, size_t size, uint64_t token,
  uint32_t addressSpace, gdr_mh_t* handle) {
  RecordProviderCallForTest("pin", {PointerValueForTest(context), address, size, token, addressSpace,
    PointerValueForTest(handle)});
  if (providerState.operationResult == 0) *handle = providerState.outputHandle;
  return providerState.operationResult;
}

/**
 * @brief Implements deterministic v2 buffer pinning.
 * @param[in] context Any GDR context recorded exactly.
 * @param[in] address Device address in [0, ULONG_MAX] recorded exactly.
 * @param[in] size Buffer size in [0, SIZE_MAX] recorded exactly.
 * @param[in] flags Pin flags in [0, UINT32_MAX] recorded exactly.
 * @param[out] handle Valid non-NULL output receiving outputHandle on success.
 * @return providerState.operationResult.
 * @note Side effects: Records all arguments and writes handle only on success.
 */
int PinV2ProviderForTest(gdr_t context, unsigned long address, size_t size, uint32_t flags,
  gdr_mh_t* handle) {
  RecordProviderCallForTest("pin_v2", {PointerValueForTest(context), address, size, flags,
    PointerValueForTest(handle)});
  if (providerState.operationResult == 0) *handle = providerState.outputHandle;
  return providerState.operationResult;
}

/**
 * @brief Implements deterministic buffer unpinning.
 * @param[in] context Any GDR context recorded exactly.
 * @param[in] handle Any GDR memory handle recorded exactly.
 * @return providerState.operationResult.
 * @note Side effects: Records context and handle.
 */
int UnpinProviderForTest(gdr_t context, gdr_mh_t handle) {
  RecordProviderCallForTest("unpin", {PointerValueForTest(context), handle.h});
  return providerState.operationResult;
}

/**
 * @brief Implements deterministic memory-information lookup.
 * @param[in] context Any GDR context recorded exactly.
 * @param[in] handle Any GDR memory handle recorded exactly.
 * @param[out] info Valid non-NULL output receiving outputInfo on success.
 * @return providerState.operationResult.
 * @note Side effects: Records all arguments and writes info only on success.
 */
int GetInfoProviderForTest(gdr_t context, gdr_mh_t handle, gdr_info_t* info) {
  RecordProviderCallForTest("get_info", {PointerValueForTest(context), handle.h, PointerValueForTest(info)});
  if (providerState.operationResult == 0) *info = providerState.outputInfo;
  return providerState.operationResult;
}

/**
 * @brief Implements deterministic memory mapping.
 * @param[in] context Any GDR context recorded exactly.
 * @param[in] handle Any GDR memory handle recorded exactly.
 * @param[out] mapping Valid non-NULL output receiving outputMapping on success.
 * @param[in] size Mapping size in [0, SIZE_MAX] recorded exactly.
 * @return providerState.operationResult.
 * @note Side effects: Records all arguments and writes mapping only on success.
 */
int MapProviderForTest(gdr_t context, gdr_mh_t handle, void** mapping, size_t size) {
  RecordProviderCallForTest("map", {PointerValueForTest(context), handle.h,
    PointerValueForTest(mapping), size});
  if (providerState.operationResult == 0) *mapping = providerState.outputMapping;
  return providerState.operationResult;
}

/**
 * @brief Implements deterministic memory unmapping.
 * @param[in] context Any GDR context recorded exactly.
 * @param[in] handle Any GDR memory handle recorded exactly.
 * @param[in] mapping Any mapped address recorded exactly.
 * @param[in] size Mapping size in [0, SIZE_MAX] recorded exactly.
 * @return providerState.operationResult.
 * @note Side effects: Records every argument.
 */
int UnmapProviderForTest(gdr_t context, gdr_mh_t handle, void* mapping, size_t size) {
  RecordProviderCallForTest("unmap", {PointerValueForTest(context), handle.h,
    PointerValueForTest(mapping), size});
  return providerState.operationResult;
}

/**
 * @brief Returns the configured runtime version.
 * @param[out] major Valid non-NULL output receiving runtimeMajor.
 * @param[out] minor Valid non-NULL output receiving runtimeMinor.
 * @note Side effects: Records both output pointers and writes both outputs.
 */
void RuntimeVersionProviderForTest(int* major, int* minor) {
  RecordProviderCallForTest("runtime_version", {PointerValueForTest(major), PointerValueForTest(minor)});
  *major = providerState.runtimeMajor;
  *minor = providerState.runtimeMinor;
}

/**
 * @brief Returns the configured driver version.
 * @param[in] context Any GDR context recorded exactly.
 * @param[out] major Valid non-NULL output receiving driverMajor.
 * @param[out] minor Valid non-NULL output receiving driverMinor.
 * @note Side effects: Records every argument and writes both outputs.
 */
void DriverVersionProviderForTest(gdr_t context, int* major, int* minor) {
  RecordProviderCallForTest("driver_version", {PointerValueForTest(context),
    PointerValueForTest(major), PointerValueForTest(minor)});
  *major = providerState.driverMajor;
  *minor = providerState.driverMinor;
}

/**
 * @brief Copies host bytes into one fake GDR mapping.
 * @param[in] handle Any GDR memory handle recorded exactly.
 * @param[out] mapping Valid non-NULL destination of size bytes.
 * @param[in] host Valid non-NULL source of size bytes.
 * @param[in] size Copy size in [0, SIZE_MAX] recorded exactly.
 * @return providerState.operationResult.
 * @note Side effects: Records all arguments and copies bytes only on success.
 */
int CopyToProviderForTest(gdr_mh_t handle, void* mapping, const void* host, size_t size) {
  RecordProviderCallForTest("copy_to", {handle.h, PointerValueForTest(mapping),
    PointerValueForTest(host), size});
  if (providerState.operationResult == 0) std::memcpy(mapping, host, size);
  return providerState.operationResult;
}

/**
 * @brief Copies bytes from one fake GDR mapping into host storage.
 * @param[in] handle Any GDR memory handle recorded exactly.
 * @param[out] host Valid non-NULL destination of size bytes.
 * @param[in] mapping Valid non-NULL source of size bytes.
 * @param[in] size Copy size in [0, SIZE_MAX] recorded exactly.
 * @return providerState.operationResult.
 * @note Side effects: Records all arguments and copies bytes only on success.
 */
int CopyFromProviderForTest(gdr_mh_t handle, void* host, const void* mapping, size_t size) {
  RecordProviderCallForTest("copy_from", {handle.h, PointerValueForTest(host),
    PointerValueForTest(mapping), size});
  if (providerState.operationResult == 0) std::memcpy(host, mapping, size);
  return providerState.operationResult;
}

/**
 * @brief Blocks inside a provider copy until the test releases it.
 * @param[in] handle Any GDR memory handle.
 * @param[out] mapping Valid non-NULL destination with capacity for @p size bytes.
 * @param[in] host Valid non-NULL source readable for @p size bytes.
 * @param[in] size Copy size in [0, SIZE_MAX].
 * @return 0 after release.
 * @note Side effects: Updates synchronization atomics and copies size bytes after release.
 */
int BlockingCopyToProviderForTest(gdr_mh_t handle, void* mapping, const void* host, size_t size) {
  (void)handle;
  ++blockingProviderCalls;
  blockingProviderEntered.store(true, std::memory_order_release);
  while (!releaseBlockingProvider.load(std::memory_order_acquire)) std::this_thread::yield();
  std::memcpy(mapping, host, size);
  return 0;
}

/**
 * @brief Opens the configured fake dynamic library.
 * @param[in] filename Valid non-NULL library name recorded exactly.
 * @param[in] mode Dynamic-loader mode recorded exactly.
 * @return A stable non-NULL handle when libraryAvailable is true; otherwise NULL.
 * @note Side effects: Records the name and mode and increments libraryOpenCalls.
 */
ncclOsLibraryHandle OpenLibraryForTest(const char* filename, int mode) {
  ++providerState.libraryOpenCalls;
  providerState.openedLibrary = filename;
  providerState.openedMode = mode;
  return providerState.libraryAvailable ? &libraryToken : nullptr;
}

/**
 * @brief Resolves one configured fake provider symbol.
 * @param[in] library Valid non-NULL fake library handle.
 * @param[in] symbol Valid non-NULL symbol name.
 * @return The provider address, or NULL when symbol is configured missing or unknown.
 * @note Side effects: Appends symbol to resolvedSymbols.
 */
void* ResolveSymbolForTest(ncclOsLibraryHandle library, const char* symbol) {
  providerState.resolvedSymbols.emplace_back(symbol);
  if (library != &libraryToken || providerState.missingSymbol == symbol) return nullptr;
  if (std::strcmp(symbol, "gdr_open") == 0) return SymbolAddressForTest(OpenProviderForTest);
  if (std::strcmp(symbol, "gdr_close") == 0) return SymbolAddressForTest(CloseProviderForTest);
  if (std::strcmp(symbol, "gdr_pin_buffer") == 0) return SymbolAddressForTest(PinProviderForTest);
  if (std::strcmp(symbol, "gdr_pin_buffer_v2") == 0) return SymbolAddressForTest(PinV2ProviderForTest);
  if (std::strcmp(symbol, "gdr_unpin_buffer") == 0) return SymbolAddressForTest(UnpinProviderForTest);
  if (std::strcmp(symbol, "gdr_get_info") == 0) return SymbolAddressForTest(GetInfoProviderForTest);
  if (std::strcmp(symbol, "gdr_map") == 0) return SymbolAddressForTest(MapProviderForTest);
  if (std::strcmp(symbol, "gdr_unmap") == 0) return SymbolAddressForTest(UnmapProviderForTest);
  if (std::strcmp(symbol, "gdr_runtime_get_version") == 0) {
    return SymbolAddressForTest(RuntimeVersionProviderForTest);
  }
  if (std::strcmp(symbol, "gdr_driver_get_version") == 0) {
    return SymbolAddressForTest(DriverVersionProviderForTest);
  }
  if (std::strcmp(symbol, "gdr_copy_to_mapping") == 0) return SymbolAddressForTest(CopyToProviderForTest);
  if (std::strcmp(symbol, "gdr_copy_from_mapping") == 0) return SymbolAddressForTest(CopyFromProviderForTest);
  return nullptr;
}

/**
 * @brief Closes one fake dynamic library.
 * @param[in] library Any library handle recorded exactly.
 * @note Side effects: Increments libraryCloseCalls and replaces closedLibrary.
 */
void CloseLibraryForTest(ncclOsLibraryHandle library) {
  ++providerState.libraryCloseCalls;
  providerState.closedLibrary = library;
}

/**
 * @brief Returns a stable diagnostic for fake dynamic-loader failures.
 * @return A non-NULL static error string.
 * @note Side effects: None.
 */
const char* LoaderErrorForTest() {
  return "synthetic GDR loader failure";
}

}  // namespace

/**
 * @brief Compiles gdrwrap.cc against a fake loader so symbol discovery does not require libgdrapi.
 */
#define ncclOsDlopen OpenLibraryForTest
#define ncclOsDlsym ResolveSymbolForTest
#define ncclOsDlclose CloseLibraryForTest
#define ncclOsDlerror LoaderErrorForTest
#include "../../../misc/gdrwrap.cc"  // NOLINT(build/include)
#undef ncclOsDlerror
#undef ncclOsDlclose
#undef ncclOsDlsym
#undef ncclOsDlopen

namespace {

/**
 * @brief Resets fake provider data and every directly configurable wrapper symbol.
 * @note Side effects: Replaces providerState, storage, synchronization state, and symbols.
 */
void ResetProviderForTest() {
  providerState = ProviderState{};
  providerState.openResult = reinterpret_cast<gdr_t>(&handleToken);
  providerState.outputInfo.va = 0x12345000;
  providerState.outputInfo.mapped_size = 0x8000;
  providerState.outputInfo.page_size = 0x10000;
  providerState.outputInfo.tm_cycles = 0x2222;
  providerState.outputInfo.cycles_per_ms = 333;
  providerState.outputInfo.mapped = 1;
  providerState.outputInfo.wc_mapping = 1;
  providerState.outputMapping = mappedStorage.data();
  mappedStorage.fill(0);
  blockingProviderEntered.store(false, std::memory_order_relaxed);
  releaseBlockingProvider.store(false, std::memory_order_relaxed);
  blockingProviderCalls.store(0, std::memory_order_relaxed);
  gdr_internal_open = nullptr;
  gdr_internal_close = nullptr;
  gdr_internal_pin_buffer = nullptr;
  gdr_internal_pin_buffer_v2 = nullptr;
  gdr_internal_unpin_buffer = nullptr;
  gdr_internal_get_info = nullptr;
  gdr_internal_map = nullptr;
  gdr_internal_unmap = nullptr;
  gdr_internal_runtime_get_version = nullptr;
  gdr_internal_driver_get_version = nullptr;
  gdr_internal_copy_to_mapping = nullptr;
  gdr_internal_copy_from_mapping = nullptr;
  initResult = ncclInternalError;
}

/**
 * @brief Reports whether every dynamically loaded provider slot is NULL.
 * @return true only when all twelve slots are NULL.
 * @note Side effects: None.
 */
bool AllProviderSymbolsAreNullForTest() {
  return gdr_internal_open == nullptr && gdr_internal_close == nullptr &&
    gdr_internal_pin_buffer == nullptr && gdr_internal_pin_buffer_v2 == nullptr &&
    gdr_internal_unpin_buffer == nullptr && gdr_internal_get_info == nullptr &&
    gdr_internal_map == nullptr && gdr_internal_unmap == nullptr &&
    gdr_internal_runtime_get_version == nullptr && gdr_internal_driver_get_version == nullptr &&
    gdr_internal_copy_to_mapping == nullptr && gdr_internal_copy_from_mapping == nullptr;
}

/**
 * @brief Reports whether every mandatory provider slot contains its matching fake implementation.
 * @return true only when all eleven required slots match their expected functions.
 * @note Side effects: None.
 */
bool AllRequiredProviderSymbolsAreLoadedForTest() {
  return gdr_internal_open == OpenProviderForTest && gdr_internal_close == CloseProviderForTest &&
    gdr_internal_pin_buffer == PinProviderForTest && gdr_internal_unpin_buffer == UnpinProviderForTest &&
    gdr_internal_get_info == GetInfoProviderForTest && gdr_internal_map == MapProviderForTest &&
    gdr_internal_unmap == UnmapProviderForTest &&
    gdr_internal_runtime_get_version == RuntimeVersionProviderForTest &&
    gdr_internal_driver_get_version == DriverVersionProviderForTest &&
    gdr_internal_copy_to_mapping == CopyToProviderForTest &&
    gdr_internal_copy_from_mapping == CopyFromProviderForTest;
}

/**
 * @brief Compares every semantic field of two GDR information records.
 * @param[in] actual Record produced or preserved by the wrapper.
 * @param[in] expected Record required by the tested contract.
 * @param[in] reason Valid non-NULL explanation attached to every failure.
 * @note Side effects: Records GoogleTest failures when any field differs.
 */
void ExpectInfoEqualsForTest(const gdr_info_t& actual, const gdr_info_t& expected, const char* reason) {
  SCOPED_TRACE(reason);
  EXPECT_EQ(actual.va, expected.va) << "The virtual address differs.";
  EXPECT_EQ(actual.mapped_size, expected.mapped_size) << "The mapped size differs.";
  EXPECT_EQ(actual.page_size, expected.page_size) << "The page size differs.";
  EXPECT_EQ(actual.tm_cycles, expected.tm_cycles) << "The timing-cycle count differs.";
  EXPECT_EQ(actual.cycles_per_ms, expected.cycles_per_ms) << "The timing frequency differs.";
  EXPECT_EQ(actual.mapped, expected.mapped) << "The mapped flag differs.";
  EXPECT_EQ(actual.wc_mapping, expected.wc_mapping) << "The write-combining flag differs.";
}

/**
 * @brief Verifies direct loader success, optional absence, and every required-symbol failure.
 * @return 0 on success or a distinct nonzero failure code.
 * @note Side effects: Repeatedly resets state and calls the private loader initializer.
 */
int VerifyDirectLoaderForTest() {
  ResetProviderForTest();
  providerState.libraryAvailable = false;
  initOnceFunc();
  if (initResult != ncclSystemError || providerState.libraryOpenCalls != 1 ||
      providerState.openedLibrary != "libgdrapi.so" || providerState.openedMode != NCCL_OS_DL_NOW ||
      providerState.libraryCloseCalls != 0 ||
      !AllProviderSymbolsAreNullForTest()) {
    std::fprintf(stderr, "missing library did not use the expected open-and-fail contract\n");
    return 1;
  }

  for (size_t index = 0; index < requiredSymbolNames.size(); ++index) {
    ResetProviderForTest();
    providerState.missingSymbol = requiredSymbolNames[index];
    initOnceFunc();
    if (initResult != ncclSystemError || providerState.libraryCloseCalls != 1 ||
        providerState.closedLibrary != &libraryToken || !AllProviderSymbolsAreNullForTest() ||
        providerState.resolvedSymbols.empty() ||
        providerState.resolvedSymbols.back() != requiredSymbolNames[index]) {
      std::fprintf(stderr, "required symbol failure was not torn down: %s\n", requiredSymbolNames[index]);
      return static_cast<int>(10 + index);
    }
  }

  ResetProviderForTest();
  providerState.missingSymbol = "gdr_pin_buffer_v2";
  initOnceFunc();
  if (initResult != ncclSuccess || !AllRequiredProviderSymbolsAreLoadedForTest() ||
      gdr_internal_pin_buffer_v2 != nullptr || providerState.libraryCloseCalls != 0) {
    std::fprintf(stderr, "optional v2 symbol absence incorrectly failed initialization\n");
    return 30;
  }

  ResetProviderForTest();
  initOnceFunc();
  if (initResult != ncclSuccess || !AllRequiredProviderSymbolsAreLoadedForTest() ||
      gdr_internal_pin_buffer_v2 != PinV2ProviderForTest || providerState.libraryCloseCalls != 0) {
    std::fprintf(stderr, "complete symbol table did not initialize successfully\n");
    return 31;
  }
  if (providerState.openedLibrary != "libgdrapi.so" || providerState.openedMode != NCCL_OS_DL_NOW) {
    std::fprintf(stderr, "loader used wrong library or mode: %s mode %d\n",
      providerState.openedLibrary.c_str(), providerState.openedMode);
    return 32;
  }
  return 0;
}

/**
 * @brief Verifies one successful or failed public symbol-load result is cached.
 * @param[in] libraryAvailable Whether the first dynamic-library open must succeed.
 * @return 0 on success or a distinct nonzero failure code.
 * @note Side effects: Resets state and invokes wrap_gdr_symbols twice.
 */
int VerifyCachedLoaderForTest(bool libraryAvailable) {
  ResetProviderForTest();
  providerState.libraryAvailable = libraryAvailable;
  const ncclResult_t expected = libraryAvailable ? ncclSuccess : ncclSystemError;
  const ncclResult_t first = wrap_gdr_symbols();
  providerState.libraryAvailable = !libraryAvailable;
  const ncclResult_t second = wrap_gdr_symbols();
  if (first != expected || second != expected) {
    std::fprintf(stderr, "symbol result was not cached: first=%d second=%d expected=%d\n",
      first, second, expected);
    return 1;
  }
  if (providerState.libraryOpenCalls != 1) {
    std::fprintf(stderr, "cached symbol loader opened the library %d times\n", providerState.libraryOpenCalls);
    return 2;
  }
  return 0;
}

/**
 * @brief Verifies one semantic GDR pin-v2 version boundary through the production
 * availability query.
 * @param[in] major Runtime major version returned by the fake provider.
 * @param[in] minor Runtime minor version returned by the fake provider.
 * @param[in] expected Expected availability result for the configured version.
 * @return 0 on success or a distinct nonzero failure code.
 * @note Side effects: Initializes the public symbol and availability caches in an isolated process.
 */
int VerifyPinV2VersionForTest(int major, int minor, bool expected) {
  ResetProviderForTest();
  providerState.runtimeMajor = major;
  providerState.runtimeMinor = minor;
  const bool available = ncclGdrPinV2Available();
  const size_t versionCalls = static_cast<size_t>(std::count_if(providerState.calls.begin(),
    providerState.calls.end(), [](const ProviderCall& call) { return call.operation == "runtime_version"; }));
  if (available != expected || versionCalls != 1) {
    std::fprintf(stderr, "v2 availability mismatch for version %d.%d: available=%d calls=%zu\n",
      major, minor, available, versionCalls);
    return 1;
  }
  return 0;
}

/**
 * @brief Verifies one public v2 availability result is computed and cached once.
 * @param[in] initiallyAvailable Whether the first queried runtime version is 2.5 instead of 2.4.
 * @return 0 on success or a distinct nonzero failure code.
 * @note Side effects: Initializes symbols and the public v2 once flag in an isolated process.
 */
int VerifyPinV2CacheForTest(bool initiallyAvailable) {
  ResetProviderForTest();
  providerState.runtimeMajor = 2;
  providerState.runtimeMinor = initiallyAvailable ? 5 : 4;
  const bool first = ncclGdrPinV2Available();
  providerState.runtimeMinor = initiallyAvailable ? 4 : 5;
  const bool second = ncclGdrPinV2Available();
  const size_t versionCalls = static_cast<size_t>(std::count_if(providerState.calls.begin(),
    providerState.calls.end(), [](const ProviderCall& call) { return call.operation == "runtime_version"; }));
  if (first != initiallyAvailable || second != initiallyAvailable || versionCalls != 1) {
    std::fprintf(stderr, "public v2 availability did not cache %d: first=%d second=%d calls=%zu\n",
      initiallyAvailable, first, second, versionCalls);
    return 1;
  }
  return 0;
}

/**
 * @brief Verifies v2 availability remains false when the GDR library cannot be loaded.
 * @return 0 on success or a distinct nonzero failure code.
 * @note Side effects: Attempts one fake library load through the public v2 availability query.
 */
int VerifyPinV2LoadFailureForTest() {
  ResetProviderForTest();
  providerState.libraryAvailable = false;
  const bool first = ncclGdrPinV2Available();
  providerState.libraryAvailable = true;
  const bool second = ncclGdrPinV2Available();
  if (first || second) {
    std::fprintf(stderr, "v2 pinning was reported available after the GDR library failed to load\n");
    return 1;
  }
  if (providerState.libraryOpenCalls != 1 ||
      std::any_of(providerState.calls.begin(), providerState.calls.end(),
        [](const ProviderCall& call) { return call.operation == "runtime_version"; })) {
    std::fprintf(stderr, "failed v2 initialization did not stop after its single library-open attempt\n");
    return 2;
  }
  return 0;
}

/**
 * @brief Verifies v2 pinning is rejected when the runtime version is too old.
 * @return 0 on success or a distinct nonzero failure code.
 * @note Side effects: Initializes symbols and calls wrap_gdr_pin_buffer_v2 once in an isolated
 * process.
 */
int VerifyPinV2WrapperForTest() {
  ResetProviderForTest();
  providerState.runtimeMajor = 2;
  providerState.runtimeMinor = 4;
  gdr_mh_t handle = {0x1111};
  if (wrap_gdr_pin_buffer_v2(providerState.openResult, 0x23450000, 8192, 0x55, &handle) !=
      ncclInternalError || handle.h != 0x1111) {
    std::fprintf(stderr, "unavailable v2 pin did not return internal error without changing output\n");
    return 1;
  }
  if (std::any_of(providerState.calls.begin(), providerState.calls.end(),
      [](const ProviderCall& call) { return call.operation == "pin_v2"; })) {
    std::fprintf(stderr, "unavailable v2 pin reached the provider\n");
    return 2;
  }
  return 0;
}

/**
 * @brief Verifies provider failure and success after v2 pinning is available.
 * @return 0 on success or a distinct nonzero failure code.
 * @note Side effects: Initializes symbols and calls the fake v2 pin provider twice.
 */
int VerifyAvailablePinV2WrapperForTest() {
  ResetProviderForTest();
  providerState.runtimeMajor = 2;
  providerState.runtimeMinor = 5;
  gdr_mh_t handle = {0x1111};
  providerState.operationResult = EIO;
  if (wrap_gdr_pin_buffer_v2(providerState.openResult, 0x23450000, 8192, 0x55, &handle) !=
      ncclSystemError) {
    std::fprintf(stderr, "failing v2 provider did not return system error\n");
    return 1;
  }
  handle.h = 0x1111;
  providerState.operationResult = 0;
  if (wrap_gdr_pin_buffer_v2(providerState.openResult, 0x23450000, 8192, 0x55, &handle) !=
      ncclSuccess || handle.h != providerState.outputHandle.h) {
    std::fprintf(stderr, "successful v2 provider did not return its output handle\n");
    return 2;
  }
  const size_t pinCalls = static_cast<size_t>(std::count_if(providerState.calls.begin(),
    providerState.calls.end(), [](const ProviderCall& call) { return call.operation == "pin_v2"; }));
  if (pinCalls != 2) {
    std::fprintf(stderr, "available v2 pin reached the provider %zu times\n", pinCalls);
    return 3;
  }
  const ProviderCall& call = providerState.calls.back();
  if (call.argumentCount != 5 || call.arguments[0] != PointerValueForTest(providerState.openResult) ||
      call.arguments[1] != 0x23450000 || call.arguments[2] != 8192 || call.arguments[3] != 0x55 ||
      call.arguments[4] != PointerValueForTest(&handle)) {
    std::fprintf(stderr, "v2 pin did not forward every argument exactly\n");
    return 4;
  }
  return 0;
}

/**
 * @brief Verifies the last fake-provider call name and every argument.
 * @param[in] operation Valid non-NULL expected operation name.
 * @param[in] arguments Zero to six expected pointer-sized argument values.
 * @param[in] reason Valid non-NULL contract explanation printed on failure.
 * @note Side effects: Records GoogleTest failures when the call differs.
 */
void ExpectLastProviderCallForTest(const char* operation, std::initializer_list<uintptr_t> arguments,
  const char* reason) {
  SCOPED_TRACE(reason);
  ASSERT_FALSE(providerState.calls.empty()) << "The expected provider call was not recorded.";
  const ProviderCall& call = providerState.calls.back();
  EXPECT_EQ(call.operation, operation) << "The final provider operation has the wrong name.";
  ASSERT_EQ(call.argumentCount, arguments.size()) << "The provider received the wrong argument count.";
  size_t index = 0;
  for (uintptr_t argument : arguments) {
    EXPECT_EQ(call.arguments[index], argument) << "Provider argument " << index << " was not forwarded exactly.";
    ++index;
  }
}

/**
 * @brief Checks missing-symbol, provider-error, and successful integer-result dispatch.
 * @tparam Symbol Provider function-pointer type stored in slot.
 * @tparam Call Callable type that invokes the wrapper and returns ncclResult_t.
 * @param[in,out] slot Function-pointer slot temporarily cleared and restored to implementation.
 * @param[in] implementation Valid non-NULL fake provider function.
 * @param[in] call Callable invoking the wrapper with valid arguments.
 * @param[in] reason Valid non-NULL contract explanation printed on failure.
 * @param[in] checkMissingOutput Optional callback checking output after missing-symbol rejection.
 * @note Side effects: Calls the wrapper three times and updates provider result and call records.
 */
template <typename Symbol, typename Call>
void ExpectStatusDispatchForTest(Symbol& slot, Symbol implementation, Call call, const char* reason,
  const std::function<void()>& checkMissingOutput = {}) {
  SCOPED_TRACE(reason);
  providerState.calls.clear();
  slot = nullptr;
  EXPECT_EQ(call(), ncclInternalError) << "A missing mandatory provider symbol must return ncclInternalError.";
  if (checkMissingOutput) checkMissingOutput();
  EXPECT_TRUE(providerState.calls.empty()) << "A missing provider symbol cannot dispatch a provider call.";
  slot = implementation;
  providerState.operationResult = EIO;
  EXPECT_EQ(call(), ncclSystemError) << "A nonzero provider result must return ncclSystemError.";
  EXPECT_EQ(providerState.calls.size(), 1U) << "A provider failure must dispatch exactly one call.";
  providerState.operationResult = 0;
  EXPECT_EQ(call(), ncclSuccess) << "Provider result zero must return ncclSuccess.";
  EXPECT_EQ(providerState.calls.size(), 2U) << "The failure and success paths must dispatch one call each.";
}

/**
 * @brief Waits for one atomic flag without allowing the test to hang indefinitely.
 * @param[in] flag Atomic flag observed with acquire ordering.
 * @param[in] timeout Maximum wait duration in [0, std::chrono::milliseconds::max()].
 * @return true when flag became true before timeout; otherwise false.
 * @note Side effects: Yields the calling thread while waiting.
 */
bool WaitForFlagForTest(const std::atomic<bool>& flag, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return flag.load(std::memory_order_acquire);
}

/** @brief Provides isolated fake-provider state for GDR wrapper tests. */
class GdrTestBase : public ::testing::Test {
 protected:
  /**
   * @brief Restores fake provider state and directly configurable production symbols before
   * each test.
   * @note Side effects: Calls ResetProviderForTest.
   */
  void SetUp() override { ResetProviderForTest(); }
};

/** @brief Groups GDR dynamic-loader scenarios. */
class GdrLoaderTest : public GdrTestBase {};
/** @brief Groups GDR handle, pin, map, and unmap scenarios. */
class GdrMemoryTest : public GdrTestBase {};
/** @brief Groups GDR version-query scenarios. */
class GdrVersionTest : public GdrTestBase {};
/** @brief Groups GDR copy and serialization scenarios. */
class GdrCopyTest : public GdrTestBase {};

/**
 * @brief Verifies loader success, optional-symbol absence, library absence, and required-symbol
 * failure.
 * @note Side effects: Runs three isolated child processes using the fake dynamic loader.
 */
TEST_F(GdrLoaderTest, LoadsAndCachesSymbols) {
  SCOPED_TRACE("The loader must distinguish optional absence from failures and cache its first completed result.");

  EXPECT_EXIT({ _exit(VerifyDirectLoaderForTest()); }, ::testing::ExitedWithCode(0), "")
    << "Direct initialization must load all valid symbols and tear down every required-symbol failure.";
  EXPECT_EXIT({ _exit(VerifyCachedLoaderForTest(true)); }, ::testing::ExitedWithCode(0), "")
    << "A successful public symbol load must be cached exactly once.";
  EXPECT_EXIT({ _exit(VerifyCachedLoaderForTest(false)); }, ::testing::ExitedWithCode(0), "")
    << "A failed public symbol load must be returned unchanged and cached exactly once.";
}

/**
 * @brief Verifies open, pin, info, map, unmap, unpin, and close dispatch in lifecycle order.
 * @note Side effects: Calls the fake memory provider and writes handle, information, and
 * mapping outputs.
 */
TEST_F(GdrMemoryTest, ManagesPinnedMemory) {
  SCOPED_TRACE(
      "Memory wrappers must translate statuses, forward arguments, preserve rejected outputs, "
      "and expose successful outputs.");

  gdr_internal_open = nullptr;
  EXPECT_EQ(wrap_gdr_open(), nullptr) << "A missing open symbol must return NULL without calling a provider.";
  EXPECT_TRUE(providerState.calls.empty()) << "A missing open symbol cannot dispatch a provider call.";
  gdr_internal_open = OpenProviderForTest;
  providerState.openResult = nullptr;
  EXPECT_EQ(wrap_gdr_open(), nullptr) << "The open wrapper must preserve a provider NULL result.";
  EXPECT_EQ(providerState.calls.size(), 1U) << "A provider NULL result must originate from one open call.";
  providerState.openResult = reinterpret_cast<gdr_t>(&handleToken);
  gdr_t context = wrap_gdr_open();
  EXPECT_EQ(context, providerState.openResult) << "The open wrapper must return the exact provider context.";
  ExpectLastProviderCallForTest("open", {}, "Successful context opening must call the provider without arguments.");

  gdr_mh_t handle = {0x1111};
  const gdr_mh_t handleBeforeMissingSymbol = handle;
  ExpectStatusDispatchForTest(gdr_internal_pin_buffer, PinProviderForTest,
    [&] { return wrap_gdr_pin_buffer(context, 0x12340000, 4096, 0x55667788, 17, &handle); },
    "Legacy pinning must translate every status and write only the successful output.", [&] {
      EXPECT_EQ(handle.h, handleBeforeMissingSymbol.h)
        << "Missing pin symbols must leave handle output unchanged.";
    });
  EXPECT_EQ(handle.h, providerState.outputHandle.h) << "Successful pinning must expose the provider handle.";
  ExpectLastProviderCallForTest("pin", {PointerValueForTest(context), 0x12340000, 4096,
    0x55667788, 17, PointerValueForTest(&handle)}, "Legacy pinning must forward every argument exactly.");

  gdr_info_t info = {};
  info.va = 0xdead;
  const gdr_info_t infoBeforeMissingSymbol = info;
  ExpectStatusDispatchForTest(gdr_internal_get_info, GetInfoProviderForTest,
    [&] { return wrap_gdr_get_info(context, handle, &info); },
    "Information lookup must translate every status and expose successful provider output.", [&] {
      ExpectInfoEqualsForTest(info, infoBeforeMissingSymbol,
        "A missing information symbol must preserve the complete output record.");
    });
  ExpectInfoEqualsForTest(info, providerState.outputInfo,
    "Successful information lookup must expose the complete provider record.");
  ExpectLastProviderCallForTest("get_info", {PointerValueForTest(context), handle.h,
    PointerValueForTest(&info)}, "Information lookup must forward context, handle, and output exactly.");

  void* mapping = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
  const void* mappingBeforeMissingSymbol = mapping;
  ExpectStatusDispatchForTest(gdr_internal_map, MapProviderForTest,
    [&] { return wrap_gdr_map(context, handle, &mapping, mappedStorage.size()); },
    "Mapping must translate every status and expose the provider address only on success.", [&] {
      EXPECT_EQ(mapping, mappingBeforeMissingSymbol)
        << "Missing mapping symbols must leave address output unchanged.";
    });
  EXPECT_EQ(mapping, mappedStorage.data()) << "Successful mapping must return the exact provider address.";
  ExpectLastProviderCallForTest("map", {PointerValueForTest(context), handle.h,
    PointerValueForTest(&mapping), mappedStorage.size()}, "Mapping must forward every argument exactly.");

  ExpectStatusDispatchForTest(gdr_internal_unmap, UnmapProviderForTest,
    [&] { return wrap_gdr_unmap(context, handle, mapping, mappedStorage.size()); },
    "Unmapping must translate every provider status.");
  ExpectLastProviderCallForTest("unmap", {PointerValueForTest(context), handle.h,
    PointerValueForTest(mapping), mappedStorage.size()}, "Unmapping must forward every argument exactly.");

  ExpectStatusDispatchForTest(gdr_internal_unpin_buffer, UnpinProviderForTest,
    [&] { return wrap_gdr_unpin_buffer(context, handle); },
    "Unpinning must translate every provider status.");
  ExpectLastProviderCallForTest("unpin", {PointerValueForTest(context), handle.h},
    "Unpinning must forward context and handle exactly.");

  ExpectStatusDispatchForTest(gdr_internal_close, CloseProviderForTest,
    [&] { return wrap_gdr_close(context); }, "Closing must translate every provider status.");
  ExpectLastProviderCallForTest("close", {PointerValueForTest(context)},
    "Closing must forward the exact context after all dependent resources are released.");
}

/**
 * @brief Verifies version outputs and pin-v2 boundary, caching, load-failure, and dispatch
 * outcomes.
 * @note Side effects: Calls fake version providers and runs nine isolated v2 child processes.
 */
TEST_F(GdrVersionTest, ReportsVersionsAndPinV2) {
  SCOPED_TRACE("Version wrappers and the 2.5 v2 gate must return exact outputs and cache only completed decisions.");

  int runtimeMajor = -1;
  int runtimeMinor = -2;
  EXPECT_EQ(wrap_gdr_runtime_get_version(&runtimeMajor, &runtimeMinor), ncclInternalError)
    << "A missing runtime-version symbol must return ncclInternalError.";
  EXPECT_EQ(runtimeMajor, -1) << "Missing runtime-version symbols must preserve the major output.";
  EXPECT_EQ(runtimeMinor, -2) << "Missing runtime-version symbols must preserve the minor output.";
  EXPECT_TRUE(providerState.calls.empty()) << "A missing runtime-version symbol cannot call a provider.";
  gdr_internal_runtime_get_version = RuntimeVersionProviderForTest;
  EXPECT_EQ(wrap_gdr_runtime_get_version(&runtimeMajor, &runtimeMinor), ncclSuccess)
    << "An installed runtime-version provider must return ncclSuccess.";
  EXPECT_EQ(runtimeMajor, providerState.runtimeMajor) << "Runtime major output must match the provider.";
  EXPECT_EQ(runtimeMinor, providerState.runtimeMinor) << "Runtime minor output must match the provider.";
  ExpectLastProviderCallForTest("runtime_version", {PointerValueForTest(&runtimeMajor),
    PointerValueForTest(&runtimeMinor)}, "Runtime-version lookup must forward both outputs exactly.");

  int driverMajor = -3;
  int driverMinor = -4;
  const size_t callsBeforeMissingDriverVersion = providerState.calls.size();
  EXPECT_EQ(wrap_gdr_driver_get_version(providerState.openResult, &driverMajor, &driverMinor), ncclInternalError)
    << "A missing driver-version symbol must return ncclInternalError.";
  EXPECT_EQ(driverMajor, -3) << "Missing driver-version symbols must preserve the major output.";
  EXPECT_EQ(driverMinor, -4) << "Missing driver-version symbols must preserve the minor output.";
  EXPECT_EQ(providerState.calls.size(), callsBeforeMissingDriverVersion)
    << "A missing driver-version symbol cannot call a provider.";
  gdr_internal_driver_get_version = DriverVersionProviderForTest;
  EXPECT_EQ(wrap_gdr_driver_get_version(providerState.openResult, &driverMajor, &driverMinor), ncclSuccess)
    << "An installed driver-version provider must return ncclSuccess.";
  EXPECT_EQ(driverMajor, providerState.driverMajor) << "Driver major output must match the provider.";
  EXPECT_EQ(driverMinor, providerState.driverMinor) << "Driver minor output must match the provider.";
  ExpectLastProviderCallForTest("driver_version", {PointerValueForTest(providerState.openResult),
    PointerValueForTest(&driverMajor), PointerValueForTest(&driverMinor)},
    "Driver-version lookup must forward context and both outputs exactly.");
  EXPECT_EXIT({ _exit(VerifyPinV2VersionForTest(1, 99, false)); }, ::testing::ExitedWithCode(0), "")
    << "Pin-v2 availability must reject major versions below 2.";
  EXPECT_EXIT({ _exit(VerifyPinV2VersionForTest(2, 4, false)); }, ::testing::ExitedWithCode(0), "")
    << "Pin-v2 availability must reject version 2.4.";
  EXPECT_EXIT({ _exit(VerifyPinV2VersionForTest(2, 5, true)); }, ::testing::ExitedWithCode(0), "")
    << "Pin-v2 availability must accept the minimum supported version 2.5.";
  EXPECT_EXIT({ _exit(VerifyPinV2VersionForTest(3, 0, true)); }, ::testing::ExitedWithCode(0), "")
    << "Pin-v2 availability must accept a runtime with a later major version.";
  EXPECT_EXIT({ _exit(VerifyPinV2CacheForTest(true)); }, ::testing::ExitedWithCode(0), "")
    << "Public pin-v2 availability must cache a true version decision.";
  EXPECT_EXIT({ _exit(VerifyPinV2CacheForTest(false)); }, ::testing::ExitedWithCode(0), "")
    << "Public pin-v2 availability must cache a false version decision.";
  EXPECT_EXIT({ _exit(VerifyPinV2LoadFailureForTest()); }, ::testing::ExitedWithCode(0), "")
    << "Pin-v2 availability must remain false when the GDR library cannot be loaded.";
  EXPECT_EXIT({ _exit(VerifyPinV2WrapperForTest()); }, ::testing::ExitedWithCode(0), "")
    << "Unavailable pin-v2 must return ncclInternalError without changing output or calling the provider.";
  EXPECT_EXIT({ _exit(VerifyAvailablePinV2WrapperForTest()); }, ::testing::ExitedWithCode(0), "")
    << "Available pin-v2 must translate provider failure and return the successful handle and arguments.";
}

/**
 * @brief Verifies bidirectional mapping copies and that copy-to holds the shared GDR mutex.
 * @note Side effects: Mutates fake provider symbols, call records, mapping bytes, and
 * synchronization state; one copy-to provider call runs on a worker thread.
 */
TEST_F(GdrCopyTest, CopiesAndSerializesMappings) {
  SCOPED_TRACE("Copy wrappers must translate statuses, preserve sources, copy exact bytes, and serialize providers.");

  gdr_mh_t handle = {0x7788};
  std::array<unsigned char, 16> hostSource = {};
  for (size_t index = 0; index < hostSource.size(); ++index) {
    hostSource[index] = static_cast<unsigned char>(index + 1);
  }
  const std::array<unsigned char, 16> hostSourceBefore = hostSource;
  std::array<unsigned char, 16> mapping = {};
  const std::array<unsigned char, 16> mappingBeforeMissingSymbol = mapping;
  ExpectStatusDispatchForTest(gdr_internal_copy_to_mapping, CopyToProviderForTest,
    [&] { return wrap_gdr_copy_to_mapping(handle, mapping.data(), hostSource.data(), hostSource.size()); },
    "Host-to-mapping copy must translate all statuses and copy only on success.", [&] {
      EXPECT_EQ(mapping, mappingBeforeMissingSymbol)
        << "Missing copy-to symbols must leave destination bytes unchanged.";
    });
  EXPECT_EQ(mapping, hostSource) << "Successful copy-to must reproduce every source byte in the mapping.";
  EXPECT_EQ(hostSource, hostSourceBefore) << "Copy-to must not change its host source bytes.";
  ExpectLastProviderCallForTest("copy_to", {handle.h, PointerValueForTest(mapping.data()),
    PointerValueForTest(hostSource.data()), hostSource.size()}, "Copy-to must forward every argument exactly.");

  std::array<unsigned char, 16> hostDestination = {};
  const std::array<unsigned char, 16> hostBeforeMissingSymbol = hostDestination;
  const std::array<unsigned char, 16> mappingBeforeCopyFrom = mapping;
  ExpectStatusDispatchForTest(gdr_internal_copy_from_mapping, CopyFromProviderForTest,
    [&] { return wrap_gdr_copy_from_mapping(handle, hostDestination.data(), mapping.data(), mapping.size()); },
    "Mapping-to-host copy must translate all statuses and copy only on success.", [&] {
      EXPECT_EQ(hostDestination, hostBeforeMissingSymbol)
        << "Missing copy-from symbols must leave destination bytes unchanged.";
    });
  EXPECT_EQ(hostDestination, mapping) << "Successful copy-from must reproduce every mapping byte on the host.";
  EXPECT_EQ(mapping, mappingBeforeCopyFrom) << "Copy-from must not change its mapping source bytes.";
  ExpectLastProviderCallForTest("copy_from", {handle.h, PointerValueForTest(hostDestination.data()),
    PointerValueForTest(mapping.data()), mapping.size()}, "Copy-from must forward every argument exactly.");

  EXPECT_EQ(&getGdrMutex(), &getGdrMutex())
      << "Repeated getGdrMutex calls must return the same process-wide mutex.";
  gdr_internal_copy_to_mapping = BlockingCopyToProviderForTest;
  std::array<unsigned char, 4> blockingSource = {{9, 8, 7, 6}};
  std::array<unsigned char, 4> blockingDestination = {};
  ncclResult_t workerResult = ncclInternalError;
  /**
   * @brief Runs one blocking copy through the serialized GDR wrapper.
   * @note Side effects: Updates workerResult and the fake destination through the provider.
   */
  std::thread worker([&] {
    workerResult = wrap_gdr_copy_to_mapping(handle, blockingDestination.data(), blockingSource.data(),
      blockingSource.size());
  });
  const bool entered = WaitForFlagForTest(blockingProviderEntered, std::chrono::milliseconds(500));
  bool acquiredWhileProviderRan = false;
  if (entered) {
    acquiredWhileProviderRan = getGdrMutex().try_lock();
    if (acquiredWhileProviderRan) getGdrMutex().unlock();
  }
  releaseBlockingProvider.store(true, std::memory_order_release);
  worker.join();
  ASSERT_TRUE(entered) << "The blocking copy provider must start before the bounded wait expires.";
  EXPECT_FALSE(acquiredWhileProviderRan) << "The shared GDR mutex must remain locked during the provider call.";
  EXPECT_EQ(workerResult, ncclSuccess) << "The serialized provider result zero must return ncclSuccess.";
  EXPECT_EQ(blockingProviderCalls.load(std::memory_order_relaxed), 1)
    << "The serialized copy must reach the provider exactly once.";
  EXPECT_EQ(blockingDestination, blockingSource)
    << "The serialized copy must still produce the exact destination bytes.";
}

/**
 * @brief Verifies the architecture write-combining fence is directly executable between
 * caller-owned stores.
 */
TEST(GdrFenceTest, ExecutesWithoutChangingStoredValues) {
  SCOPED_TRACE("The write-combining fence must complete without changing surrounding caller-owned values.");
  volatile uint32_t beforeFence = 0x12345678u;
  volatile uint32_t afterFence = 0;

  EXPECT_NO_FATAL_FAILURE(wc_store_fence())
      << "The architecture-specific write-combining fence must be executable on the target architecture";
  afterFence = 0xabcdef01u;

  EXPECT_EQ(beforeFence, 0x12345678u)
      << "The fence must not modify data stored before it";
  EXPECT_EQ(afterFence, 0xabcdef01u)
      << "The fence must not modify data stored after it";
}

}  // namespace
