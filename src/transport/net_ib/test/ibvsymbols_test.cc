/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "ibvsymbols.h"

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace {

/** @brief Describes one verbs symbol's name, ABI version, and requirement status. */
struct SymbolSpec {
  const char* name;
  const char* version;
  bool mandatory;
};

constexpr std::array<SymbolSpec, 26> kSymbolSpecs = {{
  {"ibv_get_device_list", "IBVERBS_1.1", true},
  {"ibv_free_device_list", "IBVERBS_1.1", true},
  {"ibv_get_device_name", "IBVERBS_1.1", true},
  {"ibv_open_device", "IBVERBS_1.1", true},
  {"ibv_close_device", "IBVERBS_1.1", true},
  {"ibv_get_async_event", "IBVERBS_1.1", true},
  {"ibv_ack_async_event", "IBVERBS_1.1", true},
  {"ibv_query_device", "IBVERBS_1.1", true},
  {"ibv_query_port", "IBVERBS_1.1", true},
  {"ibv_query_gid", "IBVERBS_1.1", true},
  {"ibv_query_qp", "IBVERBS_1.1", true},
  {"ibv_alloc_pd", "IBVERBS_1.1", true},
  {"ibv_dealloc_pd", "IBVERBS_1.1", true},
  {"ibv_reg_mr", "IBVERBS_1.1", true},
  {"ibv_reg_mr_iova2", "IBVERBS_1.8", false},
  {"ibv_reg_dmabuf_mr", "IBVERBS_1.12", false},
  {"ibv_dereg_mr", "IBVERBS_1.1", true},
  {"ibv_create_cq", "IBVERBS_1.1", true},
  {"ibv_destroy_cq", "IBVERBS_1.1", true},
  {"ibv_create_qp", "IBVERBS_1.1", true},
  {"ibv_modify_qp", "IBVERBS_1.1", true},
  {"ibv_destroy_qp", "IBVERBS_1.1", true},
  {"ibv_fork_init", "IBVERBS_1.1", true},
  {"ibv_event_type_str", "IBVERBS_1.1", true},
  {"ibv_query_ece", "IBVERBS_1.10", false},
  {"ibv_set_ece", "IBVERBS_1.10", false},
}};

/** @brief Selects the fake verbs library-open outcome. */
enum class LibraryMode {
  Retained,
  Primary,
  Fallback,
  Unavailable,
};

/** @brief Records one dynamic-library open request. */
struct OpenCall {
  std::string name;
  int flags;
};

/** @brief Records one versioned dynamic-symbol lookup. */
struct ResolveCall {
  void* handle;
  std::string name;
  std::string version;
};

/** @brief Scripts library loading and records all dynamic-loader calls. */
struct LoaderState {
  LibraryMode mode = LibraryMode::Primary;
  std::vector<std::string> missingSymbols;
  std::vector<OpenCall> openCalls;
  std::vector<ResolveCall> resolveCalls;
  std::vector<void*> closeCalls;
  unsigned char primaryHandle = 0;
  unsigned char fallbackHandle = 0;
  std::array<unsigned char, kSymbolSpecs.size()> symbolAddresses = {};
};

LoaderState loaderState;

/**
 * @brief Resets the fake dynamic loader to a successful primary-library configuration.
 * @note Side effects: Replaces loaderState and assigns distinct bytes to every fake symbol address.
 */
void ResetLoaderForTest() {
  loaderState = LoaderState{};
  for (size_t index = 0; index < loaderState.symbolAddresses.size(); ++index) {
    loaderState.symbolAddresses[index] = static_cast<unsigned char>(index + 1);
  }
}

/**
 * @brief Finds a symbol specification by its exact name.
 * @param[in] name Valid non-NULL null-terminated symbol name.
 * @return Its index in kSymbolSpecs, or kSymbolSpecs.size() when it is unknown.
 * @note Side effects: None.
 */
size_t FindSymbolForTest(const char* name) {
  for (size_t index = 0; index < kSymbolSpecs.size(); ++index) {
    if (std::strcmp(kSymbolSpecs[index].name, name) == 0) return index;
  }
  return kSymbolSpecs.size();
}

/**
 * @brief Reports whether the configured fake loader must hide one symbol.
 * @param[in] name Valid non-NULL null-terminated symbol name.
 * @return true when name appears in missingSymbols; otherwise false.
 * @note Side effects: None.
 */
bool IsSymbolMissingForTest(const char* name) {
  for (const std::string& missing : loaderState.missingSymbols) {
    if (missing == name) return true;
  }
  return false;
}

/**
 * @brief Returns the stable fake address assigned to one known symbol.
 * @param[in] name Valid non-NULL symbol name present in kSymbolSpecs.
 * @return A distinct non-NULL address for name.
 * @note Side effects: Records a GoogleTest failure if name is unknown.
 */
void* SymbolAddressForTest(const char* name) {
  const size_t index = FindSymbolForTest(name);
  EXPECT_LT(index, kSymbolSpecs.size()) << "The test requested an unknown fake symbol: " << name;
  return index < kSymbolSpecs.size() ? &loaderState.symbolAddresses[index] : nullptr;
}

/**
 * @brief Models a retained versioned library and both ordinary soname-loading paths.
 * @param[in] name Valid non-NULL null-terminated library name.
 * @param[in] flags Dynamic-loader flags forwarded by production code.
 * @return The configured fake handle, or NULL when that library must be unavailable.
 * @note Side effects: Appends the request to openCalls.
 */
void* OpenLibraryForTest(const char* name, int flags) {
  loaderState.openCalls.push_back({name, flags});
  if (std::strcmp(name, "libibverbs.so") == 0) {
    return loaderState.mode == LibraryMode::Primary ? &loaderState.primaryHandle : nullptr;
  }
  if (std::strcmp(name, "libibverbs.so.1") == 0) {
    if (loaderState.mode == LibraryMode::Retained && flags == (RTLD_NOW | RTLD_NOLOAD)) {
      return &loaderState.fallbackHandle;
    }
    if (loaderState.mode == LibraryMode::Fallback && flags == RTLD_NOW) {
      return &loaderState.fallbackHandle;
    }
  }
  return nullptr;
}

/**
 * @brief Implements deterministic versioned symbol lookup.
 * @param[in] handle Valid non-NULL primary or fallback fake library handle.
 * @param[in] name Valid non-NULL null-terminated symbol name.
 * @param[in] version Valid non-NULL null-terminated ABI version.
 * @return The symbol's fake address when name and version match and it is enabled; otherwise NULL.
 * @note Side effects: Appends the request to resolveCalls.
 */
void* ResolveVersionedSymbolForTest(void* handle, const char* name, const char* version) {
  loaderState.resolveCalls.push_back({handle, name, version});
  const size_t index = FindSymbolForTest(name);
  if (index == kSymbolSpecs.size()) return nullptr;
  if (std::strcmp(kSymbolSpecs[index].version, version) != 0) return nullptr;
  if (IsSymbolMissingForTest(name)) return nullptr;
  return &loaderState.symbolAddresses[index];
}

/**
 * @brief Supplies a stable diagnostic string for mandatory symbol failures.
 * @return A valid mutable null-terminated diagnostic string.
 * @note Side effects: None.
 */
char* LoaderErrorForTest() {
  static char message[] = "controlled loader failure";
  return message;
}

/**
 * @brief Implements deterministic library closure.
 * @param[in] handle Valid non-NULL fake library handle.
 * @return 0.
 * @note Side effects: Appends handle to closeCalls.
 */
int CloseLibraryForTest(void* handle) {
  loaderState.closeCalls.push_back(handle);
  return 0;
}

/**
 * @brief Copies one POSIX `dlsym` address into a function-pointer field for test setup.
 * @tparam Symbol Function-pointer field type.
 * @param[out] field Function-pointer field receiving address.
 * @param[in] address Any fake dynamic-loader address, including NULL.
 * @note Side effects: Replaces field.
 */
template <typename Symbol>
void SetSymbolFieldForTest(Symbol& field, void* address) {
  static_assert(sizeof(field) == sizeof(address), "The test requires POSIX-compatible pointer representations.");
  std::memcpy(&field, &address, sizeof(field));
}

/**
 * @brief Reads the POSIX `dlsym` address stored in one function-pointer field.
 * @tparam Symbol Function-pointer field type.
 * @param[in] field Any function-pointer field.
 * @return Its object-pointer representation.
 * @note Side effects: None.
 */
template <typename Symbol>
void* ReadSymbolFieldForTest(Symbol field) {
  static_assert(sizeof(field) == sizeof(void*), "The test requires POSIX-compatible pointer representations.");
  void* address = nullptr;
  std::memcpy(&address, &field, sizeof(field));
  return address;
}

/**
 * @brief Verifies one loaded symbol field against its configured fake address.
 * @tparam Symbol Function-pointer field type under comparison.
 * @param[in] field Function-pointer field populated by production code.
 * @param[in] name Valid non-NULL symbol name present in kSymbolSpecs.
 * @param[in] expectedLoaded Whether the field must contain the fake address instead of NULL.
 * @note Side effects: Records a GoogleTest failure when the field differs.
 */
template <typename Symbol>
void ExpectSymbolFieldForTest(Symbol field, const char* name, bool expectedLoaded) {
  SCOPED_TRACE(std::string("symbol field: ") + name);
  void* expected = expectedLoaded ? SymbolAddressForTest(name) : nullptr;
  EXPECT_EQ(ReadSymbolFieldForTest(field), expected)
    << "The symbol table field must exactly reflect the loader result for " << name;
}

/**
 * @brief Assigns non-NULL sentinels to every symbol-table field before a failure-path call.
 * @param[out] symbols Valid non-NULL symbol table receiving all fake addresses.
 * @note Side effects: Replaces all 26 fields in symbols.
 */
void FillSymbolTableForTest(struct ncclIbvSymbols* symbols) {
#define SET_TEST_SYMBOL(field, name) SetSymbolFieldForTest(symbols->field, SymbolAddressForTest(name))
  SET_TEST_SYMBOL(ibv_internal_get_device_list, "ibv_get_device_list");
  SET_TEST_SYMBOL(ibv_internal_free_device_list, "ibv_free_device_list");
  SET_TEST_SYMBOL(ibv_internal_get_device_name, "ibv_get_device_name");
  SET_TEST_SYMBOL(ibv_internal_open_device, "ibv_open_device");
  SET_TEST_SYMBOL(ibv_internal_close_device, "ibv_close_device");
  SET_TEST_SYMBOL(ibv_internal_get_async_event, "ibv_get_async_event");
  SET_TEST_SYMBOL(ibv_internal_ack_async_event, "ibv_ack_async_event");
  SET_TEST_SYMBOL(ibv_internal_query_device, "ibv_query_device");
  SET_TEST_SYMBOL(ibv_internal_query_port, "ibv_query_port");
  SET_TEST_SYMBOL(ibv_internal_query_gid, "ibv_query_gid");
  SET_TEST_SYMBOL(ibv_internal_query_qp, "ibv_query_qp");
  SET_TEST_SYMBOL(ibv_internal_alloc_pd, "ibv_alloc_pd");
  SET_TEST_SYMBOL(ibv_internal_dealloc_pd, "ibv_dealloc_pd");
  SET_TEST_SYMBOL(ibv_internal_reg_mr, "ibv_reg_mr");
  SET_TEST_SYMBOL(ibv_internal_reg_mr_iova2, "ibv_reg_mr_iova2");
  SET_TEST_SYMBOL(ibv_internal_reg_dmabuf_mr, "ibv_reg_dmabuf_mr");
  SET_TEST_SYMBOL(ibv_internal_dereg_mr, "ibv_dereg_mr");
  SET_TEST_SYMBOL(ibv_internal_create_cq, "ibv_create_cq");
  SET_TEST_SYMBOL(ibv_internal_destroy_cq, "ibv_destroy_cq");
  SET_TEST_SYMBOL(ibv_internal_create_qp, "ibv_create_qp");
  SET_TEST_SYMBOL(ibv_internal_modify_qp, "ibv_modify_qp");
  SET_TEST_SYMBOL(ibv_internal_destroy_qp, "ibv_destroy_qp");
  SET_TEST_SYMBOL(ibv_internal_fork_init, "ibv_fork_init");
  SET_TEST_SYMBOL(ibv_internal_event_type_str, "ibv_event_type_str");
  SET_TEST_SYMBOL(ibv_internal_query_ece, "ibv_query_ece");
  SET_TEST_SYMBOL(ibv_internal_set_ece, "ibv_set_ece");
#undef SET_TEST_SYMBOL
}

/**
 * @brief Verifies all symbol-table fields after a successful dynamic load.
 * @param[in] symbols Valid non-NULL populated symbol table.
 * @note Side effects: Records GoogleTest failures for incorrect fields.
 */
void ExpectLoadedSymbolTableForTest(const struct ncclIbvSymbols* symbols) {
#define EXPECT_TEST_SYMBOL(field, name, loaded) ExpectSymbolFieldForTest(symbols->field, name, loaded)
  EXPECT_TEST_SYMBOL(ibv_internal_get_device_list, "ibv_get_device_list", true);
  EXPECT_TEST_SYMBOL(ibv_internal_free_device_list, "ibv_free_device_list", true);
  EXPECT_TEST_SYMBOL(ibv_internal_get_device_name, "ibv_get_device_name", true);
  EXPECT_TEST_SYMBOL(ibv_internal_open_device, "ibv_open_device", true);
  EXPECT_TEST_SYMBOL(ibv_internal_close_device, "ibv_close_device", true);
  EXPECT_TEST_SYMBOL(ibv_internal_get_async_event, "ibv_get_async_event", true);
  EXPECT_TEST_SYMBOL(ibv_internal_ack_async_event, "ibv_ack_async_event", true);
  EXPECT_TEST_SYMBOL(ibv_internal_query_device, "ibv_query_device", true);
  EXPECT_TEST_SYMBOL(ibv_internal_query_port, "ibv_query_port", true);
  EXPECT_TEST_SYMBOL(ibv_internal_query_gid, "ibv_query_gid", true);
  EXPECT_TEST_SYMBOL(ibv_internal_query_qp, "ibv_query_qp", true);
  EXPECT_TEST_SYMBOL(ibv_internal_alloc_pd, "ibv_alloc_pd", true);
  EXPECT_TEST_SYMBOL(ibv_internal_dealloc_pd, "ibv_dealloc_pd", true);
  EXPECT_TEST_SYMBOL(ibv_internal_reg_mr, "ibv_reg_mr", true);
  EXPECT_TEST_SYMBOL(ibv_internal_reg_mr_iova2, "ibv_reg_mr_iova2", !IsSymbolMissingForTest("ibv_reg_mr_iova2"));
  EXPECT_TEST_SYMBOL(ibv_internal_reg_dmabuf_mr, "ibv_reg_dmabuf_mr",
    !IsSymbolMissingForTest("ibv_reg_dmabuf_mr"));
  EXPECT_TEST_SYMBOL(ibv_internal_dereg_mr, "ibv_dereg_mr", true);
  EXPECT_TEST_SYMBOL(ibv_internal_create_cq, "ibv_create_cq", true);
  EXPECT_TEST_SYMBOL(ibv_internal_destroy_cq, "ibv_destroy_cq", true);
  EXPECT_TEST_SYMBOL(ibv_internal_create_qp, "ibv_create_qp", true);
  EXPECT_TEST_SYMBOL(ibv_internal_modify_qp, "ibv_modify_qp", true);
  EXPECT_TEST_SYMBOL(ibv_internal_destroy_qp, "ibv_destroy_qp", true);
  EXPECT_TEST_SYMBOL(ibv_internal_fork_init, "ibv_fork_init", true);
  EXPECT_TEST_SYMBOL(ibv_internal_event_type_str, "ibv_event_type_str", true);
  EXPECT_TEST_SYMBOL(ibv_internal_query_ece, "ibv_query_ece", !IsSymbolMissingForTest("ibv_query_ece"));
  EXPECT_TEST_SYMBOL(ibv_internal_set_ece, "ibv_set_ece", !IsSymbolMissingForTest("ibv_set_ece"));
#undef EXPECT_TEST_SYMBOL
}

/**
 * @brief Verifies that teardown cleared every symbol-table field.
 * @param[in] symbols Valid non-NULL symbol table after a failed build.
 * @note Side effects: Records GoogleTest failures for non-NULL fields.
 */
void ExpectClearedSymbolTableForTest(const struct ncclIbvSymbols* symbols) {
  SCOPED_TRACE("Every symbol-table field must be NULL after teardown.");
  EXPECT_EQ(symbols->ibv_internal_get_device_list, nullptr);
  EXPECT_EQ(symbols->ibv_internal_free_device_list, nullptr);
  EXPECT_EQ(symbols->ibv_internal_get_device_name, nullptr);
  EXPECT_EQ(symbols->ibv_internal_open_device, nullptr);
  EXPECT_EQ(symbols->ibv_internal_close_device, nullptr);
  EXPECT_EQ(symbols->ibv_internal_get_async_event, nullptr);
  EXPECT_EQ(symbols->ibv_internal_ack_async_event, nullptr);
  EXPECT_EQ(symbols->ibv_internal_query_device, nullptr);
  EXPECT_EQ(symbols->ibv_internal_query_port, nullptr);
  EXPECT_EQ(symbols->ibv_internal_query_gid, nullptr);
  EXPECT_EQ(symbols->ibv_internal_query_qp, nullptr);
  EXPECT_EQ(symbols->ibv_internal_alloc_pd, nullptr);
  EXPECT_EQ(symbols->ibv_internal_dealloc_pd, nullptr);
  EXPECT_EQ(symbols->ibv_internal_reg_mr, nullptr);
  EXPECT_EQ(symbols->ibv_internal_reg_mr_iova2, nullptr);
  EXPECT_EQ(symbols->ibv_internal_reg_dmabuf_mr, nullptr);
  EXPECT_EQ(symbols->ibv_internal_dereg_mr, nullptr);
  EXPECT_EQ(symbols->ibv_internal_create_cq, nullptr);
  EXPECT_EQ(symbols->ibv_internal_destroy_cq, nullptr);
  EXPECT_EQ(symbols->ibv_internal_create_qp, nullptr);
  EXPECT_EQ(symbols->ibv_internal_modify_qp, nullptr);
  EXPECT_EQ(symbols->ibv_internal_destroy_qp, nullptr);
  EXPECT_EQ(symbols->ibv_internal_fork_init, nullptr);
  EXPECT_EQ(symbols->ibv_internal_event_type_str, nullptr);
  EXPECT_EQ(symbols->ibv_internal_query_ece, nullptr);
  EXPECT_EQ(symbols->ibv_internal_set_ece, nullptr);
}

/**
 * @brief Verifies the names, ABI versions, and handle used by symbol-resolution requests.
 * @param[in] expectedHandle Valid non-NULL fake library handle expected for every request.
 * @note Side effects: Records GoogleTest failures for missing, unknown, or incorrectly
 * versioned requests.
 */
void ExpectResolutionRequestsForTest(void* expectedHandle) {
  for (const ResolveCall& call : loaderState.resolveCalls) {
    SCOPED_TRACE(std::string("actual resolution request: ") + call.name);
    const size_t index = FindSymbolForTest(call.name.c_str());
    ASSERT_LT(index, kSymbolSpecs.size()) << "The loader must not request an unknown symbol.";
    EXPECT_EQ(call.handle, expectedHandle) << "Every request must use the handle returned by dlopen.";
    EXPECT_EQ(call.version, kSymbolSpecs[index].version)
      << "Every request must use the symbol's required ABI version.";
  }
  for (const SymbolSpec& spec : kSymbolSpecs) {
    SCOPED_TRACE(std::string("resolution request: ") + spec.name);
    size_t matchingCalls = 0;
    for (const ResolveCall& call : loaderState.resolveCalls) {
      if (call.name != spec.name) continue;
      ++matchingCalls;
      EXPECT_EQ(call.handle, expectedHandle) << "The request must use the handle returned by dlopen.";
      EXPECT_EQ(call.version, spec.version) << "The request must use the symbol's required ABI version.";
    }
    EXPECT_GT(matchingCalls, 0U) << "Every configured symbol must be requested.";
  }
}

/** @brief Provides an isolated successful loader for dynamic verbs-symbol tests. */
class DynamicSymbolTest : public ::testing::Test {
 protected:
  /**
   * @brief Restores a successful fake loader before each test.
   * @note Side effects: Resets loaderState.
   */
  void SetUp() override { ResetLoaderForTest(); }
};

}  // namespace

/**
 * @brief Compiles ibvsymbols.cc with deterministic loader seams, then undefines them.
 */
#define dlopen OpenLibraryForTest
#define dlvsym ResolveVersionedSymbolForTest
#define dlerror LoaderErrorForTest
#define dlclose CloseLibraryForTest
#include "../../../misc/ibvsymbols.cc"  // NOLINT(build/include)
#undef dlclose
#undef dlerror
#undef dlvsym
#undef dlopen
#undef LOAD_SYM_VERSION
#undef LOAD_SYM
#undef IBVERBS_VERSION

namespace {

/**
 * @brief Verifies reuse of a versioned verbs runtime already present in the link map.
 * @note Side effects: Calls the fake dynamic loader and writes a local symbol table.
 */
TEST_F(DynamicSymbolTest, ReusesRetainedVersionedLibrary) {
  SCOPED_TRACE("A retained versioned runtime must be reused before either ordinary load path.");

  loaderState.mode = LibraryMode::Retained;
  struct ncclIbvSymbols symbols = {};
  EXPECT_EQ(buildIbvSymbols(&symbols), ncclSuccess)
      << "A retained versioned verbs runtime was not reused.";
  ASSERT_EQ(loaderState.openCalls.size(), 1U) << "Retained reuse must require one probe.";
  EXPECT_EQ(loaderState.openCalls[0].name, "libibverbs.so.1");
  EXPECT_EQ(loaderState.openCalls[0].flags, RTLD_NOW | RTLD_NOLOAD);
  ExpectResolutionRequestsForTest(&loaderState.fallbackHandle);
  ExpectLoadedSymbolTableForTest(&symbols);
  EXPECT_TRUE(loaderState.closeCalls.empty()) << "A retained process-lifetime handle was closed.";
}

/**
 * @brief Verifies primary-library loading, exact ABI requests, and complete table population.
 * @note Side effects: Calls the fake dynamic loader and writes a local symbol table.
 */
TEST_F(DynamicSymbolTest, LoadsPrimaryLibrary) {
  SCOPED_TRACE("A complete primary libibverbs image must populate every field without closing its live handle.");

  struct ncclIbvSymbols symbols = {};
  EXPECT_EQ(buildIbvSymbols(&symbols), ncclSuccess)
    << "A library containing every required symbol must load successfully.";
  ASSERT_EQ(loaderState.openCalls.size(), 2U)
      << "Primary loading must follow one unsuccessful retained-runtime probe.";
  EXPECT_EQ(loaderState.openCalls[0].name, "libibverbs.so.1");
  EXPECT_EQ(loaderState.openCalls[0].flags, RTLD_NOW | RTLD_NOLOAD);
  EXPECT_EQ(loaderState.openCalls[1].name, "libibverbs.so") << "The primary library name must be exact.";
  EXPECT_EQ(loaderState.openCalls[1].flags, RTLD_NOW)
      << "The primary library must resolve dependencies immediately.";
  ExpectResolutionRequestsForTest(&loaderState.primaryHandle);
  ExpectLoadedSymbolTableForTest(&symbols);
  EXPECT_TRUE(loaderState.closeCalls.empty()) << "A successfully loaded process-lifetime handle must remain open.";
}

/**
 * @brief Verifies fallback loading with individual or all optional symbols absent.
 * @note Side effects: Calls the fake dynamic loader and writes a local symbol table.
 */
TEST_F(DynamicSymbolTest, UsesFallbackAndOptionalSymbols) {
  SCOPED_TRACE("The fallback soname must work, and unavailable versioned extensions must remain optional.");

  std::vector<std::vector<std::string>> missingSets;
  std::vector<std::string> allOptionalSymbols;
  for (const SymbolSpec& spec : kSymbolSpecs) {
    if (spec.mandatory) continue;
    missingSets.push_back({spec.name});
    allOptionalSymbols.emplace_back(spec.name);
  }
  missingSets.push_back(allOptionalSymbols);

  for (const std::vector<std::string>& missingSymbols : missingSets) {
    SCOPED_TRACE(testing::Message()
        << "missing optional symbols: count=" << missingSymbols.size()
        << ", first=" << missingSymbols.front());
    ResetLoaderForTest();
    loaderState.mode = LibraryMode::Fallback;
    loaderState.missingSymbols = missingSymbols;
    struct ncclIbvSymbols symbols = {};
    EXPECT_EQ(buildIbvSymbols(&symbols), ncclSuccess)
      << "Missing optional extensions must not reject an otherwise complete fallback library.";
    ASSERT_EQ(loaderState.openCalls.size(), 3U)
        << "Fallback loading must probe retained state before both ordinary sonames.";
    EXPECT_EQ(loaderState.openCalls[0].name, "libibverbs.so.1");
    EXPECT_EQ(loaderState.openCalls[0].flags, RTLD_NOW | RTLD_NOLOAD);
    EXPECT_EQ(loaderState.openCalls[1].name, "libibverbs.so");
    EXPECT_EQ(loaderState.openCalls[1].flags, RTLD_NOW);
    EXPECT_EQ(loaderState.openCalls[2].name, "libibverbs.so.1");
    EXPECT_EQ(loaderState.openCalls[2].flags, RTLD_NOW);
    ExpectResolutionRequestsForTest(&loaderState.fallbackHandle);
    ExpectLoadedSymbolTableForTest(&symbols);
    EXPECT_TRUE(loaderState.closeCalls.empty()) << "A successfully loaded fallback handle must remain open.";
  }
}

/**
 * @brief Verifies complete output clearing when neither supported library name can be opened.
 * @note Side effects: Calls the fake dynamic loader and clears a local symbol table.
 */
TEST_F(DynamicSymbolTest, ClearsTableWhenLibraryIsUnavailable) {
  SCOPED_TRACE("Total library-open failure must return an error and clear every pre-existing function pointer.");

  loaderState.mode = LibraryMode::Unavailable;
  struct ncclIbvSymbols symbols = {};
  FillSymbolTableForTest(&symbols);
  EXPECT_EQ(buildIbvSymbols(&symbols), ncclSystemError)
    << "Failure to open both supported sonames must return ncclSystemError.";
  ASSERT_EQ(loaderState.openCalls.size(), 3U)
      << "Total absence must include the retained probe and both ordinary load attempts.";
  EXPECT_EQ(loaderState.openCalls[0].name, "libibverbs.so.1");
  EXPECT_EQ(loaderState.openCalls[0].flags, RTLD_NOW | RTLD_NOLOAD);
  EXPECT_EQ(loaderState.openCalls[1].name, "libibverbs.so");
  EXPECT_EQ(loaderState.openCalls[1].flags, RTLD_NOW);
  EXPECT_EQ(loaderState.openCalls[2].name, "libibverbs.so.1");
  EXPECT_EQ(loaderState.openCalls[2].flags, RTLD_NOW);
  EXPECT_TRUE(loaderState.resolveCalls.empty()) << "Symbol resolution cannot begin without a library handle.";
  EXPECT_TRUE(loaderState.closeCalls.empty()) << "A NULL library handle must not be passed to dlclose.";
  ExpectClearedSymbolTableForTest(&symbols);
}

/**
 * @brief Verifies teardown after the absence of each mandatory symbol in turn.
 * @note Side effects: Calls the fake dynamic loader repeatedly and clears one local table
 * per scenario.
 */
TEST_F(DynamicSymbolTest, RejectsEveryMissingMandatorySymbol) {
  SCOPED_TRACE("Every mandatory symbol must independently gate success and trigger complete teardown.");

  for (size_t missingIndex = 0; missingIndex < kSymbolSpecs.size(); ++missingIndex) {
    if (!kSymbolSpecs[missingIndex].mandatory) continue;
    SCOPED_TRACE(std::string("missing mandatory symbol: ") + kSymbolSpecs[missingIndex].name);
    ResetLoaderForTest();
    if ((missingIndex & 1U) != 0U) loaderState.mode = LibraryMode::Fallback;
    void* expectedHandle = loaderState.mode == LibraryMode::Primary
      ? static_cast<void*>(&loaderState.primaryHandle)
      : static_cast<void*>(&loaderState.fallbackHandle);
    loaderState.missingSymbols.emplace_back(kSymbolSpecs[missingIndex].name);
    struct ncclIbvSymbols symbols = {};
    FillSymbolTableForTest(&symbols);

    EXPECT_EQ(buildIbvSymbols(&symbols), ncclSystemError)
      << "A missing mandatory symbol must return ncclSystemError.";
    ExpectClearedSymbolTableForTest(&symbols);
    size_t failedRequests = 0;
    for (const ResolveCall& call : loaderState.resolveCalls) {
      if (call.name != kSymbolSpecs[missingIndex].name) continue;
      ++failedRequests;
      EXPECT_EQ(call.handle, expectedHandle)
        << "The failed mandatory lookup must use the opened library handle.";
      EXPECT_EQ(call.version, kSymbolSpecs[missingIndex].version)
        << "The failed mandatory lookup must use its required ABI version.";
    }
    EXPECT_GT(failedRequests, 0U) << "The configured missing mandatory symbol must be requested.";
    ASSERT_EQ(loaderState.closeCalls.size(), 1U) << "Mandatory-symbol failure must close the opened handle once.";
    EXPECT_EQ(loaderState.closeCalls[0], expectedHandle)
      << "Teardown must close the exact handle returned by dlopen.";
  }
}

}  // namespace
