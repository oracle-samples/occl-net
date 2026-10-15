/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "mlx5/mlx5dvsymbols.h"

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace {

/** @brief Describes one mlx5 direct-verbs symbol and whether it is mandatory. */
struct SymbolSpec {
  const char* name;
  const char* version;
};

constexpr std::array<SymbolSpec, 3> kSymbolSpecs = {{
  {"mlx5dv_is_supported", "MLX5_1.8"},
  {"mlx5dv_get_data_direct_sysfs_path", "MLX5_1.25"},
  {"mlx5dv_reg_dmabuf_mr", "MLX5_1.25"},
}};

/** @brief Selects the fake mlx5 library-open outcome. */
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

/** @brief Records one dynamic mlx5 symbol lookup. */
struct ResolveCall {
  void* handle;
  std::string name;
  std::string version;
};

/**
 * @brief Models soname availability, records loader calls, and owns stable fake handles
 * and addresses.
 */
struct LoaderState {
  LibraryMode mode = LibraryMode::Primary;
  std::vector<std::string> missingSymbols;
  std::vector<OpenCall> openCalls;
  std::vector<ResolveCall> resolveCalls;
  std::vector<void*> closeCalls;
  unsigned char primaryHandle = 0;
  unsigned char fallbackHandle = 0;
  std::array<unsigned char, kSymbolSpecs.size()> symbolAddresses = {};
  std::array<unsigned char, kSymbolSpecs.size()> sentinelAddresses = {};
};

LoaderState loaderState;

/**
 * @brief Resets the fake loader with a successful primary library and distinct symbol and
 * sentinel addresses.
 * @note Side effects: Replaces loaderState.
 */
void ResetLoaderForTest() {
  loaderState = LoaderState{};
  for (size_t index = 0; index < kSymbolSpecs.size(); ++index) {
    loaderState.symbolAddresses[index] = static_cast<unsigned char>(index + 1);
    loaderState.sentinelAddresses[index] = static_cast<unsigned char>(index + 11);
  }
}

/**
 * @brief Finds one configured symbol by exact name.
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
 * @brief Reports whether one symbol is configured as unavailable.
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
 * @brief Returns the stable loaded address assigned to one known symbol.
 * @param[in] name Valid non-NULL symbol name present in kSymbolSpecs.
 * @return A distinct non-NULL address for name.
 * @note Side effects: Records a GoogleTest failure when name is unknown.
 */
void* SymbolAddressForTest(const char* name) {
  const size_t index = FindSymbolForTest(name);
  EXPECT_LT(index, kSymbolSpecs.size()) << "The test requested an unknown fake mlx5 symbol: " << name;
  return index < kSymbolSpecs.size() ? &loaderState.symbolAddresses[index] : nullptr;
}

/**
 * @brief Models a retained versioned mlx5 library and both ordinary soname-loading paths.
 * @param[in] name Valid non-NULL null-terminated library name.
 * @param[in] flags Dynamic-loader flags recorded exactly.
 * @return The configured fake handle, or NULL when the requested library is unavailable.
 * @note Side effects: Appends the request to openCalls.
 */
void* OpenLibraryForTest(const char* name, int flags) {
  loaderState.openCalls.push_back({name, flags});
  if (std::strcmp(name, "libmlx5.so") == 0) {
    return loaderState.mode == LibraryMode::Primary ? &loaderState.primaryHandle : nullptr;
  }
  if (std::strcmp(name, "libmlx5.so.1") == 0) {
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
 * @brief Implements deterministic versioned lookup of the three mlx5 symbols.
 * @param[in] handle Valid non-NULL primary or fallback fake handle.
 * @param[in] name Valid non-NULL null-terminated symbol name.
 * @param[in] version Valid non-NULL null-terminated ABI version.
 * @return The configured symbol address when name and version match and it is available;
 * otherwise NULL.
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
 * @brief Supplies a stable diagnostic string for symbol lookup failures.
 * @return A valid mutable null-terminated diagnostic string.
 * @note Side effects: None.
 */
char* LoaderErrorForTest() {
  static char message[] = "controlled mlx5 loader failure";
  return message;
}

/**
 * @brief Implements deterministic closure of one opened fake library.
 * @param[in] handle Valid non-NULL fake library handle.
 * @return 0.
 * @note Side effects: Appends handle to closeCalls.
 */
int CloseLibraryForTest(void* handle) {
  loaderState.closeCalls.push_back(handle);
  return 0;
}

/**
 * @brief Copies one POSIX loader address into a function-pointer field.
 * @tparam Symbol Function-pointer field type with POSIX-compatible representation.
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
 * @brief Reads the POSIX loader address stored in one function-pointer field.
 * @tparam Symbol Function-pointer type with POSIX-compatible representation.
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
 * @brief Preloads every output field with a distinct non-NULL sentinel.
 * @param[out] symbols Valid non-NULL symbol table receiving all three sentinels.
 * @note Side effects: Replaces every field in symbols.
 */
void FillSymbolTableForTest(struct ncclMlx5dvSymbols* symbols) {
  SetSymbolFieldForTest(symbols->mlx5dv_internal_is_supported, &loaderState.sentinelAddresses[0]);
  SetSymbolFieldForTest(symbols->mlx5dv_internal_get_data_direct_sysfs_path, &loaderState.sentinelAddresses[1]);
  SetSymbolFieldForTest(symbols->mlx5dv_internal_reg_dmabuf_mr, &loaderState.sentinelAddresses[2]);
}

/**
 * @brief Checks one output field against its expected loaded address or explicit NULL value.
 * @tparam Symbol Function-pointer type stored in the symbol table.
 * @param[in] field Function-pointer field returned by production code.
 * @param[in] name Valid non-NULL symbol name present in kSymbolSpecs.
 * @param[in] expectedLoaded true for the fake address or false for NULL.
 * @note Side effects: Records a GoogleTest failure when field differs.
 */
template <typename Symbol>
void ExpectSymbolFieldForTest(Symbol field, const char* name, bool expectedLoaded) {
  SCOPED_TRACE(std::string("mlx5 symbol field: ") + name);
  void* expected = expectedLoaded ? SymbolAddressForTest(name) : nullptr;
  EXPECT_EQ(ReadSymbolFieldForTest(field), expected)
    << "The symbol-table field must exactly reflect loader availability for " << name << ".";
}

/**
 * @brief Checks all three output fields after successful symbol construction.
 * @param[in] symbols Valid non-NULL returned symbol table.
 * @note Side effects: Records GoogleTest failures for incorrect fields.
 */
void ExpectLoadedSymbolTableForTest(const struct ncclMlx5dvSymbols* symbols) {
  ExpectSymbolFieldForTest(symbols->mlx5dv_internal_is_supported, "mlx5dv_is_supported", true);
  ExpectSymbolFieldForTest(symbols->mlx5dv_internal_get_data_direct_sysfs_path,
    "mlx5dv_get_data_direct_sysfs_path", !IsSymbolMissingForTest("mlx5dv_get_data_direct_sysfs_path"));
  ExpectSymbolFieldForTest(symbols->mlx5dv_internal_reg_dmabuf_mr,
    "mlx5dv_reg_dmabuf_mr", !IsSymbolMissingForTest("mlx5dv_reg_dmabuf_mr"));
}

/**
 * @brief Checks that failure teardown explicitly cleared all three preloaded output fields.
 * @param[in] symbols Valid non-NULL returned symbol table.
 * @note Side effects: Records GoogleTest failures for non-NULL fields.
 */
void ExpectClearedSymbolTableForTest(const struct ncclMlx5dvSymbols* symbols) {
  ExpectSymbolFieldForTest(symbols->mlx5dv_internal_is_supported, "mlx5dv_is_supported", false);
  ExpectSymbolFieldForTest(symbols->mlx5dv_internal_get_data_direct_sysfs_path,
    "mlx5dv_get_data_direct_sysfs_path", false);
  ExpectSymbolFieldForTest(symbols->mlx5dv_internal_reg_dmabuf_mr, "mlx5dv_reg_dmabuf_mr", false);
}

/**
 * @brief Checks request membership, ABI versions, handles, and at-least-once symbol coverage.
 * @param[in] expectedHandle Valid non-NULL handle required for every request.
 * @note Side effects: Records GoogleTest failures for missing, unknown, or malformed requests.
 */
void ExpectResolutionRequestsForTest(void* expectedHandle) {
  for (const ResolveCall& call : loaderState.resolveCalls) {
    SCOPED_TRACE(std::string("actual mlx5 resolution request: ") + call.name);
    const size_t index = FindSymbolForTest(call.name.c_str());
    ASSERT_LT(index, kSymbolSpecs.size()) << "The loader must not request an unknown mlx5 symbol.";
    EXPECT_EQ(call.handle, expectedHandle) << "Every lookup must use the handle returned by dlopen.";
    EXPECT_EQ(call.version, kSymbolSpecs[index].version)
      << "Every lookup must use the symbol's required ABI version.";
  }
  for (const SymbolSpec& spec : kSymbolSpecs) {
    SCOPED_TRACE(std::string("expected mlx5 resolution request: ") + spec.name);
    size_t matchingCalls = 0;
    for (const ResolveCall& call : loaderState.resolveCalls) {
      if (call.name == spec.name) ++matchingCalls;
    }
    EXPECT_GT(matchingCalls, 0U) << "Every configured mlx5 symbol must be requested.";
  }
}

/** @brief Provides an isolated successful loader for dynamic mlx5 symbol tests. */
class Mlx5DvDynamicSymbolTest : public ::testing::Test {
 protected:
  /**
   * @brief Restores a successful primary fake loader before each test.
   * @note Side effects: Resets loaderState.
   */
  void SetUp() override { ResetLoaderForTest(); }
};

}  // namespace

/**
 * @brief Compiles dynamic symbol loading against deterministic loader seams.
 */
#define dlopen OpenLibraryForTest
#define dlvsym ResolveVersionedSymbolForTest
#define dlerror LoaderErrorForTest
#define dlclose CloseLibraryForTest
#include "../../../misc/mlx5dvsymbols.cc"  // NOLINT(build/include)
#undef dlclose
#undef dlerror
#undef dlvsym
#undef dlopen
#undef LOAD_SYM_VERSION
#undef LOAD_SYM
#undef MLX5DV_VERSION

namespace {

/**
 * @brief Verifies reuse of a versioned mlx5 runtime already present in the link map.
 * @note Side effects: Calls the fake dynamic loader and replaces a local symbol table.
 */
TEST_F(Mlx5DvDynamicSymbolTest, ReusesRetainedVersionedLibrary) {
  SCOPED_TRACE("A retained versioned mlx5 runtime must be reused before ordinary loading.");

  loaderState.mode = LibraryMode::Retained;
  struct ncclMlx5dvSymbols symbols = {};
  FillSymbolTableForTest(&symbols);
  EXPECT_EQ(buildMlx5dvSymbols(&symbols), ncclSuccess)
      << "A retained versioned mlx5 runtime was not reused.";
  ASSERT_EQ(loaderState.openCalls.size(), 1U) << "Retained reuse must require one probe.";
  EXPECT_EQ(loaderState.openCalls[0].name, "libmlx5.so.1");
  EXPECT_EQ(loaderState.openCalls[0].flags, RTLD_NOW | RTLD_NOLOAD);
  ExpectResolutionRequestsForTest(&loaderState.fallbackHandle);
  ExpectLoadedSymbolTableForTest(&symbols);
  EXPECT_TRUE(loaderState.closeCalls.empty()) << "A retained process-lifetime handle was closed.";
}

/**
 * @brief Verifies primary-soname loading, ABI requests, and replacement of every output sentinel.
 * @note Side effects: Calls the fake dynamic loader and replaces a local symbol table.
 */
TEST_F(Mlx5DvDynamicSymbolTest, LoadsPrimaryLibrary) {
  SCOPED_TRACE("A complete primary mlx5 library must replace every sentinel and retain its live handle.");

  struct ncclMlx5dvSymbols symbols = {};
  FillSymbolTableForTest(&symbols);
  EXPECT_EQ(buildMlx5dvSymbols(&symbols), ncclSuccess)
    << "A primary library containing every symbol must load successfully.";
  ASSERT_EQ(loaderState.openCalls.size(), 2U)
      << "Primary loading must follow one unsuccessful retained-runtime probe.";
  EXPECT_EQ(loaderState.openCalls[0].name, "libmlx5.so.1");
  EXPECT_EQ(loaderState.openCalls[0].flags, RTLD_NOW | RTLD_NOLOAD);
  EXPECT_EQ(loaderState.openCalls[1].name, "libmlx5.so") << "The primary mlx5 soname must be exact.";
  EXPECT_EQ(loaderState.openCalls[1].flags, RTLD_NOW)
      << "The primary library must resolve dependencies immediately.";
  ExpectResolutionRequestsForTest(&loaderState.primaryHandle);
  ExpectLoadedSymbolTableForTest(&symbols);
  EXPECT_TRUE(loaderState.closeCalls.empty()) << "A successful process-lifetime handle must remain open.";
}

/**
 * @brief Verifies fallback loading with all meaningful combinations of optional symbol
 * availability.
 * @note Side effects: Calls the fake dynamic loader repeatedly and replaces one local table per
 * scenario.
 */
TEST_F(Mlx5DvDynamicSymbolTest, UsesFallbackWithOptionalSymbols) {
  SCOPED_TRACE("The fallback library must succeed whether either, both, or neither optional extension is absent.");

  /** @brief Describes one optional symbol and its table-field accessor. */
  struct OptionalScenario {
    const char* description;
    std::vector<std::string> missingSymbols;
  };
  const std::array<OptionalScenario, 4> scenarios = {{
    {"all optional symbols available", {}},
    {"data-direct path unavailable", {"mlx5dv_get_data_direct_sysfs_path"}},
    {"DMA-BUF registration unavailable", {"mlx5dv_reg_dmabuf_mr"}},
    {"both optional symbols unavailable",
      {"mlx5dv_get_data_direct_sysfs_path", "mlx5dv_reg_dmabuf_mr"}},
  }};
  for (const OptionalScenario& scenario : scenarios) {
    SCOPED_TRACE(scenario.description);
    ResetLoaderForTest();
    loaderState.mode = LibraryMode::Fallback;
    loaderState.missingSymbols = scenario.missingSymbols;
    struct ncclMlx5dvSymbols symbols = {};
    FillSymbolTableForTest(&symbols);

    EXPECT_EQ(buildMlx5dvSymbols(&symbols), ncclSuccess)
      << "Missing optional extensions must not reject an otherwise valid fallback library.";
    ASSERT_EQ(loaderState.openCalls.size(), 3U)
        << "Fallback loading must probe retained state before both ordinary sonames.";
    EXPECT_EQ(loaderState.openCalls[0].name, "libmlx5.so.1");
    EXPECT_EQ(loaderState.openCalls[0].flags, RTLD_NOW | RTLD_NOLOAD);
    EXPECT_EQ(loaderState.openCalls[1].name, "libmlx5.so");
    EXPECT_EQ(loaderState.openCalls[1].flags, RTLD_NOW);
    EXPECT_EQ(loaderState.openCalls[2].name, "libmlx5.so.1");
    EXPECT_EQ(loaderState.openCalls[2].flags, RTLD_NOW);
    ExpectResolutionRequestsForTest(&loaderState.fallbackHandle);
    ExpectLoadedSymbolTableForTest(&symbols);
    EXPECT_TRUE(loaderState.closeCalls.empty()) << "A successfully loaded fallback handle must remain open.";
  }
}

/**
 * @brief Verifies complete output clearing when neither supported mlx5 library can be opened.
 * @note Side effects: Calls the fake dynamic loader and clears a local symbol table.
 */
TEST_F(Mlx5DvDynamicSymbolTest, ClearsTableWhenLibrariesUnavailable) {
  SCOPED_TRACE("Total library absence must return an error and clear every preloaded output field.");

  loaderState.mode = LibraryMode::Unavailable;
  struct ncclMlx5dvSymbols symbols = {};
  FillSymbolTableForTest(&symbols);
  EXPECT_EQ(buildMlx5dvSymbols(&symbols), ncclSystemError)
    << "Failure to open both supported sonames must return ncclSystemError.";
  ASSERT_EQ(loaderState.openCalls.size(), 3U)
      << "Total absence must include the retained probe and both ordinary load attempts.";
  EXPECT_EQ(loaderState.openCalls[0].name, "libmlx5.so.1");
  EXPECT_EQ(loaderState.openCalls[0].flags, RTLD_NOW | RTLD_NOLOAD);
  EXPECT_EQ(loaderState.openCalls[1].name, "libmlx5.so");
  EXPECT_EQ(loaderState.openCalls[1].flags, RTLD_NOW);
  EXPECT_EQ(loaderState.openCalls[2].name, "libmlx5.so.1");
  EXPECT_EQ(loaderState.openCalls[2].flags, RTLD_NOW);
  EXPECT_TRUE(loaderState.resolveCalls.empty()) << "Resolution cannot begin without a valid library handle.";
  EXPECT_TRUE(loaderState.closeCalls.empty()) << "A NULL handle must not be passed to dlclose.";
  ExpectClearedSymbolTableForTest(&symbols);
}

/**
 * @brief Verifies mandatory-symbol failure, teardown, and active-handle ownership for both sonames.
 * @note Side effects: Runs two scenarios with three total open calls and clears both local tables.
 */
TEST_F(Mlx5DvDynamicSymbolTest, RejectsMissingMandatorySymbol) {
  SCOPED_TRACE("The mandatory support symbol must gate success on both primary and fallback libraries.");

  for (LibraryMode mode : {LibraryMode::Primary, LibraryMode::Fallback}) {
    ResetLoaderForTest();
    loaderState.mode = mode;
    loaderState.missingSymbols.emplace_back("mlx5dv_is_supported");
    void* expectedHandle = mode == LibraryMode::Primary
      ? static_cast<void*>(&loaderState.primaryHandle)
      : static_cast<void*>(&loaderState.fallbackHandle);
    SCOPED_TRACE(mode == LibraryMode::Primary ? "primary handle" : "fallback handle");
    struct ncclMlx5dvSymbols symbols = {};
    FillSymbolTableForTest(&symbols);

    EXPECT_EQ(buildMlx5dvSymbols(&symbols), ncclSystemError)
      << "A missing mandatory support symbol must return ncclSystemError.";
    ExpectClearedSymbolTableForTest(&symbols);
    ASSERT_FALSE(loaderState.resolveCalls.empty()) << "The mandatory support symbol must be requested before teardown.";
    for (const ResolveCall& call : loaderState.resolveCalls) {
      EXPECT_EQ(call.name, "mlx5dv_is_supported")
        << "Optional symbols must not be requested after the mandatory lookup fails.";
      EXPECT_EQ(call.handle, expectedHandle) << "The failed lookup must use the opened library handle.";
      EXPECT_EQ(call.version, "MLX5_1.8") << "The mandatory lookup must use its required ABI version.";
    }
    ASSERT_EQ(loaderState.closeCalls.size(), 1U) << "Mandatory-symbol failure must close one active handle.";
    EXPECT_EQ(loaderState.closeCalls[0], expectedHandle) << "Teardown must close the exact active handle.";
  }
}

}  // namespace
