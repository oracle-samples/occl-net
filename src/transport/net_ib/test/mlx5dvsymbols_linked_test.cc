/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "mlx5/mlx5dvsymbols.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <type_traits>

namespace {

std::array<int, 6> linkedSymbolCalls = {};

/**
 * @brief Supplies one distinct, correctly typed linked symbol and records unexpected invocation.
 * @tparam Id Unique identity that gives each symbol a distinct address.
 * @tparam Result Function return type.
 * @tparam Arguments Function argument types.
 * @param[in] arguments Values ignored by the address-only test symbol.
 * @return A value-initialized Result, or no value when Result is void.
 * @note Side effects: Increments linkedSymbolCalls[Id] when invoked.
 */
template <int Id, typename Result, typename... Arguments>
Result UnusedLinkedMlx5SymbolForTest(Arguments... arguments) {
  static_assert(Id >= 0 && Id < 6, "Linked test symbol identity must index linkedSymbolCalls.");
  ++linkedSymbolCalls[Id];
  ((void)arguments, ...);
  if constexpr (std::is_void_v<Result>) {
    return;
  } else {
    return Result{};
  }
}

}  // namespace

/**
 * @brief Compiles the linked-mode implementation with distinct typed provider symbols
 * for assignment checks.
 */
#define NCCL_BUILD_MLX5DV 1
#define mlx5dv_is_supported UnusedLinkedMlx5SymbolForTest<0, bool, struct ibv_device*>
#define mlx5dv_get_data_direct_sysfs_path \
  UnusedLinkedMlx5SymbolForTest<1, int, struct ibv_context*, char*, size_t>
#define mlx5dv_reg_dmabuf_mr \
  UnusedLinkedMlx5SymbolForTest<2, struct ibv_mr*, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int>
#include "../../../misc/mlx5dvsymbols.cc"  // NOLINT(build/include)
#undef mlx5dv_reg_dmabuf_mr
#undef mlx5dv_get_data_direct_sysfs_path
#undef mlx5dv_is_supported
#undef NCCL_BUILD_MLX5DV
#undef ASSIGN_SYM

namespace {

/**
 * @brief Checks one linked output field with a diagnostic naming the violated assignment.
 * @tparam Symbol Exact function-pointer type stored in the symbol table.
 * @param[in] actual Address written by buildMlx5dvSymbols.
 * @param[in] expected Address required by the linked-mode contract.
 * @param[in] name Valid non-NULL symbol name used in failure output.
 * @note Side effects: Records a nonfatal GoogleTest failure when the addresses differ.
 */
template <typename Symbol>
void ExpectLinkedSymbolForTest(Symbol actual, Symbol expected, const char* name) {
  SCOPED_TRACE(std::string("linked mlx5 symbol: ") + name);
  EXPECT_EQ(actual, expected) << "Linked mode must assign the exact address for " << name << ".";
}

/**
 * @brief Verifies that linked mode replaces every preloaded field with its corresponding
 * function address.
 * @note Side effects: Clears linkedSymbolCalls and replaces all fields in a local symbol table.
 */
TEST(Mlx5DvLinkedSymbolTest, AssignsEveryLinkedSymbol) {
  SCOPED_TRACE("Linked symbol construction must replace all sentinels without calling provider functions.");

  linkedSymbolCalls.fill(0);
  struct ncclMlx5dvSymbols symbols = {};
  symbols.mlx5dv_internal_is_supported = &UnusedLinkedMlx5SymbolForTest<3, bool, struct ibv_device*>;
  symbols.mlx5dv_internal_get_data_direct_sysfs_path =
    &UnusedLinkedMlx5SymbolForTest<4, int, struct ibv_context*, char*, size_t>;
  symbols.mlx5dv_internal_reg_dmabuf_mr =
    &UnusedLinkedMlx5SymbolForTest<5, struct ibv_mr*, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int>;

  EXPECT_EQ(buildMlx5dvSymbols(&symbols), ncclSuccess)
    << "Linked symbol construction must return ncclSuccess.";
  ExpectLinkedSymbolForTest(symbols.mlx5dv_internal_is_supported,
    &UnusedLinkedMlx5SymbolForTest<0, bool, struct ibv_device*>, "mlx5dv_is_supported");
  ExpectLinkedSymbolForTest(symbols.mlx5dv_internal_get_data_direct_sysfs_path,
    &UnusedLinkedMlx5SymbolForTest<1, int, struct ibv_context*, char*, size_t>,
    "mlx5dv_get_data_direct_sysfs_path");
  ExpectLinkedSymbolForTest(symbols.mlx5dv_internal_reg_dmabuf_mr,
    &UnusedLinkedMlx5SymbolForTest<2, struct ibv_mr*, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int>,
    "mlx5dv_reg_dmabuf_mr");
  const std::array<int, 6> noLinkedCalls = {};
  EXPECT_EQ(linkedSymbolCalls, noLinkedCalls)
    << "Building a linked symbol table must not invoke expected or sentinel provider functions.";
}

}  // namespace
