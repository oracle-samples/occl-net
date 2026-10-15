/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "ibvsymbols.h"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <string>
#include <type_traits>

namespace {

/** @brief Scripts and records calls through directly linked verbs symbols. */
struct LinkedProviderState {
  struct ibv_mr* registrationResult = nullptr;
  int portResult = 0;
  int registrationCalls = 0;
  int portCalls = 0;
  struct ibv_pd* registrationDomain = nullptr;
  void* registrationAddress = nullptr;
  size_t registrationLength = 0;
  int registrationAccess = 0;
  struct ibv_context* portContext = nullptr;
  uint8_t portNumber = 0;
  struct ibv_port_attr* portOutput = nullptr;
  struct ibv_port_attr portAttributes = {};
};

LinkedProviderState linkedProviderState;

/**
 * @brief Resets the linked-provider call records and deterministic port output.
 * @note Side effects: Replaces linkedProviderState.
 */
void ResetLinkedProviderForTest() {
  linkedProviderState = LinkedProviderState{};
  linkedProviderState.portAttributes.lid = 0x1234;
  linkedProviderState.portAttributes.active_mtu = IBV_MTU_2048;
}

/**
 * @brief Supplies one distinct, correctly typed linked symbol that is assigned but never called.
 * @tparam Id Unique identity that gives each symbol a distinct address.
 * @tparam Result Function return type.
 * @tparam Arguments Function argument types.
 * @param[in] arguments Values ignored because this address-only test symbol is never called.
 * @return A value-initialized Result, or no value when Result is void.
 * @note Side effects: None.
 */
template <int Id, typename Result, typename... Arguments>
Result UnusedLinkedSymbolForTest(Arguments... arguments) {
  ((void)arguments, ...);
  if constexpr (std::is_void_v<Result>) {
    return;
  } else {
    return Result{};
  }
}

/**
 * @brief Implements the provider called by the production linked port-query passthrough.
 * @param[in] context Any context pointer recorded exactly.
 * @param[in] port Port number in [0, UINT8_MAX] recorded exactly.
 * @param[out] attributes Valid non-NULL output receiving portAttributes on success.
 * @return linkedProviderState.portResult.
 * @note Side effects: Records every argument and writes attributes when portResult is 0.
 */
int QueryPortForLinkedTest(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attributes) {
  ++linkedProviderState.portCalls;
  linkedProviderState.portContext = context;
  linkedProviderState.portNumber = port;
  linkedProviderState.portOutput = attributes;
  if (linkedProviderState.portResult == 0) *attributes = linkedProviderState.portAttributes;
  return linkedProviderState.portResult;
}

/**
 * @brief Implements the provider called by the production linked memory-registration passthrough.
 * @param[in] domain Any protection-domain pointer recorded exactly.
 * @param[in] address Any buffer address recorded exactly.
 * @param[in] length Registration size in [0, SIZE_MAX] recorded exactly.
 * @param[in] access Any representable verbs access mask recorded exactly.
 * @return linkedProviderState.registrationResult.
 * @note Side effects: Records every argument.
 */
struct ibv_mr* RegisterMemoryForLinkedTest(struct ibv_pd* domain, void* address, size_t length, int access) {
  ++linkedProviderState.registrationCalls;
  linkedProviderState.registrationDomain = domain;
  linkedProviderState.registrationAddress = address;
  linkedProviderState.registrationLength = length;
  linkedProviderState.registrationAccess = access;
  return linkedProviderState.registrationResult;
}

}  // namespace

/**
 * @brief Builds the linked symbol table with distinct, correctly typed test function addresses.
 */
#define NCCL_BUILD_RDMA_CORE 1
#define ibv_fork_init UnusedLinkedSymbolForTest<0, int>
#define ibv_get_device_list UnusedLinkedSymbolForTest<1, struct ibv_device**, int*>
#define ibv_free_device_list UnusedLinkedSymbolForTest<2, void, struct ibv_device**>
#define ibv_get_device_name UnusedLinkedSymbolForTest<3, const char*, struct ibv_device*>
#define ibv_open_device UnusedLinkedSymbolForTest<4, struct ibv_context*, struct ibv_device*>
#define ibv_close_device UnusedLinkedSymbolForTest<5, int, struct ibv_context*>
#define ibv_get_async_event UnusedLinkedSymbolForTest<6, int, struct ibv_context*, struct ibv_async_event*>
#define ibv_ack_async_event UnusedLinkedSymbolForTest<7, void, struct ibv_async_event*>
#define ibv_query_device UnusedLinkedSymbolForTest<8, int, struct ibv_context*, struct ibv_device_attr*>
#define ibv_query_port QueryPortForLinkedTest
#define ibv_query_gid UnusedLinkedSymbolForTest<10, int, struct ibv_context*, uint8_t, int, union ibv_gid*>
#define ibv_query_qp \
  UnusedLinkedSymbolForTest<11, int, struct ibv_qp*, struct ibv_qp_attr*, int, struct ibv_qp_init_attr*>
#define ibv_alloc_pd UnusedLinkedSymbolForTest<12, struct ibv_pd*, struct ibv_context*>
#define ibv_dealloc_pd UnusedLinkedSymbolForTest<13, int, struct ibv_pd*>
#define ibv_reg_mr RegisterMemoryForLinkedTest
#define ibv_reg_mr_iova2 \
  UnusedLinkedSymbolForTest<15, struct ibv_mr*, struct ibv_pd*, void*, size_t, uint64_t, unsigned int>
#define ibv_reg_dmabuf_mr \
  UnusedLinkedSymbolForTest<16, struct ibv_mr*, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int>
#define ibv_dereg_mr UnusedLinkedSymbolForTest<17, int, struct ibv_mr*>
#define ibv_create_cq \
  UnusedLinkedSymbolForTest<18, struct ibv_cq*, struct ibv_context*, int, void*, struct ibv_comp_channel*, int>
#define ibv_destroy_cq UnusedLinkedSymbolForTest<19, int, struct ibv_cq*>
#define ibv_create_qp UnusedLinkedSymbolForTest<20, struct ibv_qp*, struct ibv_pd*, struct ibv_qp_init_attr*>
#define ibv_modify_qp UnusedLinkedSymbolForTest<21, int, struct ibv_qp*, struct ibv_qp_attr*, int>
#define ibv_destroy_qp UnusedLinkedSymbolForTest<22, int, struct ibv_qp*>
#define ibv_event_type_str UnusedLinkedSymbolForTest<23, const char*, enum ibv_event_type>
#define ibv_query_ece UnusedLinkedSymbolForTest<24, int, struct ibv_qp*, struct ibv_ece*>
#define ibv_set_ece UnusedLinkedSymbolForTest<25, int, struct ibv_qp*, struct ibv_ece*>
#include "../../../misc/ibvsymbols.cc"  // NOLINT(build/include)
#undef ibv_set_ece
#undef ibv_query_ece
#undef ibv_event_type_str
#undef ibv_destroy_qp
#undef ibv_modify_qp
#undef ibv_create_qp
#undef ibv_destroy_cq
#undef ibv_create_cq
#undef ibv_dereg_mr
#undef ibv_reg_dmabuf_mr
#undef ibv_reg_mr_iova2
#undef ibv_reg_mr
#undef ibv_dealloc_pd
#undef ibv_alloc_pd
#undef ibv_query_qp
#undef ibv_query_gid
#undef ibv_query_port
#undef ibv_query_device
#undef ibv_ack_async_event
#undef ibv_get_async_event
#undef ibv_close_device
#undef ibv_open_device
#undef ibv_get_device_name
#undef ibv_free_device_list
#undef ibv_get_device_list
#undef ibv_fork_init
#undef NCCL_BUILD_RDMA_CORE
#undef ASSIGN_SYM

namespace {

/** @brief Provides isolated provider state for directly linked symbol-table tests. */
class LinkedSymbolTest : public ::testing::Test {
 protected:
  /**
   * @brief Restores the linked-provider state before each test.
   * @note Side effects: Resets linkedProviderState.
   */
  void SetUp() override { ResetLinkedProviderForTest(); }
};

/**
 * @brief Checks one linked symbol address with a diagnostic naming the violated assignment.
 * @tparam Symbol Exact function-pointer type stored by the symbol table.
 * @param[in] actual Address written by buildIbvSymbols.
 * @param[in] expected Address required by the linked-mode contract.
 * @param[in] name Valid non-NULL symbol name used in failure output.
 * @note Side effects: Records a nonfatal GoogleTest failure when the addresses differ.
 */
template <typename Symbol>
void ExpectLinkedSymbolForTest(Symbol actual, Symbol expected, const char* name) {
  SCOPED_TRACE(std::string("linked symbol: ") + name);
  EXPECT_EQ(actual, expected) << "Linked mode must assign the exact address for " << name << ".";
}

/**
 * @brief Verifies that linked mode assigns the correct address to every symbol-table field.
 * @note Side effects: Writes a local symbol table.
 */
TEST_F(LinkedSymbolTest, AssignsEveryLinkedSymbol) {
  SCOPED_TRACE("Linked mode must replace every field with its corresponding linked function address.");

  struct ncclIbvSymbols symbols = {};
  EXPECT_EQ(buildIbvSymbols(&symbols), ncclSuccess) << "Linked symbol construction must return ncclSuccess.";
  ExpectLinkedSymbolForTest(symbols.ibv_internal_get_device_list,
    &UnusedLinkedSymbolForTest<1, struct ibv_device**, int*>, "ibv_get_device_list");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_free_device_list,
    &UnusedLinkedSymbolForTest<2, void, struct ibv_device**>, "ibv_free_device_list");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_get_device_name,
    &UnusedLinkedSymbolForTest<3, const char*, struct ibv_device*>, "ibv_get_device_name");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_open_device,
    &UnusedLinkedSymbolForTest<4, struct ibv_context*, struct ibv_device*>, "ibv_open_device");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_close_device,
    &UnusedLinkedSymbolForTest<5, int, struct ibv_context*>, "ibv_close_device");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_get_async_event,
    &UnusedLinkedSymbolForTest<6, int, struct ibv_context*, struct ibv_async_event*>, "ibv_get_async_event");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_ack_async_event,
    &UnusedLinkedSymbolForTest<7, void, struct ibv_async_event*>, "ibv_ack_async_event");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_query_device,
    &UnusedLinkedSymbolForTest<8, int, struct ibv_context*, struct ibv_device_attr*>, "ibv_query_device");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_query_port, &ibv_internal_query_port, "ibv_query_port");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_query_gid,
    &UnusedLinkedSymbolForTest<10, int, struct ibv_context*, uint8_t, int, union ibv_gid*>, "ibv_query_gid");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_query_qp,
    &UnusedLinkedSymbolForTest<11, int, struct ibv_qp*, struct ibv_qp_attr*, int, struct ibv_qp_init_attr*>,
    "ibv_query_qp");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_alloc_pd,
    &UnusedLinkedSymbolForTest<12, struct ibv_pd*, struct ibv_context*>, "ibv_alloc_pd");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_dealloc_pd,
    &UnusedLinkedSymbolForTest<13, int, struct ibv_pd*>, "ibv_dealloc_pd");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_reg_mr, &ibv_internal_reg_mr, "ibv_reg_mr");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_reg_mr_iova2,
    &UnusedLinkedSymbolForTest<15, struct ibv_mr*, struct ibv_pd*, void*, size_t, uint64_t, unsigned int>,
    "ibv_reg_mr_iova2");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_reg_dmabuf_mr,
    &UnusedLinkedSymbolForTest<16, struct ibv_mr*, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int>,
    "ibv_reg_dmabuf_mr");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_dereg_mr,
    &UnusedLinkedSymbolForTest<17, int, struct ibv_mr*>, "ibv_dereg_mr");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_create_cq,
    &UnusedLinkedSymbolForTest<18, struct ibv_cq*, struct ibv_context*, int, void*, struct ibv_comp_channel*, int>,
    "ibv_create_cq");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_destroy_cq,
    &UnusedLinkedSymbolForTest<19, int, struct ibv_cq*>, "ibv_destroy_cq");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_create_qp,
    &UnusedLinkedSymbolForTest<20, struct ibv_qp*, struct ibv_pd*, struct ibv_qp_init_attr*>, "ibv_create_qp");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_modify_qp,
    &UnusedLinkedSymbolForTest<21, int, struct ibv_qp*, struct ibv_qp_attr*, int>, "ibv_modify_qp");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_destroy_qp,
    &UnusedLinkedSymbolForTest<22, int, struct ibv_qp*>, "ibv_destroy_qp");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_fork_init, &UnusedLinkedSymbolForTest<0, int>, "ibv_fork_init");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_event_type_str,
    &UnusedLinkedSymbolForTest<23, const char*, enum ibv_event_type>, "ibv_event_type_str");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_query_ece,
    &UnusedLinkedSymbolForTest<24, int, struct ibv_qp*, struct ibv_ece*>, "ibv_query_ece");
  ExpectLinkedSymbolForTest(symbols.ibv_internal_set_ece,
    &UnusedLinkedSymbolForTest<25, int, struct ibv_qp*, struct ibv_ece*>, "ibv_set_ece");
}

/**
 * @brief Verifies both linked passthroughs preserve provider results, outputs, and inputs while
 * forwarding every argument.
 * @note Side effects: Calls the fake registration and port-query providers.
 */
TEST_F(LinkedSymbolTest, ForwardsMacroBackedOperations) {
  SCOPED_TRACE(
      "Linked passthroughs must forward inputs and preserve provider results and outputs unchanged.");

  struct ibv_context context = {};
  const struct ibv_context contextBefore = context;
  struct ibv_pd domain = {};
  domain.context = &context;
  struct ibv_mr memoryRegion = {};
  const struct ibv_pd domainBefore = domain;
  std::array<unsigned char, 17> buffer = {};
  for (size_t index = 0; index < buffer.size(); ++index) buffer[index] = static_cast<unsigned char>(index + 3);
  const std::array<unsigned char, 17> bufferBefore = buffer;

  linkedProviderState.registrationResult = &memoryRegion;
  EXPECT_EQ(ibv_internal_reg_mr(&domain, buffer.data(), buffer.size(), IBV_ACCESS_REMOTE_WRITE), &memoryRegion)
    << "The linked registration passthrough must return the provider memory region.";
  EXPECT_EQ(linkedProviderState.registrationCalls, 1) << "The linked registration provider must be called once.";
  EXPECT_EQ(linkedProviderState.registrationDomain, &domain) << "The protection domain must be forwarded exactly.";
  EXPECT_EQ(linkedProviderState.registrationAddress, buffer.data()) << "The buffer address must be forwarded exactly.";
  EXPECT_EQ(linkedProviderState.registrationLength, buffer.size()) << "The length must be forwarded exactly.";
  EXPECT_EQ(linkedProviderState.registrationAccess, IBV_ACCESS_REMOTE_WRITE)
    << "The access mask must be forwarded exactly.";

  linkedProviderState.registrationResult = nullptr;
  linkedProviderState.registrationCalls = 0;
  EXPECT_EQ(ibv_internal_reg_mr(&domain, buffer.data() + buffer.size() - 1, 1, IBV_ACCESS_LOCAL_WRITE), nullptr)
    << "The linked registration passthrough must preserve a NULL provider result.";
  EXPECT_EQ(linkedProviderState.registrationCalls, 1) << "A failed registration must call the provider once.";
  EXPECT_EQ(linkedProviderState.registrationDomain, &domain)
    << "A failed registration must still forward the protection domain.";
  EXPECT_EQ(linkedProviderState.registrationAddress, buffer.data() + buffer.size() - 1)
    << "A failed registration must forward its distinct buffer address.";
  EXPECT_EQ(linkedProviderState.registrationLength, 1U) << "The minimum nonempty length must be forwarded exactly.";
  EXPECT_EQ(linkedProviderState.registrationAccess, IBV_ACCESS_LOCAL_WRITE)
    << "A failed registration must forward its access mask.";
  EXPECT_EQ(std::memcmp(&domain, &domainBefore, sizeof(domain)), 0)
    << "The non-mutating provider and passthrough must not change the protection domain.";
  EXPECT_EQ(buffer, bufferBefore) << "Registration passthrough calls must not change registered bytes.";

  struct ibv_port_attr portAttributes = {};
  EXPECT_EQ(ibv_internal_query_port(&context, 7, &portAttributes), 0)
    << "The linked port-query passthrough must return provider success unchanged.";
  EXPECT_EQ(linkedProviderState.portCalls, 1) << "The linked port-query provider must be called once.";
  EXPECT_EQ(linkedProviderState.portContext, &context) << "The context must be forwarded exactly.";
  EXPECT_EQ(linkedProviderState.portNumber, 7) << "The port number must be forwarded exactly.";
  EXPECT_EQ(linkedProviderState.portOutput, &portAttributes) << "The output pointer must be forwarded exactly.";
  EXPECT_EQ(std::memcmp(&portAttributes, &linkedProviderState.portAttributes, sizeof(portAttributes)), 0)
    << "A successful port query must expose every provider output field.";
  EXPECT_EQ(std::memcmp(&context, &contextBefore, sizeof(context)), 0)
    << "A successful port query must not change its input context.";

  struct ibv_context failingContext = {};
  const struct ibv_context failingContextBefore = failingContext;
  struct ibv_port_attr failedPortAttributes;
  std::memset(&failedPortAttributes, 0xa5, sizeof(failedPortAttributes));
  const struct ibv_port_attr failedOutputBefore = failedPortAttributes;
  linkedProviderState.portResult = EIO;
  linkedProviderState.portCalls = 0;
  EXPECT_EQ(ibv_internal_query_port(&failingContext, 8, &failedPortAttributes), EIO)
    << "The linked port-query passthrough must return provider errors unchanged.";
  EXPECT_EQ(linkedProviderState.portCalls, 1) << "A failing port query must call the provider once.";
  EXPECT_EQ(linkedProviderState.portContext, &failingContext)
    << "A failing port query must forward its distinct context.";
  EXPECT_EQ(linkedProviderState.portNumber, 8) << "The failing query's valid port must be forwarded exactly.";
  EXPECT_EQ(linkedProviderState.portOutput, &failedPortAttributes)
    << "A failing port query must forward its distinct output pointer.";
  EXPECT_EQ(std::memcmp(&failedPortAttributes, &failedOutputBefore, sizeof(failedPortAttributes)), 0)
    << "The non-mutating failing provider and passthrough must leave output storage unchanged.";
  EXPECT_EQ(std::memcmp(&failingContext, &failingContextBefore, sizeof(failingContext)), 0)
    << "A failing port query must not change its input context.";
}

}  // namespace
