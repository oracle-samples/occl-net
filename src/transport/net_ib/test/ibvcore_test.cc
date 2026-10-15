/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "ibvcore.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>

namespace {

/**
 * @brief Shares a scripted status and captures header-only post-send and extended-port dispatches.
 */
struct CoreProviderState {
  int result = 0;
  int calls = 0;
  int queryPortCalls = 0;
  ibv_qp* queuePair = nullptr;
  ibv_send_wr* workRequest = nullptr;
  ibv_send_wr** badWorkRequest = nullptr;
  ibv_context* context = nullptr;
  uint8_t port = 0;
  ibv_port_attr* portAttributes = nullptr;
  size_t portAttributesSize = 0;
};

CoreProviderState coreProviderState;

/**
 * @brief Supplies a non-NULL legacy allocation callback used only as a device-layout discriminator.
 * @param[in] device Device forwarded by a hypothetical legacy provider.
 * @param[in] descriptor Command descriptor forwarded by a hypothetical legacy provider.
 * @return NULL because the callback is never expected to be invoked by these tests.
 * @note Side effects: None.
 */
ibv_context* LegacyContextAllocationForTest(ibv_device* device, int descriptor) {
  (void)device;
  (void)descriptor;
  return nullptr;
}

/**
 * @brief Records direct queue-pair send dispatch and supplies success or a provider error.
 * @param[in] queuePair Exact queue pair forwarded by the inline dispatch helper.
 * @param[in] workRequest Exact first work request forwarded to the provider.
 * @param[out] badWorkRequest Receives NULL on success or workRequest on failure.
 * @return coreProviderState.result.
 * @note Side effects: Records all arguments and increments the provider call count.
 */
int PostSendForCoreTest(ibv_qp* queuePair, ibv_send_wr* workRequest,
  ibv_send_wr** badWorkRequest) {
  ++coreProviderState.calls;
  coreProviderState.queuePair = queuePair;
  coreProviderState.workRequest = workRequest;
  coreProviderState.badWorkRequest = badWorkRequest;
  *badWorkRequest = coreProviderState.result == 0 ? nullptr : workRequest;
  return coreProviderState.result;
}

/**
 * @brief Records an extended port query and supplies the configured provider status.
 * @param[in] context Exact context forwarded by ibv_query_port_ex.
 * @param[in] port Exact port number forwarded by ibv_query_port_ex.
 * @param[out] attributes Receives a recognizable state on provider success.
 * @param[in] attributesSize Exact output size forwarded by ibv_query_port_ex.
 * @return coreProviderState.result.
 * @note Side effects: Records every argument and writes attributes only on success.
 */
int QueryExtendedPortForCoreTest(ibv_context* context, uint8_t port,
  ibv_port_attr* attributes, size_t attributesSize) {
  ++coreProviderState.queryPortCalls;
  coreProviderState.context = context;
  coreProviderState.port = port;
  coreProviderState.portAttributes = attributes;
  coreProviderState.portAttributesSize = attributesSize;
  if (coreProviderState.result == 0) {
    *attributes = {};
    attributes->state = IBV_PORT_ACTIVE;
    attributes->active_mtu = IBV_MTU_2048;
  }
  return coreProviderState.result;
}

}  // namespace

/**
 * @brief Verifies extended-device recovery and legacy-device rejection without modifying
 * either device description.
 */
TEST(IbvCoreInlineTest, RecoversExtendedDeviceContainer) {
  SCOPED_TRACE("Device recovery must distinguish embedded extended devices from legacy provider objects.");
  verbs_device extendedDevice = {};
  const verbs_device extendedSnapshot = extendedDevice;

  EXPECT_EQ(verbs_get_device(&extendedDevice.device), &extendedDevice)
      << "A device without the legacy allocation callback must resolve to its containing extended device";
  EXPECT_EQ(std::memcmp(&extendedDevice, &extendedSnapshot, sizeof(extendedDevice)), 0)
      << "Container recovery must not modify the embedded device or its owner";

  ibv_device legacyDevice = {};
  legacyDevice.ops.alloc_context = LegacyContextAllocationForTest;
  const ibv_device legacySnapshot = legacyDevice;
  EXPECT_EQ(verbs_get_device(&legacyDevice), nullptr)
      << "A legacy device with its own allocation callback has no verbs_device container";
  EXPECT_EQ(std::memcmp(&legacyDevice, &legacySnapshot, sizeof(legacyDevice)), 0)
      << "Legacy-device detection must not modify the provider object";
}

/**
 * @brief Verifies context recovery distinguishes legacy and extended ABI objects without
 * changing either input.
 */
TEST(IbvCoreInlineTest, RecoversExtendedContextContainer) {
  SCOPED_TRACE("Context recovery must accept only the explicit extended-ABI marker.");
  verbs_context extendedContext = {};
  extendedContext.context.abi_compat = __VERBS_ABI_IS_EXTENDED;
  const verbs_context extendedSnapshot = extendedContext;

  EXPECT_EQ(verbs_get_ctx(&extendedContext.context), &extendedContext)
      << "An extended ABI context must resolve to its containing verbs_context";
  EXPECT_EQ(std::memcmp(&extendedContext, &extendedSnapshot, sizeof(extendedContext)), 0)
      << "Extended context recovery must not modify caller-owned context storage";

  ibv_context legacyContext = {};
  const ibv_context legacySnapshot = legacyContext;
  EXPECT_EQ(verbs_get_ctx(&legacyContext), nullptr)
      << "A context without the extended ABI marker must not be treated as embedded";
  EXPECT_EQ(std::memcmp(&legacyContext, &legacySnapshot, sizeof(legacyContext)), 0)
      << "Legacy context rejection must preserve every caller-owned field";
}

/**
 * @brief Verifies extended port queries validate ABI support and forward provider results
 * and outputs exactly.
 */
TEST(IbvCoreInlineTest, QueriesExtendedPort) {
  SCOPED_TRACE("Extended port queries must reject unavailable operations and preserve provider behavior.");
  coreProviderState = {};
  ibv_context legacyContext = {};
  ibv_port_attr attributes = {};
  attributes.state = IBV_PORT_DOWN;
  const ibv_port_attr unavailableSnapshot = attributes;

  EXPECT_EQ(ibv_query_port_ex(&legacyContext, 3, &attributes), -1)
      << "A legacy context cannot expose an extended port-query operation";
  EXPECT_EQ(std::memcmp(&attributes, &unavailableSnapshot, sizeof(attributes)), 0)
      << "Rejecting a legacy context must preserve the output object";
  EXPECT_EQ(coreProviderState.queryPortCalls, 0)
      << "A rejected legacy context must not call an extended provider";

  verbs_context extendedContext = {};
  extendedContext.context.abi_compat = __VERBS_ABI_IS_EXTENDED;
  extendedContext.query_port = QueryExtendedPortForCoreTest;
  extendedContext.sz = sizeof(extendedContext) - offsetof(verbs_context, query_port) - 1;
  EXPECT_EQ(ibv_query_port_ex(&extendedContext.context, 3, &attributes), -1)
      << "A provider context too small to contain query_port must be rejected";
  EXPECT_EQ(std::memcmp(&attributes, &unavailableSnapshot, sizeof(attributes)), 0)
      << "Rejecting an undersized provider context must preserve the output object";
  EXPECT_EQ(coreProviderState.queryPortCalls, 0)
      << "An unavailable operation must not be dispatched";

  extendedContext.sz = sizeof(extendedContext) - offsetof(verbs_context, query_port);
  extendedContext.query_port = nullptr;
  EXPECT_EQ(ibv_query_port_ex(&extendedContext.context, 3, &attributes), -1)
      << "An extended context without a query_port callback must be rejected";
  EXPECT_EQ(std::memcmp(&attributes, &unavailableSnapshot, sizeof(attributes)), 0)
      << "Rejecting a missing callback must preserve the output object";
  EXPECT_EQ(coreProviderState.queryPortCalls, 0)
      << "A missing callback must not be dispatched";

  extendedContext.query_port = QueryExtendedPortForCoreTest;
  EXPECT_EQ(ibv_query_port_ex(&extendedContext.context, 7, &attributes), 0)
      << "A successful extended provider result must be returned unchanged";
  EXPECT_EQ(coreProviderState.queryPortCalls, 1)
      << "A supported extended query must call the provider exactly once";
  EXPECT_EQ(coreProviderState.context, &extendedContext.context)
      << "The provider must receive the caller's context";
  EXPECT_EQ(coreProviderState.port, 7)
      << "The provider must receive the caller's port number";
  EXPECT_EQ(coreProviderState.portAttributes, &attributes)
      << "The provider must receive the caller's output address";
  EXPECT_EQ(coreProviderState.portAttributesSize, sizeof(attributes))
      << "The provider must receive the complete output-structure size";
  EXPECT_EQ(attributes.state, IBV_PORT_ACTIVE)
      << "A successful provider must populate the port state";
  EXPECT_EQ(attributes.active_mtu, IBV_MTU_2048)
      << "A successful provider must populate the active MTU";

  attributes.state = IBV_PORT_ARMED;
  attributes.active_mtu = IBV_MTU_1024;
  const ibv_port_attr providerErrorSnapshot = attributes;
  coreProviderState.result = EIO;
  EXPECT_EQ(ibv_query_port_ex(&extendedContext.context, 9, &attributes), EIO)
      << "A provider error must be returned without translation";
  EXPECT_EQ(coreProviderState.queryPortCalls, 2)
      << "The failing query must make exactly one additional provider call";
  EXPECT_EQ(coreProviderState.port, 9)
      << "The failing query must still forward its port number";
  EXPECT_EQ(std::memcmp(&attributes, &providerErrorSnapshot, sizeof(attributes)), 0)
      << "An error from a non-writing provider must preserve the caller's output structure";
}

/**
 * @brief Verifies direct post-send dispatch forwards every argument, output, and provider status.
 */
TEST(IbvCoreInlineTest, DispatchesPostSend) {
  SCOPED_TRACE("Inline post-send dispatch must preserve provider arguments, inputs, output, and status.");
  coreProviderState = {};
  ibv_context context = {};
  context.ops.post_send = PostSendForCoreTest;
  ibv_qp queuePair = {};
  queuePair.context = &context;
  queuePair.qp_num = 41;
  ibv_send_wr workRequest = {};
  workRequest.wr_id = 0x1234;
  workRequest.opcode = IBV_WR_SEND;
  const ibv_qp queuePairSnapshot = queuePair;
  const ibv_send_wr workRequestSnapshot = workRequest;
  ibv_send_wr* badWorkRequest = reinterpret_cast<ibv_send_wr*>(0x1);

  EXPECT_EQ(ibv_post_send(&queuePair, &workRequest, &badWorkRequest), 0)
      << "A successful provider result must be returned unchanged";
  EXPECT_EQ(coreProviderState.calls, 1) << "Post-send dispatch must call the provider exactly once";
  EXPECT_EQ(coreProviderState.queuePair, &queuePair) << "The provider must receive the caller's queue pair";
  EXPECT_EQ(coreProviderState.workRequest, &workRequest)
      << "The provider must receive the caller's first work request";
  EXPECT_EQ(coreProviderState.badWorkRequest, &badWorkRequest)
      << "The provider must receive the caller's output address";
  EXPECT_EQ(badWorkRequest, nullptr) << "A successful provider must clear the bad-work-request output";
  EXPECT_EQ(std::memcmp(&queuePair, &queuePairSnapshot, sizeof(queuePair)), 0)
      << "Post-send dispatch must not modify queue-pair metadata";
  EXPECT_EQ(std::memcmp(&workRequest, &workRequestSnapshot, sizeof(workRequest)), 0)
      << "Post-send dispatch must not modify the submitted work request";

  coreProviderState.result = EIO;
  badWorkRequest = nullptr;
  EXPECT_EQ(ibv_post_send(&queuePair, &workRequest, &badWorkRequest), EIO)
      << "A provider error must be returned without conversion";
  EXPECT_EQ(coreProviderState.calls, 2) << "The failing dispatch must make exactly one additional provider call";
  EXPECT_EQ(coreProviderState.queuePair, &queuePair)
      << "The failing dispatch must forward the caller's queue pair";
  EXPECT_EQ(coreProviderState.workRequest, &workRequest)
      << "The failing dispatch must forward the caller's work request";
  EXPECT_EQ(coreProviderState.badWorkRequest, &badWorkRequest)
      << "The failing dispatch must forward the caller's output address";
  EXPECT_EQ(badWorkRequest, &workRequest)
      << "A failing provider must be able to identify the rejected work request";
  EXPECT_EQ(std::memcmp(&queuePair, &queuePairSnapshot, sizeof(queuePair)), 0)
      << "A provider error must not modify queue-pair metadata";
  EXPECT_EQ(std::memcmp(&workRequest, &workRequestSnapshot, sizeof(workRequest)), 0)
      << "A provider error must not modify the submitted work request";
}
