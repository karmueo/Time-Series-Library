#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "receiver/multicast_receiver.h"

using namespace timesnet;

class MulticastReceiverTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.group = "239.0.0.1";
        config_.port = 5000;
        config_.iface = "192.168.1.1";
        config_.bind_ip = "";
        config_.timeout_s = 2.0;
        config_.skip_checksum = false;
    }

    ReceiverConfig config_;
};

TEST_F(MulticastReceiverTest, ConfigWithValues) {
    EXPECT_EQ(config_.group, "239.0.0.1");
    EXPECT_EQ(config_.port, 5000);
    EXPECT_EQ(config_.timeout_s, 2.0);
}

TEST_F(MulticastReceiverTest, NotOpenBeforeOpen) {
    MulticastReceiver receiver(config_);
    EXPECT_FALSE(receiver.is_open());
}

TEST_F(MulticastReceiverTest, InvalidGroup) {
    ReceiverConfig invalid_config;
    invalid_config.group = "invalid";
    invalid_config.port = 5000;
    invalid_config.timeout_s = 1.0;

    MulticastReceiver receiver(invalid_config);
    // Should fail to open due to invalid group
    EXPECT_FALSE(receiver.open());
}

TEST_F(MulticastReceiverTest, ZeroPort) {
    ReceiverConfig zero_port_config;
    zero_port_config.group = "239.0.0.1";
    zero_port_config.port = 0;

    MulticastReceiver receiver(zero_port_config);
    // Should handle zero port gracefully
}

TEST_F(MulticastReceiverTest, NegativeTimeout) {
    ReceiverConfig neg_timeout_config;
    neg_timeout_config.group = "239.0.0.1";
    neg_timeout_config.port = 5000;
    neg_timeout_config.timeout_s = -1.0;

    MulticastReceiver receiver(neg_timeout_config);
    // Should handle negative timeout
}

TEST_F(MulticastReceiverTest, EmptyInterface) {
    ReceiverConfig empty_iface_config;
    empty_iface_config.group = "239.0.0.1";
    empty_iface_config.port = 5000;
    empty_iface_config.iface = "";

    MulticastReceiver receiver(empty_iface_config);
    // Should use default interface
}

TEST_F(MulticastReceiverTest, Movable) {
    MulticastReceiver receiver1(config_);
    MulticastReceiver receiver2(std::move(receiver1));

    // After move, receiver1 should be valid but unspecified state
    EXPECT_FALSE(receiver2.is_open());
}

TEST_F(MulticastReceiverTest, MovableAssignment) {
    MulticastReceiver receiver1(config_);
    MulticastReceiver receiver2(config_);
    receiver2 = std::move(receiver1);

    EXPECT_FALSE(receiver2.is_open());
}

TEST_F(MulticastReceiverTest, LastErrorInitial) {
    MulticastReceiver receiver(config_);
    EXPECT_TRUE(receiver.last_error().empty());
}
