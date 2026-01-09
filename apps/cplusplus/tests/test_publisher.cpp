#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "publisher/multicast_publisher.h"

using namespace timesnet;

class MulticastPublisherTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.group = "239.0.0.1";
        config_.port = 5001;
        config_.iface = "192.168.1.1";
        config_.ttl = 1;
    }

    PublisherConfig config_;
};

TEST_F(MulticastPublisherTest, ConfigWithValues) {
    EXPECT_EQ(config_.group, "239.0.0.1");
    EXPECT_EQ(config_.port, 5001);
    EXPECT_EQ(config_.ttl, 1);
}

TEST_F(MulticastPublisherTest, NotOpenBeforeOpen) {
    MulticastPublisher publisher(config_);
    EXPECT_FALSE(publisher.is_open());
}

TEST_F(MulticastPublisherTest, InvalidGroup) {
    PublisherConfig invalid_config;
    invalid_config.group = "invalid";
    invalid_config.port = 5001;
    invalid_config.ttl = 1;

    MulticastPublisher publisher(invalid_config);
    EXPECT_FALSE(publisher.open());
}

TEST_F(MulticastPublisherTest, ZeroPort) {
    PublisherConfig zero_port_config;
    zero_port_config.group = "239.0.0.1";
    zero_port_config.port = 0;
    zero_port_config.ttl = 1;

    MulticastPublisher publisher(zero_port_config);
    // Should handle zero port
}

TEST_F(MulticastPublisherTest, ZeroTTL) {
    PublisherConfig zero_ttl_config;
    zero_ttl_config.group = "239.0.0.1";
    zero_ttl_config.port = 5001;
    zero_ttl_config.ttl = 0;

    MulticastPublisher publisher(zero_ttl_config);
    // TTL 0 means local host only
}

TEST_F(MulticastPublisherTest, HighTTL) {
    PublisherConfig high_ttl_config;
    high_ttl_config.group = "239.0.0.1";
    high_ttl_config.port = 5001;
    high_ttl_config.ttl = 255;

    MulticastPublisher publisher(high_ttl_config);
    // Should handle high TTL (will be clamped to 255)
}

TEST_F(MulticastPublisherTest, Movable) {
    MulticastPublisher publisher1(config_);
    MulticastPublisher publisher2(std::move(publisher1));

    EXPECT_FALSE(publisher2.is_open());
}

TEST_F(MulticastPublisherTest, MovableAssignment) {
    MulticastPublisher publisher1(config_);
    MulticastPublisher publisher2(config_);
    publisher2 = std::move(publisher1);

    EXPECT_FALSE(publisher2.is_open());
}

TEST_F(MulticastPublisherTest, LastErrorInitial) {
    MulticastPublisher publisher(config_);
    EXPECT_TRUE(publisher.last_error().empty());
}

TEST_F(MulticastPublisherTest, PredictionResultFields) {
    PredictionResult result;
    result.batch_id = 1;
    result.track_id = 100;
    result.timestamp_ms = 1234567890;
    result.pred = 1;
    result.prob_uav = 0.95f;
    result.prob_bird = 0.05f;

    EXPECT_EQ(result.batch_id, 1);
    EXPECT_EQ(result.track_id, 100);
    EXPECT_EQ(result.pred, 1);
    EXPECT_NEAR(result.prob_uav, 0.95f, 0.001f);
    EXPECT_NEAR(result.prob_bird, 0.05f, 0.001f);
}
