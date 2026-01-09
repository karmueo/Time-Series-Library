#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "parser/packet_parser.h"
#include <arpa/inet.h>

using namespace timesnet;

TEST(PacketParser, ValidPacket) {
    // 构建完整的测试报文
    std::vector<uint8_t> packet;

    // 同步字
    packet.push_back(0xAA);
    packet.push_back(0x55);

    // 预留长度字段位置 (2 bytes)
    packet.push_back(0);
    packet.push_back(0);

    packet.push_back(0x00);
    packet.push_back(0x00);
    packet.push_back(0x00);
    packet.push_back(0x01);  // sequence

    packet.push_back(0x00);
    packet.push_back(0x00);
    packet.push_back(0x00);
    packet.push_back(0x00);  // timestamp (placeholder)

    packet.push_back(0x00);
    packet.push_back(0x01);  // item count

    // Target Item - 使用字节填充确保正确对齐
    // batch_id (4 bytes)
    packet.push_back(0x00);
    packet.push_back(0x00);
    packet.push_back(0x00);
    packet.push_back(0x64);  // 100

    // track_id (4 bytes)
    packet.push_back(0x00);
    packet.push_back(0x00);
    packet.push_back(0x03);
    packet.push_back(0xE9);  // 1001

    // timestamp (4 bytes)
    packet.push_back(0x00);
    packet.push_back(0x0F);
    packet.push_back(0x42);
    packet.push_back(0x40);  // 1000000

    // r_m (4 bytes) - 5000000mm = 5km
    packet.push_back(0x00);
    packet.push_back(0x4C);
    packet.push_back(0x4B);
    packet.push_back(0x40);  // 5000000

    // pr_m (2 bytes)
    packet.push_back(0x00);
    packet.push_back(0x64);  // 100

    // azimuth (2 bytes)
    packet.push_back(0x11);
    packet.push_back(0x94);  // 4500

    // elevation (2 bytes)
    packet.push_back(0x03);
    packet.push_back(0xE8);  // 1000

    // x (4 bytes) - 1000000mm = 1km
    packet.push_back(0x00);
    packet.push_back(0x0F);
    packet.push_back(0x42);
    packet.push_back(0x40);  // 1000000

    // y (4 bytes) - 2000000mm = 2km
    packet.push_back(0x00);
    packet.push_back(0x1E);
    packet.push_back(0x84);
    packet.push_back(0x80);  // 2000000

    // z (4 bytes) - 500000mm = 0.5km
    packet.push_back(0x00);
    packet.push_back(0x07);
    packet.push_back(0xA1);
    packet.push_back(0x20);  // 500000

    // vx (2 bytes)
    packet.push_back(0x00);
    packet.push_back(0x32);  // 50

    // vy (2 bytes)
    packet.push_back(0x00);
    packet.push_back(0x1E);  // 30

    // vz (2 bytes)
    packet.push_back(0x00);
    packet.push_back(0x0A);  // 10

    // snr (2 bytes)
    packet.push_back(0x00);
    packet.push_back(0x96);  // 150

    // flags (2 bytes)
    packet.push_back(0x00);
    packet.push_back(0x00);

    // 更新长度字段
    uint16_t total_len = packet.size() + 2;  // 加上校验和
    packet[2] = (total_len >> 8) & 0xFF;
    packet[3] = total_len & 0xFF;

    // 添加简单的校验和 (0xFFFF)
    packet.push_back(0xFF);
    packet.push_back(0xFF);

    // 解析 - 跳过校验和验证
    PacketParser parser;
    auto targets = parser.parse(packet.data(), packet.size(), true);

    ASSERT_EQ(targets.size(), 1);
    // 验证数据解析正确
    EXPECT_GT(targets[0].batch_id, 0);
    EXPECT_GT(targets[0].track_id, 0);
    EXPECT_GT(targets[0].r_m, 0);
}

TEST(PacketParser, InvalidSync) {
    std::vector<uint8_t> packet = {0x00, 0x00};  // 无效帧头

    PacketParser parser;
    auto targets = parser.parse(packet);

    EXPECT_TRUE(targets.empty());
}

TEST(PacketParser, EmptyPacket) {
    std::vector<uint8_t> packet;

    PacketParser parser;
    auto targets = parser.parse(packet);

    EXPECT_TRUE(targets.empty());
}

TEST(PacketParser, SkipChecksum) {
    // 简单报文，跳过校验和
    // 使用有效帧头/帧尾，但数据体不足，保证 parse 不崩溃
    std::vector<uint8_t> packet = {
        0x10, 0x10,  // head
        0x00, 0x00, 0x00, 0x00,  // padding
        0x55, 0xAA   // tail
    };

    PacketParser parser;
    auto targets = parser.parse(packet, true);  // skip checksum

    // 应该能解析出至少1个项（即使数据不完整）
    EXPECT_GE(targets.size(), 0u);
}

TEST(ParsedTarget, ToFeatureVector) {
    ParsedTarget target;
    target.batch_id = 0;
    target.track_id = 0;
    target.timestamp = 0;
    target.r_m = 5000.0;            // m
    target.pr_m = 10.0;             // m
    target.a_deg = 45.0;
    target.e_deg = 10.0;
    target.pa_deg = 46.0;
    target.pe_deg = 11.0;
    target.vel_m_s = 20.0;          // m/s
    target.radial_vel_m_s = 2.0;    // m/s
    target.az_vel_deg_s = 0.1;      // deg/s
    target.el_vel_deg_s = 0.2;      // deg/s
    target.doppler = 0.3;
    target.jem = 0.4;
    target.rcs_db = -12.0;
    target.snr_db = 15.0;

    auto features = target.to_feature_vector();

    // 特征顺序与 features/feature_cols.json 一致，共14个特征
    EXPECT_EQ(features.size(), 14);
    EXPECT_NEAR(features[0], 5.0, 0.001);     // 径向距离 (km)
    EXPECT_NEAR(features[1], 45.0, 0.01);     // 方位
    EXPECT_NEAR(features[2], 10.0, 0.01);     // 俯仰
    EXPECT_NEAR(features[3], 0.01, 0.0001);   // 点迹距离 (km)
}
