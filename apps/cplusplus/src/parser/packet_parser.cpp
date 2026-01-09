#include "parser/packet_parser.h"
#include "logger.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace timesnet {

namespace {

constexpr uint16_t kTrkFlagHead = 0x1010;
constexpr uint16_t kTrkFlagTail = 0x55AA;
constexpr size_t kTarStr = 26;
constexpr size_t kTarLen = 160;

uint16_t read_u16_le(const uint8_t* data, size_t offset) {
    return static_cast<uint16_t>(data[offset]) |
           (static_cast<uint16_t>(data[offset + 1]) << 8);
}

int16_t read_i16_le(const uint8_t* data, size_t offset) {
    return static_cast<int16_t>(read_u16_le(data, offset));
}

uint32_t read_u32_le(const uint8_t* data, size_t offset) {
    return static_cast<uint32_t>(data[offset]) |
           (static_cast<uint32_t>(data[offset + 1]) << 8) |
           (static_cast<uint32_t>(data[offset + 2]) << 16) |
           (static_cast<uint32_t>(data[offset + 3]) << 24);
}

int32_t read_i32_le(const uint8_t* data, size_t offset) {
    return static_cast<int32_t>(read_u32_le(data, offset));
}

float read_f32_le(const uint8_t* data, size_t offset) {
    uint32_t raw = read_u32_le(data, offset);
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

} // namespace

std::vector<ParsedTarget> PacketParser::parse(const uint8_t* data, size_t len, bool skip_checksum) {
    std::vector<ParsedTarget> targets;

    LOG_INFO("Parsing packet: len=" + std::to_string(len));

    if (len < 30) {
        last_error_ = "Packet too short: " + std::to_string(len);
        LOG_WARNING(last_error_);
        return targets;
    }

    uint16_t head = read_u16_le(data, 0);
    uint16_t tail = read_u16_le(data, len - 2);

    char head_str[16], tail_str[16];
    snprintf(head_str, sizeof(head_str), "0x%04x", head);
    snprintf(tail_str, sizeof(tail_str), "0x%04x", tail);

    LOG_INFO("Packet flags: head=" + std::string(head_str) +
             " (expected 0x1010), tail=" + std::string(tail_str) +
             " (expected 0x55aa)");

    if (head != kTrkFlagHead || tail != kTrkFlagTail) {
        last_error_ = "Invalid head/tail flags";
        LOG_WARNING(last_error_);
        return targets;
    }

    if (!skip_checksum && !verify_checksum(data, len)) {
        last_error_ = "Checksum verification failed";
        LOG_WARNING(last_error_);
        return targets;
    }

    uint16_t item_count = read_u16_le(data, 24);
    LOG_INFO("Item count: " + std::to_string(item_count));

    for (uint16_t i = 0; i < item_count; ++i) {
        size_t start = kTarStr + static_cast<size_t>(i) * kTarLen;
        size_t end = start + kTarLen;
        if (end > len) {
            LOG_WARNING("Item " + std::to_string(i) + " exceeds packet length");
            break;
        }

        ParsedTarget target;
        uint16_t tar_id = read_u16_le(data, start + 2);
        uint32_t tar_seq = read_u32_le(data, start + 6);
        uint32_t t_25us = read_u32_le(data, start + 22);
        uint32_t r_raw = read_u32_le(data, start + 26);
        uint32_t a_raw = read_u32_le(data, start + 30);
        int32_t e_raw = read_i32_le(data, start + 34);
        uint32_t pr_raw = read_u32_le(data, start + 38);
        uint32_t pa_raw = read_u32_le(data, start + 42);
        int32_t pe_raw = read_i32_le(data, start + 46);
        int16_t radial_vel_raw = read_i16_le(data, start + 50);
        int16_t az_vel_raw = read_i16_le(data, start + 54);
        int16_t el_vel_raw = read_i16_le(data, start + 56);
        uint32_t vel_raw = read_u32_le(data, start + 58);
        uint16_t snr_raw = read_u16_le(data, start + 80);
        int16_t rcs_raw = read_i16_le(data, start + 82);
        float doppler = read_f32_le(data, start + 152);
        float jem = read_f32_le(data, start + 156);

        target.batch_id = tar_id;
        target.track_id = tar_id;
        target.timestamp = static_cast<double>(t_25us) * 25e-6;
        target.r_m = static_cast<double>(r_raw) * 1e-1;
        target.a_deg = static_cast<double>(a_raw) * 1e-5;
        target.e_deg = static_cast<double>(e_raw) * 1e-5;
        target.pr_m = static_cast<double>(pr_raw) * 1e-1;
        target.pa_deg = static_cast<double>(pa_raw) * 1e-5;
        target.pe_deg = static_cast<double>(pe_raw) * 1e-5;
        target.vel_m_s = static_cast<double>(vel_raw) * 1e-1;
        target.radial_vel_m_s = static_cast<double>(radial_vel_raw) * 1e-2;
        target.az_vel_deg_s = static_cast<double>(az_vel_raw) * 1e-3;
        target.el_vel_deg_s = static_cast<double>(el_vel_raw) * 1e-3;
        target.doppler = static_cast<double>(doppler);
        target.jem = static_cast<double>(jem);
        target.rcs_db = static_cast<double>(rcs_raw) * 0.01;
        target.snr_db = static_cast<double>(snr_raw) * 0.01;

        // tar_seq 暂不参与流程，保留用于后续扩展
        (void)tar_seq;

        targets.push_back(target);
    }

    LOG_INFO("Parsed " + std::to_string(targets.size()) + " targets");
    return targets;
}

std::vector<ParsedTarget> PacketParser::parse(const std::vector<uint8_t>& data, bool skip_checksum) {
    return parse(data.data(), data.size(), skip_checksum);
}

bool PacketParser::verify_checksum(const uint8_t* data, size_t len) {
    // 校验和在倒数第4到第2字节 (与 Python 版本一致)
    if (len < 4) {
        return false;
    }

    uint16_t stored_checksum = read_u16_le(data, len - 4);
    size_t body_len = len - 4;
    uint32_t sum = 0;
    for (size_t i = 0; i < body_len; ++i) {
        sum += data[i];
    }

    return (sum & 0xFFFF) == stored_checksum;
}

std::vector<float> ParsedTarget::to_feature_vector() const {
    // 特征顺序与 Python 版本 feature_cols.json 一致 (14个特征)
    return {
        static_cast<float>(r_m / 1000.0),                // 1. 径向距离 (km)
        static_cast<float>(a_deg),                       // 2. 方位 (度)
        static_cast<float>(e_deg),                       // 3. 俯仰 (度)
        static_cast<float>(pr_m / 1000.0),               // 4. 点迹距离 (km)
        static_cast<float>(pa_deg),                      // 5. 点迹方位 (度)
        static_cast<float>(pe_deg),                      // 6. 点迹俯仰 (度)
        static_cast<float>(vel_m_s / 1000.0),            // 7. 全速度 (km/s)
        static_cast<float>(radial_vel_m_s / 1000.0),     // 8. 径向速度 (km/s)
        static_cast<float>(az_vel_deg_s),                // 9. 方位速度 (度/s)
        static_cast<float>(el_vel_deg_s),                // 10. 俯仰速度 (度/s)
        static_cast<float>(doppler),                     // 11. 多普勒展宽
        static_cast<float>(jem),                         // 12. JEM
        static_cast<float>(rcs_db),                      // 13. RCS
        static_cast<float>(snr_db)                       // 14. 目标信噪比
    };
}

} // namespace timesnet
