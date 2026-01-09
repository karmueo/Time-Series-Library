#ifndef PACKET_PARSER_H
#define PACKET_PARSER_H

#include <vector>
#include <cstdint>
#include <string>
#include <optional>

namespace timesnet {

/**
 * @brief 解析后的目标信息
 */
struct ParsedTarget {
    uint32_t batch_id;         // 批号
    uint32_t track_id;         // 轨迹 ID
    double timestamp;          // 时间戳 (秒)
    double r_m;                // 滤波径向距离 (m)
    double a_deg;              // 滤波方位 (度)
    double e_deg;              // 滤波俯仰 (度)
    double pr_m;               // 点迹距离 (m)
    double pa_deg;             // 点迹方位 (度)
    double pe_deg;             // 点迹俯仰 (度)
    double vel_m_s;            // 全速度 (m/s)
    double radial_vel_m_s;     // 径向速度 (m/s)
    double az_vel_deg_s;       // 方位速度 (度/s)
    double el_vel_deg_s;       // 俯仰速度 (度/s)
    double doppler;            // 多普勒展宽
    double jem;                // JEM 特征
    double rcs_db;             // RCS (dB)
    double snr_db;             // 目标信噪比 (dB)

    // 转换为特征向量
    std::vector<float> to_feature_vector() const;
};

/**
 * @brief 报文解析器
 *
 * 解析自定义格式的 UDP 报文
 */
class PacketParser {
public:
    PacketParser() = default;

    /**
     * @brief 解析报文
     * @param data 原始报文数据
     * @param skip_checksum 是否跳过校验和校验
     * @return 解析后的目标列表，解析失败返回空
     */
    std::vector<ParsedTarget> parse(const uint8_t* data, size_t len, bool skip_checksum = false);

    /**
     * @brief 解析报文 (vector 版本)
     */
    std::vector<ParsedTarget> parse(const std::vector<uint8_t>& data, bool skip_checksum = false);

    /**
     * @brief 获取最后一个错误信息
     */
    const std::string& last_error() const { return last_error_; }

    /**
     * @brief 检查校验和
     */
    static bool verify_checksum(const uint8_t* data, size_t len);

private:
    std::string last_error_;
};

} // namespace timesnet

#endif // PACKET_PARSER_H
