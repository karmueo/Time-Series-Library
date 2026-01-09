#ifndef MULTICAST_PUBLISHER_H
#define MULTICAST_PUBLISHER_H

#include <string>
#include <vector>
#include <cstdint>
#include <sys/socket.h>
#include <netinet/in.h>

namespace timesnet {

/**
 * @brief 组播发布器配置
 */
struct PublisherConfig {
    std::string group;   // 组播地址
    int port;            // 端口
    std::string iface;   // 网卡地址
    int ttl;             // 组播 TTL
};

/**
 * @brief 预测结果
 */
struct PredictionResult {
    uint32_t batch_id;
    uint32_t track_id;
    int64_t timestamp_ms;
    int pred;            // 预测类别 (0=bird, 1=uav)
    float prob_uav;      // UAV 概率
    float prob_bird;     // bird 概率
};

/**
 * @brief 组播发布器
 *
 * 支持 UDP 组播报文的发送
 */
class MulticastPublisher {
public:
    explicit MulticastPublisher(const PublisherConfig& config);
    ~MulticastPublisher();

    // 禁用拷贝
    MulticastPublisher(const MulticastPublisher&) = delete;
    MulticastPublisher& operator=(const MulticastPublisher&) = delete;

    // 移动语义
    MulticastPublisher(MulticastPublisher&& other) noexcept;
    MulticastPublisher& operator=(MulticastPublisher&& other) noexcept;

    /**
     * @brief 打开组播 socket
     * @return 是否成功
     */
    bool open();

    /**
     * @brief 发送预测结果
     * @param results 预测结果列表
     * @return 是否成功
     */
    bool send(const std::vector<PredictionResult>& results);

    /**
     * @brief 发送单条预测结果
     * @param result 预测结果
     * @return 是否成功
     */
    bool send(const PredictionResult& result);

    /**
     * @brief 关闭 socket
     */
    void close();

    /**
     * @brief 检查是否已打开
     */
    bool is_open() const { return socket_fd_ >= 0; }

    /**
     * @brief 获取最后一个错误信息
     */
    const std::string& last_error() const { return last_error_; }

private:
    PublisherConfig config_;
    int socket_fd_ = -1;
    std::string last_error_;
    struct sockaddr_in dest_addr_;

    bool init_socket();
    bool set_multicast_options();
    uint8_t* serialize_predictions(const std::vector<PredictionResult>& results, size_t& out_len);
};

} // namespace timesnet

#endif // MULTICAST_PUBLISHER_H
