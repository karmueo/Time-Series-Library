#ifndef MULTICAST_RECEIVER_H
#define MULTICAST_RECEIVER_H

#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <cstdint>

namespace timesnet {

/**
 * @brief 组播接收器配置
 */
struct ReceiverConfig {
    std::string group;      // 组播地址
    int port;               // 端口
    std::string iface;      // 网卡地址
    std::string bind_ip;    // 绑定地址 (空表示 0.0.0.0)
    double timeout_s;       // 接收超时秒数
    bool skip_checksum;     // 跳过校验和校验
};

/**
 * @brief 接收到的数据信息
 */
struct RecvInfo {
    std::vector<uint8_t> data;
    std::string src_addr;
    int src_port;
};

/**
 * @brief 组播接收器
 *
 * 支持 UDP 组播报文的接收，提供校验和校验和超时机制
 */
class MulticastReceiver {
public:
    explicit MulticastReceiver(const ReceiverConfig& config);
    ~MulticastReceiver();

    // 禁用拷贝
    MulticastReceiver(const MulticastReceiver&) = delete;
    MulticastReceiver& operator=(const MulticastReceiver&) = delete;

    // 移动语义
    MulticastReceiver(MulticastReceiver&& other) noexcept;
    MulticastReceiver& operator=(MulticastReceiver&& other) noexcept;

    /**
     * @brief 打开组播 socket
     * @return 是否成功
     */
    bool open();

    /**
     * @brief 接收组播报文
     * @return 接收到的数据，若超时返回 std::nullopt
     */
    std::optional<RecvInfo> recv();

    /**
     * @brief 接收报文并回调处理
     * @param callback 处理回调，返回 true 继续接收，false 停止
     * @return 实际接收次数
     */
    template<typename Func>
    size_t recv_loop(Func callback);

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
    ReceiverConfig config_;
    int socket_fd_ = -1;
    std::string last_error_;

    bool init_socket();
    bool join_multicast_group();
    uint16_t calculate_checksum(const uint8_t* data, size_t len);
    bool verify_checksum(const uint8_t* data, size_t len);
};

template<typename Func>
size_t MulticastReceiver::recv_loop(Func callback) {
    size_t count = 0;
    while (true) {
        auto result = recv();
        if (!result.has_value()) {
            break;
        }
        if (!callback(result.value())) {
            break;
        }
        ++count;
    }
    return count;
}

} // namespace timesnet

#endif // MULTICAST_RECEIVER_H
