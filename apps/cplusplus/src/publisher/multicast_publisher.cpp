#include "publisher/multicast_publisher.h"
#include "logger.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>

namespace timesnet {

MulticastPublisher::MulticastPublisher(const PublisherConfig& config)
    : config_(config), socket_fd_(-1) {
    memset(&dest_addr_, 0, sizeof(dest_addr_));
}

MulticastPublisher::~MulticastPublisher() {
    close();
}

MulticastPublisher::MulticastPublisher(MulticastPublisher&& other) noexcept
    : config_(other.config_), socket_fd_(other.socket_fd_), last_error_(other.last_error_) {
    memcpy(&dest_addr_, &other.dest_addr_, sizeof(dest_addr_));
    other.socket_fd_ = -1;
}

MulticastPublisher& MulticastPublisher::operator=(MulticastPublisher&& other) noexcept {
    if (this != &other) {
        close();
        config_ = other.config_;
        socket_fd_ = other.socket_fd_;
        last_error_ = other.last_error_;
        memcpy(&dest_addr_, &other.dest_addr_, sizeof(dest_addr_));
        other.socket_fd_ = -1;
    }
    return *this;
}

bool MulticastPublisher::init_socket() {
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        last_error_ = "Failed to create socket: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return false;
    }

    // 设置发送缓冲区
    int send_buf_size = 1024 * 1024; // 1MB
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &send_buf_size, sizeof(send_buf_size)) < 0) {
        LOG_WARNING("Failed to set SO_SNDBUF: " + std::string(strerror(errno)));
    }

    // 设置目标地址
    dest_addr_.sin_family = AF_INET;
    dest_addr_.sin_port = htons(config_.port);

    if (inet_pton(AF_INET, config_.group.c_str(), &dest_addr_.sin_addr) <= 0) {
        last_error_ = "Invalid multicast group address: " + config_.group;
        LOG_ERROR(last_error_);
        return false;
    }

    return true;
}

bool MulticastPublisher::set_multicast_options() {
    // 设置组播 TTL
    unsigned char ttl = static_cast<unsigned char>(std::min(config_.ttl, 255));
    if (setsockopt(socket_fd_, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0) {
        last_error_ = "Failed to set multicast TTL: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return false;
    }

    // 设置输出接口
    struct in_addr iface_addr;
    if (inet_pton(AF_INET, config_.iface.c_str(), &iface_addr) <= 0) {
        // 使用默认接口
        iface_addr.s_addr = htonl(INADDR_ANY);
    }

    if (setsockopt(socket_fd_, IPPROTO_IP, IP_MULTICAST_IF, &iface_addr, sizeof(iface_addr)) < 0) {
        LOG_WARNING("Failed to set multicast interface: " + std::string(strerror(errno)));
    }

    // 允许广播
    int broadcast = 1;
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) < 0) {
        LOG_WARNING("Failed to set SO_BROADCAST: " + std::string(strerror(errno)));
    }

    return true;
}

bool MulticastPublisher::open() {
    if (socket_fd_ >= 0) {
        LOG_WARNING("Socket already opened");
        return true;
    }

    if (!init_socket()) {
        return false;
    }

    if (!set_multicast_options()) {
        close();
        return false;
    }

    LOG_INFO("Multicast publisher opened: " + config_.group + ":" + std::to_string(config_.port));
    return true;
}

uint8_t* MulticastPublisher::serialize_predictions(const std::vector<PredictionResult>& results, size_t& out_len) {
    // 报文格式:
    // [item_count: 2 bytes][items...]
    // 每个 item:
    // [track_id: 4 bytes][timestamp: 8 bytes][pred: 1 byte][prob_uav: 4 bytes][prob_bird: 4 bytes]

    const size_t item_size = 4 + 8 + 1 + 4 + 4; // 21 bytes per item
    out_len = 2 + results.size() * item_size;

    uint8_t* buffer = new uint8_t[out_len];
    size_t offset = 0;

    // item count (big-endian)
    buffer[offset++] = (results.size() >> 8) & 0xFF;
    buffer[offset++] = results.size() & 0xFF;

    for (const auto& result : results) {
        // track_id (big-endian)
        buffer[offset++] = (result.track_id >> 24) & 0xFF;
        buffer[offset++] = (result.track_id >> 16) & 0xFF;
        buffer[offset++] = (result.track_id >> 8) & 0xFF;
        buffer[offset++] = result.track_id & 0xFF;

        // timestamp_ms (big-endian)
        buffer[offset++] = (result.timestamp_ms >> 56) & 0xFF;
        buffer[offset++] = (result.timestamp_ms >> 48) & 0xFF;
        buffer[offset++] = (result.timestamp_ms >> 40) & 0xFF;
        buffer[offset++] = (result.timestamp_ms >> 32) & 0xFF;
        buffer[offset++] = (result.timestamp_ms >> 24) & 0xFF;
        buffer[offset++] = (result.timestamp_ms >> 16) & 0xFF;
        buffer[offset++] = (result.timestamp_ms >> 8) & 0xFF;
        buffer[offset++] = result.timestamp_ms & 0xFF;

        // pred
        buffer[offset++] = static_cast<uint8_t>(result.pred);

        // prob_uav (float to bytes)
        uint32_t prob_uav_bits;
        memcpy(&prob_uav_bits, &result.prob_uav, sizeof(float));
        buffer[offset++] = (prob_uav_bits >> 24) & 0xFF;
        buffer[offset++] = (prob_uav_bits >> 16) & 0xFF;
        buffer[offset++] = (prob_uav_bits >> 8) & 0xFF;
        buffer[offset++] = prob_uav_bits & 0xFF;

        // prob_bird (float to bytes)
        uint32_t prob_bird_bits;
        memcpy(&prob_bird_bits, &result.prob_bird, sizeof(float));
        buffer[offset++] = (prob_bird_bits >> 24) & 0xFF;
        buffer[offset++] = (prob_bird_bits >> 16) & 0xFF;
        buffer[offset++] = (prob_bird_bits >> 8) & 0xFF;
        buffer[offset++] = prob_bird_bits & 0xFF;
    }

    return buffer;
}

bool MulticastPublisher::send(const PredictionResult& result) {
    return send(std::vector<PredictionResult>{result});
}

bool MulticastPublisher::send(const std::vector<PredictionResult>& results) {
    if (socket_fd_ < 0) {
        last_error_ = "Socket not opened";
        return false;
    }

    if (results.empty()) {
        LOG_WARNING("Empty results, skip sending");
        return true;
    }

    size_t buffer_len;
    uint8_t* buffer = serialize_predictions(results, buffer_len);

    ssize_t sent = sendto(socket_fd_, buffer, buffer_len, 0,
                          (struct sockaddr*)&dest_addr_, sizeof(dest_addr_));

    delete[] buffer;

    if (sent < 0) {
        last_error_ = "Send failed: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return false;
    }

    if (static_cast<size_t>(sent) != buffer_len) {
        LOG_WARNING("Partial send: " + std::to_string(sent) + "/" + std::to_string(buffer_len));
    }

    LOG_DEBUG("Sent " + std::to_string(results.size()) + " predictions");
    return true;
}

void MulticastPublisher::close() {
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
        LOG_INFO("Multicast publisher closed");
    }
}

} // namespace timesnet
