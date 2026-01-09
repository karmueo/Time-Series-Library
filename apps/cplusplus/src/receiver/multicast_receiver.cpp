#include "receiver/multicast_receiver.h"
#include "logger.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <algorithm>

namespace timesnet {

MulticastReceiver::MulticastReceiver(const ReceiverConfig& config)
    : config_(config), socket_fd_(-1) {}

MulticastReceiver::~MulticastReceiver() {
    close();
}

MulticastReceiver::MulticastReceiver(MulticastReceiver&& other) noexcept
    : config_(other.config_), socket_fd_(other.socket_fd_), last_error_(other.last_error_) {
    other.socket_fd_ = -1;
}

MulticastReceiver& MulticastReceiver::operator=(MulticastReceiver&& other) noexcept {
    if (this != &other) {
        close();
        config_ = other.config_;
        socket_fd_ = other.socket_fd_;
        last_error_ = other.last_error_;
        other.socket_fd_ = -1;
    }
    return *this;
}

bool MulticastReceiver::init_socket() {
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        last_error_ = "Failed to create socket: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return false;
    }

    // 设置接收缓冲区
    int recv_buf_size = 1024 * 1024; // 1MB
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &recv_buf_size, sizeof(recv_buf_size)) < 0) {
        LOG_WARNING("Failed to set SO_RCVBUF: " + std::string(strerror(errno)));
    }

    // 允许地址复用
    int reuse = 1;
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        LOG_WARNING("Failed to set SO_REUSEADDR: " + std::string(strerror(errno)));
    }

    // 设置非阻塞
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    if (fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
        last_error_ = "Failed to set non-blocking: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return false;
    }

    // 绑定地址
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.port);

    if (!config_.bind_ip.empty()) {
        inet_pton(AF_INET, config_.bind_ip.c_str(), &addr.sin_addr);
    } else {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    }

    if (bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        last_error_ = "Failed to bind: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return false;
    }

    return true;
}

bool MulticastReceiver::join_multicast_group() {
    struct ip_mreq mreq;
    memset(&mreq, 0, sizeof(mreq));

    if (inet_pton(AF_INET, config_.group.c_str(), &mreq.imr_multiaddr) <= 0) {
        last_error_ = "Invalid multicast group address: " + config_.group;
        LOG_ERROR(last_error_);
        return false;
    }

    if (inet_pton(AF_INET, config_.iface.c_str(), &mreq.imr_interface) <= 0) {
        // 如果接口地址无效，使用默认接口
        mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    }

    if (setsockopt(socket_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        last_error_ = "Failed to join multicast group: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return false;
    }

    LOG_INFO("Joined multicast group " + config_.group + " on interface " + config_.iface);
    return true;
}

bool MulticastReceiver::open() {
    if (socket_fd_ >= 0) {
        LOG_WARNING("Socket already opened");
        return true;
    }

    if (!init_socket()) {
        return false;
    }

    if (!join_multicast_group()) {
        close();
        return false;
    }

    LOG_INFO("Multicast receiver opened: " + config_.group + ":" + std::to_string(config_.port));
    return true;
}

std::optional<RecvInfo> MulticastReceiver::recv() {
    if (socket_fd_ < 0) {
        last_error_ = "Socket not opened";
        return std::nullopt;
    }

    // 设置超时
    struct timeval tv;
    tv.tv_sec = static_cast<long>(config_.timeout_s);
    tv.tv_usec = static_cast<long>((config_.timeout_s - tv.tv_sec) * 1000000);

    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(socket_fd_, &read_fds);

    int ret = select(socket_fd_ + 1, &read_fds, nullptr, nullptr, &tv);
    if (ret < 0) {
        last_error_ = "Select error: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return std::nullopt;
    }

    if (ret == 0) {
        // 超时
        return std::nullopt;
    }

    // 接收数据
    char buffer[65536];
    struct sockaddr_in src_addr;
    socklen_t addr_len = sizeof(src_addr);

    ssize_t len = recvfrom(socket_fd_, buffer, sizeof(buffer), 0,
                           (struct sockaddr*)&src_addr, &addr_len);

    if (len < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return std::nullopt;
        }
        last_error_ = "Recvfrom error: " + std::string(strerror(errno));
        LOG_ERROR(last_error_);
        return std::nullopt;
    }

    LOG_INFO("Received packet: len=" + std::to_string(len) +
             " bytes from " + std::string(inet_ntoa(src_addr.sin_addr)) +
             ":" + std::to_string(ntohs(src_addr.sin_port)));

    // 校验和校验
    if (!config_.skip_checksum && len >= 2) {
        uint16_t stored_checksum = static_cast<uint8_t>(buffer[len - 4]) |
                                   (static_cast<uint8_t>(buffer[len - 3]) << 8);
        if (!verify_checksum(reinterpret_cast<const uint8_t*>(buffer), len)) {
            LOG_WARNING("Checksum verification failed: stored=0x" +
                       std::to_string(stored_checksum) + ", len=" + std::to_string(len));
            return std::nullopt;
        }
    }

    RecvInfo info;
    info.data.assign(buffer, buffer + len);
    char addr_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &src_addr.sin_addr, addr_str, sizeof(addr_str));
    info.src_addr = addr_str;
    info.src_port = ntohs(src_addr.sin_port);

    return info;
}

uint16_t MulticastReceiver::calculate_checksum(const uint8_t* data, size_t len) {
    if (len == 0) {
        return 0;
    }

    uint32_t sum = 0;
    for (size_t i = 0; i < len; ++i) {
        sum += data[i];
    }

    return static_cast<uint16_t>(sum & 0xFFFF);
}

bool MulticastReceiver::verify_checksum(const uint8_t* data, size_t len) {
    if (len < 4) {
        return false;
    }

    // 校验和位于倒数第4到第2字节，最后2字节为帧尾
    uint16_t stored_checksum = static_cast<uint16_t>(data[len - 4]) |
                               (static_cast<uint16_t>(data[len - 3]) << 8);
    uint16_t calculated = calculate_checksum(data, len - 4);

    return stored_checksum == calculated;
}

void MulticastReceiver::close() {
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
        LOG_INFO("Multicast receiver closed");
    }
}

} // namespace timesnet
