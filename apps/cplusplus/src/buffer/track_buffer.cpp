#include "buffer/track_buffer.h"
#include "logger.h"
#include <algorithm>
#include <cmath>

namespace timesnet {

TrackWindowBuffer::TrackWindowBuffer(const Config& config) : config_(config) {}

void TrackWindowBuffer::update(uint32_t batch_id, uint32_t track_id,
                               const std::vector<float>& features, double timestamp) {
    auto& track = buffers_[batch_id];

    // 如果是新轨迹，初始化计数器
    if (track.points.empty()) {
        track.track_id = std::to_string(track_id);
        track.last_update = timestamp;
        track.since_last_infer = 0;
        track.has_inferred = false;
    }

    // 添加新点
    TrackPoint point;
    point.features = features;
    point.timestamp = timestamp;
    point.inferred = false;

    size_t size_before = track.points.size();
    track.points.push_back(std::move(point));
    track.last_update = timestamp;

    // 累积新点计数器（用于窗口步长控制）
    track.since_last_infer++;

    LOG_DEBUG("Before cleanup: batch_id=" + std::to_string(batch_id) +
              ", size_before=" + std::to_string(size_before) +
              ", size_after=" + std::to_string(track.points.size()) +
              ", timestamp=" + std::to_string(timestamp) +
              ", last_update=" + std::to_string(track.last_update) +
              ", since_last_infer=" + std::to_string(track.since_last_infer));

    // 清理过期点 (基于相对时间)
    cleanup(timestamp);

    LOG_DEBUG("After cleanup: batch_id=" + std::to_string(batch_id) +
              ", track_id=" + std::to_string(track_id) +
              ", size=" + std::to_string(track.points.size()));
}

std::unordered_map<uint32_t, PendingInfo> TrackWindowBuffer::get_pending_info(
    int min_seq_len, int window_step) const {

    std::unordered_map<uint32_t, PendingInfo> pending;

    for (const auto& [batch_id, track] : buffers_) {
        size_t length = track.points.size();

        if (!track.has_inferred) {
            // 还没推理过：显示还需要多少点才能达到 min_seq_len
            int needed = std::max(0, min_seq_len - static_cast<int>(length));
            if (length > 0 && needed > 0) {
                pending[batch_id] = {static_cast<int>(length), needed};
            }
        } else if (window_step > 0) {
            // 已经推理过：显示还需要多少新点才能达到 window_step
            int delta = track.since_last_infer;
            if (delta < window_step) {
                pending[batch_id] = {delta, window_step};
            }
        }
    }

    return pending;
}

std::optional<BatchResult> TrackWindowBuffer::build_batch(int min_seq_len, int window_step) {

    // 先清理一次，确保窗口大小正确（与 Python 版本一致）
    cleanup();

    BatchResult result;

    for (auto& [batch_id, track] : buffers_) {
        // 使用所有点（与 Python 版本一致，不过滤 inferred）
        size_t length = track.points.size();
        if (length < static_cast<size_t>(min_seq_len)) {
            continue;
        }

        // 如果启用窗口步长控制，检查距离上次推理的新点数是否足够
        if (window_step > 0 && track.has_inferred) {
            if (track.since_last_infer < window_step) {
                // 新点数不足，跳过此轨迹
                LOG_DEBUG("Skipping batch_id=" + std::to_string(batch_id) +
                         ", since_last_infer=" + std::to_string(track.since_last_infer) +
                         ", window_step=" + std::to_string(window_step));
                continue;
            }
        }

        // 收集所有点的特征
        std::vector<std::vector<float>> seq_features;
        for (const auto& point : track.points) {
            seq_features.push_back(point.features);
        }

        // 如果超过 seq_len，只取最后 seq_len 个点（模拟 deque(maxlen=seq_len)）
        if (static_cast<int>(seq_features.size()) > config_.seq_len) {
            int start = static_cast<int>(seq_features.size()) - config_.seq_len;
            seq_features = std::vector<std::vector<float>>(
                seq_features.begin() + start,
                seq_features.end());
            length = config_.seq_len;
        }

        // 如果不足 seq_len，填充 0（与 Python 版本一致）
        if (static_cast<int>(seq_features.size()) < config_.seq_len) {
            if (!seq_features.empty()) {
                size_t feature_dim = seq_features[0].size();
                size_t pad_size = config_.seq_len - seq_features.size();
                for (size_t i = 0; i < pad_size; ++i) {
                    seq_features.push_back(std::vector<float>(feature_dim, 0.0f));
                }
            }
        }

        result.track_ids.push_back(batch_id);
        result.data.push_back(std::move(seq_features));
        result.lengths.push_back(static_cast<int>(length));
    }

    if (result.track_ids.empty()) {
        return std::nullopt;
    }

    return result;
}

void TrackWindowBuffer::mark_inferred(const std::vector<uint32_t>& track_ids) {
    for (uint32_t batch_id : track_ids) {
        auto it = buffers_.find(batch_id);
        if (it != buffers_.end()) {
            // 只重置计数器，不删除点（与 Python 版本行为一致）
            // Python 的 deque(maxlen=seq_len) 会在 update 时自动挤出旧点
            it->second.since_last_infer = 0;
            it->second.has_inferred = true;

            LOG_DEBUG("Mark inferred: batch_id=" + std::to_string(batch_id) +
                     ", since_last_infer reset to 0");
        }
    }
}

void TrackWindowBuffer::cleanup(double current_time) {
    // 注意：timestamp 是从报文解析的相对时间（秒），不是 Unix 时间戳
    // 由于时间戳的含义不明确，我们只使用队列长度限制，不使用基于时间的清理

    for (auto& [batch_id, track] : buffers_) {
        // 如果队列过长，移除最老的点（保留最近 seq_len 个点）
        while (track.points.size() > static_cast<size_t>(config_.seq_len)) {
            LOG_DEBUG("Removing excess point: batch_id=" + std::to_string(batch_id) +
                     ", size=" + std::to_string(track.points.size()));
            track.points.pop_front();
        }

        // 暂时禁用基于时间戳的清理，因为时间戳的含义不明确
        // 如果需要启用，请确保时间戳是真实的绝对时间
    }

    // 移除空轨迹
    std::vector<uint32_t> to_remove;
    for (const auto& [batch_id, track] : buffers_) {
        if (track.points.empty()) {
            to_remove.push_back(batch_id);
        }
    }

    for (uint32_t batch_id : to_remove) {
        buffers_.erase(batch_id);
    }
}

std::optional<double> TrackWindowBuffer::get_last_timestamp(uint32_t track_id) const {
    auto it = buffers_.find(track_id);
    if (it != buffers_.end() && !it->second.points.empty()) {
        return it->second.points.back().timestamp;
    }
    return std::nullopt;
}

void TrackWindowBuffer::clear() {
    buffers_.clear();
}

} // namespace timesnet
