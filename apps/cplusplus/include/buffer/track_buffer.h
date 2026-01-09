#ifndef TRACK_BUFFER_H
#define TRACK_BUFFER_H

#include <vector>
#include <unordered_map>
#include <deque>
#include <cstdint>
#include <string>
#include <optional>
#include <utility>

namespace timesnet {

/**
 * @brief 单条轨迹点
 */
struct TrackPoint {
    std::vector<float> features;  // 特征向量
    double timestamp;             // 时间戳 (秒)
    bool inferred = false;        // 是否已推理
};

/**
 * @brief 轨迹数据
 */
struct Track {
    std::deque<TrackPoint> points;
    std::string track_id;
    double last_update = 0.0;
    int since_last_infer = 0;     // 自上次推理以来的新点数量
    bool has_inferred = false;    // 是否已经推理过
};

/**
 * @brief 待处理批次信息
 */
struct PendingInfo {
    int count;       // 当前数量
    int needed;      // 所需数量
};

/**
 * @brief 批次输出结果
 */
struct BatchResult {
    std::vector<uint32_t> track_ids;                          // 轨迹 ID
    std::vector<std::vector<std::vector<float>>> data;        // [batch][seq][features]
    std::vector<int> lengths;                                 // 实际序列长度
};

/**
 * @brief 轨迹窗口缓冲
 *
 * 按 batch_id 组织轨迹数据，支持滑动窗口和批次构建
 */
class TrackWindowBuffer {
public:
    struct Config {
        int seq_len = 20;       // 序列长度
        double max_age_s = 10.0; // 最大保留时间
    };

    explicit TrackWindowBuffer(const Config& config);
    ~TrackWindowBuffer() = default;

    /**
     * @brief 更新轨迹
     * @param batch_id 批号
     * @param track_id 轨迹 ID
     * @param features 特征向量
     * @param timestamp 时间戳
     */
    void update(uint32_t batch_id, uint32_t track_id,
                const std::vector<float>& features, double timestamp);

    /**
     * @brief 获取待处理轨迹信息
     * @param min_seq_len 最小序列长度
     * @param window_step 窗口步长
     * @return 待处理信息 {track_id -> (count, needed)}
     */
    std::unordered_map<uint32_t, PendingInfo> get_pending_info(
        int min_seq_len = 20, int window_step = 0) const;

    /**
     * @brief 构建批次
     * @param min_seq_len 最小序列长度
     * @param window_step 窗口步长
     * @return 批次数据，若无可用数据返回空
     */
    std::optional<BatchResult>
    build_batch(int min_seq_len = 20, int window_step = 0);

    /**
     * @brief 标记已推理
     */
    void mark_inferred(const std::vector<uint32_t>& track_ids);

    /**
     * @brief 清理过期轨迹
     */
    void cleanup(double current_time = -1.0);

    /**
     * @brief 获取轨迹最后时间戳
     */
    std::optional<double> get_last_timestamp(uint32_t track_id) const;

    /**
     * @brief 获取轨迹数量
     */
    size_t size() const { return buffers_.size(); }

    /**
     * @brief 清空所有数据
     */
    void clear();

private:
    Config config_;
    std::unordered_map<uint32_t, Track> buffers_;  // batch_id -> Track
};

} // namespace timesnet

#endif // TRACK_BUFFER_H
