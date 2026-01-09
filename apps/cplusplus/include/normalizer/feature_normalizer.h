#ifndef FEATURE_NORMALIZER_H
#define FEATURE_NORMALIZER_H

#include <vector>
#include <string>
#include <optional>
#include <map>

namespace timesnet {

/**
 * @brief 特征归一化器
 *
 * 使用训练数据的统计信息进行归一化
 */
class FeatureNormalizer {
public:
    /**
     * @brief 统计信息
     */
    struct Stats {
        std::vector<float> mean;      // 均值
        std::vector<float> std;       // 标准差
        std::vector<float> min;       // 最小值
        std::vector<float> max;       // 最大值
        std::vector<float> range;     // 范围
    };

    FeatureNormalizer() = default;

    /**
     * @brief 从统计信息构造
     */
    explicit FeatureNormalizer(const Stats& stats);

    /**
     * @brief 从 JSON 文件加载统计信息
     */
    static std::optional<FeatureNormalizer> from_json_file(const std::string& path);

    /**
     * @brief 归一化特征
     * @param raw 原始特征向量
     * @return 归一化后的特征，若特征维度不匹配返回空
     */
    std::optional<std::vector<float>> normalize(const std::vector<float>& raw);

    /**
     * @brief 反归一化 (用于调试)
     */
    std::vector<float> denormalize(const std::vector<float>& normalized) const;

    /**
     * @brief 设置统计信息
     */
    void set_stats(const Stats& stats);

    /**
     * @brief 获取统计信息
     */
    const Stats& get_stats() const { return stats_; }

    /**
     * @brief 获取特征维度
     */
    size_t num_features() const { return stats_.mean.size(); }

    /**
     * @brief 检查是否已初始化
     */
    bool is_initialized() const { return !stats_.mean.empty(); }

    /**
     * @brief 保存统计信息到 JSON 文件
     */
    bool save_to_json(const std::string& path) const;

private:
    Stats stats_;
    bool initialized_ = false;
};

/**
 * @brief 特征列定义
 */
struct FeatureConfig {
    static constexpr const char* FEATURE_R_M = "r_m";
    static constexpr const char* FEATURE_PR_M = "pr_m";
    static constexpr const char* FEATURE_AZIMUTH = "azimuth";
    static constexpr const char* FEATURE_ELEVATION = "elevation";
    static constexpr const char* FEATURE_X = "x";
    static constexpr const char* FEATURE_Y = "y";
    static constexpr const char* FEATURE_Z = "z";
    static constexpr const char* FEATURE_VX = "vx";
    static constexpr const char* FEATURE_VY = "vy";
    static constexpr const char* FEATURE_VZ = "vz";
    static constexpr const char* FEATURE_SNR = "snr";
    static constexpr const char* FEATURE_FLAGS = "flags";

    /**
     * @brief 获取所有特征列名
     */
    static std::vector<std::string> get_feature_names();
};

} // namespace timesnet

#endif // FEATURE_NORMALIZER_H
