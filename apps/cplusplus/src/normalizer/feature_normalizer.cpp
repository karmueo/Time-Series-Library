#include "normalizer/feature_normalizer.h"
#include "json_helper.h"
#include "logger.h"
#include <algorithm>
#include <cmath>

namespace timesnet {

/**
 * @brief 获取特征列顺序 (必须与 features/feature_cols.json 一致)
 */
static const std::vector<std::string>& get_feature_order() {
    static const std::vector<std::string> FEATURE_ORDER = {
        "径向距离", "方位", "俯仰", "点迹距离", "点迹方位", "点迹俯仰",
        "全速度", "径向速度", "方位速度", "俯仰速度",
        "多普勒展宽", "JEM", "RCS", "目标信噪比"
    };
    return FEATURE_ORDER;
}

FeatureNormalizer::FeatureNormalizer(const Stats& stats) : stats_(stats) {
    // 计算 range
    if (stats_.mean.size() > 0) {
        initialized_ = true;
        size_t n = stats_.mean.size();
        stats_.range.resize(n);
        // 如果 min/max 存在则使用，否则使用默认值
        bool has_min_max = !stats_.min.empty() && !stats_.max.empty();
        for (size_t i = 0; i < n; ++i) {
            if (has_min_max) {
                stats_.range[i] = stats_.max[i] - stats_.min[i];
            } else {
                stats_.range[i] = 1.0f;  // 默认 range
            }
            if (stats_.range[i] < 1e-6) {
                stats_.range[i] = 1.0f;
            }
        }
    }
}

/**
 * @brief 从对象格式的 JSON 中加载统计信息
 */
std::vector<float> load_stats_from_object(const json& obj) {
    const auto& order = get_feature_order();
    std::vector<float> result;
    for (const auto& key : order) {
        auto it = obj.find(key);
        if (it != obj.end() && it->is_number()) {
            result.push_back(it->get<float>());
        } else {
            result.push_back(0.0f);  // 默认值
        }
    }
    return result;
}

std::optional<FeatureNormalizer> FeatureNormalizer::from_json_file(const std::string& path) {
    auto json_opt = JsonHelper::load_from_file(path);
    if (!json_opt.has_value()) {
        LOG_ERROR("Failed to load stats from: " + path);
        return std::nullopt;
    }

    json j = json_opt.value();
    Stats stats;

    // 尝试加载对象格式 (键值对，如 {"径向距离": 3.5, ...})
    auto mean_obj = j.find("mean");
    if (mean_obj != j.end() && mean_obj->is_object()) {
        stats.mean = load_stats_from_object(*mean_obj);
    } else {
        // 尝试加载数组格式
        stats.mean = JsonHelper::get_float_array(j, "mean");
    }

    auto std_obj = j.find("std");
    if (std_obj != j.end() && std_obj->is_object()) {
        stats.std = load_stats_from_object(*std_obj);
    } else {
        stats.std = JsonHelper::get_float_array(j, "std");
    }

    stats.min = JsonHelper::get_float_array(j, "min");
    stats.max = JsonHelper::get_float_array(j, "max");

    if (stats.mean.empty()) {
        LOG_ERROR("Stats file is empty or invalid");
        return std::nullopt;
    }

    return FeatureNormalizer(stats);
}

std::optional<std::vector<float>> FeatureNormalizer::normalize(const std::vector<float>& raw) {
    if (raw.size() != stats_.mean.size()) {
        LOG_WARNING("Feature dimension mismatch: " + std::to_string(raw.size()) +
                    " vs " + std::to_string(stats_.mean.size()));
        return std::nullopt;
    }

    std::vector<float> normalized(raw.size());
    size_t n = raw.size();

    for (size_t i = 0; i < n; ++i) {
        if (stats_.range[i] > 1e-6) {
            // Z-score 标准化
            normalized[i] = (raw[i] - stats_.mean[i]) / stats_.std[i];
            // 处理异常值
            if (std::isnan(normalized[i]) || std::isinf(normalized[i])) {
                normalized[i] = 0.0f;
            }
        } else {
            normalized[i] = 0.0f;
        }
    }

    return normalized;
}

std::vector<float> FeatureNormalizer::denormalize(const std::vector<float>& normalized) const {
    std::vector<float> raw(normalized.size());
    size_t n = normalized.size();

    for (size_t i = 0; i < n; ++i) {
        raw[i] = normalized[i] * stats_.std[i] + stats_.mean[i];
    }

    return raw;
}

void FeatureNormalizer::set_stats(const Stats& stats) {
    stats_ = stats;
    initialized_ = !stats_.mean.empty();
}

bool FeatureNormalizer::save_to_json(const std::string& path) const {
    json j;
    j["mean"] = stats_.mean;
    j["std"] = stats_.std;
    j["min"] = stats_.min;
    j["max"] = stats_.max;

    return JsonHelper::save_to_file(j, path);
}

std::vector<std::string> FeatureConfig::get_feature_names() {
    return {
        FEATURE_R_M,
        FEATURE_PR_M,
        FEATURE_AZIMUTH,
        FEATURE_ELEVATION,
        FEATURE_X,
        FEATURE_Y,
        FEATURE_Z,
        FEATURE_VX,
        FEATURE_VY,
        FEATURE_VZ,
        FEATURE_SNR,
        FEATURE_FLAGS
    };
}

} // namespace timesnet
