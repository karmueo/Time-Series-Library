#ifndef JSON_HELPER_H
#define JSON_HELPER_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace timesnet {

using json = nlohmann::json;

/**
 * @brief JSON 辅助工具类
 */
class JsonHelper {
public:
    // 从文件加载 JSON
    static std::optional<json> load_from_file(const std::string& path);

    // 保存 JSON 到文件
    static bool save_to_file(const json& j, const std::string& path);

    // 安全的值获取
    template<typename T>
    static T get_value(const json& j, const std::string& key, const T& default_value);

    // 安全的字符串获取
    static std::string get_string(const json& j, const std::string& key, const std::string& default_value = "");

    // 安全的数组获取
    static std::vector<float> get_float_array(const json& j, const std::string& key);

    // 安全的整数获取
    static int get_int(const json& j, const std::string& key, int default_value = 0);

    // 安全的浮点数获取
    static double get_double(const json& j, const std::string& key, double default_value = 0.0);

    // 安全的布尔获取
    static bool get_bool(const json& j, const std::string& key, bool default_value = false);

    // 检查键是否存在
    static bool has_key(const json& j, const std::string& key);

    // 创建带时间的推理结果 JSON
    static json create_inference_result(
        int inference_id,
        int64_t timestamp_ms,
        const std::vector<json>& results
    );

    // 创建单条预测结果
    static json create_prediction_result(
        uint32_t track_id,
        int pred,
        float prob_uav,
        float prob_bird
    );
};

// 模板特化实现
template<typename T>
T JsonHelper::get_value(const json& j, const std::string& key, const T& default_value) {
    auto it = j.find(key);
    if (it != j.end() && !it->is_null()) {
        try {
            return it->get<T>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

} // namespace timesnet

#endif // JSON_HELPER_H
