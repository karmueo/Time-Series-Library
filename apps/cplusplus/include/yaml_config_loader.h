#ifndef YAML_CONFIG_LOADER_H
#define YAML_CONFIG_LOADER_H

#include <string>
#include <optional>
#include <vector>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

namespace timesnet {

/**
 * @brief 预测器配置 (与 PredictorConfig 保持一致)
 */
struct PredictorConfigRaw {
    std::string model_path;           // ONNX 模型路径
    int num_classes = 2;              // 类别数
    int seq_len = 20;                 // 序列长度
    int num_features = 12;            // 特征数
    bool use_gpu = true;              // 是否使用 GPU
    std::string gpu_device_id = "0";  // GPU 设备 ID
    bool use_timesnet_input = true;   // 使用 TimesNet 输入格式
};

/**
 * @brief 应用配置 (完整配置)
 */
struct AppConfig {
    // 接收器配置
    struct ReceiverConfig_ {
        std::string group;
        int port = 0;
        std::string iface = "0.0.0.0";
        std::string bind_ip = "";
        double timeout_s = 2.0;
        bool skip_checksum = false;
    } receiver;

    // 发布器配置
    struct PublisherConfig_ {
        std::string group;
        int port = 0;
        std::string iface = "0.0.0.0";
        int ttl = 1;
    } publisher;

    // 预测器配置
    PredictorConfigRaw predictor;

    // 轨迹缓冲配置
    struct BufferConfig {
        int seq_len = 20;
        int max_age_s = 10;
    } buffer;

    // 归一化配置
    struct NormalizerConfig {
        std::string stats_path;
    } normalizer;

    // 推理配置
    struct InferenceConfig {
        int min_seq_len = 20;
        int window_step = 0;
        float ema_alpha = 0.4f;
        int publish_interval_ms = 0;
        bool print_targets = false;
        bool print_features = false;
        std::string save_input_path;  // 保存推理输入向量路径
    } inference;

    // 本地测试配置
    struct LocalTestConfig {
        bool enabled = false;      // 是否启用本地文件测试
        std::string xls_path;      // 本地 .xls/.csv 路径
        int max_points = 20;       // 读取点数 (默认前 20 个)
    } local_test;

};

/**
 * @brief YAML 配置加载器
 *
 * 从 YAML 文件加载应用配置
 */
class YAMLConfigLoader {
public:
    /**
     * @brief 从文件加载配置
     * @param path YAML 文件路径
     * @return 配置对象，失败返回 nullopt
     */
    static std::optional<AppConfig> load_from_file(const std::string& path);

    /**
     * @brief 从字符串加载配置
     * @param yaml_str YAML 格式字符串
     * @return 配置对象，失败返回 nullopt
     */
    static std::optional<AppConfig> load_from_string(const std::string& yaml_str);

    /**
     * @brief 获取最后一次错误信息
     */
    const std::string& last_error() const { return last_error_; }

private:
    static bool parse_app_config(const YAML::Node& node, AppConfig& config);
    static bool parse_receiver_config(const YAML::Node& node, AppConfig::ReceiverConfig_& config);
    static bool parse_publisher_config(const YAML::Node& node, AppConfig::PublisherConfig_& config);
    static bool parse_predictor_config(const YAML::Node& node, PredictorConfigRaw& config);
    static bool parse_buffer_config(const YAML::Node& node, AppConfig::BufferConfig& config);
    static bool parse_normalizer_config(const YAML::Node& node, AppConfig::NormalizerConfig& config);
    static bool parse_inference_config(const YAML::Node& node, AppConfig::InferenceConfig& config);
    static bool parse_local_test_config(const YAML::Node& node, AppConfig::LocalTestConfig& config);

    static std::string last_error_;
};

/**
 * @brief 从 YAML 节点安全获取字符串值
 * @param node YAML 节点
 * @param key 键名
 * @param default_val 默认值
 * @return 值或默认值
 */
std::string get_string(const YAML::Node& node, const std::string& key, const std::string& default_val = "");

/**
 * @brief 从 YAML 节点安全获取整数值
 * @param node YAML 节点
 * @param key 键名
 * @param default_val 默认值
 * @return 值或默认值
 */
int get_int(const YAML::Node& node, const std::string& key, int default_val = 0);

/**
 * @brief 从 YAML 节点安全获取浮点值
 * @param node YAML 节点
 * @param key 键名
 * @param default_val 默认值
 * @return 值或默认值
 */
float get_float(const YAML::Node& node, const std::string& key, float default_val = 0.0f);

/**
 * @brief 从 YAML 节点安全获取双精度浮点值
 * @param node YAML 节点
 * @param key 键名
 * @param default_val 默认值
 * @return 值或默认值
 */
double get_double(const YAML::Node& node, const std::string& key, double default_val = 0.0);

/**
 * @brief 从 YAML 节点安全获取布尔值
 * @param node YAML 节点
 * @param key 键名
 * @param default_val 默认值
 * @return 值或默认值
 */
bool get_bool(const YAML::Node& node, const std::string& key, bool default_val = false);

} // namespace timesnet

#endif // YAML_CONFIG_LOADER_H
