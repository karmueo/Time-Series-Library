#include "yaml_config_loader.h"
#include "logger.h"

namespace timesnet {

std::string YAMLConfigLoader::last_error_ = "";

std::optional<AppConfig> YAMLConfigLoader::load_from_file(const std::string& path) {
    last_error_.clear();

    try {
        YAML::Node root = YAML::LoadFile(path);

        // 检查是否为空文档
        if (!root.IsDefined() || root.IsNull()) {
            last_error_ = "Empty YAML document in file: " + path;
            return std::nullopt;
        }

        AppConfig config;
        if (!parse_app_config(root, config)) {
            return std::nullopt;
        }
        return config;
    } catch (const YAML::Exception& e) {
        last_error_ = "YAML parse error: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return std::nullopt;
    } catch (const std::exception& e) {
        last_error_ = "Failed to load config: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return std::nullopt;
    }
}

std::optional<AppConfig> YAMLConfigLoader::load_from_string(const std::string& yaml_str) {
    last_error_.clear();

    try {
        // 跳过空字符串和只包含空白符的字符串
        if (yaml_str.find_first_not_of(" \t\n\r\f\v") == std::string::npos) {
            last_error_ = "Empty or whitespace-only YAML content";
            return std::nullopt;
        }

        YAML::Node root = YAML::Load(yaml_str);

        // 检查是否为空文档
        if (!root.IsDefined() || root.IsNull()) {
            last_error_ = "Empty YAML document";
            return std::nullopt;
        }

        AppConfig config;
        if (!parse_app_config(root, config)) {
            return std::nullopt;
        }
        return config;
    } catch (const YAML::Exception& e) {
        last_error_ = "YAML parse error: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return std::nullopt;
    } catch (const std::exception& e) {
        last_error_ = "Failed to parse config: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return std::nullopt;
    }
}

bool YAMLConfigLoader::parse_app_config(const YAML::Node& root, AppConfig& config) {
    // 解析各配置块
    if (root["receiver"]) {
        if (!parse_receiver_config(root["receiver"], config.receiver)) {
            return false;
        }
    }

    if (root["publisher"]) {
        if (!parse_publisher_config(root["publisher"], config.publisher)) {
            return false;
        }
    }

    if (root["predictor"]) {
        if (!parse_predictor_config(root["predictor"], config.predictor)) {
            return false;
        }
    }

    if (root["buffer"]) {
        if (!parse_buffer_config(root["buffer"], config.buffer)) {
            return false;
        }
    }

    if (root["normalizer"]) {
        if (!parse_normalizer_config(root["normalizer"], config.normalizer)) {
            return false;
        }
    }

    if (root["inference"]) {
        if (!parse_inference_config(root["inference"], config.inference)) {
            return false;
        }
    }

    if (root["local_test"]) {
        if (!parse_local_test_config(root["local_test"], config.local_test)) {
            return false;
        }
    }

    return true;
}

bool YAMLConfigLoader::parse_receiver_config(const YAML::Node& node, AppConfig::ReceiverConfig_& config) {
    config.group = get_string(node, "group", config.group);
    config.port = get_int(node, "port", config.port);
    config.iface = get_string(node, "iface", config.iface);
    config.bind_ip = get_string(node, "bind_ip", config.bind_ip);
    config.timeout_s = get_double(node, "timeout_s", config.timeout_s);
    config.skip_checksum = get_bool(node, "skip_checksum", config.skip_checksum);

    // 验证必填字段
    if (config.group.empty()) {
        last_error_ = "receiver.group is required";
        return false;
    }
    if (config.port <= 0 || config.port > 65535) {
        last_error_ = "receiver.port must be 1-65535";
        return false;
    }

    return true;
}

bool YAMLConfigLoader::parse_publisher_config(const YAML::Node& node, AppConfig::PublisherConfig_& config) {
    config.group = get_string(node, "group", config.group);
    config.port = get_int(node, "port", config.port);
    config.iface = get_string(node, "iface", config.iface);
    config.ttl = get_int(node, "ttl", config.ttl);

    // 验证必填字段
    if (config.group.empty()) {
        last_error_ = "publisher.group is required";
        return false;
    }
    if (config.port <= 0 || config.port > 65535) {
        last_error_ = "publisher.port must be 1-65535";
        return false;
    }

    return true;
}

bool YAMLConfigLoader::parse_predictor_config(const YAML::Node& node, PredictorConfigRaw& config) {
    config.model_path = get_string(node, "model_path", config.model_path);
    config.num_classes = get_int(node, "num_classes", config.num_classes);
    config.seq_len = get_int(node, "seq_len", config.seq_len);
    config.num_features = get_int(node, "num_features", config.num_features);
    config.use_gpu = get_bool(node, "use_gpu", config.use_gpu);
    config.gpu_device_id = get_string(node, "gpu_device_id", config.gpu_device_id);
    config.use_timesnet_input = get_bool(node, "use_timesnet_input", config.use_timesnet_input);

    // 验证必填字段
    if (config.model_path.empty()) {
        last_error_ = "predictor.model_path is required";
        return false;
    }

    return true;
}

bool YAMLConfigLoader::parse_buffer_config(const YAML::Node& node, AppConfig::BufferConfig& config) {
    config.seq_len = get_int(node, "seq_len", config.seq_len);
    config.max_age_s = get_int(node, "max_age_s", config.max_age_s);

    return true;
}

bool YAMLConfigLoader::parse_normalizer_config(const YAML::Node& node, AppConfig::NormalizerConfig& config) {
    config.stats_path = get_string(node, "stats_path", config.stats_path);
    return true;
}

bool YAMLConfigLoader::parse_inference_config(const YAML::Node& node, AppConfig::InferenceConfig& config) {
    config.min_seq_len = get_int(node, "min_seq_len", config.min_seq_len);
    config.window_step = get_int(node, "window_step", config.window_step);
    config.ema_alpha = get_float(node, "ema_alpha", config.ema_alpha);
    config.publish_interval_ms = get_int(node, "publish_interval_ms", config.publish_interval_ms);
    config.print_targets = get_bool(node, "print_targets", config.print_targets);
    config.print_features = get_bool(node, "print_features", config.print_features);
    config.save_input_path = get_string(node, "save_input_path", config.save_input_path);

    return true;
}

bool YAMLConfigLoader::parse_local_test_config(const YAML::Node& node, AppConfig::LocalTestConfig& config) {
    config.enabled = get_bool(node, "enabled", config.enabled);
    config.xls_path = get_string(node, "xls_path", config.xls_path);
    config.max_points = get_int(node, "max_points", config.max_points);

    if (config.enabled) {
        if (config.xls_path.empty()) {
            last_error_ = "local_test.xls_path is required when local_test.enabled is true";
            return false;
        }
        if (config.max_points <= 0) {
            last_error_ = "local_test.max_points must be positive";
            return false;
        }
    }

    return true;
}

// 辅助函数实现
std::string get_string(const YAML::Node& node, const std::string& key, const std::string& default_val) {
    if (!node[key]) {
        return default_val;
    }
    try {
        return node[key].as<std::string>();
    } catch (const YAML::Exception&) {
        return default_val;
    }
}

int get_int(const YAML::Node& node, const std::string& key, int default_val) {
    if (!node[key]) {
        return default_val;
    }
    try {
        return node[key].as<int>();
    } catch (const YAML::Exception&) {
        return default_val;
    }
}

float get_float(const YAML::Node& node, const std::string& key, float default_val) {
    if (!node[key]) {
        return default_val;
    }
    try {
        return node[key].as<float>();
    } catch (const YAML::Exception&) {
        return default_val;
    }
}

double get_double(const YAML::Node& node, const std::string& key, double default_val) {
    if (!node[key]) {
        return default_val;
    }
    try {
        return node[key].as<double>();
    } catch (const YAML::Exception&) {
        return default_val;
    }
}

bool get_bool(const YAML::Node& node, const std::string& key, bool default_val) {
    if (!node[key]) {
        return default_val;
    }
    try {
        return node[key].as<bool>();
    } catch (const YAML::Exception&) {
        return default_val;
    }
}

} // namespace timesnet
