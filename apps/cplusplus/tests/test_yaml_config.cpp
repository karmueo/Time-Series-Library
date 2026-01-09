#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "yaml_config_loader.h"
#include <fstream>
#include <filesystem>

using namespace timesnet;

class YAMLConfigLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建临时目录
        temp_dir_ = std::filesystem::temp_directory_path() / "yaml_config_test_XXXXXX";
        std::filesystem::create_directories(temp_dir_);
    }

    void TearDown() override {
        // 清理临时文件
        std::filesystem::remove_all(temp_dir_);
    }

    std::filesystem::path temp_dir_;
};

TEST_F(YAMLConfigLoaderTest, LoadFromEmptyString) {
    auto result = YAMLConfigLoader::load_from_string("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(YAMLConfigLoaderTest, LoadFromInvalidYaml) {
    auto result = YAMLConfigLoader::load_from_string("invalid: yaml: content: [");
    EXPECT_FALSE(result.has_value());
}

TEST_F(YAMLConfigLoaderTest, LoadMinimalConfig) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->receiver.group, "230.1.1.22");
    EXPECT_EQ(result->receiver.port, 8002);
    EXPECT_EQ(result->publisher.group, "230.1.1.24");
    EXPECT_EQ(result->publisher.port, 8011);
    EXPECT_EQ(result->predictor.model_path, "models/test.onnx");
}

TEST_F(YAMLConfigLoaderTest, LoadFullConfig) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
  iface: "192.168.1.1"
  bind_ip: "0.0.0.0"
  timeout_s: 5.0
  skip_checksum: true
publisher:
  group: "230.1.1.24"
  port: 8011
  iface: "192.168.1.1"
  ttl: 2
predictor:
  model_path: "models/timesnet.onnx"
  num_classes: 2
  seq_len: 20
  num_features: 14
  use_gpu: false
  gpu_device_id: "0"
  use_timesnet_input: true
buffer:
  seq_len: 20
  max_age_s: 15
normalizer:
  stats_path: "data/stats.json"
inference:
  min_seq_len: 10
  window_step: 5
  ema_alpha: 0.5
  publish_interval_ms: 100
  print_targets: true
  print_features: false
local_test:
  enabled: true
  xls_path: "data/test.xls"
  max_points: 20
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    ASSERT_TRUE(result.has_value());

    // 验证接收器配置
    EXPECT_EQ(result->receiver.group, "230.1.1.22");
    EXPECT_EQ(result->receiver.port, 8002);
    EXPECT_EQ(result->receiver.iface, "192.168.1.1");
    EXPECT_EQ(result->receiver.bind_ip, "0.0.0.0");
    EXPECT_DOUBLE_EQ(result->receiver.timeout_s, 5.0);
    EXPECT_TRUE(result->receiver.skip_checksum);

    // 验证发布器配置
    EXPECT_EQ(result->publisher.group, "230.1.1.24");
    EXPECT_EQ(result->publisher.port, 8011);
    EXPECT_EQ(result->publisher.iface, "192.168.1.1");
    EXPECT_EQ(result->publisher.ttl, 2);

    // 验证预测器配置
    EXPECT_EQ(result->predictor.model_path, "models/timesnet.onnx");
    EXPECT_EQ(result->predictor.num_classes, 2);
    EXPECT_EQ(result->predictor.seq_len, 20);
    EXPECT_EQ(result->predictor.num_features, 14);
    EXPECT_FALSE(result->predictor.use_gpu);
    EXPECT_EQ(result->predictor.gpu_device_id, "0");
    EXPECT_TRUE(result->predictor.use_timesnet_input);

    // 验证缓冲配置
    EXPECT_EQ(result->buffer.seq_len, 20);
    EXPECT_EQ(result->buffer.max_age_s, 15);

    // 验证归一化配置
    EXPECT_EQ(result->normalizer.stats_path, "data/stats.json");

    // 验证推理配置
    EXPECT_EQ(result->inference.min_seq_len, 10);
    EXPECT_EQ(result->inference.window_step, 5);
    EXPECT_FLOAT_EQ(result->inference.ema_alpha, 0.5f);
    EXPECT_EQ(result->inference.publish_interval_ms, 100);
    EXPECT_TRUE(result->inference.print_targets);
    EXPECT_FALSE(result->inference.print_features);

    // 验证本地测试配置
    EXPECT_TRUE(result->local_test.enabled);
    EXPECT_EQ(result->local_test.xls_path, "data/test.xls");
    EXPECT_EQ(result->local_test.max_points, 20);
}

TEST_F(YAMLConfigLoaderTest, MissingReceiverGroup) {
    std::string yaml = R"(
receiver:
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(YAMLConfigLoader().last_error().empty());
}

TEST_F(YAMLConfigLoaderTest, MissingReceiverPort) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    EXPECT_FALSE(result.has_value());
}

TEST_F(YAMLConfigLoaderTest, InvalidReceiverPort) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 99999
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    EXPECT_FALSE(result.has_value());
}

TEST_F(YAMLConfigLoaderTest, MissingPublisherGroup) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    EXPECT_FALSE(result.has_value());
}

TEST_F(YAMLConfigLoaderTest, MissingPredictorModelPath) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  num_classes: 2
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    EXPECT_FALSE(result.has_value());
}

TEST_F(YAMLConfigLoaderTest, DefaultValues) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    ASSERT_TRUE(result.has_value());

    // 验证默认值
    EXPECT_EQ(result->receiver.iface, "0.0.0.0");
    EXPECT_EQ(result->receiver.bind_ip, "");
    EXPECT_DOUBLE_EQ(result->receiver.timeout_s, 2.0);
    EXPECT_FALSE(result->receiver.skip_checksum);

    EXPECT_EQ(result->publisher.iface, "0.0.0.0");
    EXPECT_EQ(result->publisher.ttl, 1);

    EXPECT_EQ(result->predictor.num_classes, 2);
    EXPECT_EQ(result->predictor.seq_len, 20);
    EXPECT_EQ(result->predictor.num_features, 12);
    EXPECT_TRUE(result->predictor.use_gpu);
    EXPECT_EQ(result->predictor.gpu_device_id, "0");
    EXPECT_TRUE(result->predictor.use_timesnet_input);

    EXPECT_EQ(result->buffer.seq_len, 20);
    EXPECT_EQ(result->buffer.max_age_s, 10);

    EXPECT_EQ(result->normalizer.stats_path, "");

    EXPECT_EQ(result->inference.min_seq_len, 20);
    EXPECT_EQ(result->inference.window_step, 0);
    EXPECT_FLOAT_EQ(result->inference.ema_alpha, 0.4f);
    EXPECT_EQ(result->inference.publish_interval_ms, 0);
    EXPECT_FALSE(result->inference.print_targets);
    EXPECT_FALSE(result->inference.print_features);

    EXPECT_FALSE(result->local_test.enabled);
    EXPECT_EQ(result->local_test.xls_path, "");
    EXPECT_EQ(result->local_test.max_points, 20);
}

TEST_F(YAMLConfigLoaderTest, LocalTestRequiresPath) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
local_test:
  enabled: true
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    EXPECT_FALSE(result.has_value());
}

TEST_F(YAMLConfigLoaderTest, LoadFromFile) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto config_path = temp_dir_ / "test_config.yaml";
    std::ofstream(config_path) << yaml;

    auto result = YAMLConfigLoader::load_from_file(config_path.string());
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->receiver.group, "230.1.1.22");
    EXPECT_EQ(result->receiver.port, 8002);
    EXPECT_EQ(result->publisher.group, "230.1.1.24");
    EXPECT_EQ(result->publisher.port, 8011);
    EXPECT_EQ(result->predictor.model_path, "models/test.onnx");
}

TEST_F(YAMLConfigLoaderTest, LoadNonexistentFile) {
    auto result = YAMLConfigLoader::load_from_file("/nonexistent/path/config.yaml");
    EXPECT_FALSE(result.has_value());
}

TEST_F(YAMLConfigLoaderTest, GPUEnabled) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
  use_gpu: true
  gpu_device_id: "1"
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    ASSERT_TRUE(result.has_value());

    EXPECT_TRUE(result->predictor.use_gpu);
    EXPECT_EQ(result->predictor.gpu_device_id, "1");
}

TEST_F(YAMLConfigLoaderTest, AllBooleanFields) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
  skip_checksum: true
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
  use_gpu: true
  use_timesnet_input: false
inference:
  print_targets: true
  print_features: true
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    ASSERT_TRUE(result.has_value());

    EXPECT_TRUE(result->receiver.skip_checksum);
    EXPECT_TRUE(result->predictor.use_gpu);
    EXPECT_FALSE(result->predictor.use_timesnet_input);
    EXPECT_TRUE(result->inference.print_targets);
    EXPECT_TRUE(result->inference.print_features);
}

TEST_F(YAMLConfigLoaderTest, FloatValues) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
  timeout_s: 3.5
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
inference:
  ema_alpha: 0.8
)";

    auto result = YAMLConfigLoader::load_from_string(yaml);
    ASSERT_TRUE(result.has_value());

    EXPECT_DOUBLE_EQ(result->receiver.timeout_s, 3.5);
    EXPECT_FLOAT_EQ(result->inference.ema_alpha, 0.8f);
}

TEST_F(YAMLConfigLoaderTest, LastErrorMessage) {
    YAMLConfigLoader loader;

    // 加载无效配置
    std::string invalid_yaml = "receiver:\n  port: 99999";
    auto result = YAMLConfigLoader::load_from_string(invalid_yaml);
    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(loader.last_error().empty());
}

TEST_F(YAMLConfigLoaderTest, AppConfigCopyable) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto result1 = YAMLConfigLoader::load_from_string(yaml);
    ASSERT_TRUE(result1.has_value());

    // 测试拷贝构造
    AppConfig config2 = result1.value();
    EXPECT_EQ(config2.receiver.group, "230.1.1.22");
    EXPECT_EQ(config2.receiver.port, 8002);

    // 测试赋值
    AppConfig config3;
    config3 = result1.value();
    EXPECT_EQ(config3.publisher.group, "230.1.1.24");
    EXPECT_EQ(config3.publisher.port, 8011);
}

TEST_F(YAMLConfigLoaderTest, PredictorConfigRawDefaults) {
    PredictorConfigRaw config;
    EXPECT_TRUE(config.model_path.empty());
    EXPECT_EQ(config.num_classes, 2);
    EXPECT_EQ(config.seq_len, 20);
    EXPECT_EQ(config.num_features, 12);
    EXPECT_TRUE(config.use_gpu);
    EXPECT_EQ(config.gpu_device_id, "0");
    EXPECT_TRUE(config.use_timesnet_input);
}

TEST_F(YAMLConfigLoaderTest, PredictorConfigRawWithValues) {
    PredictorConfigRaw config;
    config.model_path = "models/timesnet.onnx";
    config.num_classes = 3;
    config.seq_len = 30;
    config.num_features = 16;
    config.use_gpu = false;
    config.gpu_device_id = "1";
    config.use_timesnet_input = false;

    EXPECT_EQ(config.model_path, "models/timesnet.onnx");
    EXPECT_EQ(config.num_classes, 3);
    EXPECT_EQ(config.seq_len, 30);
    EXPECT_EQ(config.num_features, 16);
    EXPECT_FALSE(config.use_gpu);
    EXPECT_EQ(config.gpu_device_id, "1");
    EXPECT_FALSE(config.use_timesnet_input);
}

TEST_F(YAMLConfigLoaderTest, ConfigEquality) {
    std::string yaml = R"(
receiver:
  group: "230.1.1.22"
  port: 8002
publisher:
  group: "230.1.1.24"
  port: 8011
predictor:
  model_path: "models/test.onnx"
)";

    auto result1 = YAMLConfigLoader::load_from_string(yaml);
    auto result2 = YAMLConfigLoader::load_from_string(yaml);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // 两次加载的配置应该相等
    EXPECT_EQ(result1->receiver.group, result2->receiver.group);
    EXPECT_EQ(result1->receiver.port, result2->receiver.port);
    EXPECT_EQ(result1->publisher.group, result2->publisher.group);
    EXPECT_EQ(result1->publisher.port, result2->publisher.port);
    EXPECT_EQ(result1->predictor.model_path, result2->predictor.model_path);
}
