#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "predictor/onnx_predictor.h"

using namespace timesnet;

class OnnxPredictorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.model_path = "/nonexistent/model.onnx";
        config_.num_classes = 2;
        config_.seq_len = 20;
        config_.num_features = 12;
        config_.use_gpu = false;
        config_.gpu_device_id = "0";
    }

    PredictorConfig config_;
};

TEST_F(OnnxPredictorTest, DefaultConfig) {
    PredictorConfig default_config;
    EXPECT_TRUE(default_config.model_path.empty());
    EXPECT_EQ(default_config.num_classes, 2);
    EXPECT_EQ(default_config.seq_len, 20);
    EXPECT_EQ(default_config.num_features, 12);
    EXPECT_TRUE(default_config.use_gpu);
}

TEST_F(OnnxPredictorTest, ConfigWithValues) {
    EXPECT_EQ(config_.model_path, "/nonexistent/model.onnx");
    EXPECT_EQ(config_.num_classes, 2);
    EXPECT_EQ(config_.seq_len, 20);
    EXPECT_EQ(config_.num_features, 12);
    EXPECT_FALSE(config_.use_gpu);
}

TEST_F(OnnxPredictorTest, NotLoadedBeforeLoad) {
    OnnxPredictor predictor(config_);
    EXPECT_FALSE(predictor.is_loaded());
}

TEST_F(OnnxPredictorTest, LoadNonexistentModel) {
    OnnxPredictor predictor(config_);
    bool result = predictor.load();
    EXPECT_FALSE(result);
    EXPECT_FALSE(predictor.last_error().empty());
}

TEST_F(OnnxPredictorTest, PredictBeforeLoad) {
    PredictorConfig test_config;
    test_config.model_path = "";
    OnnxPredictor predictor(test_config);

    BatchData batch;
    auto result = predictor.predict(batch);
    EXPECT_TRUE(result.first.empty());
    EXPECT_TRUE(result.second.empty());
}

TEST_F(OnnxPredictorTest, PredictEmptyBatch) {
    OnnxPredictor predictor(config_);
    // Won't load, but test prediction with empty data
    BatchData batch;
    auto result = predictor.predict(batch);
    EXPECT_TRUE(result.first.empty());
    EXPECT_TRUE(result.second.empty());
}

TEST_F(OnnxPredictorTest, Movable) {
    OnnxPredictor predictor1(config_);
    OnnxPredictor predictor2(std::move(predictor1));

    EXPECT_FALSE(predictor2.is_loaded());
}

TEST_F(OnnxPredictorTest, MovableAssignment) {
    OnnxPredictor predictor1(config_);
    OnnxPredictor predictor2(config_);
    predictor2 = std::move(predictor1);

    EXPECT_FALSE(predictor2.is_loaded());
}

TEST_F(OnnxPredictorTest, LastErrorInitial) {
    OnnxPredictor predictor(config_);
    EXPECT_TRUE(predictor.last_error().empty());
}

TEST_F(OnnxPredictorTest, InferenceTimeInitial) {
    OnnxPredictor predictor(config_);
    EXPECT_EQ(predictor.last_inference_time_ms(), 0.0);
}

TEST_F(OnnxPredictorTest, InputOutputShapeInitial) {
    OnnxPredictor predictor(config_);
    EXPECT_TRUE(predictor.get_input_shape().empty());
    EXPECT_TRUE(predictor.get_output_shape().empty());
}

TEST_F(OnnxPredictorTest, BatchDataDefault) {
    BatchData batch;
    EXPECT_TRUE(batch.data.empty());
    EXPECT_TRUE(batch.lengths.empty());
}

TEST_F(OnnxPredictorTest, BatchDataWithContent) {
    BatchData batch;
    batch.data = {
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    };
    batch.lengths = {2, 2};

    EXPECT_EQ(batch.data.size(), 2);
    EXPECT_EQ(batch.lengths.size(), 2);
    EXPECT_EQ(batch.data[0].size(), 2);
    EXPECT_EQ(batch.data[0][0].size(), 2);
}

TEST_F(OnnxPredictorTest, PredictTimesnetFormat) {
    OnnxPredictor predictor(config_);

    // 准备 TimesNet 格式数据
    std::vector<std::vector<std::vector<float>>> x_enc(2);
    std::vector<std::vector<float>> x_mark_enc(2);

    for (int b = 0; b < 2; ++b) {
        x_enc[b].resize(10);
        x_mark_enc[b].resize(10);

        for (int t = 0; t < 10; ++t) {
            x_enc[b][t].resize(3);
            for (int f = 0; f < 3; ++f) {
                x_enc[b][t][f] = static_cast<float>(b + t + f) / 10.0f;
            }
            x_mark_enc[b][t] = 1.0f;
        }
    }

    // 即使模型未加载，调用也不应崩溃
    auto result = predictor.predict_timesnet(x_enc, x_mark_enc);

    EXPECT_TRUE(result.first.empty());  // 未加载时应返回空
    EXPECT_TRUE(result.second.empty());
}

TEST_F(OnnxPredictorTest, LastOutputInitial) {
    OnnxPredictor predictor(config_);
    EXPECT_TRUE(predictor.last_output().empty());
}

TEST_F(OnnxPredictorTest, ConfigUseTimesnetInput) {
    PredictorConfig config;
    config.use_timesnet_input = true;
    EXPECT_TRUE(config.use_timesnet_input);

    config.use_timesnet_input = false;
    EXPECT_FALSE(config.use_timesnet_input);
}
