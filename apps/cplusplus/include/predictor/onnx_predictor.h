#ifndef ONNX_PREDICTOR_H
#define ONNX_PREDICTOR_H

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <cstdint>

// ONNX Runtime: 通过 CMake 配置 include 路径
#include <onnxruntime_cxx_api.h>

namespace timesnet {

/**
 * @brief ONNX 预测器配置
 */
struct PredictorConfig {
    std::string model_path;           // ONNX 模型路径
    int num_classes = 2;              // 类别数
    int seq_len = 20;                 // 序列长度
    int num_features = 12;            // 特征数
    bool use_gpu = true;              // 是否使用 GPU
    std::string gpu_device_id = "0";  // GPU 设备 ID
    bool use_timesnet_input = true;   // 使用 TimesNet 输入格式 (x_enc, x_mark_enc)
};

/**
 * @brief 批数据
 */
struct BatchData {
    std::vector<std::vector<std::vector<float>>> data;  // [batch][seq][features]
    std::vector<int> lengths;                            // 实际序列长度
};

/**
 * @brief ONNX 预测器
 *
 * 使用 ONNX Runtime 进行模型推理
 * 支持 TimesNet 分类模型的输入格式
 */
class OnnxPredictor {
public:
    explicit OnnxPredictor(const PredictorConfig& config);
    ~OnnxPredictor();

    // 禁用拷贝
    OnnxPredictor(const OnnxPredictor&) = delete;
    OnnxPredictor& operator=(const OnnxPredictor&) = delete;

    // 移动语义
    OnnxPredictor(OnnxPredictor&& other) noexcept;
    OnnxPredictor& operator=(OnnxPredictor&& other) noexcept;

    /**
     * @brief 加载模型
     * @return 是否成功
     */
    bool load();

    /**
     * @brief 推理预测 (TimesNet 格式)
     * @param x_enc 输入数据 [batch][seq][features]
     * @param x_mark_enc 掩码 [batch][seq] (1=有效, 0=填充)
     * @return 预测结果 {predictions, probabilities(无人机概率)}
     */
    std::pair<std::vector<int>, std::vector<float>> predict_timesnet(
        const std::vector<std::vector<std::vector<float>>>& x_enc,
        const std::vector<std::vector<float>>& x_mark_enc
    );

    /**
     * @brief 推理预测 (通用格式)
     * @param batch 批数据
     * @return 预测结果 {predictions, probabilities(无人机概率)}
     */
    std::pair<std::vector<int>, std::vector<float>> predict(const BatchData& batch);

    /**
     * @brief 推理预测 (便利接口)
     * @param data 输入数据 [batch][seq][features]
     * @param lengths 实际序列长度
     * @return 预测结果 {predictions, probabilities(无人机概率)}
     */
    std::pair<std::vector<int>, std::vector<float>> predict(
        const std::vector<std::vector<std::vector<float>>>& data,
        const std::vector<int>& lengths
    );

    /**
     * @brief 获取模型输入形状
     */
    std::vector<int64_t> get_input_shape() const { return input_shape_; }

    /**
     * @brief 获取模型输出形状
     */
    std::vector<int64_t> get_output_shape() const { return output_shape_; }

    /**
     * @brief 检查是否已加载
     */
    bool is_loaded() const { return session_ != nullptr; }

    /**
     * @brief 获取最后一个错误信息
     */
    const std::string& last_error() const { return last_error_; }

    /**
     * @brief 获取推理时间 (毫秒)
     */
    double last_inference_time_ms() const { return last_inference_time_ms_; }

    /**
     * @brief 获取最后一个输出 (用于调试)
     */
    const std::vector<float>& last_output() const { return last_output_; }

private:
    PredictorConfig config_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    std::string last_error_;
    double last_inference_time_ms_ = 0.0;
    std::vector<float> last_output_;

    bool init_session();
    bool get_input_output_info();
};

/**
 * @brief 推理结果
 */
struct InferenceResult {
    std::vector<int> predictions;    // 预测类别
    std::vector<float> probabilities; // 无人机概率（class=1）
    double inference_time_ms;        // 推理耗时
};

} // namespace timesnet

#endif // ONNX_PREDICTOR_H
