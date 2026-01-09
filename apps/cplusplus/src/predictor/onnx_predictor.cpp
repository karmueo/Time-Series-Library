#include "predictor/onnx_predictor.h"
#include "logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace timesnet {

OnnxPredictor::OnnxPredictor(const PredictorConfig& config)
    : config_(config),
      env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING)),
      memory_info_(std::make_unique<Ort::MemoryInfo>(
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))) {}

OnnxPredictor::~OnnxPredictor() {
    session_.reset();
    env_.reset();
}

OnnxPredictor::OnnxPredictor(OnnxPredictor&& other) noexcept
    : config_(other.config_),
      env_(std::move(other.env_)),
      session_(std::move(other.session_)),
      memory_info_(std::move(other.memory_info_)),
      input_names_(std::move(other.input_names_)),
      output_names_(std::move(other.output_names_)),
      input_shape_(std::move(other.input_shape_)),
      output_shape_(std::move(other.output_shape_)),
      last_error_(other.last_error_),
      last_inference_time_ms_(other.last_inference_time_ms_),
      last_output_(std::move(other.last_output_)) {}

OnnxPredictor& OnnxPredictor::operator=(OnnxPredictor&& other) noexcept {
    if (this != &other) {
        session_.reset();
        config_ = other.config_;
        env_ = std::move(other.env_);
        session_ = std::move(other.session_);
        memory_info_ = std::move(other.memory_info_);
        input_names_ = std::move(other.input_names_);
        output_names_ = std::move(other.output_names_);
        input_shape_ = std::move(other.input_shape_);
        output_shape_ = std::move(other.output_shape_);
        last_error_ = other.last_error_;
        last_inference_time_ms_ = other.last_inference_time_ms_;
        last_output_ = std::move(other.last_output_);
    }
    return *this;
}

bool OnnxPredictor::init_session() {
    try {
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // GPU 支持将在后续版本中添加
        if (config_.use_gpu) {
            LOG_INFO("GPU inference support requires additional configuration. Using CPU for now.");
        } else {
            LOG_INFO("Using CPU for inference");
        }

        session_ = std::make_unique<Ort::Session>(
            *env_,
            config_.model_path.c_str(),
            session_options);

        LOG_INFO("ONNX model loaded: " + config_.model_path);
        return true;

    } catch (const Ort::Exception& e) {
        last_error_ = "ONNX Exception: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return false;
    } catch (const std::exception& e) {
        last_error_ = "Failed to load model: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return false;
    }
}

bool OnnxPredictor::get_input_output_info() {
    if (!session_) {
        last_error_ = "Session not initialized";
        return false;
    }

    try {
        // 获取输入信息
        size_t num_inputs = session_->GetInputCount();
        input_names_.clear();
        input_shape_.clear();

        LOG_INFO("Model has " + std::to_string(num_inputs) + " inputs:");

        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            input_names_.push_back(strdup(input_name.get()));

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
            input_shape_.insert(input_shape_.end(), tensor_shape.begin(), tensor_shape.end());

            std::string shape_str = "[";
            for (size_t j = 0; j < tensor_shape.size(); ++j) {
                shape_str += std::to_string(tensor_shape[j]);
                if (j < tensor_shape.size() - 1) shape_str += ",";
            }
            shape_str += "]";

            LOG_INFO("  Input " + std::to_string(i) + ": " + input_names_[i] + " shape: " + shape_str);
        }

        // 获取输出信息
        size_t num_outputs = session_->GetOutputCount();
        output_names_.clear();
        output_shape_.clear();

        LOG_INFO("Model has " + std::to_string(num_outputs) + " outputs:");

        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            output_names_.push_back(strdup(output_name.get()));

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
            output_shape_.insert(output_shape_.end(), tensor_shape.begin(), tensor_shape.end());

            std::string shape_str = "[";
            for (size_t j = 0; j < tensor_shape.size(); ++j) {
                shape_str += std::to_string(tensor_shape[j]);
                if (j < tensor_shape.size() - 1) shape_str += ",";
            }
            shape_str += "]";

            LOG_INFO("  Output " + std::to_string(i) + ": " + output_names_[i] + " shape: " + shape_str);
        }

        return true;

    } catch (const Ort::Exception& e) {
        last_error_ = "ONNX Exception: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return false;
    }
}

bool OnnxPredictor::load() {
    if (!init_session()) {
        return false;
    }

    if (!get_input_output_info()) {
        return false;
    }

    LOG_INFO("ONNX predictor loaded successfully, inputs: " + std::to_string(input_names_.size()));
    return true;
}

std::pair<std::vector<int>, std::vector<float>> OnnxPredictor::predict_timesnet(
    const std::vector<std::vector<std::vector<float>>>& x_enc,
    const std::vector<std::vector<float>>& x_mark_enc
) {
    if (!session_) {
        last_error_ = "Model not loaded";
        LOG_ERROR(last_error_);
        return {{}, {}};
    }

    size_t batch_size = x_enc.size();
    if (batch_size == 0) {
        return {{}, {}};
    }

    // 检查输入维度
    if (x_enc[0].size() != static_cast<size_t>(config_.seq_len) ||
        x_enc[0][0].size() != static_cast<size_t>(config_.num_features)) {
        last_error_ = "Input dimension mismatch";
        LOG_ERROR(last_error_);
        return {{}, {}};
    }

    auto start = std::chrono::high_resolution_clock::now();

    try {
        // 准备输入张量
        std::vector<Ort::Value> input_tensors;

        // x_enc: [batch, seq_len, num_features] float
        std::vector<float> x_enc_flat;
        x_enc_flat.reserve(batch_size * config_.seq_len * config_.num_features);
        for (const auto& seq : x_enc) {
            for (const auto& features : seq) {
                for (float f : features) {
                    x_enc_flat.push_back(f);
                }
            }
        }

        std::vector<int64_t> x_enc_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(config_.seq_len),
            static_cast<int64_t>(config_.num_features)
        };

        LOG_INFO("Creating x_enc tensor: shape=[" + std::to_string(x_enc_shape[0]) + "," +
                 std::to_string(x_enc_shape[1]) + "," + std::to_string(x_enc_shape[2]) + "], " +
                 "size=" + std::to_string(x_enc_flat.size()));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            *memory_info_,
            x_enc_flat.data(),
            x_enc_flat.size(),
            x_enc_shape.data(), x_enc_shape.size()));

        // x_mark_enc: [batch, seq_len] float (掩码)
        std::vector<float> x_mark_flat;
        x_mark_flat.reserve(batch_size * config_.seq_len);

        if (x_mark_enc.size() == batch_size && x_mark_enc[0].size() == static_cast<size_t>(config_.seq_len)) {
            for (const auto& mask : x_mark_enc) {
                for (float m : mask) {
                    x_mark_flat.push_back(m);
                }
            }
        } else {
            // 默认全 1
            for (size_t i = 0; i < batch_size; ++i) {
                for (int t = 0; t < config_.seq_len; ++t) {
                    x_mark_flat.push_back(1.0f);
                }
            }
        }

        std::vector<int64_t> x_mark_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(config_.seq_len)
        };

        LOG_INFO("Creating x_mark_enc tensor: shape=[" + std::to_string(x_mark_shape[0]) + "," +
                 std::to_string(x_mark_shape[1]) + "], size=" + std::to_string(x_mark_flat.size()));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            *memory_info_,
            x_mark_flat.data(),
            x_mark_flat.size(),
            x_mark_shape.data(), x_mark_shape.size()));

        // 检查输入数量是否匹配
        if (input_tensors.size() != input_names_.size()) {
            last_error_ = "Input count mismatch: expected " + std::to_string(input_names_.size()) +
                         ", got " + std::to_string(input_tensors.size());
            LOG_ERROR(last_error_);
            return {{}, {}};
        }

        LOG_INFO("Running inference with " + std::to_string(input_tensors.size()) + " inputs");

        // 运行推理
        std::vector<Ort::Value> output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            input_tensors.data(), input_tensors.size(),
            output_names_.data(), output_names_.size());

        auto end = std::chrono::high_resolution_clock::now();
        last_inference_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_INFO("Inference completed in " + std::to_string(last_inference_time_ms_) + "ms");

        // 提取输出
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        // 计算输出大小：batch_size * num_classes
        // 注意：output_shape_[0] 可能是 -1（动态维度），所以使用实际的 batch_size
        size_t num_classes = static_cast<size_t>(output_shape_[1]);
        size_t output_size = batch_size * num_classes;

        LOG_INFO("output_shape_[0]=" + std::to_string(output_shape_[0]) + ", output_shape_[1]=" + std::to_string(output_shape_[1]));
        LOG_INFO("batch_size=" + std::to_string(batch_size) + ", num_classes=" + std::to_string(num_classes) + ", output_size=" + std::to_string(output_size));

        last_output_.clear();
        last_output_.insert(last_output_.end(), output_data, output_data + output_size);

        LOG_INFO("last_output_.size()=" + std::to_string(last_output_.size()));

        // 解析结果
        std::vector<int> predictions;
        std::vector<float> probabilities;  // 每个样本的无人机概率（class=1）
        for (size_t i = 0; i < batch_size; ++i) {
            float* out_ptr = last_output_.data() + i * num_classes;

            // 计算 softmax 概率
            std::vector<float> exp_probs(num_classes);
            float max_logit = out_ptr[0];
            for (int c = 1; c < static_cast<int>(num_classes); ++c) {
                max_logit = std::max(max_logit, out_ptr[c]);
            }
            float sum_exp = 0.0f;
            for (int c = 0; c < static_cast<int>(num_classes); ++c) {
                exp_probs[c] = std::exp(out_ptr[c] - max_logit);
                sum_exp += exp_probs[c];
            }
            for (int c = 0; c < static_cast<int>(num_classes); ++c) {
                exp_probs[c] /= sum_exp;
            }

            // 找最大概率的类别
            int pred = 0;
            float max_prob = exp_probs[0];
            for (int c = 1; c < static_cast<int>(num_classes); ++c) {
                if (exp_probs[c] > max_prob) {
                    max_prob = exp_probs[c];
                    pred = c;
                }
            }

            predictions.push_back(pred);
            if (num_classes > 1) {
                probabilities.push_back(exp_probs[1]);
            } else {
                probabilities.push_back(exp_probs[0]);
            }
        }

        LOG_DEBUG("TimesNet inference: batch=" + std::to_string(batch_size) +
                  ", time=" + std::to_string(last_inference_time_ms_) + "ms");

        return {predictions, probabilities};

    } catch (const Ort::Exception& e) {
        last_error_ = "ONNX Exception: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return {{}, {}};
    }
}

std::pair<std::vector<int>, std::vector<float>> OnnxPredictor::predict(const BatchData& batch) {
    size_t batch_size = batch.data.size();
    if (batch_size == 0) {
        return {{}, {}};
    }

    // 转换为 TimesNet 格式
    std::vector<std::vector<std::vector<float>>> x_enc(batch_size);
    std::vector<std::vector<float>> x_mark_enc(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        int actual_len = batch.lengths.empty() ?
            static_cast<int>(batch.data[i].size()) : batch.lengths[i];
        actual_len = std::min(actual_len, config_.seq_len);

        x_enc[i].reserve(config_.seq_len);
        x_mark_enc[i].reserve(config_.seq_len);

        for (int t = 0; t < config_.seq_len; ++t) {
            if (t < actual_len && t < static_cast<int>(batch.data[i].size())) {
                x_enc[i].push_back(batch.data[i][t]);
                x_mark_enc[i].push_back(1.0f);
            } else {
                // 填充零
                x_enc[i].push_back(std::vector<float>(config_.num_features, 0.0f));
                x_mark_enc[i].push_back(0.0f);
            }
        }
    }

    return predict_timesnet(x_enc, x_mark_enc);
}

std::pair<std::vector<int>, std::vector<float>> OnnxPredictor::predict(
    const std::vector<std::vector<std::vector<float>>>& data,
    const std::vector<int>& lengths
) {
    BatchData batch;
    batch.data = data;
    batch.lengths = lengths;
    return predict(batch);
}

} // namespace timesnet
