// ONNX 推理工具 - 用于与 Python ONNX 对比
// 编译: g++ -std=c++17 -O2 -I../include -I../build-debug/_deps/onnxruntime-src/include
//      -I../build-debug/_deps/nlohmann_json-src/include
//      -L../build-debug/lib -lonnxruntime -lpthread
//      -o run_onnx_inference run_onnx_inference.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct InferenceResult {
    int pred;
    float prob_bird;
    float prob_uav;
    double inference_time_ms;
};

class OnnxClassifier {
public:
    OnnxClassifier(const std::string& model_path, bool use_gpu = false)
        : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxClassifier"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {

        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_gpu) {
            // 尝试使用 GPU
            try {
                Ort::ProviderOptions provider_options;
                provider_options["device_id"] = 0;
                session_options.AppendExecutionProvider_CUDA(provider_options);
                std::cout << "Using CUDA GPU" << std::endl;
            } catch (...) {
                std::cout << "CUDA not available, using CPU" << std::endl;
            }
        }

        session_ = std::make_unique<Ort::Session>(
            env_, model_path.c_str(), session_options);

        // 获取输入输出信息
        input_names_.push_back("x_enc");
        input_names_.push_back("x_mark_enc");

        output_names_.push_back("output");

        // 获取形状信息
        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        std::cout << "Input shape: ";
        for (auto s : input_shape) std::cout << s << " ";
        std::cout << std::endl;

        std::cout << "Output shape: ";
        for (auto s : output_shape) std::cout << s << " ";
        std::cout << std::endl;
    }

    InferenceResult predict(const std::vector<std::vector<std::vector<float>>>& x_enc,
                            const std::vector<std::vector<float>>& x_mark_enc) {
        auto start = std::chrono::high_resolution_clock::now();

        // 准备输入张量
        std::vector<int64_t> input_shape = {
            static_cast<int64_t>(x_enc.size()),
            static_cast<int64_t>(x_enc[0].size()),
            static_cast<int64_t>(x_enc[0][0].size())
        };

        // 展平输入数据
        std::vector<float> input_data(input_shape[0] * input_shape[1] * input_shape[2]);
        std::vector<float> mark_data(input_shape[0] * input_shape[1]);

        size_t idx = 0;
        for (const auto& batch : x_enc) {
            for (const auto& seq : batch) {
                for (const auto& feat : seq) {
                    input_data[idx++] = feat;
                }
            }
        }

        idx = 0;
        for (const auto& batch : x_mark_enc) {
            for (const auto& val : batch) {
                mark_data[idx++] = val;
            }
        }

        // 创建张量
        Ort::Value x_enc_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        std::vector<int64_t> mark_shape = {static_cast<int64_t>(x_mark_enc.size()),
                                           static_cast<int64_t>(x_mark_enc[0].size())};
        Ort::Value x_mark_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, mark_data.data(), mark_data.size(),
            mark_shape.data(), mark_shape.size());

        // 运行推理
        auto output = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &x_enc_tensor, 2,
            output_names_.data(), 1);

        auto end = std::chrono::high_resolution_clock::now();
        double inference_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // 处理输出
        float* output_data = output[0].GetTensorMutableData<float>();
        auto output_shape_info = output[0].GetTensorTypeAndShapeInfo();
        auto output_dim = output_shape_info.GetShape();

        // Softmax
        std::vector<float> probs(output_dim[1]);
        float sum = 0.0f;
        for (int i = 0; i < output_dim[1]; i++) {
            probs[i] = std::exp(output_data[i]);
            sum += probs[i];
        }
        for (int i = 0; i < output_dim[1]; i++) {
            probs[i] /= sum;
        }

        // 找最大概率的类别
        int pred = std::max_element(probs.begin(), probs.end()) - probs.begin();

        return {pred, probs[0], probs[1], inference_time_ms};
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <model.onnx> <test_data.json> <output.json>" << std::endl;
        std::cout << "Or: " << argv[0] << " <model.onnx> <data_dir> <output.json> --folder" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string data_path = argv[2];
    std::string output_path = argv[3];
    bool use_folder = (argc > 4 && std::string(argv[4]) == "--folder");

    std::cout << "=== ONNX Inference Comparison Tool ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Data: " << data_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;

    // 加载测试数据
    json test_data;
    if (use_folder) {
        // 从文件夹加载
        std::cout << "Loading data from folder: " << data_path << std::endl;
        // 这里简化为从 JSON 文件加载
        return 1;
    } else {
        std::ifstream f(data_path);
        if (!f.is_open()) {
            std::cerr << "Error: Cannot open " << data_path << std::endl;
            return 1;
        }
        f >> test_data;
    }

    // 创建分类器
    OnnxClassifier classifier(model_path, false);

    // 运行推理
    std::vector<json> results;

    auto start_total = std::chrono::high_resolution_clock::now();

    for (const auto& item : test_data["results"]) {
        // 重建输入数据
        std::vector<std::vector<std::vector<float>>> x_enc;
        std::vector<std::vector<float>> x_mark_enc;

        int batch_size = item.get("batch_size", 1);
        int seq_len = item.get("seq_len", 20);
        int num_features = item.get("num_features", 14);

        x_enc.resize(batch_size);
        x_mark_enc.resize(batch_size);

        const auto& data = item["data"];
        for (int b = 0; b < batch_size; b++) {
            x_enc[b].resize(seq_len);
            x_mark_enc[b].resize(seq_len);

            for (int t = 0; t < seq_len; t++) {
                x_enc[b][t].resize(num_features);
                for (int f = 0; f < num_features; f++) {
                    x_enc[b][t][f] = data[b][t][f];
                }
                x_mark_enc[b][t] = 1.0f;
            }
        }

        auto result = classifier.predict(x_enc, x_mark_enc);

        json res_item = {
            {"sample_idx", item.value("sample_idx", 0)},
            {"pred", result.pred},
            {"prob_bird", result.prob_bird},
            {"prob_uav", result.prob_uav},
            {"inference_time_ms", result.inference_time_ms}
        };
        results.push_back(res_item);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    // 保存结果
    json output = {
        {"source", "cpp_onnx"},
        {"model_path", model_path},
        {"num_samples", results.size()},
        {"total_time_ms", total_time_ms},
        {"avg_time_per_sample_ms", total_time_ms / results.size()},
        {"results", results}
    };

    std::ofstream of(output_path);
    of << output.dump(2) << std::endl;

    std::cout << "=== Results ===" << std::endl;
    std::cout << "Total samples: " << results.size() << std::endl;
    std::cout << "Total time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Avg per sample: " << (total_time_ms / results.size()) << " ms" << std::endl;
    std::cout << "Results saved to: " << output_path << std::endl;

    return 0;
}
