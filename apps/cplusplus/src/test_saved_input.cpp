/**
 * @brief 测试程序：加载保存的输入向量并进行推理
 *
 * 使用方法：
 *   ./test_saved_input <input.json> <model.onnx>
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include "json_helper.h"
#include "logger.h"
#include "predictor/onnx_predictor.h"

using namespace timesnet;

/**
 * @brief 从 JSON 文件加载输入数据
 */
bool load_input_from_file(
    const std::string& filepath,
    std::vector<std::vector<std::vector<float>>>& data,
    std::vector<int>& lengths,
    std::vector<uint32_t>& track_ids) {

    try {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            std::cerr << "Error: Failed to open file: " << filepath << std::endl;
            return false;
        }

        json input;
        ifs >> input;
        ifs.close();

        // 读取数据
        if (input.contains("data") && input["data"].is_array()) {
            data = input["data"].get<std::vector<std::vector<std::vector<float>>>>();
        }

        if (input.contains("lengths") && input["lengths"].is_array()) {
            lengths = input["lengths"].get<std::vector<int>>();
        }

        if (input.contains("track_ids") && input["track_ids"].is_array()) {
            track_ids = input["track_ids"].get<std::vector<uint32_t>>();
        }

        std::cout << "Loaded input data from: " << filepath << std::endl;
        std::cout << "  batch_size: " << data.size() << std::endl;
        if (!data.empty()) {
            std::cout << "  seq_len: " << data[0].size() << std::endl;
            if (!data[0].empty()) {
                std::cout << "  num_features: " << data[0][0].size() << std::endl;
            }
        }
        std::cout << "  track_ids: ";
        for (auto id : track_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to load input data: " << e.what() << std::endl;
        return false;
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <input.json> <model.onnx> [OPTIONS]\n"
              << "\n"
              << "Options:\n"
              << "  --num_classes N   Number of classes (default: 2)\n"
              << "  --num_features N  Number of features (default: 14)\n"
              << "  --seq_len LEN     Sequence length (default: 20)\n"
              << "  --use_gpu         Use GPU for inference\n"
              << "  --gpu_device_id ID GPU device ID (default: 0)\n"
              << "\n"
              << "Example:\n"
              << "  " << prog << " input.json models/timesnet.onnx\n"
              << "  " << prog << " input.json models/timesnet.onnx --num_features 14\n";
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    std::string model_path = argv[2];
    int num_classes = 2;
    int num_features = 14;
    int seq_len = 20;
    bool use_gpu = false;
    std::string gpu_device_id = "0";

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num_classes" && i + 1 < argc) {
            num_classes = std::stoi(argv[++i]);
        } else if (arg == "--num_features" && i + 1 < argc) {
            num_features = std::stoi(argv[++i]);
        } else if (arg == "--seq_len" && i + 1 < argc) {
            seq_len = std::stoi(argv[++i]);
        } else if (arg == "--use_gpu") {
            use_gpu = true;
        } else if (arg == "--gpu_device_id" && i + 1 < argc) {
            gpu_device_id = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "=== Test Saved Input ===" << std::endl;
    std::cout << "Input file: " << input_path << std::endl;
    std::cout << "Model file: " << model_path << std::endl;
    std::cout << "num_classes: " << num_classes << std::endl;
    std::cout << "num_features: " << num_features << std::endl;
    std::cout << "seq_len: " << seq_len << std::endl;
    std::cout << "use_gpu: " << (use_gpu ? "true" : "false") << std::endl;
    std::cout << std::endl;

    // 初始化日志
    LoggerConfig log_config;
    log_config.level = LogLevel::INFO;
    log_config.target = LogTarget::CONSOLE;
    log_config.enable_timestamp = true;
    log_config.enable_thread_id = false;
    Logger::instance() = Logger(log_config);

    // 加载输入数据
    std::vector<std::vector<std::vector<float>>> batch_data;
    std::vector<int> lengths;
    std::vector<uint32_t> track_ids;

    if (!load_input_from_file(input_path, batch_data, lengths, track_ids)) {
        return 1;
    }

    // 初始化预测器
    PredictorConfig pred_config;
    pred_config.model_path = model_path;
    pred_config.num_classes = num_classes;
    pred_config.seq_len = seq_len;
    pred_config.num_features = num_features;
    pred_config.use_gpu = use_gpu;
    pred_config.gpu_device_id = gpu_device_id;

    OnnxPredictor predictor(pred_config);
    if (!predictor.load()) {
        std::cerr << "Error: Failed to load model: " << predictor.last_error() << std::endl;
        return 1;
    }

    std::cout << "\n=== Running Inference ===" << std::endl;

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 推理
    auto result = predictor.predict(batch_data, lengths);
    std::vector<int> predictions = result.first;
    std::vector<float> probs = result.second;

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 打印结果
    std::cout << "\n=== Inference Results ===" << std::endl;
    std::cout << "Inference time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Batch size: " << predictions.size() << std::endl;
    std::cout << std::endl;

    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Track " << track_ids[i] << ":" << std::endl;
        std::cout << "  Prediction: " << predictions[i]
                  << " (" << (predictions[i] == 1 ? "UAV" : "Bird") << ")" << std::endl;
        std::cout << "  Prob(UAV): " << probs[i] << std::endl;
        std::cout << "  Prob(Bird): " << (1.0f - probs[i]) << std::endl;
        std::cout << std::endl;
    }

    // 以 JSON 格式输出完整结果
    json output;
    output["inference_time_ms"] = elapsed_ms;
    output["batch_size"] = predictions.size();
    output["results"] = json::array();

    for (size_t i = 0; i < predictions.size(); ++i) {
        json item;
        item["track_id"] = track_ids[i];
        item["prediction"] = predictions[i];
        item["prediction_label"] = (predictions[i] == 1 ? "UAV" : "Bird");
        item["prob_uav"] = probs[i];
        item["prob_bird"] = 1.0f - probs[i];
        output["results"].push_back(item);
    }

    std::cout << "=== JSON Output ===" << std::endl;
    std::cout << output.dump(2) << std::endl;

    return 0;
}
