/**
 * @brief TimesNet 航迹预测 C++ 主程序
 *
 * 组播接收 -> 预测 -> 组播发送
 */

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>

#include "logger.h"
#include "json_helper.h"
#include "yaml_config_loader.h"
#include "receiver/multicast_receiver.h"
#include "publisher/multicast_publisher.h"
#include "parser/packet_parser.h"
#include "parser/trajectory_file_reader.h"
#include "predictor/onnx_predictor.h"
#include "buffer/track_buffer.h"
#include "normalizer/feature_normalizer.h"

using namespace timesnet;

// 全局标志，用于优雅退出
volatile sig_atomic_t g_running = 1;

// 保存文件的计数器（用于生成唯一文件名）
static std::atomic<uint64_t> g_save_counter{0};

void signal_handler(int sig) {
    g_running = 0;
    LOG_INFO("Received signal " + std::to_string(sig) + ", shutting down...");
}

/**
 * @brief 保存推理输入向量到 JSON 文件
 * @param batch_data 输入数据 [batch][seq][features]
 * @param lengths 序列长度
 * @param track_ids 轨迹 ID 列表
 * @param dir_path 输出文件夹路径
 * @return 是否保存成功
 */
bool save_input_to_file(
    const std::vector<std::vector<std::vector<float>>>& batch_data,
    const std::vector<int>& lengths,
    const std::vector<uint32_t>& track_ids,
    const std::string& dir_path) {

    try {
        // 创建文件夹（如果不存在）
        std::filesystem::path dir(dir_path);
        if (!std::filesystem::exists(dir)) {
            if (!std::filesystem::create_directories(dir)) {
                LOG_ERROR("Failed to create directory: " + dir_path);
                return false;
            }
        }

        // 生成带时间戳和计数器的文件名
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << dir_path << "/input_"
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
           << "_" << std::setfill('0') << std::setw(3) << ms.count()
           << "_" << g_save_counter.fetch_add(1)
           << ".json";

        std::string filepath = ss.str();

        // 构建输出 JSON
        json output;
        output["batch_size"] = batch_data.size();
        output["seq_len"] = batch_data.empty() ? 0 : batch_data[0].size();
        output["num_features"] = batch_data.empty() ? 0 :
            (batch_data[0].empty() ? 0 : batch_data[0][0].size());
        output["track_ids"] = track_ids;
        output["lengths"] = lengths;
        output["data"] = batch_data;

        // 写入文件
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            LOG_ERROR("Failed to open file for writing: " + filepath);
            return false;
        }

        ofs << output.dump(2);
        ofs.close();

        LOG_INFO("Saved input data to: " + filepath);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to save input data: " + std::string(e.what()));
        return false;
    }
}

/**
 * @brief 命令行参数
 */
struct Args {
    // 配置文件
    std::string config_path = "config.yaml";

    // 输入组播
    std::string in_group;
    int in_port = 0;
    std::string in_iface = "0.0.0.0";
    std::string bind_ip = "";
    double timeout = 2.0;
    bool skip_checksum = false;

    // 输出组播
    std::string out_group;
    int out_port = 0;
    std::string out_iface = "0.0.0.0";
    int ttl = 1;

    // 模型
    std::string model_path;
    int num_classes = 2;
    int seq_len = 20;
    int min_seq_len = 20;
    int num_features = 14;
    std::string stats_path;
    bool use_gpu = false;  // 默认使用 CPU
    std::string gpu_device_id = "0";
    float ema_alpha = 0.4f;  // EMA 平滑因子，与 Python 版本一致

    // 缓冲
    double max_age_s = 10.0;
    int publish_interval_ms = 0;
    int window_step = 0;

    // 其他
    bool print_targets = false;
    bool print_features = false;
    std::string save_input_path;  // 保存推理输入向量路径

    // 本地测试
    bool local_test_enabled = false;
    std::string local_test_path;
    int local_test_points = 20;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "\n"
              << "Options:\n"
              << "  --config PATH            Config file path (default: config.yaml)\n"
              << "  --in_group GROUP         Input multicast group\n"
              << "  --in_port PORT           Input port\n"
              << "  --in_iface IFACE         Input interface IP\n"
              << "  --bind_ip IP             Bind IP (default: 0.0.0.0)\n"
              << "  --timeout SECONDS        Receive timeout\n"
              << "  --skip_checksum          Skip checksum verification\n"
              << "\n"
              << "  --out_group GROUP        Output multicast group\n"
              << "  --out_port PORT          Output port\n"
              << "  --out_iface IFACE        Output interface IP\n"
              << "  --ttl TTL                Multicast TTL\n"
              << "\n"
              << "  --model_path PATH        ONNX model path\n"
              << "  --num_classes N          Number of classes\n"
              << "  --seq_len LEN            Sequence length (default: 20)\n"
              << "  --min_seq_len LEN        Minimum sequence length (default: 20)\n"
              << "  --num_features N         Number of features (default: 14)\n"
              << "  --stats_path PATH        Stats JSON path\n"
              << "  --use_gpu                Use GPU for inference\n"
              << "  --gpu_device_id ID       GPU device ID\n"
              << "  --ema_alpha ALPHA        EMA smoothing factor (default: 0.4)\n"
              << "\n"
              << "  --max_age_s SECONDS      Maximum track age\n"
              << "  --publish_interval_ms MS Publish interval\n"
              << "  --window_step STEP       Window step (0=infer all)\n"
              << "\n"
              << "  --print_targets          Print parsed targets\n"
              << "  --print_features         Print features\n"
              << "  --help                   Show this help\n"
              << "\n"
              << "Note: Use config.yaml for configuration. Command line args override config file.\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--config" && i + 1 < argc) {
            args.config_path = argv[++i];
        } else if (arg == "--in_group" && i + 1 < argc) {
            args.in_group = argv[++i];
        } else if (arg == "--in_port" && i + 1 < argc) {
            args.in_port = std::stoi(argv[++i]);
        } else if (arg == "--in_iface" && i + 1 < argc) {
            args.in_iface = argv[++i];
        } else if (arg == "--bind_ip" && i + 1 < argc) {
            args.bind_ip = argv[++i];
        } else if (arg == "--timeout" && i + 1 < argc) {
            args.timeout = std::stod(argv[++i]);
        } else if (arg == "--skip_checksum") {
            args.skip_checksum = true;
        } else if (arg == "--out_group" && i + 1 < argc) {
            args.out_group = argv[++i];
        } else if (arg == "--out_port" && i + 1 < argc) {
            args.out_port = std::stoi(argv[++i]);
        } else if (arg == "--out_iface" && i + 1 < argc) {
            args.out_iface = argv[++i];
        } else if (arg == "--ttl" && i + 1 < argc) {
            args.ttl = std::stoi(argv[++i]);
        } else if (arg == "--model_path" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--num_classes" && i + 1 < argc) {
            args.num_classes = std::stoi(argv[++i]);
        } else if (arg == "--seq_len" && i + 1 < argc) {
            args.seq_len = std::stoi(argv[++i]);
        } else if (arg == "--min_seq_len" && i + 1 < argc) {
            args.min_seq_len = std::stoi(argv[++i]);
        } else if (arg == "--num_features" && i + 1 < argc) {
            args.num_features = std::stoi(argv[++i]);
        } else if (arg == "--stats_path" && i + 1 < argc) {
            args.stats_path = argv[++i];
        } else if (arg == "--use_gpu") {
            args.use_gpu = true;
        } else if (arg == "--gpu_device_id" && i + 1 < argc) {
            args.gpu_device_id = argv[++i];
        } else if (arg == "--ema_alpha" && i + 1 < argc) {
            args.ema_alpha = std::stof(argv[++i]);
        } else if (arg == "--max_age_s" && i + 1 < argc) {
            args.max_age_s = std::stod(argv[++i]);
        } else if (arg == "--publish_interval_ms" && i + 1 < argc) {
            args.publish_interval_ms = std::stoi(argv[++i]);
        } else if (arg == "--window_step" && i + 1 < argc) {
            args.window_step = std::stoi(argv[++i]);
        } else if (arg == "--print_targets") {
            args.print_targets = true;
        } else if (arg == "--print_features") {
            args.print_features = true;
        }
    }

    return args;
}

bool validate_args(const Args& args) {
    if (args.local_test_enabled) {
        if (args.local_test_path.empty()) {
            std::cerr << "Error: local_test_path is required when local_test_enabled is true\n";
            return false;
        }
        if (args.local_test_points <= 0) {
            std::cerr << "Error: local_test_points must be positive\n";
            return false;
        }
        if (args.model_path.empty()) {
            std::cerr << "Error: --model_path is required (set in config or via --model_path)\n";
            return false;
        }
        return true;
    }

    // 如果配置文件中已有值，则不检查必填项
    bool has_in_group = !args.in_group.empty();
    bool has_in_port = args.in_port > 0;
    bool has_out_group = !args.out_group.empty();
    bool has_out_port = args.out_port > 0;
    bool has_model_path = !args.model_path.empty();

    // 检查必填参数
    if (!has_in_group) {
        std::cerr << "Error: --in_group is required (set in config or via --in_group)\n";
        return false;
    }
    if (!has_in_port) {
        std::cerr << "Error: --in_port is required (set in config or via --in_port)\n";
        return false;
    }
    if (!has_out_group) {
        std::cerr << "Error: --out_group is required (set in config or via --out_group)\n";
        return false;
    }
    if (!has_out_port) {
        std::cerr << "Error: --out_port is required (set in config or via --out_port)\n";
        return false;
    }
    if (!has_model_path) {
        std::cerr << "Error: --model_path is required (set in config or via --model_path)\n";
        return false;
    }
    return true;
}

/**
 * @brief 从 YAML 配置文件加载参数
 */
bool load_config(Args& args) {
    if (args.config_path.empty()) {
        return true;  // 不使用配置文件
    }

    auto config_opt = YAMLConfigLoader::load_from_file(args.config_path);
    if (!config_opt.has_value()) {
        LOG_WARNING("Failed to load config file: " + YAMLConfigLoader().last_error());
        return true;  // 配置文件不存在也继续
    }

    const auto& cfg = config_opt.value();

    // 合并配置：命令行参数优先
    if (args.in_group.empty() && !cfg.receiver.group.empty()) {
        args.in_group = cfg.receiver.group;
    }
    if (args.in_port == 0 && cfg.receiver.port > 0) {
        args.in_port = cfg.receiver.port;
    }
    if (args.in_iface == "0.0.0.0" && !cfg.receiver.iface.empty()) {
        args.in_iface = cfg.receiver.iface;
    }
    if (args.bind_ip.empty() && !cfg.receiver.bind_ip.empty()) {
        args.bind_ip = cfg.receiver.bind_ip;
    }
    if (args.timeout == 2.0 && cfg.receiver.timeout_s > 0) {
        args.timeout = cfg.receiver.timeout_s;
    }
    if (cfg.receiver.skip_checksum) {
        args.skip_checksum = true;
    }

    if (args.out_group.empty() && !cfg.publisher.group.empty()) {
        args.out_group = cfg.publisher.group;
    }
    if (args.out_port == 0 && cfg.publisher.port > 0) {
        args.out_port = cfg.publisher.port;
    }
    if (args.out_iface == "0.0.0.0" && !cfg.publisher.iface.empty()) {
        args.out_iface = cfg.publisher.iface;
    }
    if (args.ttl == 1 && cfg.publisher.ttl > 0) {
        args.ttl = cfg.publisher.ttl;
    }

    if (args.model_path.empty() && !cfg.predictor.model_path.empty()) {
        args.model_path = cfg.predictor.model_path;
    }
    if (args.num_classes == 2 && cfg.predictor.num_classes > 0) {
        args.num_classes = cfg.predictor.num_classes;
    }
    if (args.seq_len == 20 && cfg.predictor.seq_len > 0) {
        args.seq_len = cfg.predictor.seq_len;
    }
    if (args.num_features == 14 && cfg.predictor.num_features > 0) {
        args.num_features = cfg.predictor.num_features;
    }
    if (!cfg.predictor.use_gpu) {
        args.use_gpu = cfg.predictor.use_gpu;
    }
    if (args.gpu_device_id == "0" && !cfg.predictor.gpu_device_id.empty()) {
        args.gpu_device_id = cfg.predictor.gpu_device_id;
    }

    if (args.max_age_s == 10.0 && cfg.buffer.max_age_s > 0) {
        args.max_age_s = cfg.buffer.max_age_s;
    }

    if (args.stats_path.empty() && !cfg.normalizer.stats_path.empty()) {
        args.stats_path = cfg.normalizer.stats_path;
    }

    if (args.min_seq_len == 20 && cfg.inference.min_seq_len > 0) {
        args.min_seq_len = cfg.inference.min_seq_len;
    }
    if (args.publish_interval_ms == 0 && cfg.inference.publish_interval_ms > 0) {
        args.publish_interval_ms = cfg.inference.publish_interval_ms;
    }
    if (args.window_step == 0 && cfg.inference.window_step > 0) {
        args.window_step = cfg.inference.window_step;
    }
    if (args.ema_alpha == 0.4f && cfg.inference.ema_alpha > 0) {
        args.ema_alpha = cfg.inference.ema_alpha;
    }
    if (cfg.inference.print_targets) {
        args.print_targets = cfg.inference.print_targets;
    }
    if (cfg.inference.print_features) {
        args.print_features = cfg.inference.print_features;
    }
    if (!cfg.inference.save_input_path.empty()) {
        args.save_input_path = cfg.inference.save_input_path;
    }

    if (cfg.local_test.enabled) {
        args.local_test_enabled = cfg.local_test.enabled;
    }
    if (args.local_test_path.empty() && !cfg.local_test.xls_path.empty()) {
        args.local_test_path = cfg.local_test.xls_path;
    }
    if (args.local_test_points == 20 && cfg.local_test.max_points > 0) {
        args.local_test_points = cfg.local_test.max_points;
    }

    return true;
}

bool run_local_file_inference(const Args& args, FeatureNormalizer& normalizer, OnnxPredictor& predictor) {
    std::string error;
    auto data_opt = TrajectoryFileReader::load(args.local_test_path, args.local_test_points, &error);
    if (!data_opt.has_value()) {
        LOG_ERROR("Failed to load local file: " + error);
        return false;
    }

    const auto& raw_features = data_opt.value().features;
    if (raw_features.size() < static_cast<size_t>(args.seq_len)) {
        LOG_ERROR("Local file has insufficient points: " +
                  std::to_string(raw_features.size()) +
                  " < seq_len=" + std::to_string(args.seq_len));
        return false;
    }

    std::vector<std::vector<float>> seq_features;
    seq_features.reserve(args.seq_len);
    for (int i = 0; i < args.seq_len; ++i) {
        auto features_opt = normalizer.normalize(raw_features[i]);
        if (!features_opt.has_value()) {
            seq_features.push_back(raw_features[i]);
        } else {
            seq_features.push_back(features_opt.value());
        }
    }

    std::vector<std::vector<std::vector<float>>> batch_data = {seq_features};
    std::vector<int> lengths = {args.seq_len};

    if (!args.save_input_path.empty()) {
        std::vector<uint32_t> track_ids = {1};
        save_input_to_file(batch_data, lengths, track_ids, args.save_input_path);
    }

    auto infer_result = predictor.predict(batch_data, lengths);
    if (infer_result.first.empty() || infer_result.second.empty()) {
        LOG_ERROR("Inference failed: empty result");
        return false;
    }

    PredictionResult result;
    result.batch_id = 1;
    result.track_id = 1;
    result.timestamp_ms = 0;
    result.pred = infer_result.first[0];
    result.prob_uav = infer_result.second[0];
    result.prob_bird = 1.0f - result.prob_uav;

    json out;
    out["batch_id"] = result.batch_id;
    out["count"] = 1;
    out["items"] = json::array();
    json item;
    item["batch_id"] = result.batch_id;
    item["track_id"] = result.track_id;
    item["timestamp_ms"] = result.timestamp_ms;
    item["time_25us"] = static_cast<uint64_t>(0);
    item["pred"] = result.pred;
    item["prob_uav"] = result.prob_uav;
    item["prob_bird"] = result.prob_bird;
    out["items"].push_back(item);
    std::cout << out.dump() << std::endl;

    return true;
}

int main(int argc, char* argv[]) {
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 解析命令行参数
    Args args = parse_args(argc, argv);

    // 初始化日志（必须在最早时刻，以便后续使用）
    LoggerConfig log_config;
    log_config.level = LogLevel::WARNING;  // 只输出警告和错误，减少日志
    log_config.target = LogTarget::CONSOLE;
    log_config.enable_timestamp = true;
    log_config.enable_thread_id = false;
    Logger::instance() = Logger(log_config);

    // 加载配置文件 (如果存在)
    if (!load_config(args)) {
        return 1;
    }

    // 验证必填参数（加载配置文件后验证）
    if (!validate_args(args)) {
        print_usage(argv[0]);
        return 1;
    }

    // 加载归一化器
    FeatureNormalizer normalizer;
    if (!args.stats_path.empty()) {
        auto norm_opt = FeatureNormalizer::from_json_file(args.stats_path);
        if (norm_opt.has_value()) {
            normalizer = std::move(norm_opt.value());
        } else {
            LOG_WARNING("Failed to load stats, using raw features");
        }
    }

    // 初始化接收器
    ReceiverConfig recv_config;
    recv_config.group = args.in_group;
    recv_config.port = args.in_port;
    recv_config.iface = args.in_iface;
    recv_config.bind_ip = args.bind_ip;
    recv_config.timeout_s = args.timeout;
    recv_config.skip_checksum = args.skip_checksum;

    MulticastReceiver receiver(recv_config);
    if (!receiver.open()) {
        LOG_ERROR("Failed to open receiver: " + receiver.last_error());
        return 1;
    }

    // 初始化发布器
    PublisherConfig pub_config;
    pub_config.group = args.out_group;
    pub_config.port = args.out_port;
    pub_config.iface = args.out_iface;
    pub_config.ttl = args.ttl;

    MulticastPublisher publisher(pub_config);
    if (!publisher.open()) {
        LOG_ERROR("Failed to open publisher: " + publisher.last_error());
        return 1;
    }

    // 初始化预测器
    PredictorConfig pred_config;
    pred_config.model_path = args.model_path;
    pred_config.num_classes = args.num_classes;
    pred_config.seq_len = args.seq_len;
    pred_config.num_features = args.num_features;
    pred_config.use_gpu = args.use_gpu;
    pred_config.gpu_device_id = args.gpu_device_id;

    OnnxPredictor predictor(pred_config);
    if (!predictor.load()) {
        LOG_ERROR("Failed to load model: " + predictor.last_error());
        return 1;
    }

    if (args.local_test_enabled) {
        std::cout << "本地文件测试模式已开启，使用文件进行推理..." << std::endl;
        if (!run_local_file_inference(args, normalizer, predictor)) {
            return 1;
        }
        return 0;
    }

    // 初始化轨迹缓冲
    TrackWindowBuffer::Config buf_config;
    buf_config.seq_len = args.seq_len;
    buf_config.max_age_s = args.max_age_s;
    TrackWindowBuffer buffer(buf_config);

    // 状态跟踪
    std::unordered_map<uint32_t, double> batch_last_seen;
    std::unordered_map<uint32_t, double> batch_last_publish;
    std::unordered_map<uint32_t, float> batch_ema_prob_uav;
    float ema_alpha = args.ema_alpha;

    std::cout << "开始接收组播并预测..." << std::endl;

    // 收包统计与心跳日志
    constexpr double report_interval_s = 2.0;
    auto last_report = std::chrono::steady_clock::now();
    auto last_recv = last_report;
    uint64_t recv_packets = 0;
    uint64_t parsed_packets = 0;
    uint64_t parsed_targets = 0;
    uint64_t empty_packets = 0;

    // 主循环
    while (g_running) {
        // 接收报文
        auto recv_opt = receiver.recv();
        auto loop_now = std::chrono::steady_clock::now();
        if (!recv_opt.has_value()) {
            auto since_last = std::chrono::duration<double>(loop_now - last_recv).count();
            auto since_report = std::chrono::duration<double>(loop_now - last_report).count();
            if (since_report >= report_interval_s) {
                if (since_last >= report_interval_s) {
                    LOG_WARNING("No multicast packets received in the last " +
                                std::to_string(static_cast<int>(since_last)) + "s");
                }
                LOG_INFO("Receiver stats (last " + std::to_string(static_cast<int>(since_report)) +
                         "s): recv=" + std::to_string(recv_packets) +
                         ", parsed=" + std::to_string(parsed_packets) +
                         ", targets=" + std::to_string(parsed_targets) +
                         ", empty=" + std::to_string(empty_packets));
                recv_packets = 0;
                parsed_packets = 0;
                parsed_targets = 0;
                empty_packets = 0;
                last_report = loop_now;
            }
            continue;
        }

        const auto& recv_info = recv_opt.value();
        recv_packets += 1;
        last_recv = loop_now;

        // 解析报文
        PacketParser parser;
        auto targets = parser.parse(recv_info.data.data(), recv_info.data.size(), args.skip_checksum);
        if (targets.empty()) {
            empty_packets += 1;
            continue;
        }
        parsed_packets += 1;
        parsed_targets += targets.size();

        // 打印目标信息
        if (args.print_targets) {
            json out;
            out["src"] = recv_info.src_addr + ":" + std::to_string(recv_info.src_port);
            out["count"] = targets.size();
            LOG_INFO(out.dump(2));
        }

        double now = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // 处理每个目标
        for (const auto& target : targets) {
            auto features_opt = normalizer.normalize(target.to_feature_vector());
            std::vector<float> features_to_use;

            if (!features_opt.has_value()) {
                // 归一化失败时使用原始特征
                features_to_use = target.to_feature_vector();
            } else {
                features_to_use = features_opt.value();
            }

            if (args.print_features) {
                json out;
                out["raw"] = target.to_feature_vector();
                if (features_opt.has_value()) {
                    out["normalized"] = features_opt.value();
                }
                LOG_DEBUG(out.dump());
            }

            // 更新轨迹缓冲
            buffer.update(target.batch_id, target.track_id, features_to_use, target.timestamp);
            batch_last_seen[target.batch_id] = now;
        }

        // 清理过期批次
        std::vector<uint32_t> expired;
        for (const auto& [batch_id, ts] : batch_last_seen) {
            if (now - ts > args.max_age_s) {
                expired.push_back(batch_id);
            }
        }
        for (auto batch_id : expired) {
            batch_last_seen.erase(batch_id);
            batch_last_publish.erase(batch_id);
            batch_ema_prob_uav.erase(batch_id);
        }

        // 构建批次并推理
        auto batch_opt = buffer.build_batch(args.min_seq_len, args.window_step);
        if (!batch_opt.has_value()) {
            // 打印待处理航迹信息（与 Python 版本一致）
            auto pending_info = buffer.get_pending_info(args.min_seq_len, args.window_step);
            for (const auto& [track_id, info] : pending_info) {
                // info.count 是当前已收集的点数，args.seq_len 是总共需要的点数
                std::cout << "batch_id=" << track_id << " track_id=" << track_id
                         << " [" << info.count << "/" << args.seq_len << "]" << std::endl;
            }
            continue;
        }

        const auto& batch_result = batch_opt.value();
        const auto& track_ids = batch_result.track_ids;
        const auto& batch_data = batch_result.data;
        const auto& lengths = batch_result.lengths;

        // 检查发布间隔
        if (args.publish_interval_ms > 0) {
            bool skip = false;
            for (auto track_id : track_ids) {
                auto it = batch_last_publish.find(track_id);
                if (it != batch_last_publish.end()) {
                    if ((now - it->second) * 1000 < args.publish_interval_ms) {
                        skip = true;
                        break;
                    }
                }
            }
            if (skip) {
                continue;
            }
        }

        // 保存输入向量（如果配置了保存路径）
        if (!args.save_input_path.empty()) {
            save_input_to_file(batch_data, lengths, track_ids, args.save_input_path);
        }

        // 推理
        std::pair<std::vector<int>, std::vector<float>> infer_result = predictor.predict(batch_data, lengths);
        std::vector<int> predictions = infer_result.first;
        std::vector<float> probs = infer_result.second;

        // 构建结果
        std::vector<PredictionResult> results;
        for (size_t i = 0; i < track_ids.size(); ++i) {
            uint32_t batch_id = track_ids[i];
            auto last_ts_opt = buffer.get_last_timestamp(batch_id);

            PredictionResult result;
            result.batch_id = batch_id;
            result.track_id = batch_id;
            result.timestamp_ms = last_ts_opt.has_value() ?
                static_cast<int64_t>(last_ts_opt.value() * 1000) : 0;
            result.pred = predictions[i];
            result.prob_uav = probs[i];
            result.prob_bird = 1.0f - probs[i];
            results.push_back(result);
        }

        // EMA 平滑
        if (!results.empty()) {
            float current_prob = 0.0f;
            for (const auto& r : results) {
                current_prob += r.prob_uav;
            }
            current_prob /= results.size();

            uint32_t first_batch_id = track_ids[0];
            auto it = batch_ema_prob_uav.find(first_batch_id);
            float ema_prob;
            if (it == batch_ema_prob_uav.end()) {
                ema_prob = current_prob;
            } else {
                ema_prob = ema_alpha * current_prob + (1.0f - ema_alpha) * it->second;
            }
            batch_ema_prob_uav[first_batch_id] = ema_prob;

            int batch_pred = ema_prob >= 0.5f ? 1 : 0;
            for (auto& r : results) {
                r.pred = batch_pred;
                r.prob_uav = ema_prob;
                r.prob_bird = 1.0f - ema_prob;
            }
        }

        // 打印结果（与 Python 版本一致的 JSON 格式）
        json out;
        out["batch_id"] = track_ids[0];
        out["count"] = results.size();
        out["items"] = json::array();
        for (const auto& r : results) {
            json item;
            item["batch_id"] = r.batch_id;
            item["track_id"] = r.track_id;
            item["timestamp_ms"] = r.timestamp_ms;
            auto ts_opt = buffer.get_last_timestamp(r.batch_id);
            item["time_25us"] = ts_opt.has_value() ?
                static_cast<uint64_t>(std::llround(ts_opt.value() / 25e-6)) :
                static_cast<uint64_t>(std::llround(r.timestamp_ms * 40.0));
            item["pred"] = r.pred;
            item["prob_uav"] = r.prob_uav;
            item["prob_bird"] = r.prob_bird;
            out["items"].push_back(item);
        }
        std::cout << out.dump() << std::endl;

        // 发送结果
        publisher.send(results);

        // 标记已推理
        buffer.mark_inferred(track_ids);
        for (auto track_id : track_ids) {
            batch_last_publish[track_id] = now;
        }

        auto since_report = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_report).count();
        if (since_report >= report_interval_s) {
            LOG_INFO("Receiver stats (last " + std::to_string(static_cast<int>(since_report)) +
                     "s): recv=" + std::to_string(recv_packets) +
                     ", parsed=" + std::to_string(parsed_packets) +
                     ", targets=" + std::to_string(parsed_targets) +
                     ", empty=" + std::to_string(empty_packets));
            recv_packets = 0;
            parsed_packets = 0;
            parsed_targets = 0;
            empty_packets = 0;
            last_report = std::chrono::steady_clock::now();
        }
    }

    // 清理
    receiver.close();
    publisher.close();

    return 0;
}
