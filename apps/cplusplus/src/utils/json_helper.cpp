#include "json_helper.h"
#include <fstream>
#include <iostream>

namespace timesnet {

std::optional<json> JsonHelper::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file: " << path << std::endl;
        return std::nullopt;
    }

    try {
        json j;
        file >> j;
        return j;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return std::nullopt;
    }
}

bool JsonHelper::save_to_file(const json& j, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return false;
    }

    try {
        file << j.dump(2); // 格式化输出
        return true;
    } catch (const json::type_error& e) {
        std::cerr << "JSON type error: " << e.what() << std::endl;
        return false;
    }
}

std::string JsonHelper::get_string(const json& j, const std::string& key, const std::string& default_value) {
    auto it = j.find(key);
    if (it != j.end() && it->is_string()) {
        return it->get<std::string>();
    }
    return default_value;
}

std::vector<float> JsonHelper::get_float_array(const json& j, const std::string& key) {
    std::vector<float> result;
    auto it = j.find(key);
    if (it != j.end() && it->is_array()) {
        for (const auto& elem : *it) {
            if (elem.is_number()) {
                result.push_back(elem.get<float>());
            }
        }
    }
    return result;
}

int JsonHelper::get_int(const json& j, const std::string& key, int default_value) {
    auto it = j.find(key);
    if (it != j.end() && it->is_number_integer()) {
        return it->get<int>();
    }
    return default_value;
}

double JsonHelper::get_double(const json& j, const std::string& key, double default_value) {
    auto it = j.find(key);
    if (it != j.end() && it->is_number()) {
        return it->get<double>();
    }
    return default_value;
}

bool JsonHelper::get_bool(const json& j, const std::string& key, bool default_value) {
    auto it = j.find(key);
    if (it != j.end() && it->is_boolean()) {
        return it->get<bool>();
    }
    return default_value;
}

bool JsonHelper::has_key(const json& j, const std::string& key) {
    return j.find(key) != j.end();
}

json JsonHelper::create_inference_result(
    int inference_id,
    int64_t timestamp_ms,
    const std::vector<json>& results
) {
    json j;
    j["inference_id"] = inference_id;
    j["timestamp_ms"] = timestamp_ms;
    j["results"] = results;
    return j;
}

json JsonHelper::create_prediction_result(
    uint32_t track_id,
    int pred,
    float prob_uav,
    float prob_bird
) {
    json j;
    j["track_id"] = track_id;
    j["pred"] = pred;
    j["prob_uav"] = prob_uav;
    j["prob_bird"] = prob_bird;
    return j;
}

} // namespace timesnet
