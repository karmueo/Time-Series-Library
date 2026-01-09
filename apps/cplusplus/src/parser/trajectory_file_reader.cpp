#include "parser/trajectory_file_reader.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace timesnet {
namespace {

bool ends_with(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

char detect_delimiter(const std::string& path) {
    if (ends_with(path, ".xls")) {
        return '\t';
    }
    return ',';
}

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

std::vector<std::string> split_line(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(line);
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

bool parse_double(const std::string& token, double& value) {
    const char* str = token.c_str();
    char* end = nullptr;
    errno = 0;
    double v = std::strtod(str, &end);
    if (errno != 0 || end == str) {
        return false;
    }
    while (*end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) {
            return false;
        }
        ++end;
    }
    value = v;
    return true;
}

bool parse_row(const std::string& line, char delimiter, std::vector<double>& values) {
    auto tokens = split_line(line, delimiter);
    if (tokens.size() < 14) {
        return false;
    }

    values.clear();
    values.reserve(14);
    for (size_t i = 0; i < 14; ++i) {
        double v = 0.0;
        if (!parse_double(tokens[i], v)) {
            return false;
        }
        values.push_back(v);
    }
    return true;
}

std::vector<float> to_feature_vector(const std::vector<double>& raw) {
    // 与 ParsedTarget::to_feature_vector 一致，距离/速度从 m 转为 km
    return {
        static_cast<float>(raw[0] / 1000.0),  // r_m
        static_cast<float>(raw[1]),           // a_deg
        static_cast<float>(raw[2]),           // e_deg
        static_cast<float>(raw[3] / 1000.0),  // pr_m
        static_cast<float>(raw[4]),           // pa_deg
        static_cast<float>(raw[5]),           // pe_deg
        static_cast<float>(raw[6] / 1000.0),  // vel_m_s
        static_cast<float>(raw[7] / 1000.0),  // radial_vel_m_s
        static_cast<float>(raw[8]),           // az_vel_deg_s
        static_cast<float>(raw[9]),           // el_vel_deg_s
        static_cast<float>(raw[10]),          // doppler
        static_cast<float>(raw[11]),          // jem
        static_cast<float>(raw[12]),          // rcs_db
        static_cast<float>(raw[13])           // snr_db
    };
}

} // namespace

std::optional<TrajectoryFileResult> TrajectoryFileReader::load(
    const std::string& path,
    int max_points,
    std::string* error) {

    if (max_points <= 0) {
        if (error) {
            *error = "max_points must be positive";
        }
        return std::nullopt;
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        if (error) {
            *error = "Failed to open file: " + path;
        }
        return std::nullopt;
    }

    char delimiter = detect_delimiter(path);
    TrajectoryFileResult result;
    std::string line;
    bool has_data = false;

    while (std::getline(ifs, line)) {
        if (result.features.size() >= static_cast<size_t>(max_points)) {
            break;
        }
        std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }

        std::vector<double> raw_values;
        if (!parse_row(trimmed, delimiter, raw_values)) {
            if (!has_data) {
                // 首行非数字，视为表头
                continue;
            }
            if (error) {
                *error = "Invalid data row: " + trimmed;
            }
            return std::nullopt;
        }

        has_data = true;
        result.features.push_back(to_feature_vector(raw_values));
    }

    if (result.features.empty()) {
        if (error) {
            *error = "No valid data rows in file: " + path;
        }
        return std::nullopt;
    }

    return result;
}

} // namespace timesnet
