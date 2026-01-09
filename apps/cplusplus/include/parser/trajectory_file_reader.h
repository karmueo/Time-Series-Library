#ifndef TRAJECTORY_FILE_READER_H
#define TRAJECTORY_FILE_READER_H

#include <optional>
#include <string>
#include <vector>

namespace timesnet {

/**
 * @brief 本地航迹文件读取结果
 */
struct TrajectoryFileResult {
    std::vector<std::vector<float>> features;  // 每行一个特征向量 (已按 TimesNet 特征顺序与单位处理)
};

/**
 * @brief 本地航迹文件读取器
 *
 * 支持 .xls (GBK 编码的 TSV) 与 .csv
 */
class TrajectoryFileReader {
public:
    /**
     * @brief 读取航迹文件并转换为特征向量
     * @param path 文件路径
     * @param max_points 最大读取点数
     * @param error 输出错误信息（可选）
     * @return 读取结果，失败返回 nullopt
     */
    static std::optional<TrajectoryFileResult> load(
        const std::string& path,
        int max_points,
        std::string* error = nullptr);
};

} // namespace timesnet

#endif // TRAJECTORY_FILE_READER_H
