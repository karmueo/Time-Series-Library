#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>

namespace timesnet {

/**
 * @brief 日志级别
 */
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};

/**
 * @brief 日志输出目标
 */
enum class LogTarget {
    CONSOLE,
    FILE,
    BOTH
};

/**
 * @brief 日志配置
 */
struct LoggerConfig {
    LogLevel level = LogLevel::INFO;
    LogTarget target = LogTarget::CONSOLE;
    std::string file_path;
    bool enable_timestamp = true;
    bool enable_thread_id = false;
};

/**
 * @brief 日志类
 */
class Logger {
public:
    explicit Logger(const LoggerConfig& config);
    ~Logger();

    // 禁用拷贝
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // 移动语义
    Logger(Logger&&) noexcept;
    Logger& operator=(Logger&&) noexcept;

    static Logger& instance();

    void set_level(LogLevel level);
    void set_target(LogTarget target);
    void set_output_file(const std::string& path);

    void debug(const std::string& msg);
    void info(const std::string& msg);
    void warning(const std::string& msg);
    void error(const std::string& msg);
    void fatal(const std::string& msg);

    // 格式化日志
    void log(LogLevel level, const std::string& msg);

    // 清空日志文件
    void flush();
    void close();

private:
    LoggerConfig config_;
    std::mutex mutex_;
    std::ofstream file_stream_;
    std::unique_ptr<std::mutex> file_mutex_;

    std::string format_message(LogLevel level, const std::string& msg) const;
    void write_to_console(const std::string& msg);
    void write_to_file(const std::string& msg);
    std::string level_to_string(LogLevel level) const;
    std::string get_timestamp() const;
    std::string get_thread_id() const;
};

/**
 * @brief RAII 作用域日志器
 */
class ScopedLogger {
public:
    explicit ScopedLogger(const std::string& name);
    ~ScopedLogger();

    void log(const std::string& msg);

private:
    std::string name_;
};

} // namespace timesnet

// 便捷宏
#define LOG_DEBUG(msg) timesnet::Logger::instance().debug(msg)
#define LOG_INFO(msg) timesnet::Logger::instance().info(msg)
#define LOG_WARNING(msg) timesnet::Logger::instance().warning(msg)
#define LOG_ERROR(msg) timesnet::Logger::instance().error(msg)
#define LOG_FATAL(msg) timesnet::Logger::instance().fatal(msg)
#define LOG_SCOPED(name) timesnet::ScopedLogger logger(name)

#endif // LOGGER_H
