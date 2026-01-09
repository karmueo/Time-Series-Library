#include "logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>

namespace timesnet {

Logger::Logger(const LoggerConfig& config) : config_(config), file_mutex_(std::make_unique<std::mutex>()) {
    if (config_.target == LogTarget::FILE || config_.target == LogTarget::BOTH) {
        if (!config_.file_path.empty()) {
            file_stream_.open(config_.file_path, std::ios::out | std::ios::app);
            if (!file_stream_.is_open()) {
                std::cerr << "Failed to open log file: " << config_.file_path << std::endl;
            }
        }
    }
}

Logger::~Logger() {
    close();
}

Logger::Logger(Logger&& other) noexcept
    : config_(other.config_),
      mutex_(),
      file_stream_(std::move(other.file_stream_)),
      file_mutex_(std::move(other.file_mutex_)) {}

Logger& Logger::operator=(Logger&& other) noexcept {
    if (this != &other) {
        close();
        config_ = other.config_;
        file_stream_ = std::move(other.file_stream_);
        file_mutex_ = std::move(other.file_mutex_);
    }
    return *this;
}

Logger& Logger::instance() {
    static Logger instance(LoggerConfig{});
    return instance;
}

void Logger::set_level(LogLevel level) {
    config_.level = level;
}

void Logger::set_target(LogTarget target) {
    config_.target = target;
}

void Logger::set_output_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    config_.file_path = path;
    if (!path.empty()) {
        file_stream_.open(path, std::ios::out | std::ios::app);
    }
}

void Logger::debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
void Logger::info(const std::string& msg) { log(LogLevel::INFO, msg); }
void Logger::warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
void Logger::error(const std::string& msg) { log(LogLevel::ERROR, msg); }
void Logger::fatal(const std::string& msg) { log(LogLevel::FATAL, msg); }

void Logger::log(LogLevel level, const std::string& msg) {
    if (level < config_.level) return;

    std::lock_guard<std::mutex> lock(mutex_);
    std::string formatted = format_message(level, msg);

    if (config_.target == LogTarget::CONSOLE || config_.target == LogTarget::BOTH) {
        write_to_console(formatted);
    }
    if (config_.target == LogTarget::FILE || config_.target == LogTarget::BOTH) {
        write_to_file(formatted);
    }
}

std::string Logger::format_message(LogLevel level, const std::string& msg) const {
    std::ostringstream oss;

    if (config_.enable_timestamp) {
        oss << get_timestamp() << " ";
    }

    oss << "[" << level_to_string(level) << "] ";

    if (config_.enable_thread_id) {
        oss << "[tid:" << get_thread_id() << "] ";
    }

    oss << msg;
    return oss.str();
}

void Logger::write_to_console(const std::string& msg) {
    std::cout << msg << std::endl;
}

void Logger::write_to_file(const std::string& msg) {
    if (file_stream_.is_open()) {
        std::lock_guard<std::mutex> lock(*file_mutex_);
        file_stream_ << msg << std::endl;
        file_stream_.flush();
    }
}

std::string Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "UNKNOWN";
    }
}

std::string Logger::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    std::tm tm_buf;
    oss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::get_thread_id() const {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

void Logger::flush() {
    if (file_stream_.is_open()) {
        std::lock_guard<std::mutex> lock(*file_mutex_);
        file_stream_.flush();
    }
}

void Logger::close() {
    if (file_stream_.is_open()) {
        std::lock_guard<std::mutex> lock(*file_mutex_);
        file_stream_.close();
    }
}

ScopedLogger::ScopedLogger(const std::string& name) : name_(name) {
    LOG_INFO("=== Entering scope: " + name_ + " ===");
}

ScopedLogger::~ScopedLogger() {
    LOG_INFO("=== Leaving scope: " +name_ + " ===");
}

void ScopedLogger::log(const std::string& msg) {
    LOG_INFO("[" + name_ + "] " + msg);
}

} // namespace timesnet
