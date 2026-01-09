#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "logger.h"

using namespace timesnet;

TEST(Logger, Singleton) {
    auto& logger1 = Logger::instance();
    auto& logger2 = Logger::instance();
    EXPECT_EQ(&logger1, &logger2);
}

TEST(Logger, SetLevel) {
    Logger& logger = Logger::instance();
    logger.set_level(LogLevel::DEBUG);
    // 不应该崩溃
    logger.debug("Test debug");
    logger.info("Test info");
    logger.warning("Test warning");
    logger.error("Test error");
    logger.fatal("Test fatal");
}

TEST(Logger, SetTarget) {
    Logger& logger = Logger::instance();
    logger.set_target(LogTarget::CONSOLE);
    logger.set_target(LogTarget::FILE);
    logger.set_output_file("/tmp/test_log.txt");
}

TEST(Logger, ScopedLogger) {
    LOG_SCOPED("TestScope");
    ScopedLogger scoped_logger("TestScope2");
    scoped_logger.log("Test message");
}

TEST(Logger, Flush) {
    Logger& logger = Logger::instance();
    logger.set_output_file("/tmp/test_log_flush.txt");
    logger.info("Test flush");
    logger.flush();
    logger.close();
}

TEST(Logger, Timestamp) {
    LoggerConfig config;
    config.level = LogLevel::DEBUG;
    config.target = LogTarget::CONSOLE;
    config.enable_timestamp = true;

    Logger test_logger(config);
    test_logger.info("Test with timestamp");
}

TEST(Logger, ThreadId) {
    LoggerConfig config;
    config.level = LogLevel::DEBUG;
    config.target = LogTarget::CONSOLE;
    config.enable_timestamp = false;
    config.enable_thread_id = true;

    Logger test_logger(config);
    test_logger.info("Test with thread id");
}

TEST(Logger, LevelFiltering) {
    LoggerConfig config;
    config.level = LogLevel::WARNING;  // 只显示 WARNING 及以上
    config.target = LogTarget::CONSOLE;

    Logger test_logger(config);
    test_logger.debug("This should not appear");
    test_logger.info("This should not appear");
    test_logger.warning("This should appear");
    test_logger.error("This should appear");
}
