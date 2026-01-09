cmake_minimum_required(VERSION 3.20)
project(TimesNetPredictor CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 编译选项
if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# Include FetchContent
include(FetchContent)

# 选项
option(USE_GPU "Enable GPU inference with ONNX Runtime" ON)
option(BUILD_TESTS "Build unit tests" ON)

# ONNX Runtime 配置
# 支持多种方式:
# 1. 环境变量 ONNXRUNTIME_DIR
# 2. 默认位置 third_party/onnxruntime-linux-x64-gpu
# 3. Conda 环境 (onnxruntime/lib/libonnxruntime.so + onnxruntime/include)

# 尝试从环境变量获取
set(ORT_DIR_ENV "$ENV{ONNXRUNTIME_DIR}")

# 默认位置
set(ORT_PREFIX "onnxruntime-linux-x64-gpu")

# 确定 ONNX Runtime 位置
if(ORT_DIR_ENV)
    set(ORT_ROOT "${ORT_DIR_ENV}")
    message(STATUS "Using ONNX Runtime from environment: ${ORT_ROOT}")
else()
    set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/${ORT_PREFIX}")
    message(STATUS "Using ONNX Runtime from default location: ${ORT_ROOT}")
endif()

set(ORT_LIB_DIR "${ORT_ROOT}/lib")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")

# 检查 ONNX Runtime 是否可用 (支持多种目录结构)
# 1. 标准 C++ SDK 目录结构
if(EXISTS "${ORT_INCLUDE_DIR}/onnxruntime/core/session/onnxruntime_cxx_api.h")
    set(ORT_AVAILABLE TRUE)
    message(STATUS "ONNX Runtime found at ${ORT_ROOT}/include/onnxruntime/core/session")
    set(ORT_INCLUDE_DIR "${ORT_INCLUDE_DIR}/onnxruntime")
# 2. 简化目录结构
elseif(EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    set(ORT_AVAILABLE TRUE)
    message(STATUS "ONNX Runtime found at ${ORT_ROOT}/include")
# 3. Conda 环境结构 (onnxruntime package)
elseif(EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    set(ORT_AVAILABLE TRUE)
    message(STATUS "ONNX Runtime found at ${ORT_ROOT}/include")
elseif(EXISTS "/home/tl/miniconda3/envs/time-series/include/onnxruntime_cxx_api.h")
    # 使用 time-series conda 环境的 ONNX Runtime
    set(ORT_ROOT "/home/tl/miniconda3/envs/time-series")
    set(ORT_AVAILABLE TRUE)
    message(STATUS "Using ONNX Runtime from conda env: ${ORT_ROOT}")
else()
    set(ORT_AVAILABLE FALSE)
    message(WARNING "ONNX Runtime not found at ${ORT_ROOT}")
    message(WARNING "")
    message(WARNING "请选择以下方式之一:")
    message(WARNING "1. 下载 C++ SDK: https://github.com/microsoft/onnxruntime/releases")
    message(WARNING "2. 设置环境变量: export ONNXRUNTIME_DIR=/path/to/onnxruntime")
    message(WARNING "3. 安装 conda 包: conda install -c conda-forge onnxruntime")
    message(WARNING "")
    message(WARNING "Predictor 模块将被禁用")
endif()

# 查找库文件
if(ORT_AVAILABLE)
    if(EXISTS "${ORT_LIB_DIR}/libonnxruntime.so")
        set(ORT_LIB_FOUND TRUE)
    elseif(EXISTS "${ORT_ROOT}/lib/libonnxruntime.so")
        set(ORT_LIB_DIR "${ORT_ROOT}/lib")
        set(ORT_LIB_FOUND TRUE)
    elseif(EXISTS "${ORT_ROOT}/libonnxruntime.so")
        set(ORT_LIB_DIR "${ORT_ROOT}")
        set(ORT_LIB_FOUND TRUE)
    endif()

    if(NOT ORT_LIB_FOUND)
        message(WARNING "ONNX Runtime library not found in ${ORT_LIB_DIR}")
        set(ORT_AVAILABLE FALSE)
    endif()
endif()

# 源文件列表 (不包含 predictor)
set(SOURCES
    src/receiver/multicast_receiver.cpp
    src/publisher/multicast_publisher.cpp
    src/parser/packet_parser.cpp
    src/parser/trajectory_file_reader.cpp
    src/buffer/track_buffer.cpp
    src/normalizer/feature_normalizer.cpp
    src/utils/logger.cpp
    src/utils/json_helper.cpp
    src/yaml_config_loader.cpp
)

# 如果 ONNX 可用，添加 predictor
if(ORT_AVAILABLE)
    list(APPEND SOURCES src/predictor/onnx_predictor.cpp)
endif()

# 创建核心库
add_library(timesnet_lib STATIC ${SOURCES})

target_include_directories(timesnet_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${ORT_INCLUDE_DIR}
)

target_link_directories(timesnet_lib
    PRIVATE
        ${ORT_LIB_DIR}
)

# 如果 ONNX 可用，链接库
if(ORT_AVAILABLE)
    target_link_libraries(timesnet_lib
        PRIVATE
            ${ORT_LIB_DIR}/libonnxruntime.so
    )
endif()

target_link_libraries(timesnet_lib
    PRIVATE
        pthread
)

# nlohmann/json (header-only) - must be before it's used
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

target_link_libraries(timesnet_lib
    PRIVATE
        nlohmann_json::nlohmann_json
)

# yaml-cpp for YAML configuration support
FetchContent_Declare(
    yaml_cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml_cpp)

target_link_libraries(timesnet_lib
    PRIVATE
        yaml-cpp
)

# GoogleTest (must be before tests subdirectory)
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# 主可执行文件 (如果 ONNX 不可用，不编译 main)
if(ORT_AVAILABLE)
    add_executable(${PROJECT_NAME}
        src/main.cpp
    )
    target_include_directories(${PROJECT_NAME}
        PRIVATE
            ${ORT_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            timesnet_lib
            ${ORT_LIB_DIR}/libonnxruntime.so
    )

    # 测试程序：加载保存的输入向量进行推理
    add_executable(test_saved_input
        src/test_saved_input.cpp
    )
    target_include_directories(test_saved_input
        PRIVATE
            ${ORT_INCLUDE_DIR}
    )
    target_link_libraries(test_saved_input
        PRIVATE
            timesnet_lib
            ${ORT_LIB_DIR}/libonnxruntime.so
    )
else()
    message(WARNING "Main executable not built - ONNX Runtime required")
endif()

# Enable testing and add tests subdirectory
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 安装
include(GNUInstallDirs)
if(ORT_AVAILABLE)
    install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDDIR}
    )
endif()
install(TARGETS timesnet_lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_ARCHIVEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
