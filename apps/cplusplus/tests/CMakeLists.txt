cmake_minimum_required(VERSION 3.20)

# 检查 ONNX Runtime 是否可用
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-gpu/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
    set(ORT_AVAILABLE TRUE)
else()
    set(ORT_AVAILABLE FALSE)
endif()

# 基础测试文件
set(TEST_SOURCES
    test_parser.cpp
    test_trajectory_reader.cpp
    test_buffer.cpp
    test_normalizer.cpp
    test_logger.cpp
    test_yaml_config.cpp
)

# 添加 receiver/publisher 测试 (不需要 ONNX)
list(APPEND TEST_SOURCES
    test_receiver.cpp
    test_publisher.cpp
)

# 如果 ONNX 可用，添加 predictor 测试
if(ORT_AVAILABLE)
    list(APPEND TEST_SOURCES test_predictor.cpp)
endif()

foreach(source IN LISTS TEST_SOURCES)
    get_filename_component(name ${source} NAME_WE)
    add_executable(test_${name} ${source})
    target_link_libraries(test_${name}
        PRIVATE
            timesnet_lib
            yaml-cpp
            GTest::gtest_main
            GTest::gmock_main
    )
    target_include_directories(test_${name}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
            ${CMAKE_SOURCE_DIR}/third_party/googletest/googlemock/include
    )
    add_test(NAME test_${name} COMMAND test_${name})
endforeach()
