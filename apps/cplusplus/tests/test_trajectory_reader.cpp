#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "parser/trajectory_file_reader.h"

using namespace timesnet;

class TrajectoryFileReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        temp_dir_ = std::filesystem::temp_directory_path() / "traj_reader_test";
        std::filesystem::create_directories(temp_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(temp_dir_);
    }

    std::filesystem::path temp_dir_;
};

TEST_F(TrajectoryFileReaderTest, LoadTsvWithHeader) {
    std::string content =
        "r_m\taz\tel\tpr_m\tpa\tpe\tvel\tvr\tvaz\tvel\t"
        "doppler\tjem\trcs\tsnr\n"
        "1000\t1\t2\t2000\t3\t4\t500\t600\t7\t8\t9\t10\t11\t12\n"
        "1100\t2\t3\t2100\t4\t5\t510\t610\t8\t9\t10\t11\t12\t13\n";

    auto file_path = temp_dir_ / "sample.xls";
    std::ofstream(file_path) << content;

    std::string error;
    auto result = TrajectoryFileReader::load(file_path.string(), 1, &error);
    ASSERT_TRUE(result.has_value()) << error;
    ASSERT_EQ(result->features.size(), 1u);

    const auto& feature = result->features[0];
    ASSERT_EQ(feature.size(), 14u);
    EXPECT_FLOAT_EQ(feature[0], 1.0f);
    EXPECT_FLOAT_EQ(feature[3], 2.0f);
    EXPECT_FLOAT_EQ(feature[6], 0.5f);
    EXPECT_FLOAT_EQ(feature[7], 0.6f);
    EXPECT_FLOAT_EQ(feature[12], 11.0f);
    EXPECT_FLOAT_EQ(feature[13], 12.0f);
}
