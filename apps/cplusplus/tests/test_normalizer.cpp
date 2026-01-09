#include <gtest/gtest.h>
#include "normalizer/feature_normalizer.h"
#include <cmath>

using namespace timesnet;

TEST(FeatureNormalizer, DefaultInitialized) {
    FeatureNormalizer normalizer;
    EXPECT_FALSE(normalizer.is_initialized());
    EXPECT_EQ(normalizer.num_features(), 0);
}

TEST(FeatureNormalizer, WithStats) {
    FeatureNormalizer::Stats stats;
    stats.mean = {1.0f, 2.0f, 3.0f};
    stats.std = {0.5f, 0.5f, 0.5f};
    stats.min = {0.0f, 1.0f, 2.0f};
    stats.max = {2.0f, 3.0f, 4.0f};

    FeatureNormalizer normalizer(stats);

    EXPECT_TRUE(normalizer.is_initialized());
    EXPECT_EQ(normalizer.num_features(), 3);
}

TEST(FeatureNormalizer, Normalize) {
    FeatureNormalizer::Stats stats;
    stats.mean = {1.0f, 2.0f};
    stats.std = {1.0f, 2.0f};
    stats.min = {0.0f, 0.0f};
    stats.max = {2.0f, 4.0f};

    FeatureNormalizer normalizer(stats);

    // 归一化 mean 值应该得到 0
    std::vector<float> input = {1.0f, 2.0f};
    auto result = normalizer.normalize(input);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)[0], 0.0f, 0.001f);
    EXPECT_NEAR((*result)[1], 0.0f, 0.001f);

    // 归一化 (mean + std) 值应该得到 1
    input = {2.0f, 4.0f};
    result = normalizer.normalize(input);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)[0], 1.0f, 0.001f);
    EXPECT_NEAR((*result)[1], 1.0f, 0.001f);
}

TEST(FeatureNormalizer, Denormalize) {
    FeatureNormalizer::Stats stats;
    stats.mean = {1.0f, 2.0f};
    stats.std = {1.0f, 2.0f};
    stats.min = {0.0f, 0.0f};
    stats.max = {2.0f, 4.0f};

    FeatureNormalizer normalizer(stats);

    // 反归一化
    std::vector<float> normalized = {0.0f, 0.0f};
    auto result = normalizer.denormalize(normalized);

    EXPECT_NEAR(result[0], 1.0f, 0.001f);
    EXPECT_NEAR(result[1], 2.0f, 0.001f);

    normalized = {1.0f, 1.0f};
    result = normalizer.denormalize(normalized);

    EXPECT_NEAR(result[0], 2.0f, 0.001f);
    EXPECT_NEAR(result[1], 4.0f, 0.001f);
}

TEST(FeatureNormalizer, DimensionMismatch) {
    FeatureNormalizer::Stats stats;
    stats.mean = {1.0f, 2.0f, 3.0f};
    stats.std = {1.0f, 1.0f, 1.0f};
    stats.min = {0.0f, 0.0f, 0.0f};
    stats.max = {3.0f, 3.0f, 3.0f};

    FeatureNormalizer normalizer(stats);

    std::vector<float> input = {1.0f, 2.0f};  // 2 个特征
    auto result = normalizer.normalize(input);

    EXPECT_FALSE(result.has_value());
}

TEST(FeatureNormalizer, SetStats) {
    FeatureNormalizer normalizer;

    FeatureNormalizer::Stats stats;
    stats.mean = {1.0f, 2.0f};
    stats.std = {1.0f, 1.0f};
    stats.min = {0.0f, 0.0f};
    stats.max = {2.0f, 2.0f};

    normalizer.set_stats(stats);

    EXPECT_TRUE(normalizer.is_initialized());
    EXPECT_EQ(normalizer.num_features(), 2);
}

TEST(FeatureNormalizer, SaveLoadJson) {
    FeatureNormalizer::Stats stats;
    stats.mean = {1.0f, 2.0f};
    stats.std = {0.5f, 0.5f};
    stats.min = {0.0f, 1.0f};
    stats.max = {2.0f, 3.0f};

    FeatureNormalizer normalizer(stats);

    // 保存
    bool saved = normalizer.save_to_json("/tmp/test_stats.json");
    EXPECT_TRUE(saved);

    // 加载
    auto loaded_opt = FeatureNormalizer::from_json_file("/tmp/test_stats.json");
    ASSERT_TRUE(loaded_opt.has_value());

    const auto& loaded = loaded_opt.value();
    EXPECT_EQ(loaded.num_features(), 2);

    // 清理
    std::remove("/tmp/test_stats.json");
}

TEST(FeatureNormalizer, HandleOutliers) {
    FeatureNormalizer::Stats stats;
    stats.mean = {1.0f};
    stats.std = {1.0f};
    stats.min = {0.0f};
    stats.max = {2.0f};

    FeatureNormalizer normalizer(stats);

    // 输入包含 NaN
    std::vector<float> input = {std::numeric_limits<float>::quiet_NaN()};
    auto result = normalizer.normalize(input);

    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(std::isnan((*result)[0]));

    // 输入包含 Inf
    input = {std::numeric_limits<float>::infinity()};
    result = normalizer.normalize(input);

    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(std::isinf((*result)[0]));
}

TEST(FeatureConfig, GetFeatureNames) {
    auto names = FeatureConfig::get_feature_names();

    EXPECT_EQ(names.size(), 12);
    EXPECT_EQ(names[0], "r_m");
    EXPECT_EQ(names[1], "pr_m");
    EXPECT_EQ(names[11], "flags");
}
