#include <gtest/gtest.h>
#include "buffer/track_buffer.h"

using namespace timesnet;

TEST(TrackBuffer, UpdateSingle) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);
    buffer.update(1, 100, features, 1.0);

    EXPECT_EQ(buffer.size(), 1);

    auto last_ts = buffer.get_last_timestamp(1);
    ASSERT_TRUE(last_ts.has_value());
    EXPECT_NEAR(last_ts.value(), 1.0, 0.001);
}

TEST(TrackBuffer, UpdateMultipleTracks) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);

    // 添加多个批次
    for (int batch_id = 0; batch_id < 5; ++batch_id) {
        for (int i = 0; i < 25; ++i) {
            buffer.update(batch_id, 100 + batch_id, features, 1.0 + i * 0.1);
        }
    }

    EXPECT_EQ(buffer.size(), 5);
}

TEST(TrackBuffer, BuildBatch) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);

    // 添加足够的数据
    for (int i = 0; i < 25; ++i) {
        buffer.update(1, 100, features, 1.0 + i * 0.1);
    }

    auto batch_opt = buffer.build_batch(20, 0);

    ASSERT_TRUE(batch_opt.has_value());
    const auto& batch_result = batch_opt.value();

    EXPECT_EQ(batch_result.track_ids.size(), 1);
    EXPECT_EQ(batch_result.data.size(), 1);
    EXPECT_EQ(batch_result.data[0].size(), 20);
}

TEST(TrackBuffer, BuildBatchInsufficientData) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);

    // 只添加 10 个点
    for (int i = 0; i < 10; ++i) {
        buffer.update(1, 100, features, 1.0 + i * 0.1);
    }

    auto batch_opt = buffer.build_batch(20, 0);

    EXPECT_FALSE(batch_opt.has_value());
}

TEST(TrackBuffer, MarkInferred) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);

    for (int i = 0; i < 25; ++i) {
        buffer.update(1, 100, features, 1.0 + i * 0.1);
    }

    // 第一次构建（应该有 20 个点，cleanup 删除了 5 个）
    auto batch_opt1 = buffer.build_batch(20, 0);
    ASSERT_TRUE(batch_opt1.has_value());
    EXPECT_EQ(batch_opt1->data[0].size(), 20);  // 20 个点

    // 标记已推理（不清空 buffer，只重置计数器）
    buffer.mark_inferred({1});

    // 第二次构建（与 Python 版本一致：仍然有 20 个点）
    // Python 版本：deque(maxlen=20) 保留所有点，推理后不清空
    auto batch_opt2 = buffer.build_batch(20, 0);
    EXPECT_TRUE(batch_opt2.has_value());  // 仍然有数据
    EXPECT_EQ(batch_opt2->data[0].size(), 20);  // 仍然是 20 个点
}

TEST(TrackBuffer, Clear) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);

    for (int batch_id = 0; batch_id < 5; ++batch_id) {
        for (int i = 0; i < 25; ++i) {
            buffer.update(batch_id, 100 + batch_id, features, 1.0 + i * 0.1);
        }
    }

    EXPECT_EQ(buffer.size(), 5);

    buffer.clear();

    EXPECT_EQ(buffer.size(), 0);
}

TEST(TrackBuffer, PendingInfo) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);

    // 只添加 15 个点
    for (int i = 0; i < 15; ++i) {
        buffer.update(1, 100, features, 1.0 + i * 0.1);
    }

    auto pending = buffer.get_pending_info(20, 0);

    EXPECT_EQ(pending.size(), 1);
    EXPECT_EQ(pending[1].count, 15);
    EXPECT_EQ(pending[1].needed, 5);
}

TEST(TrackBuffer, MultiBatch) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);

    // 添加 3 个批次
    for (int batch_id = 0; batch_id < 3; ++batch_id) {
        for (int i = 0; i < 25; ++i) {
            buffer.update(batch_id, 100 + batch_id, features, 1.0 + i * 0.1);
        }
    }

    auto batch_opt = buffer.build_batch(20, 0);

    ASSERT_TRUE(batch_opt.has_value());
    const auto& batch_result = batch_opt.value();

    EXPECT_EQ(batch_result.track_ids.size(), 3);
    EXPECT_EQ(batch_result.data.size(), 3);
}

TEST(TrackBuffer, UpdateWithFeatures) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    // 创建特定的特征值
    std::vector<float> features(12);
    features[0] = 1.0f;  // r_m
    features[1] = 2.0f;  // pr_m
    features[2] = 3.0f;  // azimuth
    // ... 其他特征

    // 添加足够多的点以满足最小序列长度
    for (int i = 0; i < 20; ++i) {
        buffer.update(1, 100, features, 1.0 + i * 0.1);
    }

    auto batch_opt = buffer.build_batch(20, 0);
    ASSERT_TRUE(batch_opt.has_value());

    const auto& batch_result = batch_opt.value();
    EXPECT_EQ(batch_result.data[0][0][0], 1.0f);
    EXPECT_EQ(batch_result.data[0][0][1], 2.0f);
    EXPECT_EQ(batch_result.data[0][0][2], 3.0f);
}

TEST(TrackBuffer, WindowStep) {
    TrackWindowBuffer::Config config;
    config.seq_len = 20;
    config.max_age_s = 10.0;

    TrackWindowBuffer buffer(config);

    std::vector<float> features(12, 0.5f);

    // 添加 20 个点
    for (int i = 0; i < 20; ++i) {
        buffer.update(1, 100, features, 1.0 + i * 0.1);
    }

    // 第一次构建（应该成功）
    auto batch_opt1 = buffer.build_batch(20, 0);
    ASSERT_TRUE(batch_opt1.has_value());

    // 标记已推理
    buffer.mark_inferred({1});

    // 添加 3 个新点（小于 window_step=5）
    for (int i = 20; i < 23; ++i) {
        buffer.update(1, 100, features, 1.0 + i * 0.1);
    }

    // 使用 window_step=5，应该不推理（因为只添加了 3 个新点）
    auto batch_opt2 = buffer.build_batch(20, 5);
    EXPECT_FALSE(batch_opt2.has_value());  // 新点数不足

    // 再添加 2 个新点（总共 5 个新点）
    for (int i = 23; i < 25; ++i) {
        buffer.update(1, 100, features, 1.0 + i * 0.1);
    }

    // 现在应该推理（因为添加了 5 个新点）
    auto batch_opt3 = buffer.build_batch(20, 5);
    EXPECT_TRUE(batch_opt3.has_value());
}
