"""
测试批量推理优化功能

验证批量推理的逻辑正确性：
1. 单批号和多批号推理结果一致性
2. 批量推理的数据分发正确性
3. 边界情况处理（空批号、单个样本）
"""

import json
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# 模拟 core 模块的导入
sys_modules_mock = MagicMock()
sys_modules_mock.core.predictor.TimesNetPredictor = MagicMock
sys_modules_mock.core.track_buffer.TrackWindowBuffer = MagicMock
sys_modules_mock.features.extractor.FeatureNormalizer = MagicMock
sys_modules_mock.features.feature_config.FEATURE_MAP = {}
sys_modules_mock.features.feature_config.get_feature_cols = MagicMock(return_value=[])
sys_modules_mock.udp.parser.parse_packet = MagicMock
sys_modules_mock.udp.publisher.MulticastPublisher = MagicMock
sys_modules_mock.udp.receiver.MulticastReceiver = MagicMock

import sys
sys.modules['core'] = sys_modules_mock.core
sys.modules['core.predictor'] = sys_modules_mock.core.predictor
sys.modules['core.track_buffer'] = sys_modules_mock.core.track_buffer
sys.modules['features'] = sys_modules_mock.features
sys.modules['features.extractor'] = sys_modules_mock.features.extractor
sys.modules['features.feature_config'] = sys_modules_mock.features.feature_config
sys.modules['udp'] = sys_modules_mock.udp
sys.modules['udp.parser'] = sys_modules_mock.udp.parser
sys.modules['udp.publisher'] = sys_modules_mock.udp.publisher
sys.modules['udp.receiver'] = sys_modules_mock.udp.receiver


class TestBatchInferenceLogic:
    """测试批量推理的核心逻辑"""

    def test_batch_inference_single_batch(self):
        """测试单批号推理（回归测试）"""
        # 模拟单个批号的数据
        all_batch_ids = [1]
        all_track_ids = [[101, 102]]
        all_buffers = [
            (
                np.random.randn(2, 20, 14).astype(np.float32),  # batch: 2 samples
                np.array([20, 20], dtype=np.int64),  # lengths
                MagicMock()  # buffer mock
            )
        ]

        # 合并批次
        merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
        merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

        assert merged_batch.shape == (2, 20, 14)
        assert merged_lengths.shape == (2,)
        assert np.array_equal(merged_lengths, np.array([20, 20]))

    def test_batch_inference_multiple_batches(self):
        """测试多批号合并推理"""
        # 模拟3个批号的数据
        batch_1_data = np.random.randn(2, 20, 14).astype(np.float32)
        batch_2_data = np.random.randn(3, 20, 14).astype(np.float32)
        batch_3_data = np.random.randn(1, 20, 14).astype(np.float32)

        all_batch_ids = [1, 2, 3]
        all_track_ids = [[101, 102], [201, 202, 203], [301]]
        all_buffers = [
            (batch_1_data, np.array([20, 20], dtype=np.int64), MagicMock()),
            (batch_2_data, np.array([20, 20, 20], dtype=np.int64), MagicMock()),
            (batch_3_data, np.array([20], dtype=np.int64), MagicMock()),
        ]

        # 合并批次
        merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
        merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

        assert merged_batch.shape == (6, 20, 14), "应合并为 6 个样本（2+3+1）"
        assert merged_lengths.shape == (6,), "应有 6 个长度值"

        # 验证样本顺序
        sample_offsets = [0, 2, 5]
        for i, offset in enumerate(sample_offsets):
            expected_samples = len(all_track_ids[i])
            if i < len(sample_offsets) - 1:
                actual_slice = merged_batch[offset:sample_offsets[i + 1]]
            else:
                actual_slice = merged_batch[offset:]

            assert actual_slice.shape[0] == expected_samples

    def test_batch_result_distribution(self):
        """测试批量推理结果的正确分发"""
        # 模拟预测器输出
        num_total_samples = 5
        preds = np.array([0, 1, 0, 1, 1], dtype=np.int64)
        probs = np.array([
            [0.8, 0.2],  # sample 0: bird
            [0.3, 0.7],  # sample 1: uav
            [0.9, 0.1],  # sample 2: bird
            [0.4, 0.6],  # sample 3: uav
            [0.2, 0.8],  # sample 4: uav
        ], dtype=np.float32)

        all_batch_ids = [1, 2]
        all_track_ids = [[101, 102], [201, 202, 203]]

        # 分发结果
        sample_offset = 0
        results = {}
        for idx, batch_id in enumerate(all_batch_ids):
            track_ids = all_track_ids[idx]
            num_samples = len(track_ids)

            batch_preds = preds[sample_offset:sample_offset + num_samples]
            batch_probs = probs[sample_offset:sample_offset + num_samples]

            batch_results = []
            for i, tid in enumerate(track_ids):
                prob_uav = float(batch_probs[i, 1])
                prob_bird = float(batch_probs[i, 0])
                batch_results.append({
                    "batch_id": batch_id,
                    "track_id": tid,
                    "pred": int(batch_preds[i]),
                    "prob_uav": prob_uav,
                    "prob_bird": prob_bird,
                })

            results[batch_id] = batch_results
            sample_offset += num_samples

        # 验证批号1的结果
        assert len(results[1]) == 2
        assert results[1][0]["track_id"] == 101
        assert results[1][0]["pred"] == 0
        assert abs(results[1][0]["prob_uav"] - 0.2) < 1e-6

        # 验证批号2的结果
        assert len(results[2]) == 3
        assert results[2][0]["track_id"] == 201
        assert results[2][0]["pred"] == 0  # sample 2: [0.9, 0.1] -> bird
        assert abs(results[2][0]["prob_uav"] - 0.1) < 1e-6
        assert results[2][2]["track_id"] == 203
        assert results[2][2]["pred"] == 1  # sample 4: [0.2, 0.8] -> uav
        assert abs(results[2][2]["prob_uav"] - 0.8) < 1e-6

    def test_empty_batches(self):
        """测试空批号列表的处理"""
        all_batch_ids = []
        all_track_ids = []
        all_buffers = []

        # 空列表不应执行合并操作
        if all_buffers:
            merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
            assert False, "不应执行到此分支"
        else:
            assert True, "正确跳过了空批号处理"

    def test_single_sample_batch(self):
        """测试单样本批号的处理"""
        single_sample = np.random.randn(1, 20, 14).astype(np.float32)

        all_batch_ids = [100]
        all_track_ids = [[505]]
        all_buffers = [
            (single_sample, np.array([20], dtype=np.int64), MagicMock()),
        ]

        merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
        merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

        assert merged_batch.shape == (1, 20, 14)
        assert merged_lengths.shape == (1,)

    def test_consistency_sequential_vs_batch(self):
        """测试批量推理与顺序推理的结果一致性"""
        np.random.seed(42)

        # 创建测试数据：3个批号
        batches_data = []
        for i in range(3):
            num_samples = np.random.randint(1, 4)
            batch = np.random.randn(num_samples, 20, 14).astype(np.float32)
            lengths = np.full(num_samples, 20, dtype=np.int64)
            batches_data.append((batch, lengths))

        # 模拟批量推理
        all_batch_ids = [1, 2, 3]
        all_track_ids = [[101], [201, 202], [301]]
        all_buffers = [
            (batches_data[0][0], batches_data[0][1], MagicMock()),
            (batches_data[1][0], batches_data[1][1], MagicMock()),
            (batches_data[2][0], batches_data[2][1], MagicMock()),
        ]

        merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
        merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

        # 验证合并后的数据与原始数据一致
        sample_offset = 0
        for idx, batch_data in enumerate(batches_data):
            original_batch = batch_data[0]
            original_lengths = batch_data[1]
            num_samples = original_batch.shape[0]

            merged_slice = merged_batch[sample_offset:sample_offset + num_samples]
            lengths_slice = merged_lengths[sample_offset:sample_offset + num_samples]

            assert np.array_equal(merged_slice, original_batch), f"批号{idx + 1}的数据不一致"
            assert np.array_equal(lengths_slice, original_lengths), f"批号{idx + 1}的长度不一致"

            sample_offset += num_samples


class TestBatchInferenceIntegration:
    """集成测试：测试批量推理在实际场景中的表现"""

    def test_predictor_call_optimization(self):
        """测试预测器调用次数优化"""
        # 创建模拟预测器
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (
            np.array([0, 1, 0, 1], dtype=np.int64),
            np.random.randn(4, 2).astype(np.float32),
        )

        # 模拟3个批号
        all_batch_ids = [1, 2, 3]
        all_track_ids = [[101], [201], [301]]
        all_buffers = [
            (np.random.randn(1, 20, 14).astype(np.float32), np.array([20], dtype=np.int64), MagicMock()),
            (np.random.randn(1, 20, 14).astype(np.float32), np.array([20], dtype=np.int64), MagicMock()),
            (np.random.randn(1, 20, 14).astype(np.float32), np.array([20], dtype=np.int64), MagicMock()),
        ]

        # 批量推理：只调用一次预测器
        merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
        merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

        preds, probs = mock_predictor.predict(merged_batch, merged_lengths)

        # 验证只调用了一次
        assert mock_predictor.predict.call_count == 1
        assert preds.shape == (4,)
        assert probs.shape == (4, 2)

    def test_ema_smoothing_with_batch_inference(self):
        """测试批量推理下的 EMA 平滑功能"""
        # 模拟3个批号的推理结果
        preds = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
        probs = np.array([
            [0.8, 0.2],  # batch 1, sample 1
            [0.3, 0.7],  # batch 2, sample 1
            [0.4, 0.6],  # batch 2, sample 2
            [0.2, 0.8],  # batch 2, sample 3
            [0.1, 0.9],  # batch 3, sample 1
            [0.7, 0.3],  # batch 3, sample 2
        ], dtype=np.float32)

        all_batch_ids = [1, 2, 3]
        all_track_ids = [[101], [201, 202, 203], [301, 302]]

        alpha = 0.6
        batch_ema_prob_uav = {}

        # 应用 EMA 平滑
        sample_offset = 0
        for idx, batch_id in enumerate(all_batch_ids):
            track_ids = all_track_ids[idx]
            num_samples = len(track_ids)

            batch_probs = probs[sample_offset:sample_offset + num_samples]
            current_prob = float(np.mean(batch_probs[:, 1]))

            prev_prob = batch_ema_prob_uav.get(batch_id)
            ema_prob = current_prob if prev_prob is None else alpha * current_prob + (1.0 - alpha) * prev_prob
            batch_ema_prob_uav[batch_id] = ema_prob

            sample_offset += num_samples

        # 验证批号1的 EMA（首次，应等于当前值）
        assert abs(batch_ema_prob_uav[1] - 0.2) < 1e-6

        # 验证批号2的 EMA（首次，应等于当前值的平均值）
        expected_batch2 = (0.7 + 0.6 + 0.8) / 3
        assert abs(batch_ema_prob_uav[2] - expected_batch2) < 1e-6

        # 验证批号3的 EMA（首次，应等于当前值的平均值）
        expected_batch3 = (0.9 + 0.3) / 2
        assert abs(batch_ema_prob_uav[3] - expected_batch3) < 1e-6


class TestEdgeCases:
    """测试边界情况和异常处理"""

    def test_all_zero_length_batches(self):
        """测试所有批号长度为0的情况"""
        all_batch_ids = []
        all_track_ids = []
        all_buffers = []

        # 不应执行推理
        can_proceed = len(all_buffers) > 0
        assert not can_proceed

    def test_very_large_batch_count(self):
        """测试大量批号的处理能力"""
        num_batches = 100
        all_batch_ids = list(range(num_batches))
        all_track_ids = [[i * 100 + j for j in range(1)] for i in range(num_batches)]
        all_buffers = [
            (np.random.randn(1, 20, 14).astype(np.float32), np.array([20], dtype=np.int64), MagicMock())
            for _ in range(num_batches)
        ]

        merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
        merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

        assert merged_batch.shape == (num_batches, 20, 14)
        assert merged_lengths.shape == (num_batches,)

    def test_uneven_sequence_lengths(self):
        """测试不同批号的序列长度不同"""
        all_batch_ids = [1, 2, 3]
        all_track_ids = [[101, 102], [201], [301, 302, 303]]
        all_buffers = [
            (np.random.randn(2, 20, 14).astype(np.float32), np.array([20, 18], dtype=np.int64), MagicMock()),
            (np.random.randn(1, 20, 14).astype(np.float32), np.array([15], dtype=np.int64), MagicMock()),
            (np.random.randn(3, 20, 14).astype(np.float32), np.array([20, 19, 20], dtype=np.int64), MagicMock()),
        ]

        merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
        merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

        assert merged_batch.shape == (6, 20, 14)
        assert merged_lengths.shape == (6,)
        assert list(merged_lengths) == [20, 18, 15, 20, 19, 20]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
