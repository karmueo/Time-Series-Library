"""
支持固定 period_list 的 TimesNet 预测器

用于确保 Python PyTorch 和 ONNX 模型使用相同的 period_list，
从而获得完全一致的推理结果。
"""

import importlib

import numpy as np
import torch

from data_provider.uea import padding_mask


class ModelArgs:
    def __init__(self, seq_len, num_features, num_classes, model_cfg):
        self.task_name = "classification"
        self.seq_len = seq_len
        self.label_len = model_cfg.get("label_len", 48)
        self.pred_len = model_cfg.get("pred_len", 0)
        self.d_model = model_cfg.get("d_model", 64)
        self.n_heads = model_cfg.get("n_heads", 8)
        self.enc_in = num_features
        self.e_layers = model_cfg.get("e_layers", 2)
        self.d_layers = model_cfg.get("d_layers", 1)
        self.d_ff = model_cfg.get("d_ff", 256)
        self.top_k = model_cfg.get("top_k", 2)
        self.num_kernels = model_cfg.get("num_kernels", 6)
        self.embed = model_cfg.get("embed", "timeF")
        self.freq = model_cfg.get("freq", "h")
        self.dropout = model_cfg.get("dropout", 0.1)
        self.num_class = num_classes


class TimesNetPredictorFixedPeriod:
    """
    使用固定 period_list 的 TimesNet 预测器

    与 ONNX 模型使用相同的 period_list，确保推理结果完全一致。
    """

    def __init__(
        self,
        model_path,
        period_list,
        model_name="TimesNet",
        num_classes=2,
        device="cpu",
        model_cfg=None,
    ):
        """
        Args:
            model_path: 模型 checkpoint 路径
            period_list: 固定的周期列表，如 [10, 2, 1]
            model_name: 模型名称，默认 "TimesNet"
            num_classes: 类别数
            device: 设备 ("cpu" 或 "cuda")
            model_cfg: 模型配置字典
        """
        self.model_path = model_path
        self.period_list = period_list
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.model_cfg = model_cfg or {}
        self.model = None
        self.num_features = None

    def load(self, num_features):
        """
        加载模型并使用固定的 period_list

        这个方法会导入 Model_Precomputed 而不是原始的 Model，
        从而使用固定的 period_list 进行推理。
        """
        self.num_features = num_features

        # 导入预计算周期的模型
        from apps.cplusplus.scripts.export_onnx_accurate import Model_Precomputed

        args = ModelArgs(
            self.model_cfg["seq_len"], num_features, self.num_classes, self.model_cfg
        )
        args.period_list = self.period_list  # 添加 period_list

        # 创建预计算周期的模型
        model = Model_Precomputed(args, self.period_list).float()

        # 加载权重
        state = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

        print(f"[TimesNetPredictorFixedPeriod] 已加载模型")
        print(f"  period_list = {self.period_list}")
        print(f"  device = {self.device}")

    def predict(self, batch, lengths):
        """
        推理预测

        Args:
            batch: 输入数据 [batch_size, seq_len, num_features]
            lengths: 实际序列长度 [batch_size]

        Returns:
            preds: 预测类别 [batch_size]
            probs: 类别概率 [batch_size, num_classes]
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        if isinstance(lengths, np.ndarray):
            lengths = torch.from_numpy(lengths)

        batch = batch.to(self.device)
        lengths = lengths.to(self.device)

        # 构建 mask
        mask = padding_mask(lengths, max_len=batch.shape[1]).to(self.device)

        with torch.no_grad():
            output = self.model(batch, mask, None, None)
            prob = torch.nn.functional.softmax(output, dim=1)

        pred = torch.argmax(prob, dim=1)
        return pred.cpu().numpy(), prob.cpu().numpy()
