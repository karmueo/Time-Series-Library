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


class TimesNetPredictor:
    def __init__(self, model_path, model_name="TimesNet", num_classes=2, device="cpu", model_cfg=None):
        self.model_path = model_path
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.model_cfg = model_cfg or {}
        self.model = None
        self.num_features = None

    def load(self, num_features):
        self.num_features = num_features
        model_module = importlib.import_module(f"models.{self.model_name}")
        model_class = getattr(model_module, "Model")
        args = ModelArgs(self.model_cfg["seq_len"], num_features, self.num_classes, self.model_cfg)
        model = model_class(args).float()
        state = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def predict(self, batch, lengths):
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        if isinstance(lengths, np.ndarray):
            lengths = torch.from_numpy(lengths)
        batch = batch.to(self.device)
        lengths = lengths.to(self.device)
        mask = padding_mask(lengths, max_len=batch.shape[1]).to(self.device)
        with torch.no_grad():
            output = self.model(batch, mask, None, None)
            prob = torch.nn.functional.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        return pred.cpu().numpy(), prob.cpu().numpy()


def _softmax_numpy(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class OnnxTimesNetPredictor:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.session = None
        self.input_info = {}
        self.output_names = []
        self.num_features = None
        self.x_enc_name = None
        self.x_mark_enc_name = None
        self.extra_inputs = []

    def load(self, num_features):
        self.num_features = num_features
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime 未安装，无法加载 ONNX 模型") from exc

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]

        try:
            session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as exc:
            raise RuntimeError(f"ONNX 模型加载失败: {exc}") from exc

        inputs = session.get_inputs()
        self.input_info = {info.name: info for info in inputs}
        input_names = [info.name for info in inputs]
        self.x_enc_name = "x_enc" if "x_enc" in input_names else (input_names[0] if input_names else None)
        self.x_mark_enc_name = (
            "x_mark_enc" if "x_mark_enc" in input_names else (input_names[1] if len(input_names) > 1 else None)
        )
        self.extra_inputs = [name for name in input_names if name not in {self.x_enc_name, self.x_mark_enc_name}]
        self.output_names = [info.name for info in session.get_outputs()]
        self.session = session

    def _build_mask(self, lengths, seq_len):
        lengths = np.asarray(lengths, dtype=np.int64)
        steps = np.arange(seq_len)[None, :]
        return (steps < lengths[:, None]).astype(np.float32)

    def _placeholder_for_input(self, name, batch_size, seq_len):
        info = self.input_info.get(name)
        if info is None:
            return np.zeros((batch_size, seq_len), dtype=np.float32)
        if not info.shape:
            return np.zeros((), dtype=np.float32)
        shape = []
        for dim_idx, dim in enumerate(info.shape):
            if isinstance(dim, str) or dim is None:
                if dim_idx == 0:
                    shape.append(batch_size)
                elif dim_idx == 1:
                    shape.append(seq_len)
                elif dim_idx == 2:
                    shape.append(self.num_features)
                else:
                    shape.append(1)
            else:
                shape.append(int(dim))
        return np.zeros(shape, dtype=np.float32)

    def predict(self, batch, lengths):
        if self.session is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        batch = np.asarray(batch, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.int64)
        batch_size, seq_len, _ = batch.shape

        inputs = {}
        if self.x_enc_name:
            inputs[self.x_enc_name] = batch
        if self.x_mark_enc_name:
            # ONNX 版本使用显式 mask 表示有效时序
            inputs[self.x_mark_enc_name] = self._build_mask(lengths, seq_len)
        for name in self.extra_inputs:
            inputs[name] = self._placeholder_for_input(name, batch_size, seq_len)

        outputs = self.session.run(self.output_names or None, inputs)
        logits = outputs[0]
        probs = _softmax_numpy(logits)
        preds = np.argmax(probs, axis=1)
        return preds, probs
