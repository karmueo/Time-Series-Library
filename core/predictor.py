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
