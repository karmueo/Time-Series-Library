# Repository Guidelines

## 项目结构与模块组织
- `run.py` 为统一入口，按 `--task_name` 调度 `exp/exp_*` 中的实验类（长短期预测、插值、异常检测、分类、零样本）。
- `models/` 存放模型定义，新模型需在 `exp/exp_basic.py` 的 `model_dict` 注册。
- `data_provider/` 为数据加载，`dataset/` 存放下载的数据；`layers/` 为可复用网络模块；`utils/` 提供指标、日志与工具函数。
- `scripts/` 归档可复现实验的 bash 命令（如 `scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh`）；`tutorial/` 含示例 notebook；`pic/` 存文档配图。

## 架构与扩展建议
- 每个任务的实验逻辑封装在 `exp/exp_*`，共享训练/测试接口；新增任务应复用基础组件并对齐命令行参数。
- 模型注册后即可通过 `--model` 切换；如需额外超参，请在对应 `Exp` 类和脚本中补充，保持默认值可复现。
- 日志与 checkpoint 路径默认在 `./checkpoints/`，可通过参数覆盖；提交前清理大文件与无关缓存。

## 构建、测试与开发命令
- 虚拟环境：`conda activate time-series`.
- 最小训练示例：  
  `python run.py --task_name long_term_forecast --is_training 1 --model_id TimesNet_ETTh1 --model TimesNet --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --features M --seq_len 96 --label_len 48 --pred_len 336 --itr 1`
- 更推荐直接运行脚本：如 `bash scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh`、`bash scripts/imputation/ETT_script/TimesNet_ETTh1.sh`、`bash scripts/anomaly_detection/PSM/TimesNet.sh`。
- 数据需先按 README 下载并放置于 `dataset/` 相应子目录，路径在脚本中已给出，可按需调整。
- GPU 相关：`--use_gpu/--use_multi_gpu`、`--devices`、`--gpu_type cuda|mps`；评估模式用 `--is_training 0`。

## 代码风格与命名
- Python 采用 4 空格缩进，遵循 PEP 8；类名使用驼峰，函数与变量用 snake_case，与现有 `models/*.py`、`exp/*.py` 保持一致。
- 日志/打印优先 f-string，遵循 `run.py` 的固定随机种子；脚本命名沿用 `Model_dataset.sh`，按任务目录归档。

## 测试准则
- 无独立单测框架，默认以对应脚本或 `--itr 1` 运行快速验证，确认指标与日志。
- 新模型需在 `dataset/` 中至少一个公开数据上完成训练+测试闭环，并在 PR 说明中粘贴完整命令与关键输出。
- 使用 `--is_training 0` 可快速检查数据管线与推理流程，无需完整训练。
- 关注异常检测与分类任务的日志阈值设置，提交前确认评估指标与脚本参数一致。

## 安全与配置提示
- 数据集体积较大，放置于 `dataset/`，提交代码时避免上传原始数据与中间缓存；必要时在 PR 说明下载链接或使用占位符文件。
- 依赖固定在 `requirements.txt`（Torch 1.7.1 基线），如需更高版本请说明兼容性；建议在虚拟环境中验证后再提交。
- 多 GPU 训练务必标注 `--use_multi_gpu` 与 `--devices` 设置，确保日志中记录显存需求；在本地先用小批量与 `--itr 1` 做冒烟测试以降低资源开销，必要时附上数据包校验信息。

## 提交与 PR 准则
- 提交应小而聚焦，使用祈使句（如 `Add TimeXer demo script`、`Fix ETT dataloader null handling`）。
- PR 需简述目的与任务，若是新模型请附 Issue（参见 `CONTRIBUTING.md`），列出运行命令、关键指标或日志，包含所需脚本与 `model_dict` 注册以便复现。
- 补充数据来源路径（位于 `dataset/`）、硬件信息（GPU 型号/数量）以及可能影响复现的参数说明。必要时附截图或日志片段，方便快速审阅。
- 评审前可自检格式与 lints（若有），保证新增脚本可直接运行，不引入无关文件。

## 回答说明
回答时用中文