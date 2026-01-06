import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_MAP = {
    "批号": "目标批号",
    "目标类型": "目标大类",
    "时间": "时间",
    "航迹历史": "航迹历史",
    "目标状态": "目标状态",
    "高（目标-滤波后）": "height_m",
    "径向距离": "r_m",
    "方位": "a_deg",
    "俯仰": "e_deg",
    "点迹距离": "pr_m",
    "点迹方位": "pa_deg",
    "点迹俯仰": "pe_deg",
    "全速度": "vel_m_s",
    "径向速度": "radial_vel_m_s",
    "方位速度": "az_vel_deg_s",
    "俯仰速度": "el_vel_deg_s",
    "航向": "course_deg",
    "目标信噪比": "snr_db",
    "多普勒展宽": "doppler",
    "JEM": "jem",
    "RCS": "rcs_db",
    "目标流水号": "目标流水号",
    "识别信息大类": "目标大类",
}

DEFAULT_FEATURE_COLS_PATH = PROJECT_ROOT / "features" / "feature_cols.json"


def get_feature_cols(path: str | Path | None = None) -> list[str]:
    feature_path = Path(path) if path else DEFAULT_FEATURE_COLS_PATH
    if not feature_path.is_file():
        raise FileNotFoundError(f"特征列配置不存在: {feature_path}")
    with feature_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("feature_cols.json 需要是 JSON 数组")
    cols = [str(c).strip() for c in data if str(c).strip()]
    if not cols:
        raise ValueError("feature_cols.json 不能为空")
    return cols
