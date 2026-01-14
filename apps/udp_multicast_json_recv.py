import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from udp.receiver import MulticastReceiver


DEFAULT_GROUP = "230.1.88.51"
DEFAULT_PORT = 8003
TARGET_MSG_ID = 0x7111

# 按报文说明定义字段与类型约束
REQUIRED_FIELDS = {
    "yr": "int",
    "mo": "int",
    "dy": "int",
    "h": "int",
    "min": "int",
    "sec": "int",
    "msec": "float",
    "tar_id": "int",
    "tar_category": "int",
    "guid_stat": "int",
    "ecef_x": "float",
    "ecef_y": "float",
    "ecef_z": "float",
    "ecef_vx": "float",
    "ecef_vy": "float",
    "ecef_vz": "float",
    "h_dvi_pct": "float",
    "v_dvi_pct": "float",
    "enu_r": "float",
    "enu_a": "float",
    "enu_e": "float",
    "enu_v": "float",
    "enu_h": "float",
    "lon": "float",
    "lat": "float",
    "alt": "float",
}


def parse_args():
    parser = argparse.ArgumentParser(description="接收并解析组播 JSON 报文")
    parser.add_argument("--group", default=DEFAULT_GROUP, help="组播地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="端口号")
    parser.add_argument("--iface", default="0.0.0.0", help="加入组播的网卡IP")
    parser.add_argument("--bind_ip", default="", help="绑定IP，默认0.0.0.0")
    parser.add_argument("--timeout", type=float, default=2.0, help="接收超时秒数")
    parser.add_argument("--max_packets", type=int, default=0, help="最多接收包数(0=不限)")
    parser.add_argument("--encoding", default="utf-8", help="JSON 字符串编码")
    parser.add_argument("--strict", action="store_true", help="字段缺失或类型错误时标记为失败")
    return parser.parse_args()


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _decode_payload(data, encoding):
    text = data.decode(encoding, errors="ignore").strip()
    if not text:
        return "", "空报文"
    return text, None


def _extract_json_text(text):
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def validate_payload(payload):
    missing = [key for key in REQUIRED_FIELDS if key not in payload]
    type_errors = []
    for key, expected in REQUIRED_FIELDS.items():
        if key not in payload:
            continue
        value = payload[key]
        if expected == "int" and not _is_int(value):
            type_errors.append(f"{key} 期望 int，实际 {type(value).__name__}")
        if expected == "float" and not _is_number(value):
            type_errors.append(f"{key} 期望 float，实际 {type(value).__name__}")
    return missing, type_errors


def parse_packet(data, encoding):
    text, err = _decode_payload(data, encoding)
    if err:
        return None, err
    json_text = _extract_json_text(text)
    if not json_text:
        return None, "未找到 JSON 片段"
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, f"JSON 解析失败: {exc}"
    if not isinstance(payload, dict):
        return None, "JSON 顶层不是对象"
    missing, type_errors = validate_payload(payload)
    return {
        "payload": payload,
        "missing": missing,
        "type_errors": type_errors,
    }, None


def main():
    args = parse_args()
    receiver = MulticastReceiver(
        group=args.group,
        port=args.port,
        iface=args.iface,
        bind_ip=args.bind_ip,
        timeout_s=args.timeout,
    ).open()

    count = 0
    print(f"开始接收组播 JSON 报文: {args.group}:{args.port}")
    while True:
        try:
            data, addr = receiver.recv()
        except Exception:
            continue
        result, err = parse_packet(data, args.encoding)
        ok = err is None
        if ok:
            msg_id = result["payload"].get("msg_id")
            if msg_id != TARGET_MSG_ID:
                continue
        if ok and args.strict:
            ok = not result["missing"] and not result["type_errors"]
        out = {
            "src": f"{addr[0]}:{addr[1]}",
            "ok": ok,
            "error": err,
            "result": result,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
        count += 1
        if args.max_packets > 0 and count >= args.max_packets:
            break


if __name__ == "__main__":
    main()
