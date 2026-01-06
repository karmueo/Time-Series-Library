import argparse
import json
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from udp.receiver import MulticastReceiver


FRAME_HEADER = 0xA999
FRAME_TAIL = 0x55AA
FRAME_WORDS = 34

CLASS_MAP = {
    1: "鸟群",
    2: "空飘",
    3: "飞机",
    4: "汽车",
    5: "大鸟",
    6: "小鸟",
    7: "无人机",
    0xF: "未知",
}


def parse_args():
    parser = argparse.ArgumentParser(description="接收识别结果组播报文并解析")
    parser.add_argument("--group", required=True, help="组播地址，如 230.1.1.24")
    parser.add_argument("--port", type=int, required=True, help="端口，如 8011")
    parser.add_argument("--iface", default="0.0.0.0", help="加入组播的网卡IP")
    parser.add_argument("--bind_ip", default="", help="绑定IP，默认0.0.0.0")
    parser.add_argument("--timeout", type=float, default=2.0, help="接收超时秒数")
    parser.add_argument("--max_packets", type=int, default=0, help="最多接收包数(0=不限)")
    return parser.parse_args()


def _from_bcd(value):
    return ((value >> 4) & 0x0F) * 10 + (value & 0x0F)


def _verify_checksum(data):
    if len(data) < 4:
        return False
    expected = struct.unpack("=H", data[-4:-2])[0]
    calc = sum(data[:-4]) & 0xFFFF
    return expected == calc


def parse_packet(data):
    if len(data) < FRAME_WORDS * 2:
        return None, "报文长度不足"
    frame_header = struct.unpack("=H", data[0:2])[0]
    frame_tail = struct.unpack("=H", data[-2:])[0]
    if frame_header != FRAME_HEADER or frame_tail != FRAME_TAIL:
        return None, "帧头/帧尾不匹配"
    if not _verify_checksum(data):
        return None, "校验和错误"

    header_words = struct.unpack("=12H", data[0:24])
    msg_type = header_words[1]
    frame_length = header_words[2]
    frame_seq = header_words[3]
    system_id = header_words[4]
    radar_word = header_words[5]
    radar_id = radar_word & 0xFF
    year = _from_bcd(header_words[6] & 0xFF)
    month = _from_bcd((header_words[6] >> 8) & 0xFF)
    day = _from_bcd(header_words[7] & 0xFF)
    hour = _from_bcd((header_words[7] >> 8) & 0xFF)
    minute = _from_bcd(header_words[8] & 0xFF)
    second = _from_bcd((header_words[8] >> 8) & 0xFF)
    sub_second = header_words[9]

    body_words = struct.unpack("=20H", data[24:64])
    target_count = body_words[0]
    batch_id = body_words[1]
    base_day = struct.unpack("=I", data[28:32])[0]
    time_25us = struct.unpack("=I", data[32:36])[0]
    class_major = body_words[6]
    confidence = body_words[7]

    return {
        "msg_type": msg_type,
        "frame_length": frame_length,
        "frame_seq": frame_seq,
        "system_id": system_id,
        "radar_id": radar_id,
        "time_bcd": f"20{year:02d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}",
        "sub_second_25us": sub_second,
        "target_count": target_count,
        "batch_id": batch_id,
        "base_day": base_day,
        "time_25us": time_25us,
        "time_s": time_25us * 25e-6,
        "class_major": class_major,
        "class_name": CLASS_MAP.get(class_major, "未知"),
        "confidence": confidence / 1000.0,
        "frame_length_ok": frame_length == FRAME_WORDS,
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
    print("开始接收识别结果组播报文...")
    while True:
        try:
            data, addr = receiver.recv()
        except Exception:
            continue
        result, err = parse_packet(data)
        out = {
            "src": f"{addr[0]}:{addr[1]}",
            "ok": err is None,
            "error": err,
            "result": result,
        }
        print(json.dumps(out, ensure_ascii=False))
        count += 1
        if args.max_packets > 0 and count >= args.max_packets:
            break


if __name__ == "__main__":
    main()
