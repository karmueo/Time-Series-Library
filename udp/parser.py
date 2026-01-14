import math
import struct


TRK_FLAG_HEAD = 0x1010
TRK_FLAG_TAIL = 0x55AA

TAR_STR = 26
TAR_LEN = 160


# WGS84 constants for ECEF -> LLA conversion
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def _ecef_to_lla(x_m, y_m, z_m):
    # Iterative conversion for stable latitude/height estimation
    lon = math.atan2(y_m, x_m)
    p = math.hypot(x_m, y_m)
    lat = math.atan2(z_m, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z_m, p * (1.0 - WGS84_E2 * (n / (n + alt))))
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lon), math.degrees(lat), alt


def parse_target(data, start):
    # Word 13: 目标状态 (b0-b3)
    status_raw = struct.unpack("=B", data[start:start + 1])[0]
    status = status_raw % 16
    # Word 14: 批号信息（目标批号）
    tar_id = struct.unpack("=H", data[start + 2:start + 4])[0]
    # Word 16: 目标流水号
    tar_seq = struct.unpack("=I", data[start + 6:start + 10])[0]
    # Word 18: 航迹历史
    trk_cn = struct.unpack("=H", data[start + 10:start + 12])[0]

    # Word 24: 时间 (25us)
    t1 = struct.unpack("=I", data[start + 22:start + 26])[0]
    tim = t1 * 25e-6

    # Word 26, 28, 30: 滤波位置信息 (径向距离, 方位, 俯仰)
    r, a, e = struct.unpack("=2I1i", data[start + 26:start + 38])
    r = r * 1e-1  # 0.1m
    a = a * 1e-5  # 0.00001°
    e = e * 1e-5  # 0.00001°

    # Word 32, 34, 36: 点迹位置信息 (径向距离, 方位, 俯仰)
    pr, pa, pe = struct.unpack("=2I1i", data[start + 38:start + 50])
    pr = pr * 1e-1  # 0.1m
    pa = pa * 1e-5  # 0.00001°
    pe = pe * 1e-5  # 0.00001°

    # Word 38: 速度信息：径向速度 (0.01 m/s)
    t1 = struct.unpack("=h", data[start + 50:start + 52])[0]
    radial_vel = t1 * 1e-2
    # Word 40: 方位速度 (0.001 °/s)
    t1 = struct.unpack("=h", data[start + 54:start + 56])[0]
    az_vel = t1 * 1e-3
    # Word 41: 俯仰速度 (0.001 °/s)
    t1 = struct.unpack("=h", data[start + 56:start + 58])[0]
    el_vel = t1 * 1e-3

    # 计算高度
    height = pr * math.sin(math.radians(pe)) + pr * pr / 17000000

    # Word 42: 全速度 (0.1m/s)
    t1 = struct.unpack("=I", data[start + 58:start + 62])[0]
    vel = t1 * 1e-1

    # Word 53: 目标信噪比 (0.01 dB), Word 54: RCS (0.01 dB)
    snr, rcs = struct.unpack("=Hh", data[start + 80:start + 84])
    snr = snr * 0.01
    rcs = rcs * 0.01

    # Word 44: 空间加速度 (0.01 m/s²)
    t1 = struct.unpack("=H", data[start + 62:start + 64])[0]
    acc = t1 * 1e-2

    # Word 45: 航向 (0.1°)
    t1 = struct.unpack("=H", data[start + 64:start + 66])[0]
    course_angle = t1 * 1e-1

    # Word 55: 识别信息 (大类 b0-b7, 小类 b8-b15)
    t1 = struct.unpack("=H", data[start + 84:start + 86])[0]
    tar_big = t1 % 256
    tar_small = t1 // 256

    # Word 89: 多普勒展宽, Word 91: Jem特征
    feat1 = struct.unpack("=f", data[start + 152:start + 156])[0]
    feat2 = struct.unpack("=f", data[start + 156:start + 160])[0]
    doppler = feat1
    jem = feat2

    # Word 57/59/61: 站址信息（经度/纬度 1e-5°，高度 0.01m）
    site_lon = struct.unpack("=i", data[start + 88:start + 92])[0] * 1e-5
    site_lat = struct.unpack("=i", data[start + 92:start + 96])[0] * 1e-5
    site_alt = struct.unpack("=i", data[start + 96:start + 100])[0] * 1e-2

    # Word 73/75/77: 地心坐标（32-bit signed，0.01m）
    x_ecef = struct.unpack("=i", data[start + 120:start + 124])[0] * 1e-2
    y_ecef = struct.unpack("=i", data[start + 124:start + 128])[0] * 1e-2
    z_ecef = struct.unpack("=i", data[start + 128:start + 132])[0] * 1e-2
    tar_lon, tar_lat, tar_alt = _ecef_to_lla(x_ecef, y_ecef, z_ecef)

    return {
        "status": status,
        "trk_stat": status,
        "trk_cn": trk_cn,
        "track_id": tar_id,
        "tar_seq": tar_seq,
        "timestamp": tim,
        "rcs_db": rcs,
        "snr_db": snr,
        "r_m": r,
        "a_deg": a,
        "e_deg": e,
        "pr_m": pr,
        "pa_deg": pa,
        "pe_deg": pe,
        "height_m": height,
        "vel_m_s": vel,
        "radial_vel_m_s": radial_vel,
        "az_vel_deg_s": az_vel,
        "el_vel_deg_s": el_vel,
        "acc_m_s2": acc,
        "course_deg": course_angle,
        "feat1": feat1,
        "feat5": feat2,
        "doppler": doppler,
        "jem": jem,
        "tar_big": tar_big,
        "tar_small": tar_small,
        "目标状态": status,
        "航迹状态": status,
        "航迹历史": trk_cn,
        "目标批号": tar_id,
        "目标流水号": tar_seq,
        "时间": tim,
        "RCS": rcs,
        "目标信噪比": snr,
        "滤波径向距离": r,
        "滤波方位": a,
        "滤波俯仰": e,
        "点迹距离": pr,
        "点迹方位": pa,
        "点迹俯仰": pe,
        "高度": height,
        "全速度": vel,
        "径向速度": radial_vel,
        "方位速度": az_vel,
        "俯仰速度": el_vel,
        "加速度": acc,
        "航向": course_angle,
        "Feature1": feat1,
        "Feature5": feat2,
        "多普勒展宽": doppler,
        "JEM": jem,
        "目标大类": tar_big,
        "目标小类": tar_small,
        "经（目标-滤波后）": tar_lon,
        "纬（目标-滤波后）": tar_lat,
        "高（目标-滤波后）": tar_alt,
        "经（站址）": site_lon,
        "纬（站址）": site_lat,
        "高（站址）": site_alt,
    }


def verify_checksum(data):
    if len(data) < 4:
        return False
    checksum = struct.unpack("=H", data[-4:-2])[0]
    body = data[:-4]
    calc = sum(body) & 0xFFFF
    return calc == checksum


def parse_packet(data, skip_checksum=False):
    if len(data) < 30:
        return None
    head = struct.unpack("=H", data[0:2])[0]
    tail = struct.unpack("=H", data[-2:])[0]
    if head != TRK_FLAG_HEAD or tail != TRK_FLAG_TAIL:
        return None
    if not skip_checksum and not verify_checksum(data):
        return None

    num = struct.unpack("=H", data[24:26])[0]
    targets = []
    for i in range(num):
        start = TAR_STR + i * TAR_LEN
        end = start + TAR_LEN
        if end > len(data):
            break
        tar = parse_target(data, start)
        targets.append(tar)
    return targets
