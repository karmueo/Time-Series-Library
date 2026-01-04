import math
import struct


TRK_FLAG_HEAD = 0x1010
TRK_FLAG_TAIL = 0x55AA

TAR_STR = 26
TAR_LEN = 160


def parse_target(data, start):
    status_raw = struct.unpack("=B", data[start:start + 1])[0]
    status = status_raw % 16
    tar_id = struct.unpack("=H", data[start + 2:start + 4])[0]
    tar_seq = struct.unpack("=I", data[start + 6:start + 10])[0]
    trk_cn = struct.unpack("=H", data[start + 10:start + 12])[0]

    t1 = struct.unpack("=I", data[start + 22:start + 26])[0]
    tim = t1 * 25e-6

    r, a, e = struct.unpack("=2I1i", data[start + 26:start + 38])
    r = r * 1e-1
    a = a * 1e-5
    e = e * 1e-5

    pr, pa, pe = struct.unpack("=2I1i", data[start + 38:start + 50])
    pr = pr * 1e-1
    pa = pa * 1e-5
    pe = pe * 1e-5

    t1 = struct.unpack("=h", data[start + 50:start + 52])[0]
    radial_vel = t1 * 1e-2
    t1 = struct.unpack("=h", data[start + 54:start + 56])[0]
    az_vel = t1 * 1e-3
    t1 = struct.unpack("=h", data[start + 56:start + 58])[0]
    el_vel = t1 * 1e-3

    height = pr * math.sin(math.radians(pe)) + pr * pr / 17000000

    t1 = struct.unpack("=I", data[start + 58:start + 62])[0]
    vel = t1 * 1e-1

    snr, rcs = struct.unpack("=Hh", data[start + 80:start + 84])
    snr = snr * 0.01
    rcs = rcs * 0.01

    t1 = struct.unpack("=H", data[start + 62:start + 64])[0]
    acc = t1 * 1e-2

    t1 = struct.unpack("=H", data[start + 64:start + 66])[0]
    course_angle = t1 * 1e-1

    t1 = struct.unpack("=H", data[start + 84:start + 86])[0]
    tar_big = t1 % 256
    tar_small = t1 // 256

    feat1 = struct.unpack("=f", data[start + 152:start + 156])[0]
    feat2 = struct.unpack("=f", data[start + 156:start + 160])[0]
    doppler = feat1
    jem = feat2

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
