import socket
import struct
import time


class MulticastPublisher:
    def __init__(self, group, port, iface="0.0.0.0", ttl=1):
        self.group = group
        self.port = port
        self.iface = iface
        self.ttl = ttl
        self.sock = None
        self.seq = 0

    def open(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.ttl)
        if self.iface:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(self.iface))
        self.sock = sock
        return self

    @staticmethod
    def _to_bcd(value):
        value = int(value) % 100
        return ((value // 10) << 4) | (value % 10)

    def _checksum(self, payload):
        return sum(payload) & 0xFFFF

    def _build_packet(self, item):
        self.seq = (self.seq + 1) & 0xFFFF
        now = time.localtime()
        frac = time.time() % 1.0
        sub_second = int(frac / 25e-6) & 0xFFFF

        year = self._to_bcd(now.tm_year % 100)
        month = self._to_bcd(now.tm_mon)
        day = self._to_bcd(now.tm_mday)
        hour = self._to_bcd(now.tm_hour)
        minute = self._to_bcd(now.tm_min)
        second = self._to_bcd(now.tm_sec)

        frame_header = 0xA999
        msg_type = 0
        frame_length = 34
        frame_seq = self.seq
        system_id = 0
        radar_id = 0
        reserve = 0

        header = struct.pack(
            "=6H",
            frame_header,
            msg_type,
            frame_length,
            frame_seq,
            system_id,
            ((reserve & 0xFF) << 8) | (radar_id & 0xFF),
        )
        header += struct.pack("=H", (month << 8) | year)
        header += struct.pack("=H", (hour << 8) | day)
        header += struct.pack("=H", (second << 8) | minute)
        header += struct.pack("=H", sub_second)
        header += struct.pack("=HH", 0, 0)

        pred = int(item.get("pred", -1))
        if pred == 0:
            class_major = 6
            prob = float(item.get("prob_bird", 0.0))
        elif pred == 1:
            class_major = 7
            prob = float(item.get("prob_uav", 0.0))
        else:
            class_major = 0xF
            prob = 0.0
        confidence = max(0, min(0xFFFF, int(round(prob * 1000))))

        batch_id = int(item.get("batch_id", item.get("track_id", 0))) & 0xFFFF
        base_day = int(item.get("base_day", now.tm_year * 10000 + now.tm_mon * 100 + now.tm_mday)) & 0xFFFFFFFF
        time_25us = int(item.get("time_25us", 0)) & 0xFFFFFFFF

        body = struct.pack("=H", 1)
        body += struct.pack("=H", batch_id)
        body += struct.pack("=I", base_day)
        body += struct.pack("=I", time_25us)
        body += struct.pack("=H", class_major)
        body += struct.pack("=H", confidence)
        body += struct.pack("=12H", *([0] * 12))

        checksum = self._checksum(header + body)
        tail = struct.pack("=HH", checksum, 0x55AA)
        return header + body + tail

    def send(self, items):
        if self.sock is None:
            raise RuntimeError("socket 未初始化，请先调用 open()")
        if not items:
            return
        for item in items:
            packet = self._build_packet(item)
            self.sock.sendto(packet, (self.group, self.port))
