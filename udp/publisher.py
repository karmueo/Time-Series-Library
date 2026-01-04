import socket
import struct


PRED_FLAG_HEAD = 0xBEEF
PRED_FLAG_TAIL = 0x55AA


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

    def _checksum(self, payload):
        return sum(payload) & 0xFFFF

    def build_packet(self, items):
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        count = len(items)
        header = struct.pack("=HBBIH", PRED_FLAG_HEAD, 1, 0, self.seq, count)

        body = bytearray()
        for item in items:
            track_id = int(item["track_id"]) & 0xFFFF
            ts_ms = int(item.get("timestamp_ms", 0)) & 0xFFFFFFFF
            pred = int(item["pred"]) & 0xFF
            prob_uav = float(item.get("prob_uav", 0.0))
            prob_bird = float(item.get("prob_bird", 0.0))
            body.extend(struct.pack("=HIBff", track_id, ts_ms, pred, prob_uav, prob_bird))

        checksum = self._checksum(header + body)
        tail = struct.pack("=HH", checksum, PRED_FLAG_TAIL)
        return header + body + tail

    def send(self, items):
        if self.sock is None:
            raise RuntimeError("socket 未初始化，请先调用 open()")
        if not items:
            return
        packet = self.build_packet(items)
        self.sock.sendto(packet, (self.group, self.port))
