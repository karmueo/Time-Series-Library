import socket


class MulticastReceiver:
    def __init__(self, group, port, iface="0.0.0.0", bind_ip="", timeout_s=2.0):
        self.group = group
        self.port = port
        self.iface = iface
        self.bind_ip = bind_ip
        self.timeout_s = timeout_s
        self.sock = None

    def open(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.bind_ip:
            sock.bind((self.bind_ip, self.port))
        else:
            sock.bind(("", self.port))
        mreq = socket.inet_aton(self.group) + socket.inet_aton(self.iface)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(self.timeout_s)
        self.sock = sock
        return self

    def recv(self, bufsize=1024 * 64):
        if self.sock is None:
            raise RuntimeError("socket 未初始化，请先调用 open()")
        return self.sock.recvfrom(bufsize)

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
