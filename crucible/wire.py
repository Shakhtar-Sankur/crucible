"""Messages between the client, the zygote and the sandboxes: length-prefixed JSON over a
Unix stream socket, with file descriptors passed alongside (SCM_RIGHTS) when a new
sandbox's socket is handed to the client."""

import json
import socket
import struct

_HDR = struct.Struct("!I")
MAX_MESSAGE = 64 << 20


def send(sock, obj, fds=()):
    body = json.dumps(obj).encode()
    data = _HDR.pack(len(body)) + body
    if fds:
        socket.send_fds(sock, [data[:_HDR.size]], list(fds))
        sock.sendall(data[_HDR.size:])
    else:
        sock.sendall(data)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv(sock, want_fds=0):
    """Returns (message, fds)."""
    fds = []
    if want_fds:
        head, fds, _, _ = socket.recv_fds(sock, _HDR.size, want_fds)
        if not head:
            raise ConnectionError("peer closed")
        head += _recv_exact(sock, _HDR.size - len(head))
    else:
        head = _recv_exact(sock, _HDR.size)
    (n,) = _HDR.unpack(head)
    if n > MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes")
    return json.loads(_recv_exact(sock, n)), fds
