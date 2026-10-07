"""Messages between the client, the zygote and the sandboxes: length-prefixed JSON over a
Unix stream socket, with file descriptors passed alongside (SCM_RIGHTS) when a new
sandbox's socket is handed to the client."""

import array
import json
import os
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
        # room for the descriptors and for credentials, which a socket with SO_PASSCRED
        # receives on every message (without it the descriptors would be cut off)
        head, anc, _, _ = sock.recvmsg(_HDR.size, socket.CMSG_SPACE(want_fds * 4) + socket.CMSG_SPACE(_UCRED.size))
        if not head:
            raise ConnectionError("peer closed")
        for level, kind, data in anc:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds += list(array.array("i", data[:len(data) - len(data) % 4]))
        head += _recv_exact(sock, _HDR.size - len(head))
    else:
        head = _recv_exact(sock, _HDR.size)
    (n,) = _HDR.unpack(head)
    if n > MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes")
    return json.loads(_recv_exact(sock, n)), fds


# A sender's identity, translated by the kernel into the receiver's view: inside a PID
# namespace a process knows only its own inner pid, while the client needs the pid it can
# signal. The receiving socket must have SO_PASSCRED set before the message is sent.
_UCRED = struct.Struct("3i")


def enable_creds(sock):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)


def send_creds(sock, obj):
    body = json.dumps(obj).encode()
    cred = _UCRED.pack(os.getpid(), os.getuid(), os.getgid())
    sock.sendmsg([_HDR.pack(len(body))], [(socket.SOL_SOCKET, socket.SCM_CREDENTIALS, cred)])
    sock.sendall(body)


def recv_creds(sock):
    """Returns (message, sender pid as this process sees it)."""
    head, anc, _, _ = sock.recvmsg(_HDR.size, socket.CMSG_SPACE(_UCRED.size))
    if not head:
        raise ConnectionError("peer closed")
    pid = None
    for level, kind, data in anc:
        if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS:
            pid = _UCRED.unpack(data[:_UCRED.size])[0]
    head += _recv_exact(sock, _HDR.size - len(head))
    (n,) = _HDR.unpack(head)
    return json.loads(_recv_exact(sock, n)), pid
