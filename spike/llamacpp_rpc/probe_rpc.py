"""What can an anonymous stranger do to a llama.cpp `rpc-server`?

The README says "Never run the RPC server on an open network" and calls the implementation
"fragile and insecure". This measures what that actually means for a volunteer, because
"insecure" is doing a lot of unexamined work in a backend comparison.

The probe is a **raw TCP socket** speaking the wire protocol directly. It links no
llama.cpp code, presents no credential, and performs no handshake beyond the protocol's own
version exchange. Whatever it gets back is what any host on the network gets back.

Wire format, read from ggml/src/ggml-rpc/ggml-rpc.cpp:

    request   [cmd:1][input_size:8 LE][input_data]
    response  [output_size:8 LE][output_data]

Usage:
    python3 probe_rpc.py 127.0.0.1 50052
"""

from __future__ import annotations

import socket
import struct
import sys

# enum rpc_cmd, in declaration order (ggml-rpc.cpp).
RPC_CMD_ALLOC_BUFFER = 0
RPC_CMD_GET_ALIGNMENT = 1
RPC_CMD_GET_MAX_SIZE = 2
RPC_CMD_FREE_BUFFER = 4
RPC_CMD_GET_DEVICE_MEMORY = 11
RPC_CMD_HELLO = 14
RPC_CMD_DEVICE_COUNT = 15

RPC_CONN_CAPS_SIZE = 24  # transport.h


def recv_exact(sock: socket.socket, count: int) -> bytes:
    buffer = b""
    while len(buffer) < count:
        chunk = sock.recv(count - len(buffer))
        if not chunk:
            raise ConnectionError(f"connection closed after {len(buffer)}/{count} bytes")
        buffer += chunk
    return buffer


def call(sock: socket.socket, cmd: int, payload: bytes = b"") -> bytes:
    sock.sendall(bytes([cmd]) + struct.pack("<Q", len(payload)) + payload)
    (size,) = struct.unpack("<Q", recv_exact(sock, 8))
    return recv_exact(sock, size) if size else b""


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 50052

    print(f"connecting to {host}:{port} as an unauthenticated stranger")
    print("(raw socket, no llama.cpp libraries, no credentials)\n")

    sock = socket.create_connection((host, port), timeout=10)
    findings: list[tuple[str, str]] = []

    try:
        # 1. HELLO -- does the server talk to anyone who asks?
        response = call(sock, RPC_CMD_HELLO, b"\x00" * RPC_CONN_CAPS_SIZE)
        major, minor, patch, _pad = struct.unpack("<BBBB", response[:4])
        print(f"1. HELLO      -> protocol v{major}.{minor}.{patch}")
        findings.append(("server answers an anonymous client", f"protocol v{major}.{minor}.{patch}"))

        # 2. How many accelerators does this volunteer have?
        response = call(sock, RPC_CMD_DEVICE_COUNT)
        (device_count,) = struct.unpack("<I", response[:4])
        print(f"2. DEVICE_COUNT -> {device_count} device(s) enumerated")
        findings.append(("hardware inventory readable", f"{device_count} device(s)"))

        # 3. How much memory does it have, and how much is free right now?
        for device in range(device_count):
            response = call(sock, RPC_CMD_GET_DEVICE_MEMORY, struct.pack("<I", device))
            free_mem, total_mem = struct.unpack("<QQ", response[:16])
            print(f"3. GET_DEVICE_MEMORY[{device}] -> free {free_mem / 2**30:.2f} GiB "
                  f"/ total {total_mem / 2**30:.2f} GiB")
            findings.append((f"device {device} memory readable",
                             f"{free_mem / 2**30:.2f}/{total_mem / 2**30:.2f} GiB"))

        # 4. Allocation limits, then an actual allocation on someone else's machine.
        response = call(sock, RPC_CMD_GET_MAX_SIZE, struct.pack("<I", 0))
        (max_size,) = struct.unpack("<Q", response[:8])
        print(f"4. GET_MAX_SIZE -> {max_size / 2**30:.2f} GiB max single buffer")

        alloc_size = 256 * 2**20  # 256 MiB
        response = call(sock, RPC_CMD_ALLOC_BUFFER, struct.pack("<IQ", 0, alloc_size))
        if len(response) >= 16:
            remote_ptr, remote_size = struct.unpack("<QQ", response[:16])
            allocated = remote_ptr != 0
            print(f"5. ALLOC_BUFFER(256 MiB) -> {'granted' if allocated else 'refused'}"
                  + (f", remote ptr 0x{remote_ptr:x}, size {remote_size / 2**20:.0f} MiB"
                     if allocated else ""))
            findings.append(("can allocate memory on the host",
                             "yes, 256 MiB" if allocated else "refused"))
            if allocated:
                call(sock, RPC_CMD_FREE_BUFFER, struct.pack("<Q", remote_ptr))
                print("   (freed)")
        else:
            print(f"5. ALLOC_BUFFER -> unexpected response ({len(response)} bytes)")

    except Exception as exc:
        print(f"\nprobe failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        sock.close()

    print("\n" + "=" * 66)
    print("What an anonymous stranger obtained, with no credential of any kind:")
    for name, detail in findings:
        print(f"  - {name}: {detail}")
    print()
    print("Not exercised here, but in the same unauthenticated command set:")
    print("  RPC_CMD_SET_TENSOR / GET_TENSOR   write and read buffer memory")
    print("  RPC_CMD_GRAPH_COMPUTE             execute a client-supplied compute graph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
