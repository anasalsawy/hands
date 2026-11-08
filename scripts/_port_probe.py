"""Tiny port liveness probe: python scripts/_port_probe.py 8802 8800"""
import socket
import sys

for a in sys.argv[1:]:
    p = int(a)
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", p))
        print(f"{p}: LISTENING")
    except OSError:
        print(f"{p}: free")
    finally:
        s.close()