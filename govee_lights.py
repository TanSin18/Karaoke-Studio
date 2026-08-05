"""
Govee LAN Control API client — no cloud, no API key, just UDP on the local
network. This is the only way lights can react to your voice in anything
close to real time: Govee's cloud/Developer API is rate-limited to roughly
one command per device per ~10 seconds, which would look like nothing while
you sing. LAN Control has to be turned on per-device in the Govee Home app
(Settings -> LAN Control) before a device will respond to any of this.

Protocol (same for every LAN-Control-capable Govee device):
  - Discovery: UDP multicast to 239.255.255.250:4001, a JSON "scan" request;
    each device answers on port 4002 with its IP, MAC-based device id, and
    model (sku).
  - Control: UDP unicast to <device ip>:4003 with a small JSON command
    ("turn" / "brightness" / "colorwc"). No acknowledgement is sent back for
    control commands — this is fire-and-forget, which is exactly the low-
    latency behavior we want for a live reactive effect, but it does mean we
    can't tell a command was dropped.
"""

import json
import socket
import time

MCAST_GRP = "239.255.255.250"
MCAST_PORT = 4001
LISTEN_PORT = 4002
CONTROL_PORT = 4003

# The WiFi MCU inside these lights can lag or drop commands if flooded much
# faster than this — this is a floor on the interval between two commands
# to the SAME device, not a global cap (each device gets its own budget).
MIN_INTERVAL_SEC = 0.09  # ~11/sec max per device

_last_sent = {}  # device ip -> monotonic timestamp of last command sent


def _local_ip_for_multicast():
    """Best-effort local interface IP to bind multicast sends to — same
    trick used elsewhere in this app (_lan_ip in studio.py): open a UDP
    socket "toward" a public IP purely to see which interface the OS would
    route through, no packet actually leaves."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def discover(timeout=3.0):
    """Scan the LAN for Govee devices with LAN Control enabled. Returns a
    list of {ip, device, sku} dicts, deduplicated by device id (a device can
    answer more than once during the scan window)."""
    local_ip = _local_ip_for_multicast()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if local_ip:
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                             socket.inet_aton(local_ip))
        except OSError:
            pass
    try:
        sock.bind(("", LISTEN_PORT))
    except OSError:
        # something else on this machine is already listening on 4002 (e.g.
        # the Govee Home app itself, if it's running here) — scan still
        # works over a plain unbound socket, we just might miss replies that
        # arrive while another process's bind steals them.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.settimeout(0.5)

    msg = json.dumps({"msg": {"cmd": "scan", "data": {"account_topic": "reserve"}}}).encode()
    found = {}
    start = time.monotonic()
    next_probe = 0.0
    while time.monotonic() - start < timeout:
        if time.monotonic() >= next_probe:
            try:
                sock.sendto(msg, (MCAST_GRP, MCAST_PORT))
            except OSError:
                pass
            next_probe = time.monotonic() + 1.0
        try:
            data, _addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            d = json.loads(data)["msg"]["data"]
            found[d["device"]] = {"ip": d["ip"], "device": d["device"], "sku": d.get("sku", "")}
        except (KeyError, ValueError, TypeError):
            continue
    sock.close()
    return list(found.values())


def _raw_send(ip, payload):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.sendto(json.dumps(payload).encode(), (ip, CONTROL_PORT))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _send(ip, payload):
    """Rate-limited send — for the high-frequency reactive color/brightness
    stream ONLY. A dropped color update just means the lights are a beat
    behind, harmless; silently dropping a deliberate one-off action (turn
    on/off) would be a real bug, so that goes through _raw_send instead and
    always fires regardless of how recently a reactive update landed."""
    lk = _last_sent.get(ip, 0)
    now = time.monotonic()
    if now - lk < MIN_INTERVAL_SEC:
        return False  # dropped — too soon after the last command to this device
    _last_sent[ip] = now
    return _raw_send(ip, payload)


def turn(ip, on):
    return _raw_send(ip, {"msg": {"cmd": "turn", "data": {"value": 1 if on else 0}}})


def set_brightness(ip, pct):
    pct = max(1, min(100, int(pct)))
    return _send(ip, {"msg": {"cmd": "brightness", "data": {"value": pct}}})


def set_color(ip, r, g, b):
    r, g, b = (max(0, min(255, int(v))) for v in (r, g, b))
    return _send(ip, {"msg": {"cmd": "colorwc",
                 "data": {"color": {"r": r, "g": g, "b": b}, "colorTemInKelvin": 0}}})
