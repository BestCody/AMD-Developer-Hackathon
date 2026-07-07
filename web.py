"""web.py -- root launcher for the Phase N web UI.

Usage:
    python web.py                              # http://0.0.0.0:5050 (LAN-visible; default)
    HOST=127.0.0.1 python web.py               # loopback-only (private dev)
    PORT=8080 python web.py                    # custom port (5050 default avoids macOS AirPlay :5000 clash)
    UPLOAD_DIR=/tmp/foo python web.py          # custom upload dir
    OUTPUT_DIR=/tmp/bar python web.py          # custom output dir
    LOG_LEVEL=DEBUG python web.py              # verbose logging

LAN discoverability:
    The launcher enumerates the host's non-loopback IPv4 addresses and
    prints the URLs at startup so you can browse from your phone or a
    sibling laptop.  No mDNS, no port forwarding -- plain ``0.0.0.0``
    bind.  (MVP: no auth.  Anyone on the LAN can upload PDFs.)

Heavy-dep startup (BGE / spaCy / pdfplumber) happens lazily on the first
job, not at server start -- so the server is up in <2s even on a cold box.
"""
from __future__ import annotations

import logging
import os
import socket
import sys
from pathlib import Path

# Ensure ``src/`` is on sys.path when launched from a clone.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def list_lan_urls(port: int) -> list[str]:
    """Discover non-loopback IPv4 URLs this host responds on for ``port``.

    Uses two complementary techniques so it works on hosts where
    ``gethostname()`` only resolves to loopback (common on Docker /
    headless macOS without an explicit hostname):

    1. ``socket.getaddrinfo(hostname)`` -- the IPs the local hostname
       resolves to via the system resolver.
    2. The classic UDP-connect trick: open a UDP ``socket`` against a
       public IP (no data is actually sent) and read ``getsockname()``,
       which returns the local interface IP the kernel would route
       through.

    Loopback / 0.0.0.0 are filtered.  Returns a sorted, deduplicated list.
    """
    addrs: set[str] = set()
    # Technique 1: getaddrinfo on the hostname.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127.") and addr != "0.0.0.0":
                addrs.add(addr)
    except socket.gaierror:
        pass
    # Technique 2: UDP-connect to 8.8.8.8 (no data sent); ``getsockname``
    # returns the local interface IP.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            addr = s.getsockname()[0]
            if not addr.startswith("127.") and addr != "0.0.0.0":
                addrs.add(addr)
    except OSError:
        pass
    return [f"http://{a}:{port}" for a in sorted(addrs)]


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from uir_pipeline.logging_config import configure as configure_pipeline_logging
    configure_pipeline_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

    upload_dir = Path(os.environ.get("UPLOAD_DIR", "/tmp/uir_web_uploads"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/tmp/uir_web_outputs"))
    # Default port 5050 instead of 5000: macOS 12+ reserves :5000 for the
    # AirPlay Receiver (ControlCe), which silently intercepts our bind.
    port = int(os.environ.get("PORT", "5050"))
    host = os.environ.get("HOST", "0.0.0.0")  # LAN-visible by default

    from uir_pipeline.web import create_app
    app = create_app(upload_dir=upload_dir, output_dir=output_dir)
    log = logging.getLogger("web_cli")
    log.info("UIR web UI starting on %s:%d", host, port)
    log.info("uploads  -> %s", upload_dir)
    log.info("outputs  -> %s", output_dir)
    if host == "0.0.0.0":
        urls = list_lan_urls(port)
        if urls:
            log.info("reachable from your LAN at: %s", ", ".join(urls))
            log.info("on this machine: http://127.0.0.1:%d", port)
        else:
            log.info("on this machine: http://127.0.0.1:%d (no LAN IP discovered)", port)
    else:
        log.info("on this machine: http://%s:%d", host, port)
    # use_reloader=False to avoid spawning the heavy-dep imports twice.
    app.run(host=host, port=port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
