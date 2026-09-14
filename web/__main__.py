"""Dev server: python -m web [--port 5000]. Loads a local .env if present."""

import argparse
import logging
import os
import re
from pathlib import Path

from dataclasses import replace

from . import create_app
from .config import Settings


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), _dotenv_value(value))


def _dotenv_value(raw: str) -> str:
    """A .env value: quoted values are taken as-is; unquoted ones end at an
    inline comment (whitespace then #), as .env.example writes them."""
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return re.split(r"\s+#", value, maxsplit=1)[0].strip() if not value.startswith("#") else ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    _load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    debug = os.environ.get("YOHO_DEV") == "1"
    settings = Settings.from_env()
    if debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        # The debug reloader's parent only watches files; the child it spawns (with
        # WERKZEUG_RUN_MAIN=true) serves requests, so only the child warms up.
        settings = replace(settings, warmup=False)
    # localhost, not 127.0.0.1: must match the redirect URI registered with Google
    # or the session cookie set before the redirect won't come back.
    create_app(settings).run(host="localhost", port=args.port, debug=debug)
