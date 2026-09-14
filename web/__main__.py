"""Dev server: python -m web [--port 5000]. Loads a local .env if present."""

import argparse
import os
from pathlib import Path

from . import create_app


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    _load_dotenv()
    # localhost, not 127.0.0.1: must match the redirect URI registered with Google
    # or the session cookie set before the redirect won't come back.
    create_app().run(host="localhost", port=args.port, debug=os.environ.get("YOHO_DEV") == "1")
