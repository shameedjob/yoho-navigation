"""Production entry point: gunicorn -b 127.0.0.1:8000 --timeout 120 web.wsgi:app

Loads a local .env the way `python -m web` does (real environment variables win),
then builds the app. Run it from the repo root so .env and data/ resolve.
"""

import logging

from . import create_app
from .__main__ import _load_dotenv

_load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
