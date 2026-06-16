"""Railway 배포용 WSGI 진입점 — gunicorn app:app 으로 실행."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.server import create_app  # noqa: E402

app = create_app()
