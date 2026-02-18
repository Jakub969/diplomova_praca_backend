from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

FFMPEG_PATH = "/usr/bin/ffmpeg"

REDIS_URL = "redis://localhost:6379/0"
