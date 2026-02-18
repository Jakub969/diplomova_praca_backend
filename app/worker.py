from celery import Celery
from settings import REDIS_URL

celery = Celery(
    "colmap_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery.conf.task_routes = {
    "tasks.process_video": {"queue": "gpu_queue"}
}
##celery -A worker.celery worker --loglevel=info -Q gpu_queue --concurrency=1
