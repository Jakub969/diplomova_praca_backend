from fastapi import FastAPI, UploadFile, File
from uuid import uuid4
from settings import JOBS_DIR
from tasks import process_video

app = FastAPI()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    job_id = uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    video_path = job_dir / file.filename

    with open(video_path, "wb") as f:
        f.write(await file.read())

    task = process_video.delay(job_id, str(video_path))

    return {
        "job_id": job_id,
        "task_id": task.id
    }


@app.get("/status/{task_id}")
def check_status(task_id: str):
    from worker import celery
    task = celery.AsyncResult(task_id)
    return {
        "state": task.state,
        "result": task.result
    }
##uvicorn main:app --host 0.0.0.0 --port 8000
