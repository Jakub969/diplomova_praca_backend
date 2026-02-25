import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
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


@app.get("/result/{job_id}/{quality}")
def get_result(job_id: str, quality: str):

    if quality == "sparse":
        path = JOBS_DIR / job_id / "sparse" / "sparse_prediction.glb"
    else:
        path = JOBS_DIR / job_id / "dense" / "dense_prediction.glb"

    return FileResponse(path, media_type="model/gltf-binary")

@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
    return {"deleted": True}

@app.get("/status/{task_id}")
def check_status(task_id: str):
    from worker import celery
    task = celery.AsyncResult(task_id)
    return {
        "state": task.state,
        "result": task.result
    }

from celery.result import AsyncResult

@app.get("/progress/{task_id}")
def get_progress(task_id: str):

    task = AsyncResult(task_id)

    if task.state == "PROGRESS":
        return {
            "state": "PROGRESS",
            "progress": task.info["progress"]
        }

    if task.state == "SUCCESS":
        return {
            "state": "SUCCESS",
            "progress": 100
        }

    return {"state": task.state}
##uvicorn main:app --host 0.0.0.0 --port 8000
