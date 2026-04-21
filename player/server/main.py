import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

DATA_DIR = Path(__file__).parent.parent.parent / "data"

app = FastAPI(title="MovieRay Player")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _find_video_file(video_name: str) -> Optional[Path]:
    for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
        path = DATA_DIR / f"{video_name}{ext}"
        if path.exists():
            return path
    path = DATA_DIR / video_name
    if path.exists():
        return path
    return None


@app.get("/api/videos")
def list_videos():
    videos = []
    if not DATA_DIR.exists():
        return {"videos": []}

    for json_file in DATA_DIR.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        video_name = json_file.stem
        video_file = _find_video_file(data.get("video", video_name))
        videos.append({
            "id": video_name,
            "name": data.get("video", video_name),
            "duration": data.get("duration", 0),
            "segment_count": len(data.get("segments", [])),
            "has_video_file": video_file is not None,
        })

    return {"videos": videos}


@app.get("/api/videos/{video_id}/metadata")
def get_metadata(video_id: str):
    json_path = DATA_DIR / f"{video_id}.json"
    if not json_path.exists():
        raise HTTPException(404, f"No metadata found for '{video_id}'")

    with open(json_path) as f:
        return json.load(f)


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str):
    json_path = DATA_DIR / f"{video_id}.json"
    if not json_path.exists():
        raise HTTPException(404, f"No metadata found for '{video_id}'")

    with open(json_path) as f:
        data = json.load(f)

    video_file = _find_video_file(data.get("video", video_id))
    if not video_file:
        raise HTTPException(404, f"Video file not found for '{video_id}'")

    return FileResponse(str(video_file), media_type="video/mp4", filename=video_file.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
