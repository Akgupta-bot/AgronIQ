# signaling_server.py
import os
import uuid
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

UPLOAD_DIR = "chat_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="AgronIQ Signaling & File Server")

# Allow local Streamlit origins (adjust host/port if different)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for prototype only; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory mapping of room -> set of WebSockets
rooms = {}  # {room_id: {client_id: websocket}}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), uploader: str = "user"):
    """
    Accept file upload and return download URL.
    """
    ext = os.path.splitext(file.filename)[1]
    uid = uuid.uuid4().hex
    safe_name = f"{uploader}_{uid}{ext}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as f:
        while True:
            chunk = await file.read(1024*1024)
            if not chunk:
                break
            f.write(chunk)
    download_url = f"/files/{safe_name}"
    return {"file_name": file.filename, "download_url": download_url, "saved_name": safe_name}

@app.get("/files/{fname}")
async def download_file(fname: str):
    path = os.path.join(UPLOAD_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/octet-stream", filename=fname)

# WebSocket signaling
@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    await websocket.accept()
    if room_id not in rooms:
        rooms[room_id] = {}
    rooms[room_id][client_id] = websocket
    print(f"[signaling] {client_id} joined {room_id}; participants: {list(rooms[room_id].keys())}")
    try:
        while True:
            msg = await websocket.receive_text()
            # Expect JSON string with fields: {"to": <client_id or "all">, "type": "...", "data": {...}}
            # For simplicity forward as-is to target(s)
            import json
            try:
                obj = json.loads(msg)
            except Exception:
                continue
            target = obj.get("to", "all")
            if target == "all":
                # broadcast to all except sender
                for cid, ws in list(rooms.get(room_id, {}).items()):
                    if cid == client_id:
                        continue
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        pass
            else:
                ws = rooms.get(room_id, {}).get(target)
                if ws:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        pass
    except WebSocketDisconnect:
        print(f"[signaling] {client_id} left {room_id}")
    finally:
        # cleanup
        if room_id in rooms and client_id in rooms[room_id]:
            del rooms[room_id][client_id]
        if room_id in rooms and not rooms[room_id]:
            del rooms[room_id]

if __name__ == "__main__":
    # Run with: python signaling_server.py
    uvicorn.run("signaling_server:app", host="0.0.0.0", port=8765, reload=False)
