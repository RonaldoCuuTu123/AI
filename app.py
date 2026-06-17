from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from src.rag_pipeline import RAGPipeline

app = FastAPI(title="Football Hybrid RAG API")

# Setup static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Khởi tạo RAG Pipeline
rag_system = RAGPipeline()

class ChatRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not rag_system.is_ready:
        return {"error": "Hệ thống chưa được huấn luyện. Vui lòng chạy train.py trước."}
        
    result = rag_system.generate_answer(request.query)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
