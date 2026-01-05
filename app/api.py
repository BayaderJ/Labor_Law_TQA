from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from app.schemas import QuestionRequest, AnswerResponse
from app.services import RAGService

app = FastAPI(title="Saudi Labor Law Chatbot API")



BASE_DIR = Path(__file__).resolve().parent.parent

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

svc = RAGService()

@app.on_event("startup")
async def startup_event():
    svc.startup()

@app.on_event("shutdown")
async def shutdown_event():
    svc.shutdown()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    try:
        out = svc.ask(req.question, top_k=3)
        return AnswerResponse(**out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
