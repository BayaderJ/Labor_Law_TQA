from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    case_class: Optional[str] = None
    articles: List[str]
    context_chunks: List[Dict[str, Any]]
