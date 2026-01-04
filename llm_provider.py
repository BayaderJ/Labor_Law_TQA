from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch

class LLMProvider:
    """LLM Provider - handles generation and embeddings"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.generation_model = None
        self.embedding_model = None
    
    def load_generation_model(self, model_id: str):
        """Load text generation model"""
        print(f"Loading generation model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.generation_model.to(self.device)
        print("✅ Generation model loaded!")
    
    def load_embedding_model(self, model_id: str):
        """Load embedding model"""
        print(f"Loading embedding model: {model_id}")
        self.embedding_model = SentenceTransformer(model_id)
        print("✅ Embedding model loaded!")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text response"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        outputs = self.generation_model.generate(
            **inputs,
            max_length=max_tokens,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings"""
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()