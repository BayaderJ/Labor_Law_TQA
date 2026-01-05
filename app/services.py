import os
import re
import hashlib
from typing import List, Dict, Any, Optional
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from vectordb import VectorDB
from document_processor import DocumentProcessor
from sentence_transformers import SentenceTransformer
from groq import Groq
import pickle
import tensorflow as tf

import json
import h5py
import numpy as np



ART_DIR = "artifacts"
GRU_MODEL_PATH = os.path.join(ART_DIR, "gru_model.h5")
TOKENIZER_PATH = os.path.join(ART_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(ART_DIR, "label_encoder.pkl")

MAX_LEN = int(os.getenv("MAX_LEN", "40")) 


def get_pdf_hash(pdf_path: str) -> str:
    """Calculate MD5 hash of PDF to detect changes"""
    md5 = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


class RAGService:
    def _load_gru_assets(self):
        try:
            import keras  # ✅ Keras 3 (مو tf.keras)

            GRU_MODEL_PATH = "artifacts/gru_model.keras"
            TOKENIZER_PATH = "artifacts/tokenizer.pkl"
            LABEL_ENCODER_PATH = "artifacts/label_encoder.pkl"

        # ✅ تحميل مودل .keras مباشرة
            self.gru_model = keras.models.load_model(GRU_MODEL_PATH, compile=False)
            print("✅ GRU model loaded (keras .keras)")

            import pickle
            with open(TOKENIZER_PATH, "rb") as f:
                self.tokenizer = pickle.load(f)
            with open(LABEL_ENCODER_PATH, "rb") as f:
                self.label_encoder = pickle.load(f)

            self.gru_maxlen = int(self.gru_model.input_shape[1])
            print(f"✅ GRU maxlen detected: {self.gru_maxlen}")

        except Exception as e:
            print(f"⚠️ Failed to load GRU assets: {e} (GRU disabled)")
            self.gru_model = None
            self.tokenizer = None
            self.label_encoder = None
            self.gru_maxlen = None


    def predict_case_class(self, text: str) -> Optional[str]:
        if self.gru_model is None:
           return None

        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.gru_maxlen, padding="post", truncating="post")

        probs = self.gru_model.predict(padded, verbose=0)
        idx = int(np.argmax(probs, axis=1)[0])
        return self.label_encoder.inverse_transform([idx])[0]

    """
    Encapsulates:
    - Qdrant VectorDB connect + cached embeddings logic
    - OCR document processing
    - Query embedding + search
    - Groq generation
    """

    def __init__(self, pdf_path: str = "data/saudi_labor_law.pdf"):
        self.pdf_path = pdf_path
        self.vectordb: Optional[VectorDB] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.groq_client: Optional[Groq] = None
        self.gru_model = None
        self.tokenizer = None
        self.label_encoder = None


    def startup(self) -> None:
        # 1) Groq
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)

        # 2) Qdrant
        self.vectordb = VectorDB()
        self.vectordb.connect()
        self._load_gru_assets()


        # 3) PDF exists?
        if not os.path.exists(self.pdf_path):
            # نخلي السيرفر يقوم بس بدون فهرسة
            print(f"⚠️ PDF not found at {self.pdf_path} - RAG indexing skipped.")
            # برضو نحمل embedding model عشان لو تبين لاحقًا
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            return

        current_pdf_hash = get_pdf_hash(self.pdf_path)
        print(f"🔍 PDF Hash: {current_pdf_hash[:8]}...")

        # 4) Decide reuse vs rebuild
        if self._collection_up_to_date(current_pdf_hash):
            print("✅ Using cached embeddings in Qdrant (no rebuild).")
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            return
        self._load_gru_assets()

        print("🔄 Rebuild needed (first run or PDF changed).")
        self._rebuild_index(current_pdf_hash)

    def shutdown(self) -> None:
        if self.vectordb:
            self.vectordb.close()

    # ---------------- internal helpers ----------------

    def _collection_up_to_date(self, current_pdf_hash: str) -> bool:
        """
        Check:
        - collection exists
        - metadata stored in collection matches PDF hash
        - collection has points
        """
        if not self.vectordb:
            return False

        if not self.vectordb.collection_exists(config.COLLECTION_NAME):
            return False

        # IMPORTANT:
        # Your VectorDB wrapper stores metadata as a special point id=0 (payload _metadata)
        # so we read it from get_collection_metadata().
        try:
            meta = self.vectordb.get_collection_metadata(config.COLLECTION_NAME)
        except Exception:
            meta = None

        stored_hash = (meta or {}).get("pdf_hash")
        if stored_hash != current_pdf_hash:
            return False

        # Check count > 0
        try:
            count = self.vectordb.client.count(config.COLLECTION_NAME)
            return count.count > 0
        except Exception:
            return False

    def _rebuild_index(self, current_pdf_hash: str) -> None:
        if not self.vectordb:
            raise RuntimeError("VectorDB is not initialized")

        # delete old if exists
        try:
            self.vectordb.client.delete_collection(config.COLLECTION_NAME)
            print("🗑️ Deleted old collection")
        except Exception:
            print("ℹ️ No old collection to delete")

        # load embedding model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        # create collection with metadata (stored as special point id=0)
        self.vectordb.create_collection(
            config.COLLECTION_NAME,
            config.EMBEDDING_SIZE,
            metadata={"pdf_hash": current_pdf_hash},
        )

        # OCR + chunking
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        chunks = processor.process(self.pdf_path)
        if not chunks:
            raise RuntimeError("No chunks extracted from PDF")

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        payloads = [
            {
                "text": c["text"],
                "article_number": c.get("article_number", "غير محدد"),
                "chunk_id": str(c.get("chunk_id")),
            }
            for c in chunks
        ]

        self.vectordb.insert(config.COLLECTION_NAME, embeddings.tolist(), payloads)
        print(f"✅ Indexed {len(chunks)} chunks into Qdrant")

    # ---------------- public API ----------------

    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        if not self.embedding_model or not self.vectordb or not self.groq_client:
            raise RuntimeError("Service not ready (startup not completed)")

        # 1) embed question
        q_vec = self.embedding_model.encode([question])[0].tolist()

        # 2) search
        results = self.vectordb.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=q_vec,
            limit=top_k
        )

        # 3) build context
        context_parts = []
        articles_mentioned = set()

        for i, r in enumerate(results, 1):
            text = r.get("text", "")
            # match "المادة ..."
            for m in re.findall(r"المادة\s+[^\n:.]{1,50}", text[:300]):
                articles_mentioned.add(m.strip())

            context_parts.append(f"[مقتطف {i}]:\n{text}\n")

        context = "\n".join(context_parts)

        # 4) prompt (نفس أسلوب main.py تقريبًا)
        system_prompt = """
أنت مساعد قانوني متخصص في نظام العمل السعودي.
مهمتك: تقديم إجابات دقيقة ومفصلة للموظفين بناءً على نصوص نظام العمل فقط.

قواعد أساسية:
- أجب باللغة العربية الفصحى فقط
- استند فقط للسياق المقدم
- اذكر رقم المادة عند كل استشهاد
- أجب بما هو موجود فقط

هيكل الإجابة (اذكر فقط ما هو موجود):
1) القاعدة الأساسية (مع رقم المادة)
2) الاستثناءات (إن وُجدت)
3) الحالات الخاصة (إن وُجدت)
4) ربط المواد ذات العلاقة (إن وُجدت)
""".strip()

        user_prompt = f"""السياق من نظام العمل السعودي:
{context}

سؤال: {question}

قدم إجابة شاملة مع ذكر المواد المستخدمة."""

        # 5) Groq
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9,
        )
        answer = chat_completion.choices[0].message.content

        # 6) extract citations
        answer_articles = re.findall(r"المادة\s+[^\n:.،]{1,50}", answer)
        all_articles = list(set(articles_mentioned) | set(answer_articles))

        cleaned_articles = []
        for art in all_articles:
            art_clean = art.replace("المادة ", "").strip()
            if art_clean and art_clean != "غير محدد":
                cleaned_articles.append(f"المادة {art_clean}")

        # 7) response
        context_chunks = [
            {
                "text": (r.get("text", "")[:200] + "...") if r.get("text") else "",
                "article": r.get("article_number", "غير محدد"),
                "score": round(float(r.get("score", 0.0)), 3),
            }
            for r in results
        ]
        case_class = self.predict_case_class(question)

        return {
            "answer": answer,
            "case_class": case_class,
            "articles": cleaned_articles[:5],
            "context_chunks": context_chunks,
        }
