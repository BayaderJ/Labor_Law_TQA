"""
Saudi Labor Law Chatbot - FastAPI Backend
OPTIMIZED VERSION - Persistent Embeddings (No Re-processing)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import re
import hashlib

import config
from vectordb import VectorDB
from document_processor import DocumentProcessor
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(title="Saudi Labor Law Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

vectordb = None
embedding_model = None
groq_client = None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    articles: List[str]
    context_chunks: List[Dict[str, Any]]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pdf_hash(pdf_path: str) -> str:
    """Calculate MD5 hash of PDF to detect changes"""
    md5 = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

def collection_exists_and_valid(vectordb: VectorDB, collection_name: str, pdf_hash: str) -> bool:
    """Check if collection exists and matches current PDF"""
    try:
        # Check if collection exists
        collection_info = vectordb.client.get_collection(collection_name)
        
        # Check if it has the PDF hash stored
        stored_hash = collection_info.config.params.get('metadata', {}).get('pdf_hash')
        
        if stored_hash == pdf_hash:
            # Check if collection has data
            count = vectordb.client.count(collection_name)
            return count.count > 0
        
        return False
    except:
        return False

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global vectordb, embedding_model, groq_client
    
    print("\n" + "="*70)
    print("🚀 STARTING SAUDI LABOR LAW CHATBOT (OPTIMIZED)")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Initialize Groq Client
    # ========================================================================
    print("\n📡 Initializing Groq API Client...")
    try:
        groq_client = Groq(api_key=config.GROQ_API_KEY)
        print(f"✅ Groq client initialized with model: {config.GROQ_MODEL}")
    except Exception as e:
        print(f"❌ Failed to initialize Groq: {e}")
        raise
    
    # ========================================================================
    # STEP 2: Connect to Vector Database
    # ========================================================================
    print("\n💾 Setting up Vector Database...")
    vectordb = VectorDB(config.QDRANT_HOST, config.QDRANT_PORT)
    vectordb.connect()
    
    # ========================================================================
    # STEP 3: Check if we need to rebuild
    # ========================================================================
    pdf_path = "data/saudi_labor_law.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  WARNING: PDF not found at {pdf_path}")
        return
    
    # Calculate current PDF hash
    current_pdf_hash = get_pdf_hash(pdf_path)
    print(f"\n🔍 PDF Hash: {current_pdf_hash[:8]}...")
    
    # Check if collection exists and is up-to-date
    if collection_exists_and_valid(vectordb, config.COLLECTION_NAME, current_pdf_hash):
        print("\n✅ FOUND EXISTING EMBEDDINGS - SKIPPING REBUILD!")
        print("   (Collection is up-to-date)")
        
        # Just load the embedding model for queries
        print("\n🔤 Loading Embedding Model...")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("✅ Embedding model loaded!")
        
        print("\n" + "="*70)
        print("✅ SYSTEM READY! (Using cached embeddings)")
        print("="*70)
        print(f"🤖 Using: {config.GROQ_MODEL}")
        print(f"🌐 API: http://localhost:8000")
        print("="*70 + "\n")
        return
    
    # ========================================================================
    # STEP 4: Rebuild needed - Delete old collection
    # ========================================================================
    print("\n🔄 REBUILD NEEDED (PDF changed or first run)")
    
    try:
        vectordb.client.delete_collection(config.COLLECTION_NAME)
        print("🗑️  Deleted old collection")
    except:
        print("ℹ️  No old collection to delete")
    
    # ========================================================================
    # STEP 5: Load Embedding Model
    # ========================================================================
    print("\n🔤 Loading Embedding Model...")
    print(f"   Model: {config.EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print("✅ Embedding model loaded!")
    
    # ========================================================================
    # STEP 6: Process PDF and Index
    # ========================================================================
    try:
        # Create collection with PDF hash in metadata
        print("\n📦 Creating collection with metadata...")
        vectordb.create_collection(
            config.COLLECTION_NAME, 
            config.EMBEDDING_SIZE,
            metadata={'pdf_hash': current_pdf_hash}
        )
        
        # Process PDF using DocumentProcessor
        print("\n📄 Processing PDF with DocumentProcessor...")
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        chunks = processor.process(pdf_path)
        
        if not chunks:
            print("❌ No chunks extracted")
            return
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Embed chunks
        print("\n🔄 Embedding chunks...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        
        # Prepare payloads
        payloads = [
            {
                'text': chunk['text'],
                'article_number': chunk.get('article_number', 'غير محدد'),
                'chunk_id': str(chunk['chunk_id'])
            }
            for chunk in chunks
        ]
        
        # Insert into Qdrant
        print("\n💾 Inserting into Vector Database...")
        vectordb.insert(
            config.COLLECTION_NAME,
            embeddings.tolist(),
            payloads
        )
        
        print("\n" + "="*70)
        print("✅ SYSTEM READY! (Embeddings saved)")
        print("="*70)
        print(f"📊 Indexed {len(chunks)} chunks")
        print(f"🤖 Using: {config.GROQ_MODEL}")
        print(f"🌐 API: http://localhost:8000")
        print(f"💾 Next startup will be INSTANT (using cached embeddings)")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def serve_ui():
    """Serve the HTML UI"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": config.GROQ_MODEL,
        "vector_db": "connected" if vectordb else "not connected"
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint - Process user question
    """
    
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    print(f"\n{'='*70}")
    print(f"❓ Question: {question}")
    print(f"{'='*70}")
    
    try:
        # ====================================================================
        # STEP 1: Embed question
        # ====================================================================
        print("🔍 Embedding question...")
        question_embedding = embedding_model.encode([question])[0].tolist()
        
        # ====================================================================
        # STEP 2: Search vector DB
        # ====================================================================
        print("🔎 Searching vector database...")
        results = vectordb.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=question_embedding,
            limit=3
        )
        
        print(f"✅ Found {len(results)} relevant chunks")
        
        # ====================================================================
        # STEP 3: Build context
        # ====================================================================
        context_parts = []
        articles_mentioned = set()
        
        for i, result in enumerate(results, 1):
            text = result['text']
            article = result.get('article', 'غير محدد')
            
            # Extract article mentions from text
            article_matches = re.findall(r'المادة\s+[^\n:.]{1,50}', text[:300])
            for match in article_matches:
                articles_mentioned.add(match.strip())
            
            context_parts.append(f"[مقتطف {i}]:\n{text}\n")
        
        context = "\n".join(context_parts)
        
        # ====================================================================
        # STEP 4: Build prompt
        # ====================================================================
        system_prompt  = """
أنت مساعد قانوني متخصص في نظام العمل السعودي.
مهمتك: تقديم إجابات دقيقة ومفصلة للموظفين بناءً على نصوص نظام العمل فقط.

 قواعد أساسية:
- أجب باللغة العربية الفصحى فقط (ممنوع أي لغة أخرى)
- استند فقط للسياق المقدم
- اذكر رقم المادة عند كل استشهاد
- أجب بما هو موجود فقط، لا تذكر ما هو غير موجود
- لا تستخدم عبارات: "لا توجد استثناءات" أو "لا توجد إشارات" أو "لا توجد مواد أخرى"

 هيكل الإجابة (اذكر فقط ما هو موجود):
1. القاعدة الأساسية (مع رقم المادة)
2. الاستثناءات (إن وُجدت)
3. الحالات الخاصة (إن وُجدت)
4. ربط المواد ذات العلاقة (إن وُجدت)

 أسلوب الإجابة:
- واضح ومباشر ومختصر
- لغة بسيطة يفهمها غير المتخصص
- توقف عند انتهاء المعلومات المفيدة
- لا حشو ولا تكرار

---
### مثال:
**سؤال:** كم مدة فترة التجربة؟

**إجابة:**
وفقًا لـ **المادة (53)** من نظام العمل السعودي، فترة التجربة لا تزيد على **90 يومًا**، 
ويجب النص عليها صراحة في العقد.

ويجوز بالاتفاق المكتوب **تمديدها** لـ **180 يومًا كحد أقصى**.

لا تدخل ضمن الحساب:
- إجازة عيدي الفطر والأضحى
- الإجازة المرضية

يحق للطرفين إنهاء العقد خلالها، ما لم ينص العقد على خلاف ذلك.

وبحسب **المادة (54)**، لا يجوز تكرار فترة التجربة لدى نفس صاحب العمل، 
إلا في مهنة مختلفة أو بعد مرور 6 أشهر من انتهاء العلاقة السابقة.

لا يستحق أي طرف تعويضًا أو مكافأة نهاية خدمة عند الإنهاء خلال التجربة.

---
 تذكر: أجب فقط بما هو موجود، واختم عند انتهاء المعلومات المفيدة.
"""

        user_prompt = f"""السياق من نظام العمل السعودي:
{context}

سؤال: {question}

قدم إجابة شاملة مع ذكر المواد المستخدمة."""

        # ====================================================================
        # STEP 5: Call Groq API
        # ====================================================================
        print("🤖 Generating answer...")
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9
        )
        
        answer = chat_completion.choices[0].message.content
        
        print(f"✅ Answer generated")
        
        # ====================================================================
        # STEP 6: Extract article citations from answer
        # ====================================================================
        answer_articles = re.findall(r'المادة\s+[^\n:.،]{1,50}', answer)
        all_articles = list(set(articles_mentioned) | set(answer_articles))
        
        # Clean up article list
        cleaned_articles = []
        for art in all_articles:
            # Extract just the article reference
            art_clean = art.replace('المادة ', '').strip()
            if art_clean and art_clean != 'غير محدد':
                cleaned_articles.append(f"المادة {art_clean}")
        
        print(f"📚 Articles cited: {cleaned_articles}")
        
        # ====================================================================
        # STEP 7: Return response
        # ====================================================================
        return AnswerResponse(
            answer=answer,
            articles=cleaned_articles[:5],  # Limit to 5
            context_chunks=[
                {
                    'text': r['text'][:200] + "...",
                    'article': r.get('article', 'غير محدد'),
                    'score': round(r['score'], 3)
                }
                for r in results
            ]
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FORCE REBUILD ENDPOINT (Optional - for manual rebuild)
# ============================================================================

@app.post("/rebuild")
async def force_rebuild():
    """Force rebuild of embeddings (useful for testing)"""
    try:
        vectordb.client.delete_collection(config.COLLECTION_NAME)
        await startup_event()
        return {"status": "success", "message": "Embeddings rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SHUTDOWN EVENT
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if vectordb:
        vectordb.close()
    print("👋 Server shutdown complete")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# """
# Saudi Labor Law Chatbot - FastAPI Backend
# SIMPLIFIED VERSION - NO COMPLEX ARTICLE EXTRACTION
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from typing import List, Dict, Any
# import os
# import re

# import config
# from vectordb import VectorDB
# from document_processor import DocumentProcessor
# from sentence_transformers import SentenceTransformer
# from groq import Groq

# # ============================================================================
# # FASTAPI APP SETUP
# # ============================================================================

# app = FastAPI(title="Saudi Labor Law Chatbot API")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================================================
# # GLOBAL VARIABLES
# # ============================================================================

# vectordb = None
# embedding_model = None
# groq_client = None

# # ============================================================================
# # PYDANTIC MODELS
# # ============================================================================

# class QuestionRequest(BaseModel):
#     question: str

# class AnswerResponse(BaseModel):
#     answer: str
#     articles: List[str]
#     context_chunks: List[Dict[str, Any]]

# # ============================================================================
# # STARTUP EVENT
# # ============================================================================

# @app.on_event("startup")
# async def startup_event():
#     global vectordb, embedding_model, groq_client
    
#     print("\n" + "="*70)
#     print("🚀 STARTING SAUDI LABOR LAW CHATBOT")
#     print("="*70)
    
#     # ========================================================================
#     # STEP 1: Initialize Groq Client
#     # ========================================================================
#     print("\n📡 Initializing Groq API Client...")
#     try:
#         groq_client = Groq(api_key=config.GROQ_API_KEY)
#         print(f"✅ Groq client initialized with model: {config.GROQ_MODEL}")
#     except Exception as e:
#         print(f"❌ Failed to initialize Groq: {e}")
#         raise
    
#     # ========================================================================
#     # STEP 2: Connect to Vector Database
#     # ========================================================================
#     print("\n💾 Setting up Vector Database...")
#     vectordb = VectorDB(config.QDRANT_HOST, config.QDRANT_PORT)
#     vectordb.connect()
    
#     # DELETE OLD COLLECTION (Fresh start)
#     try:
#         vectordb.client.delete_collection(config.COLLECTION_NAME)
#         print("🗑️  Deleted old collection")
#     except:
#         print("ℹ️  No old collection to delete")
    
#     # ========================================================================
#     # STEP 3: Load Embedding Model
#     # ========================================================================
#     print("\n🔤 Loading Embedding Model...")
#     print(f"   Model: {config.EMBEDDING_MODEL}")
#     embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
#     print("✅ Embedding model loaded!")
    
#     # ========================================================================
#     # STEP 4: Process PDF and Index
#     # ========================================================================
#     pdf_path = "data/saudi_labor_law.pdf"
    
#     if not os.path.exists(pdf_path):
#         print(f"\n⚠️  WARNING: PDF not found at {pdf_path}")
#         return
    
#     try:
#         # Create collection
#         vectordb.create_collection(config.COLLECTION_NAME, config.EMBEDDING_SIZE)
        
#         # Process PDF using YOUR DocumentProcessor class
#         print("\n📄 Processing PDF with DocumentProcessor...")
#         processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
#         chunks = processor.process(pdf_path)
        
#         if not chunks:
#             print("❌ No chunks extracted")
#             return
        
#         print(f"✅ Created {len(chunks)} chunks")
        
#         # Embed chunks
#         print("\n🔄 Embedding chunks...")
#         texts = [chunk['text'] for chunk in chunks]
#         embeddings = embedding_model.encode(texts, show_progress_bar=True)
        
#         # Prepare payloads
#         payloads = [
#             {
#                 'text': chunk['text'],
#                 'article_number': chunk.get('article_number', 'غير محدد'),
#                 'chunk_id': str(chunk['chunk_id'])
#             }
#             for chunk in chunks
#         ]
        
#         # Insert into Qdrant
#         print("\n💾 Inserting into Vector Database...")
#         vectordb.insert(
#             config.COLLECTION_NAME,
#             embeddings.tolist(),
#             payloads
#         )
        
#         print("\n" + "="*70)
#         print("✅ SYSTEM READY!")
#         print("="*70)
#         print(f"📊 Indexed {len(chunks)} chunks")
#         print(f"🤖 Using: {config.GROQ_MODEL}")
#         print(f"🌐 API: http://localhost:8000")
#         print("="*70 + "\n")
        
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         raise

# # ============================================================================
# # API ENDPOINTS
# # ============================================================================

# @app.get("/")
# async def serve_ui():
#     """Serve the HTML UI"""
#     return FileResponse("index.html")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "model": config.GROQ_MODEL,
#         "vector_db": "connected" if vectordb else "not connected"
#     }

# @app.post("/ask", response_model=AnswerResponse)
# async def ask_question(request: QuestionRequest):
#     """
#     Main endpoint - Process user question
#     """
    
#     question = request.question.strip()
    
#     if not question:
#         raise HTTPException(status_code=400, detail="Question cannot be empty")
    
#     print(f"\n{'='*70}")
#     print(f"❓ Question: {question}")
#     print(f"{'='*70}")
    
#     try:
#         # ====================================================================
#         # STEP 1: Embed question
#         # ====================================================================
#         print("🔍 Embedding question...")
#         question_embedding = embedding_model.encode([question])[0].tolist()
        
#         # ====================================================================
#         # STEP 2: Search vector DB
#         # ====================================================================
#         print("🔎 Searching vector database...")
#         results = vectordb.search(
#             collection_name=config.COLLECTION_NAME,
#             query_vector=question_embedding,
#             limit=3
#         )
        
#         print(f"✅ Found {len(results)} relevant chunks")
        
#         # ====================================================================
#         # STEP 3: Build context
#         # ====================================================================
#         context_parts = []
#         articles_mentioned = set()
        
#         for i, result in enumerate(results, 1):
#             text = result['text']
#             article = result.get('article', 'غير محدد')
            
#             # Extract article mentions from text
#             article_matches = re.findall(r'المادة\s+[^\n:.]{1,50}', text[:300])
#             for match in article_matches:
#                 articles_mentioned.add(match.strip())
            
#             context_parts.append(f"[مقتطف {i}]:\n{text}\n")
        
#         context = "\n".join(context_parts)
        
#         # ====================================================================
#         # STEP 4: Build prompt
#         # ====================================================================
#         system_prompt  = """
# أنت مساعد قانوني متخصص في نظام العمل السعودي.
# مهمتك: تقديم إجابات دقيقة ومفصلة للموظفين بناءً على نصوص نظام العمل فقط.

#  قواعد أساسية:
# - أجب باللغة العربية الفصحى فقط (ممنوع أي لغة أخرى)
# - استند فقط للسياق المقدم
# - اذكر رقم المادة عند كل استشهاد
# - أجب بما هو موجود فقط، لا تذكر ما هو غير موجود
# - لا تستخدم عبارات: "لا توجد استثناءات" أو "لا توجد إشارات" أو "لا توجد مواد أخرى"

#  هيكل الإجابة (اذكر فقط ما هو موجود):
# 1. القاعدة الأساسية (مع رقم المادة)
# 2. الاستثناءات (إن وُجدت)
# 3. الحالات الخاصة (إن وُجدت)
# 4. ربط المواد ذات العلاقة (إن وُجدت)

#  أسلوب الإجابة:
# - واضح ومباشر ومختصر
# - لغة بسيطة يفهمها غير المتخصص
# - توقف عند انتهاء المعلومات المفيدة
# - لا حشو ولا تكرار

# ---
# ### مثال:
# **سؤال:** كم مدة فترة التجربة؟

# **إجابة:**
# وفقًا لـ **المادة (53)** من نظام العمل السعودي، فترة التجربة لا تزيد على **90 يومًا**، 
# ويجب النص عليها صراحة في العقد.

# ويجوز بالاتفاق المكتوب **تمديدها** لـ **180 يومًا كحد أقصى**.

# لا تدخل ضمن الحساب:
# - إجازة عيدي الفطر والأضحى
# - الإجازة المرضية

# يحق للطرفين إنهاء العقد خلالها، ما لم ينص العقد على خلاف ذلك.

# وبحسب **المادة (54)**، لا يجوز تكرار فترة التجربة لدى نفس صاحب العمل، 
# إلا في مهنة مختلفة أو بعد مرور 6 أشهر من انتهاء العلاقة السابقة.

# لا يستحق أي طرف تعويضًا أو مكافأة نهاية خدمة عند الإنهاء خلال التجربة.

# ---
#  تذكر: أجب فقط بما هو موجود، واختم عند انتهاء المعلومات المفيدة.
# """

#         user_prompt = f"""السياق من نظام العمل السعودي:
# {context}

# سؤال: {question}

# قدم إجابة شاملة مع ذكر المواد المستخدمة."""

#         # ====================================================================
#         # STEP 5: Call Groq API
#         # ====================================================================
#         print(" Generating answer...")
        
#         chat_completion = groq_client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             model=config.GROQ_MODEL,
#             temperature=0.3,
#             max_tokens=1000,
#             top_p=0.9
#         )
        
#         answer = chat_completion.choices[0].message.content
        
#         print(f" Answer generated")
        
#         # ====================================================================
#         # STEP 6: Extract article citations from answer
#         # ====================================================================
#         answer_articles = re.findall(r'المادة\s+[^\n:.،]{1,50}', answer)
#         all_articles = list(set(articles_mentioned) | set(answer_articles))
        
#         # Clean up article list
#         cleaned_articles = []
#         for art in all_articles:
#             # Extract just the article reference
#             art_clean = art.replace('المادة ', '').strip()
#             if art_clean and art_clean != 'غير محدد':
#                 cleaned_articles.append(f"المادة {art_clean}")
        
#         print(f" Articles cited: {cleaned_articles}")
        
#         # ====================================================================
#         # STEP 7: Return response
#         # ====================================================================
#         return AnswerResponse(
#             answer=answer,
#             articles=cleaned_articles[:5],  # Limit to 5
#             context_chunks=[
#                 {
#                     'text': r['text'][:200] + "...",
#                     'article': r.get('article', 'غير محدد'),
#                     'score': round(r['score'], 3)
#                 }
#                 for r in results
#             ]
#         )
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ============================================================================
# # SHUTDOWN EVENT
# # ============================================================================

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     if vectordb:
#         vectordb.close()
#     print(" Server shutdown complete")

# # ============================================================================
# # RUN SERVER
# # ============================================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


