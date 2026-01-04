"""
Document Processor using OCR
Extracts Arabic text using pytesseract
"""

import re
from typing import List, Dict
from pdf2image import convert_from_path
import pytesseract


class DocumentProcessor:
    """Process PDF with OCR for proper Arabic extraction"""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def preprocess_arabic_for_search(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub("[إأآٱ]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "و", text)
        text = re.sub("ئ", "ي", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("[ًٌٍَُِّْـ]", "", text)
        return text.strip()
    
    def extract_text_from_pdf_ocr(self, pdf_path: str, dpi: int = 300) -> str:
        print(f"📄 Extracting text with OCR (dpi={dpi})...")
        text = ""
        
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            
            for i, img in enumerate(images, start=1):
                print(f"   Processing page {i}/{len(images)}")
                ocr_text = pytesseract.image_to_string(
                    img, 
                    lang="ara", 
                    config=custom_config
                )
                text += ocr_text + "\n"
            
            print(f"✅ Extracted {len(text)} characters")
            return text
            
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            return ""
    
    def extract_articles_flexible(self, raw_text: str) -> List[str]:
        """Extract articles using flexible patterns - keeps ORIGINAL text"""
        patterns = [
            r'المادة\s+[^\s:]+[:\s]',
            r'المادة\s+\d+[:\s]',
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        matches = list(re.finditer(combined_pattern, raw_text, re.IGNORECASE))
        
        articles = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(raw_text)
            
            article_text = raw_text[start:end].strip()
            
            if len(article_text) > 50:
                articles.append(article_text)
        
        print(f"✅ Extracted {len(articles)} articles")
        return articles
    
    def extract_article_number(self, text: str) -> str:
        """Extract article number/name from text"""
        # Look for المادة followed by text
        match = re.search(r'المادة\s+([^\n:.]{1,50})', text[:200])
        if match:
            return match.group(1).strip()
        return "غير محدد"
    
    def create_chunks(self, text: str) -> List[str]:
        """Split text into word-based chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        
        return chunks
    
    def process(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        FULL PIPELINE with OCR:
        1. Extract text with OCR (keeps original formatting)
        2. Extract articles (original text)
        3. Create normalized version for search
        4. Return chunks with both versions
        """
        print("\n" + "="*60)
        print("DOCUMENT PROCESSING WITH OCR")
        print("="*60)
        
        # Step 1: Extract with OCR
        raw_text = self.extract_text_from_pdf_ocr(pdf_path, dpi=300)
        
        if not raw_text:
            print("❌ Failed to extract text")
            return []
        
        # Step 2: Extract articles (ORIGINAL text)
        original_articles = self.extract_articles_flexible(raw_text)
        
        if not original_articles:
            print("❌ No articles found")
            return []
        
        # Step 3: Create chunks with both original and normalized text
        chunks = []
        
        for i, article in enumerate(original_articles):
            # Normalize ONLY for search
            normalized = self.preprocess_arabic_for_search(article)
            
            # Extract article number
            article_num = self.extract_article_number(article)
            
            chunks.append({
                'text': article,  # ORIGINAL text for display
                'text_normalized': normalized,  # For search only
                'article_number': article_num,
                'chunk_id': i
            })
        
        print(f"✅ Created {len(chunks)} chunks")
        print(f"\n Sample chunk:")
        if chunks:
            sample = chunks[0]
            print(f"   Article: {sample['article_number']}")
            print(f"   Text: {sample['text'][:100]}...")
        
        print("="*60 + "\n")
        return chunks


# Testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        processor = DocumentProcessor(chunk_size=300)
        chunks = processor.process(sys.argv[1])
        print(f"\n✅ DONE: {len(chunks)} chunks")
