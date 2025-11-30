# file_search.py
from google import genai
from google.genai import types
import os
import time
import json
import re
from dotenv import load_dotenv

load_dotenv()

class FileSearchEngine:
    def __init__(self, store_name="pdf_rag_store", logger=None):
        # Set up logger
        self.logger = logger
        
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        
        os.environ['GEMINI_API_KEY'] = self.api_key
        self.client = genai.Client()
        self.store_name = store_name
        self.store = self._get_or_create_store()
        
        self.chunking_config = {
            'white_space_config': {
                'max_tokens_per_chunk': 512,
                'max_overlap_tokens': 50
            }
        }

    def _log(self, msg, level="info"):
        """Helper to log if logger exists"""
        if self.logger:
            if level == "info": self.logger.info(f"[Search] {msg}")
            elif level == "error": self.logger.error(f"[Search] {msg}")

    def _get_or_create_store(self):
        try:
            stores = list(self.client.file_search_stores.list())
            if stores:
                self._log(f"Using existing store: {stores[0].name}")
                return stores[0]
            else:
                store = self.client.file_search_stores.create(config={'display_name': self.store_name})
                self._log(f"Created new store: {store.name}")
                return store
        except Exception as e:
            # Fallback creation if list fails
            store = self.client.file_search_stores.create(config={'display_name': self.store_name})
            return store

    def extract_metadata(self, file_path):
        filename = os.path.basename(file_path)
        self._log(f"Extracting metadata for {filename}")
        
        try:
            # Upload file temporarily for Gemini to analyze structure
            uploaded_file = self.client.files.upload(file=file_path)
            
            # Wait for processing (Gemini needs to process the PDF/Image before inference)
            while uploaded_file.state == "PROCESSING":
                time.sleep(1)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state == "FAILED":
                self._log("Metadata extraction file processing failed", "error")
                return filename, "N/A"

            # --- COMPREHENSIVE PROMPT FOR ID AND TITLE EXTRACTION ---
            prompt = f"""
            You are an expert Document Analyst. Your task is to extract the **Official Title** and the **Unique Identification Number (ID)** from the provided medical/academic document.

            **CONTEXT INFORMATION:**
            - **Filename:** "{filename}"

            **INSTRUCTIONS:**

            **1. EXTRACTING THE ID:**
            * **Priority A (Inside Document):** Scan the document text for explicit ID labels. Look for keywords like:
                - "CONTROL ID"
                - "Abstract ID"
                - "Paper ID"
                - "Submission ID"
                - "ID:"
                (e.g., if text says "CONTROL ID: 1956957", extract "1956957").
            * **Priority B (Filename Fallback):** If NO ID is found inside the text, look at the **Filename** provided above. Extract the leading ID number.
                - (e.g., if filename is "354.pdf", extract "354")
                - (e.g., if filename is "#354 - CERVICAL outcomes.pdf", extract "354")
            * If no ID is found in text or filename, use "N/A".

            **2. EXTRACTING THE TITLE:**
            * The title is ALWAYS inside the document text.
            * **Priority A (Explicit Label):** Look for "TITLE:", "Abstract Title:", or "Title:". Extract the text immediately following it.
            * **Priority B (Visual Hierarchy):** If no label exists, identify the **Main Heading**. 
                - This is usually the first substantial block of text.
                - It is often in UPPERCASE or Bold.
                - It appears *before* the Author names or Institutions.
                - **CRITICAL:** Do NOT select the conference name (e.g., "Scoliosis Research Society", "Annual Meeting") as the title.

            **OUTPUT FORMAT:**
            Return ONLY a raw JSON object. Do not use Markdown code blocks.
            {{
                "title": "The Extracted Title String",
                "id": "The Extracted ID String"
            }}
            """
            
            # Call Gemini Model (Flash is sufficient and fast for this)
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
                    prompt
                ]
            )
            
            # Clean up the file from Gemini Cloud
            self.client.files.delete(name=uploaded_file.name)
            
            # Parse Response
            text = response.text.strip()
            # Clean markdown if model adds it (e.g., ```json ... ```)
            if text.startswith("```"):
                text = re.sub(r"^```json|^```", "", text).strip()
                text = re.sub(r"```$", "", text).strip()
            
            try:
                data = json.loads(text)
                title = data.get("title", "Unknown Title")
                doc_id = data.get("id", "N/A")
                
                # Final cleanup of title if it grabbed the label
                title = title.replace("TITLE:", "").replace("Abstract Title:", "").strip()
                
                self._log(f"Extracted -> ID: {doc_id} | Title: {title[:30]}...")
                return title, doc_id
            except json.JSONDecodeError:
                self._log(f"Failed to parse JSON response: {text}", "error")
                return filename, "N/A"
            
        except Exception as e:
            self._log(f"Metadata error: {e}", "error")
            return filename, "N/A"

    def upload_document(self, file_path, title, doc_id, filename):
        self._log(f"Uploading to Vector Store: {title}")
        
        # Prepare metadata for filtering later
        metadata_list = [
            {"key": "title", "string_value": title},
            {"key": "file_name", "string_value": filename},
        ]
        # Only add ID if it's not N/A to keep metadata clean
        if doc_id and doc_id != "N/A":
            metadata_list.append({"key": "ID", "string_value": str(doc_id)})

        operation = self.client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=self.store.name,
            file=file_path,
            config={
                'display_name': title,
                'chunking_config': self.chunking_config,
                'custom_metadata': metadata_list,
            }
        )
        
        while not operation.done:
            time.sleep(1)
            operation = self.client.operations.get(operation)
        
        self._log("Upload complete")
        return operation

    def query(self, question):
        self._log(f"Querying model...")
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{question}\n(Return answer in concise markdown with citations if available)",
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[self.store.name])
                )]
            )
        )
        
        citations = []
        if hasattr(response.candidates[0], 'grounding_metadata'):
            gm = response.candidates[0].grounding_metadata
            if hasattr(gm, 'grounding_chunks'):
                for chunk in gm.grounding_chunks:
                    if hasattr(chunk, 'web') and hasattr(chunk.web, 'title'):
                        citations.append(chunk.web.title)
                        
        return response.text, citations

    def list_documents(self):
        documents = []
        try:
            for doc in self.client.file_search_stores.documents.list(parent=self.store.name):
                meta = {'name': doc.name, 'title': doc.display_name, 'id': 'N/A', 'file_name': 'Unknown'}
                
                # Robustly check metadata
                if hasattr(doc, 'custom_metadata'):
                    for m in doc.custom_metadata:
                        if m.key == 'ID': meta['id'] = m.string_value
                        if m.key == 'file_name': meta['file_name'] = m.string_value
                        if m.key == 'title': meta['title'] = m.string_value
                
                documents.append(meta)
        except Exception as e:
            self._log(f"List docs error: {e}", "error")
        return documents

    def delete_document(self, doc_name):
        self.client.file_search_stores.documents.delete(
            name=doc_name,
            config={'force': True}
        )