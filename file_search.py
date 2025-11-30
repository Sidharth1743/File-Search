# file_search.py
from google import genai
from google.genai import types
import os
import time
import json
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
            store = self.client.file_search_stores.create(config={'display_name': self.store_name})
            return store

    def extract_metadata(self, file_path):
        self._log(f"Extracting metadata for {os.path.basename(file_path)}")
        try:
            uploaded_file = self.client.files.upload(file=file_path)
            
            while uploaded_file.state == "PROCESSING":
                time.sleep(1)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state == "FAILED":
                self._log("Metadata extraction file processing failed", "error")
                return os.path.basename(file_path), "N/A"

            prompt = """Analyze this PDF. Return JSON: {"title": "Title", "id": "ID or N/A"}"""
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
                    prompt
                ]
            )
            
            self.client.files.delete(name=uploaded_file.name)
            
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
            
            data = json.loads(text)
            title = data.get("title", "Unknown")
            return title, data.get("id", "N/A")
            
        except Exception as e:
            self._log(f"Metadata error: {e}", "error")
            return os.path.basename(file_path), "N/A"

    def upload_document(self, file_path, title, doc_id, filename):
        self._log(f"Uploading to Vector Store: {title}")
        operation = self.client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=self.store.name,
            file=file_path,
            config={
                'display_name': title,
                'chunking_config': self.chunking_config,
                'custom_metadata': [
                    {"key": "title", "string_value": title},
                    {"key": "ID", "string_value": doc_id},
                    {"key": "file_name", "string_value": filename},
                ],
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
            contents=f"{question}\n(Return answer in concise markdown)",
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