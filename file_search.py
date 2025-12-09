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
        if self.logger:
            if level == "info": self.logger.info(f"[Search] {msg}")
            elif level == "error": self.logger.error(f"[Search] {msg}")

    def _get_or_create_store(self):
        try:
            stores = list(self.client.file_search_stores.list())

            # Look for a store with the matching name
            target_store_name = f"{self.store_name}"
            for store in stores:
                if target_store_name in store.name or store.display_name == target_store_name:
                    self._log(f"Using existing store: {store.name}")
                    return store

            # If no matching store found, create a new one
            store = self.client.file_search_stores.create(config={'display_name': target_store_name})
            self._log(f"Created new store: {store.name}")
            return store
        except Exception as e:
            # Fallback creation if list fails
            self._log(f"Fallback store creation due to error: {e}", "error")
            store = self.client.file_search_stores.create(config={'display_name': self.store_name})
            return store

    def _get_existing_short_names(self):
        existing = set()
        try:
            self._log("Checking store for existing manuscripts...")
            for doc in self.client.file_search_stores.documents.list(parent=self.store.name):
                if hasattr(doc, 'custom_metadata'):
                    for m in doc.custom_metadata:
                        if m.key == 'short_name':
                            existing.add(m.string_value)
            self._log(f"Found {len(existing)} existing manuscripts in store.")
        except Exception as e:
            self._log(f"Error checking existing docs: {e}", "error")
        return existing

    def extract_metadata(self, file_path):
        filename = os.path.basename(file_path)
        self._log(f"Extracting metadata for {filename}")
        
        try:
            # Upload file temporarily for Gemini to analyze structure
            uploaded_file = self.client.files.upload(file=file_path)
            
            # Wait for processing
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
            * Look for "CONTROL ID", "Abstract ID", "Paper ID", "ID:".
            * Fallback: Extract ID from filename if present.
            * If no ID found, use "N/A".

            **2. EXTRACTING THE TITLE:**
            * Look for "TITLE:", "Abstract Title:".
            * Fallback: Identify the Main Heading (First substantial block, UPPERCASE/Bold).
            * Do NOT select conference names.

            **OUTPUT FORMAT:**
            Return ONLY a raw JSON object:
            {{
                "title": "The Extracted Title String",
                "id": "The Extracted ID String"
            }}
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
                    prompt
                ]
            )
            
            self.client.files.delete(name=uploaded_file.name)
            
            text = response.text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```json|^```", "", text).strip()
                text = re.sub(r"```$", "", text).strip()
            
            try:
                data = json.loads(text)
                title = data.get("title", "Unknown Title")
                doc_id = data.get("id", "N/A")
                title = title.replace("TITLE:", "").replace("Abstract Title:", "").strip()
                self._log(f"Extracted -> ID: {doc_id} | Title: {title[:30]}...")
                return title, doc_id
            except json.JSONDecodeError:
                self._log(f"Failed to parse JSON response: {text}", "error")
                return filename, "N/A"
            
        except Exception as e:
            self._log(f"Metadata error: {e}", "error")
            return filename, "N/A"

    def _build_metadata(self, title, doc_id, filename):
        metadata_list = [
            {"key": "title", "string_value": title},
            {"key": "file_name", "string_value": filename},
        ]

        if "manuscripts" in self.store_name:
            short_name = os.path.splitext(filename)[0]
            metadata_list.append({"key": "short_name", "string_value": short_name})

        if doc_id and doc_id != "N/A":
            metadata_list.append({"key": "ID", "string_value": str(doc_id)})

        return metadata_list

    def _upload_to_store(self, file_path, title, doc_id, filename):
        metadata_list = self._build_metadata(title, doc_id, filename)

        try:
            # The upload_to_file_search_store method handles both upload and import operations
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=self.store.name,
                file=file_path,
                config={
                    'display_name': title,
                    'chunking_config': self.chunking_config,
                    'custom_metadata': metadata_list,
                }
            )

            # Wait for operation completion with proper error handling
            while not operation.done:
                time.sleep(1)
                operation = self.client.operations.get(operation)

                # Check for operation errors during processing
                if hasattr(operation, 'error') and operation.error:
                    error_msg = f"Operation failed: {operation.error}"
                    self._log(error_msg, "error")
                    raise Exception(error_msg)

                # Additional error checking for Google API response structure
                if hasattr(operation, 'result') and hasattr(operation.result, 'error'):
                    error_msg = f"Operation result error: {operation.result.error}"
                    self._log(error_msg, "error")
                    raise Exception(error_msg)

            # Final validation after operation completion
            if hasattr(operation, 'error') and operation.error:
                error_msg = f"Final operation error: {operation.error}"
                self._log(error_msg, "error")
                raise Exception(error_msg)

            if hasattr(operation, 'result') and hasattr(operation.result, 'error'):
                error_msg = f"Final operation result error: {operation.result.error}"
                self._log(error_msg, "error")
                raise Exception(error_msg)

            self._log(f"Document processing complete: {filename}")
            return operation

        except Exception as e:
            self._log(f"Upload error for {filename}: {str(e)}", "error")
            raise

    def upload_document(self, file_path, title, doc_id, filename):
        self._log(f"Uploading to Vector Store: {title}")
        return self._upload_to_store(file_path, title, doc_id, filename)

    def query(self, question):
        self._log(f"Querying model...")
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
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
                if hasattr(doc, 'custom_metadata'):
                    for m in doc.custom_metadata:
                        if m.key == 'ID': meta['id'] = m.string_value
                        if m.key == 'file_name': meta['file_name'] = m.string_value
                        if m.key == 'title': meta['title'] = m.string_value
                        # We can also expose short_name here if needed
                documents.append(meta)
        except Exception as e:
            self._log(f"List docs error: {e}", "error")
        return documents

    def delete_document(self, doc_name):
        try:
            self._log(f"Deleting document: {doc_name}")
            if not doc_name or not isinstance(doc_name, str):
                error_msg = f"Invalid document name: {doc_name}"
                self._log(error_msg, "error")
                raise ValueError(error_msg)

            if not doc_name.startswith('fileSearchStores/'):
                error_msg = f"Document name must start with 'fileSearchStores/': {doc_name}"
                self._log(error_msg, "error")
                raise ValueError(error_msg)

            self.client.file_search_stores.documents.delete(
                name=doc_name,
                config={'force': True}
            )
            self._log(f"Successfully deleted document: {doc_name}")
            return True
        except Exception as e:
            self._log(f"Failed to delete document {doc_name}: {str(e)}", "error")
            raise

    def bulk_upload_folder(self, folder_path, document_type="abstracts", progress_callback=None):
        if document_type not in ["abstracts", "manuscripts"]:
            raise ValueError("document_type must be 'abstracts' or 'manuscripts'")

        store_name = f"{document_type}_store"
        if self.store_name != store_name:
            self._log(f"Switching from {self.store_name} to {store_name}")
            self.store_name = store_name
            self.store = self._get_or_create_store()
            self._log(f"Now using store: {self.store.name}")

        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            return {'success': False, 'message': 'No PDF files found', 'processed': 0}

        results = {
            'success': True,
            'message': f'Processing {len(pdf_files)} PDF files',
            'processed': len(pdf_files),
            'successful': 0,
            'failed': 0,
            'errors': [],
            'processed_files': []
        }

        self._log(f"Starting bulk upload of {len(pdf_files)} files to {store_name}")

        existing_short_names = set()
        if document_type == "manuscripts":
            existing_short_names = self._get_existing_short_names()

        for i, pdf_path in enumerate(pdf_files):
            filename = os.path.basename(pdf_path)

            if document_type == "manuscripts":
                current_short_name = os.path.splitext(filename)[0]
                if current_short_name in existing_short_names:
                    msg = f"Skipping {filename} - Already exists in store ({current_short_name})"
                    self._log(msg)
                    if progress_callback:
                        progress_callback(i + 1, len(pdf_files), filename, 'skipped')
                    continue

            try:
                self._log(f"Processing file {i+1}/{len(pdf_files)}: {filename}")
                title, doc_id = self.extract_metadata(pdf_path)

                self.upload_document(pdf_path, title, doc_id, filename)

                results['successful'] += 1
                results['processed_files'].append({
                    'filename': filename, 'title': title, 'status': 'success'
                })

                if progress_callback:
                    progress_callback(i + 1, len(pdf_files), filename, 'success')

            except Exception as e:
                error_msg = f"Failed to process {filename}: {str(e)}"
                self._log(error_msg, "error")
                results['failed'] += 1
                results['errors'].append(error_msg)
                if progress_callback:
                    progress_callback(i + 1, len(pdf_files), filename, 'failed', str(e))

        return results

    def get_store_documents(self):
        """Get documents from the current store"""
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