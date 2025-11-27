from flask import Flask, render_template, request, jsonify, send_from_directory
from google import genai
from google.genai import types
import time
import os
import json
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Set the API key as environment variable for the Google client
os.environ['GEMINI_API_KEY'] = api_key
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Gemini client
client = genai.Client()

# Create the file search store (reuse existing or create new)
try:
    stores = list(client.file_search_stores.list())
    if stores:
        file_search_store = stores[0]
        print(f"Using existing store: {file_search_store.name}")
    else:
        file_search_store = client.file_search_stores.create(config={'display_name': 'pdf_rag_store'})
        print(f"Created new store: {file_search_store.name}")
except Exception as e:
    file_search_store = client.file_search_stores.create(config={'display_name': 'pdf_rag_store'})
    print(f"Created store: {file_search_store.name}")

# Chunking configuration
chunking_config = {
    'white_space_config': {
        'max_tokens_per_chunk': 512,
        'max_overlap_tokens': 10
    }
}

def extract_metadata(file_path):
    """Extract title and ID from PDF using Gemini"""
    uploaded_file = client.files.upload(file=file_path)
    
    # Wait for file to be processed
    while uploaded_file.state == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    
    if uploaded_file.state == "FAILED":
        raise ValueError(f"File processing failed")
    
    prompt = """
    Analyze this PDF document and extract the following information:
    1. Title of the document/paper/study
    2. Any ID present (Control ID, Abstract ID, Problem Statement ID, Paper ID, Study ID, etc.)
    
    Return ONLY a valid JSON object:
    {"title": "extracted title here", "id": "extracted id here"}
    
    If you cannot find the title, use the first heading.
    If you cannot find an ID, use "N/A".
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
            prompt
        ]
    )
    
    try:
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        metadata = json.loads(response_text)
        title = metadata.get("title", "Unknown Title")
        doc_id = metadata.get("id", "N/A")
        
        client.files.delete(name=uploaded_file.name)
        return title, doc_id
    except Exception as e:
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
        return os.path.basename(file_path).replace(".pdf", ""), "N/A"

def upload_to_store(file_path, title, doc_id, file_name):
    """Upload document to file search store"""
    operation = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=file_search_store.name,
        file=file_path,
        config={
            'display_name': title,
            'chunking_config': chunking_config,
            'custom_metadata': [
                {"key": "title", "string_value": title},
                {"key": "ID", "string_value": doc_id},
                {"key": "file_name", "string_value": file_name},
            ],
        }
    )
    
    while not operation.done:
        time.sleep(3)
        operation = client.operations.get(operation)
    
    return operation

def get_uploaded_documents():
    """Retrieve list of documents from file search store"""
    try:
        documents = []
        for doc in client.file_search_stores.documents.list(parent=file_search_store.name):
            # Extract metadata
            metadata = {
                'name': doc.name,
                'display_name': doc.display_name if hasattr(doc, 'display_name') else 'Unknown',
                'title': 'Unknown',
                'id': 'N/A',
                'file_name': 'Unknown'
            }
            
            if hasattr(doc, 'custom_metadata') and doc.custom_metadata:
                for meta in doc.custom_metadata:
                    if meta.key == 'title':
                        metadata['title'] = meta.string_value
                    elif meta.key == 'ID':
                        metadata['id'] = meta.string_value
                    elif meta.key == 'file_name':
                        metadata['file_name'] = meta.string_value
            
            documents.append(metadata)
        
        return documents
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract metadata
        title, doc_id = extract_metadata(filepath)
        
        # Upload to file search store
        upload_to_store(filepath, title, doc_id, filename)
        
        # Clean up local file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'metadata': {
                'title': title,
                'id': doc_id,
                'file_name': filename
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    try:
        documents = get_uploaded_documents()
        return jsonify({'documents': documents})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{question}\n(return your answer in markdown as concise bullet points)\nANSWER:\n",
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[file_search_store.name]
                        )
                    )
                ]
            )
        )
        
        # Extract citations if available
        citations = []
        if hasattr(response.candidates[0], 'grounding_metadata'):
            grounding = response.candidates[0].grounding_metadata
            if hasattr(grounding, 'grounding_chunks'):
                for chunk in grounding.grounding_chunks:
                    if hasattr(chunk, 'web') and hasattr(chunk.web, 'title'):
                        citations.append(chunk.web.title)
        
        return jsonify({
            'answer': response.text,
            'citations': citations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<path:doc_name>', methods=['DELETE'])
def delete_document(doc_name):
    try:
        # Extract the document name from the full path
        # Expected format: fileSearchStores/{store_name}/documents/{document_name}
        doc_name = doc_name.strip()
        if not doc_name.startswith('fileSearchStores/'):
            return jsonify({'error': 'Invalid document name format'}), 400
        
        # Use the delete method with force=True to delete related chunks and objects
        operation = client.file_search_stores.documents.delete(
            name=doc_name,
            config={'force': True}
        )
        
        # # Wait for the delete operation to complete
        # while not operation.done:
        #     time.sleep(1)
        #     operation = client.operations.get(operation)
        
        return jsonify({
            'success': True, 
            'message': 'Document and related data deleted successfully'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to delete document: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)