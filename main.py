# main.py - ENHANCED ADMIN BACKEND
import logging
import sys
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid
from datetime import datetime

# --- CAMEL & Neo4j Imports ---
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from unstructured.documents.elements import Text 

# --- Local Imports ---
from file_search import FileSearchEngine
from kg_agents import KnowledgeGraphAgent
from ocr_engine import OCREngine

# ---------------------------------------------------------------------------
# 1. SETUP LOGGING
# ---------------------------------------------------------------------------
logger = logging.getLogger("SpineDAO_App")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler(sys.stdout)
c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

logger.info("🚀 SpineDAO Application Starting...")

# ---------------------------------------------------------------------------
# 2. CONFIGURATION & INIT
# ---------------------------------------------------------------------------
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max total
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add Content Security Policy headers to fix JavaScript eval errors
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:"
    return response

# Task tracking for bulk uploads (in-memory, use Redis for production)
active_tasks = {}

# Safe Init
fs_engine = None
ocr_engine = None
kg_agent = None
neo4j_db = None

try:
    logger.info("Initializing Custom OCR Engine...")
    ocr_engine = OCREngine(api_key=os.getenv('GEMINI_API_KEY'), use_advanced_model=True, logger=logger)
    logger.info("✓ OCR Engine Ready")
except Exception as e:
    logger.error(f"⚠️ OCR Engine failed: {e}")

try:
    logger.info("Initializing File Search...")
    fs_engine = FileSearchEngine(logger=logger)
    logger.info("✓ File Search Ready")
except Exception as e:
    logger.error(f"⚠️ File Search failed: {e}")

try:
    logger.info("Initializing Neo4j & Llama...")
    neo4j_db = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    
    llama_model = ModelFactory.create(
        model_platform=ModelPlatformType.GROQ,
        model_type=ModelType.GROQ_LLAMA_3_3_70B,
        api_key=os.getenv("GROQ_API_KEY"),
        model_config_dict={"temperature": 0.0}
    )
    
    uio = UnstructuredIO() 
    kg_agent = KnowledgeGraphAgent(model=llama_model)
    logger.info("✓ KG Agent Ready")
except Exception as e:
    logger.error(f"⚠️ KG Agent failed: {e}")


# ---------------------------------------------------------------------------
# 3. PROCESSING LOGIC
# ---------------------------------------------------------------------------

def run_advanced_ocr(pdf_path):
    try:
        filename = os.path.basename(pdf_path)
        logger.info(f"👁️ [OCR] Running Advanced OCR on {filename}...")
        extracted_text = ocr_engine.process_file(
            pdf_path, 
            use_preprocessing=True, 
            enhancement_level="medium", 
            medical_context=True
        )
        return extracted_text
    except Exception as e:
        logger.error(f"❌ [OCR] Failed: {e}", exc_info=True)
        return None

def process_kg_from_text(text_content, metadata):
    """
    Feeds text to Camel Agent and injects metadata into nodes/rels
    """
    try:
        logger.info(f"🕸️ [KG] Structuring text for Knowledge Graph...")
        logger.info(f"    - Attaching metadata: {metadata}")
        
        element = Text(text_content)
        graph_elements = kg_agent.run(element, parse_graph_elements=True, metadata=metadata)
        
        nodes = len(graph_elements.nodes)
        rels = len(graph_elements.relationships)
        
        logger.info(f"🕸️ [KG] Storing {nodes} nodes and {rels} relationships in Neo4j...")
        neo4j_db.add_graph_elements(graph_elements=[graph_elements])
        
        return True, f"Success: {nodes} Nodes, {rels} Rels extracted."
        
    except Exception as e:
        logger.error(f"❌ [KG] Error: {e}", exc_info=True)
        return False, str(e)


# ---------------------------------------------------------------------------
# 4. REGULAR USER ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin_panel():
    """Render admin interface"""
    return render_template('admin.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if not ocr_engine or not fs_engine or not kg_agent:
        return jsonify({
            'success': False,
            'error': 'Service unavailable',
            'details': 'System components failed to initialize',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'No file provided in request',
            'code': 'NO_FILE_PROVIDED'
        }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'No file selected for upload',
            'code': 'NO_FILE_SELECTED'
        }), 400

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({
            'success': False,
            'error': 'Unsupported file type',
            'details': 'Only PDF files are supported',
            'code': 'UNSUPPORTED_FILE_TYPE'
        }), 415
    
    try:
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        
        text_content = run_advanced_ocr(pdf_path)
        if not text_content:
             return jsonify({'error': 'OCR failed'}), 500

        logger.info("🔍 [Search] Indexing PDF...")
        title, doc_id = fs_engine.extract_metadata(pdf_path)
        fs_engine.upload_document(pdf_path, title, doc_id, filename)
        search_msg = "Indexed in Google File Search."

        kg_metadata = {
            'document_title': title,
            'document_id': doc_id,
            'file_name': filename,
            'source': 'SpineDAO_Pipeline'
        }

        logger.info("🕸️ [KG] Processing Graph...")
        kg_success, kg_msg = process_kg_from_text(text_content, kg_metadata)
        
        if os.path.exists(pdf_path): os.remove(pdf_path)

        return jsonify({
            'success': True,
            'metadata': {'title': title, 'id': doc_id},
            'status': {
                'search': search_msg,
                'kg': kg_msg,
                'kg_success': kg_success
            }
        })

    except Exception as e:
        logger.error(f"❌ Route Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to process document upload',
            'details': str(e),
            'code': 'UPLOAD_ERROR'
        }), 500

@app.route('/query', methods=['POST'])
def query():
    if not fs_engine:
        return jsonify({
            'success': False,
            'error': 'Service unavailable',
            'details': 'Search engine not initialized',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'No JSON data provided',
            'code': 'NO_JSON_DATA'
        }), 400

    question = data.get('question', '')
    if not question or not question.strip():
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'Question cannot be empty',
            'code': 'EMPTY_QUESTION'
        }), 400

    # Validate question length
    if len(question) > 1000:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'Question exceeds maximum length of 1000 characters',
            'code': 'QUESTION_TOO_LONG'
        }), 400
    
    try:
        answer, citations = fs_engine.query(question)
        return jsonify({'answer': answer, 'citations': citations})
    except Exception as e:
        logger.error(f"❌ Query Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to process query',
            'details': str(e),
            'code': 'QUERY_ERROR'
        }), 500

@app.route('/documents', methods=['GET'])
def list_docs():
    if not fs_engine: return jsonify({'documents': []})

    # Get store type from query parameter, default to current engine store
    store_type = request.args.get('store_type', None)

    if store_type and store_type in ['abstracts', 'manuscripts']:
        try:
            # Create temporary engine for the requested store type
            temp_engine = FileSearchEngine(store_name=f"{store_type}_store", logger=logger)
            documents = temp_engine.list_documents()
            return jsonify({
                'documents': documents,
                'store': store_type,
                'store_name': f"{store_type}_store"
            })
        except Exception as e:
            logger.error(f"❌ Error listing documents from {store_type} store: {e}", exc_info=True)
            return jsonify({'documents': []})
    else:
        # Use current engine store
        documents = fs_engine.list_documents()
        return jsonify({
            'documents': documents,
            'store': 'current',
            'store_name': fs_engine.store_name
        })

@app.route('/delete/<path:doc_name>', methods=['DELETE'])
def delete_doc(doc_name):
    if not fs_engine: return jsonify({'error': 'Search engine down'}), 500
    try:
        fs_engine.delete_document(doc_name)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"❌ Delete Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to delete document',
            'details': str(e),
            'code': 'DELETE_ERROR'
        }), 500

# ---------------------------------------------------------------------------
# 5. ENHANCED ADMIN ROUTES FOR BULK UPLOADS
# ---------------------------------------------------------------------------

@app.route('/admin/upload-folder-path', methods=['POST'])
def admin_upload_folder_path():
    """
    OPTION 1: Server-side folder path (original method)
    Use when folders already exist on server
    """
    if not ocr_engine or not fs_engine or not kg_agent:
        return jsonify({
            'success': False,
            'error': 'Service unavailable',
            'details': 'System components failed to initialize',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'No JSON data provided',
            'code': 'NO_JSON_DATA'
        }), 400

    folder_path = data.get('folder_path')
    document_type = data.get('document_type', 'abstracts')

    if not folder_path:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'folder_path is required',
            'code': 'MISSING_FOLDER_PATH'
        }), 400

    if document_type not in ['abstracts', 'manuscripts']:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'document_type must be "abstracts" or "manuscripts"',
            'code': 'INVALID_DOCUMENT_TYPE'
        }), 400

    if not os.path.exists(folder_path):
        return jsonify({
            'success': False,
            'error': 'Not found',
            'details': f'Folder not found: {folder_path}',
            'code': 'FOLDER_NOT_FOUND'
        }), 404

    if not os.path.isdir(folder_path):
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': f'Path is not a directory: {folder_path}',
            'code': 'NOT_A_DIRECTORY'
        }), 400
    
    try:
        task_id = str(uuid.uuid4())
        logger.info(f"Admin bulk upload started [Task: {task_id}]: {folder_path} -> {document_type}")
        
        # Initialize task tracking
        active_tasks[task_id] = {
            'status': 'processing',
            'current': 0,
            'total': 0,
            'current_file': '',
            'started_at': datetime.now().isoformat(),
            'errors': [],
            'processed_files': []
        }
        
        def progress_callback(current, total, filename, status, error=None):
            active_tasks[task_id].update({
                'current': current,
                'total': total,
                'current_file': filename,
                'status': status
            })
            if error:
                active_tasks[task_id]['errors'].append(f"{filename}: {error}")
            logger.info(f"[Task {task_id}] Progress: {current}/{total} - {filename} - {status}")
        
        results = fs_engine.bulk_upload_folder(
            folder_path=folder_path,
            document_type=document_type,
            progress_callback=progress_callback
        )
        
        active_tasks[task_id]['status'] = 'completed'
        active_tasks[task_id]['completed_at'] = datetime.now().isoformat()
        active_tasks[task_id]['results'] = results
        
        return jsonify({
            'success': results['success'],
            'task_id': task_id,
            'message': results['message'],
            'summary': {
                'processed': results['processed'],
                'successful': results['successful'],
                'failed': results['failed']
            },
            'errors': results['errors']
        })
        
    except Exception as e:
        logger.error(f"Admin bulk upload failed: {e}", exc_info=True)
        if task_id in active_tasks:
            active_tasks[task_id]['status'] = 'failed'
            active_tasks[task_id]['error'] = str(e)
        return jsonify({
            'success': False,
            'error': 'Bulk upload failed',
            'details': str(e),
            'code': 'BULK_UPLOAD_ERROR'
        }), 500


@app.route('/admin/upload-folder-files', methods=['POST'])
def admin_upload_folder_files():
    """
    OPTION 2: Multiple file upload (browser folder selection)
    User selects folder in browser, all files uploaded
    """
    if not ocr_engine or not fs_engine or not kg_agent:
        return jsonify({
            'success': False,
            'error': 'Service unavailable',
            'details': 'System components failed to initialize',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    if 'files' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'No files provided in request',
            'code': 'NO_FILES_PROVIDED'
        }), 400

    files = request.files.getlist('files')
    document_type = request.form.get('document_type', 'abstracts')

    if document_type not in ['abstracts', 'manuscripts']:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'document_type must be "abstracts" or "manuscripts"',
            'code': 'INVALID_DOCUMENT_TYPE'
        }), 400

    if not files or all(f.filename == '' for f in files):
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'No files selected for upload',
            'code': 'NO_FILES_SELECTED'
        }), 400

    # Filter only PDF files
    pdf_files = [f for f in files if f.filename.lower().endswith('.pdf')]

    if not pdf_files:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'details': 'No PDF files found in upload',
            'code': 'NO_PDF_FILES'
        }), 400
    
    try:
        task_id = str(uuid.uuid4())
        logger.info(f"Admin bulk upload via browser [Task: {task_id}]: {len(pdf_files)} files -> {document_type}")
        
        # Create temporary folder for this batch
        batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'batch_{task_id}')
        os.makedirs(batch_folder, exist_ok=True)
        
        # Save all files first
        saved_files = []
        for file in pdf_files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(batch_folder, filename)
            file.save(file_path)
            saved_files.append(file_path)
        
        # Initialize task tracking
        active_tasks[task_id] = {
            'status': 'processing',
            'current': 0,
            'total': len(saved_files),
            'current_file': '',
            'started_at': datetime.now().isoformat(),
            'errors': [],
            'processed_files': []
        }
        
        def progress_callback(current, total, filename, status, error=None):
            active_tasks[task_id].update({
                'current': current,
                'total': total,
                'current_file': filename,
                'status': status
            })
            if error:
                active_tasks[task_id]['errors'].append(f"{filename}: {error}")
        
        # Process using existing bulk upload logic
        results = fs_engine.bulk_upload_folder(
            folder_path=batch_folder,
            document_type=document_type,
            progress_callback=progress_callback
        )
        
        # Cleanup temporary folder
        import shutil
        shutil.rmtree(batch_folder, ignore_errors=True)
        
        active_tasks[task_id]['status'] = 'completed'
        active_tasks[task_id]['completed_at'] = datetime.now().isoformat()
        active_tasks[task_id]['results'] = results
        
        return jsonify({
            'success': results['success'],
            'task_id': task_id,
            'message': f"Processed {len(pdf_files)} files",
            'summary': {
                'processed': results['processed'],
                'successful': results['successful'],
                'failed': results['failed']
            },
            'errors': results['errors']
        })
        
    except Exception as e:
        logger.error(f"Admin bulk upload via browser failed: {e}", exc_info=True)
        if task_id in active_tasks:
            active_tasks[task_id]['status'] = 'failed'
            active_tasks[task_id]['error'] = str(e)
        return jsonify({
            'success': False,
            'error': 'Browser bulk upload failed',
            'details': str(e),
            'code': 'BROWSER_BULK_UPLOAD_ERROR'
        }), 500


@app.route('/admin/progress/<task_id>', methods=['GET'])
def admin_get_progress(task_id):
    """Get real-time progress for a bulk upload task"""
    if task_id not in active_tasks:
        return jsonify({
            'success': False,
            'error': 'Not found',
            'details': 'Task not found',
            'code': 'TASK_NOT_FOUND'
        }), 404

    return jsonify({
        'success': True,
        'task': active_tasks[task_id]
    })


@app.route('/admin/tasks', methods=['GET'])
def admin_list_tasks():
    """List all tasks (recent first)"""
    tasks = []
    for task_id, task_data in sorted(
        active_tasks.items(), 
        key=lambda x: x[1].get('started_at', ''), 
        reverse=True
    ):
        tasks.append({
            'task_id': task_id,
            **task_data
        })
    return jsonify({'tasks': tasks})


@app.route('/admin/stores', methods=['GET'])
def admin_list_stores():
    """List all file search stores and their document counts"""
    try:
        stores_info = []
        
        for store_type in ['abstracts', 'manuscripts']:
            try:
                temp_engine = FileSearchEngine(store_name=f"{store_type}_store", logger=logger)
                documents = temp_engine.get_store_documents()
                
                stores_info.append({
                    'name': f"{store_type}_store",
                    'type': store_type,
                    'document_count': len(documents),
                    'documents': documents[:10]  # First 10 for preview
                })
            except Exception as e:
                stores_info.append({
                    'name': f"{store_type}_store",
                    'type': store_type,
                    'document_count': 0,
                    'documents': [],
                    'error': str(e)
                })
        
        return jsonify({'stores': stores_info})
    except Exception as e:
        logger.error(f"❌ Store listing error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to list stores',
            'details': str(e),
            'code': 'STORE_LIST_ERROR'
        }), 500


@app.route('/admin/folders', methods=['GET'])
def admin_list_folders():
    """List available folders on server (for path-based upload)"""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        folders = []
        
        if os.path.exists(upload_dir):
            for item in os.listdir(upload_dir):
                item_path = os.path.join(upload_dir, item)
                if os.path.isdir(item_path) and not item.startswith('batch_'):
                    pdf_count = len([f for f in os.listdir(item_path) if f.lower().endswith('.pdf')])
                    folders.append({
                        'name': item,
                        'path': item_path,
                        'pdf_count': pdf_count
                    })
        
        return jsonify({'folders': folders})
    except Exception as e:
        logger.error(f"❌ Folder listing error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to list folders',
            'details': str(e),
            'code': 'FOLDER_LIST_ERROR'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)