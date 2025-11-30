# main.py
import logging
import sys
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

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
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Safe Init
fs_engine = None
ocr_engine = None
kg_agent = None
neo4j_db = None

try:
    logger.info("Initializing Custom OCR Engine...")
    ocr_engine = OCREngine(api_key=os.getenv('GEMINI_API_KEY'), use_advanced_model=True)
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
        
        # Wrap text in Element
        element = Text(text_content)
        
        # Run Agent with Metadata
        graph_elements = kg_agent.run(element, parse_graph_elements=True, metadata=metadata)
        
        nodes = len(graph_elements.nodes)
        rels = len(graph_elements.relationships)
        
        # Store in Neo4j
        logger.info(f"🕸️ [KG] Storing {nodes} nodes and {rels} relationships in Neo4j...")
        neo4j_db.add_graph_elements(graph_elements=[graph_elements])
        
        return True, f"Success: {nodes} Nodes, {rels} Rels extracted."
        
    except Exception as e:
        logger.error(f"❌ [KG] Error: {e}", exc_info=True)
        return False, str(e)


# ---------------------------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if not ocr_engine or not fs_engine or not kg_agent:
        return jsonify({'error': 'System components failed to initialize.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # 1. Save PDF locally
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        
        # 2. Run OCR
        text_content = run_advanced_ocr(pdf_path)
        if not text_content:
             return jsonify({'error': 'OCR failed'}), 500

        # 3. Index & Get Metadata (Title, ID)
        logger.info("🔍 [Search] Indexing PDF...")
        title, doc_id = fs_engine.extract_metadata(pdf_path)
        fs_engine.upload_document(pdf_path, title, doc_id, filename)
        search_msg = "Indexed in Google File Search."

        # 4. Prepare Metadata for KG
        # This dict will be attached to every Node/Rel in Neo4j
        kg_metadata = {
            'document_title': title,
            'document_id': doc_id,
            'file_name': filename,
            'source': 'SpineDAO_Pipeline' # Overwrite the default source
        }

        # 5. Run KG Agent
        logger.info("🕸️ [KG] Processing Graph...")
        kg_success, kg_msg = process_kg_from_text(text_content, kg_metadata)
        
        # Cleanup
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
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question', '')
    if not fs_engine: return jsonify({'error': 'Search engine down'}), 500
    
    try:
        answer, citations = fs_engine.query(question)
        return jsonify({'answer': answer, 'citations': citations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_docs():
    if not fs_engine: return jsonify({'documents': []})
    return jsonify({'documents': fs_engine.list_documents()})

@app.route('/delete/<path:doc_name>', methods=['DELETE'])
def delete_doc(doc_name):
    if not fs_engine: return jsonify({'error': 'Search engine down'}), 500
    try:
        fs_engine.delete_document(doc_name)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)