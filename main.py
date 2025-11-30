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
from ocr_engine import OCREngine  # <--- Importing your custom engine

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

# A. Init Your Custom OCR Engine
try:
    logger.info("Initializing Custom OCR Engine (Fitz + Gemini Vision)...")
    # We use the GEMINI_API_KEY from .env
    ocr_engine = OCREngine(api_key=os.getenv('GEMINI_API_KEY'), use_advanced_model=True)
    logger.info("✓ OCR Engine Ready")
except Exception as e:
    logger.critical(f"✗ Failed to init OCR Engine: {e}", exc_info=True)

# B. Init File Search
try:
    fs_engine = FileSearchEngine(logger=logger)
    logger.info("✓ Google File Search Engine Ready")
except Exception as e:
    logger.critical(f"✗ Failed to init File Search: {e}", exc_info=True)

# C. Init Knowledge Graph
try:
    neo4j_db = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    
    gemini = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_2_0_FLASH,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_config_dict={"temperature": 0.2}
    )
    
    uio = UnstructuredIO() # Keep for utility, though we use your text
    kg_agent = KnowledgeGraphAgent(model=gemini)
    logger.info("✓ Knowledge Graph Agent Ready")
except Exception as e:
    logger.critical(f"✗ Failed to init KG Components: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# 3. PROCESSING LOGIC
# ---------------------------------------------------------------------------

def run_advanced_ocr(pdf_path):
    """
    Uses your custom OCREngine to process the PDF.
    """
    try:
        filename = os.path.basename(pdf_path)
        logger.info(f"👁️ [OCR] Running Advanced OCR on {filename}...")
        
        # Call your engine's process_file method
        # We enable medical_context=True as per your class definition
        extracted_text = ocr_engine.process_file(
            pdf_path, 
            use_preprocessing=True, 
            enhancement_level="medium", 
            medical_context=True
        )
        
        # Save to .txt for verification/debugging
        txt_filename = filename.replace('.pdf', '.txt')
        txt_path = os.path.join(os.path.dirname(pdf_path), txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
            
        logger.info(f"✅ [OCR] Text extracted and saved to {txt_filename} ({len(extracted_text)} chars)")
        return txt_path, extracted_text
        
    except Exception as e:
        logger.error(f"❌ [OCR] Failed: {e}", exc_info=True)
        return None, None

def process_kg_from_text(text_content):
    """
    Feeds the OCR'd text into the Camel Agent
    """
    try:
        logger.info(f"🕸️ [KG] Structuring text for Knowledge Graph...")
        
        # Wrap the text in a Camel/Unstructured Text Element
        element = Text(text_content)
        element.metadata.filename = "ocr_extracted.txt"
        
        # Run the Agent
        logger.info(f"🕸️ [KG] Extracting Nodes & Relationships ...")
        graph_elements = kg_agent.run(element, parse_graph_elements=True)
        
        nodes = len(graph_elements.nodes)
        rels = len(graph_elements.relationships)
        
        # Store in Neo4j
        logger.info(f"🕸️ [KG] Storing {nodes} nodes and {rels} relationships in Neo4j...")
        neo4j_db.add_graph_elements(graph_elements=[graph_elements])
        
        return True, f"Success: {nodes} Nodes, {rels} Relationships extracted."
        
    except Exception as e:
        logger.error(f"❌ [KG] Error: {e}", exc_info=True)
        return False, str(e)


# ---------------------------------------------------------------------------
# 4. FLASK ROUTES
# ---------------------------------------------------------------------------
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
    
    try:
        # 1. Save PDF locally
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        logger.info(f"💾 PDF saved: {filename}")

        # 2. Run YOUR Advanced OCR
        txt_path, text_content = run_advanced_ocr(pdf_path)
        
        if not text_content:
             return jsonify({'error': 'OCR failed to extract text'}), 500

        # 3. Index Original PDF in File Search (For RAG)
        logger.info("🔍 [Search] Indexing PDF...")
        title, doc_id = fs_engine.extract_metadata(pdf_path)
        fs_engine.upload_document(pdf_path, title, doc_id, filename)
        search_msg = "Indexed in Google File Search."

        # 4. Run KG Agent on the Extracted Text
        logger.info("🕸️ [KG] Processing Graph...")
        kg_success, kg_msg = process_kg_from_text(text_content)
        
        # Cleanup
        if os.path.exists(pdf_path): os.remove(pdf_path)
        if os.path.exists(txt_path): os.remove(txt_path)
        logger.info("🧹 Cleanup complete.")

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

# Copy-paste the rest of your routes (/query, /documents, etc.) from previous main.py
# They remain exactly the same.
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question', '')
    logger.info(f"❓ Query: {question}")
    try:
        answer, citations = fs_engine.query(question)
        return jsonify({'answer': answer, 'citations': citations})
    except Exception as e:
        logger.error(f"❌ Query Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_docs():
    return jsonify({'documents': fs_engine.list_documents()})

@app.route('/delete/<path:doc_name>', methods=['DELETE'])
def delete_doc(doc_name):
    try:
        fs_engine.delete_document(doc_name)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)