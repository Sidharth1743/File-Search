# Google's File Search + SpineDAO HeritageNEt

> Advanced OCR, Vector Search & Knowledge Graph Extraction for Historical Medical Documents

## 🚀 Features

### Core Capabilities
- **Advanced OCR Engine**: Gemini AI-powered text extraction with image preprocessing
- **Vector Search**: Google File Search for semantic document retrieval
- **Knowledge Graph**: CAMEL-AI + Neo4j for entity extraction and relationship mapping
- **RAG Support**: Retrieval Augmented Generation for contextual answers


## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Flask Backend  │    │   Neo4j Graph   │
│  (Drag & Drop)  │◄──►│   (Processing)  │◄──►│  (Relationships)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   OCR Engine    │
                    │ (Gemini Vision) │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ File Search API │
                    │  (Vector Store) │
                    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- Google Cloud Platform account (for Gemini AI)
- Neo4j database 
- Groq account

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd file-search
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   # Google AI Services
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Neo4j Database
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   
   # Groq (for Llama model)
   GROQ_API_KEY=your_groq_api_key
   
   # Optional
   PORT=5000
   ```

## 🚀 Quick Start

1. **Run the application**
   ```bash
   uv run main.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

## 📚 Usage

### Web Interface

1. **Upload Documents**: Drag and drop PDF files onto the upload area
2. **Ask Questions**: Use the chat interface to query processed documents
3. **View Library**: Monitor processed documents in the sidebar
4. **Delete Documents**: Remove documents from the search index

### OCR Engine Options
- **Enhancement Levels**: `light`, `medium`, `aggressive`
- **DPI Settings**: 200 or 300 for scan quality
- **Medical Context**: Specialized prompts for medical terminology
- **Preprocessing**: Image enhancement and deskewing

### File Search Settings
- **Chunk Size**: 512 tokens per chunk
- **Overlap**: 50 tokens between chunks
- **Model**: Gemini-2.0-flash for search and responses

### Knowledge Graph Schema
- **Node Types**: 
  - `ClinicalObservation`
  - `TherapeuticOutcome`
  - `ContextualFactor`
  - `MechanisticConcept`
  - `TherapeuticApproach`
  - `SourceText`

- **Relationship Types**:
  - `co_occurs_with`
  - `preceded_by/followed_by`
  - `modified_by`
  - `responds_to`
  - `associated_with`
  - `results_in`
  - `described_in`
  - `contradicts/corroborates`

## 📁 Project Structure

```
├── main.py                 # Flask application entry point
├── file_search.py          # Vector search engine (Google AI)
├── ocr_engine.py           # Advanced OCR with Gemini Vision
├── kg_agents.py            # Knowledge graph extraction agent
├── templates/
│   └── index.html          # Web interface
├── uploads/                # Temporary file storage
├── debug_images/           # OCR preprocessing debug output
├── pyproject.toml          # Python dependencies
└── Procfile               # Heroku deployment config
```
