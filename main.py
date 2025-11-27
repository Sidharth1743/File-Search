from google import genai
from google.genai import types
import time
import os
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
from google import genai
from google.genai import types
import time
import os
import json

# Initialize client
client = genai.Client()

# Create the file search store
file_search_store = client.file_search_stores.create(config={'display_name': 'abstract-2'})
print(f"File search store created: {file_search_store.name}\n")

# Chunking configuration
chunking_config = {
    'chunking_config': {
        'white_space_config': {
            'max_tokens_per_chunk': 512,
            'max_overlap_tokens': 100
        }
    }
}

def extract_metadata(file_path):
    """Extract title and ID from PDF using Gemini"""
    print("Extracting metadata from PDF...")
    
    # Upload file to Gemini for analysis
    uploaded_file = client.files.upload(file=file_path)
    
    # Wait for file to be processed
    print("Processing file...")
    while uploaded_file.state == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    
    if uploaded_file.state == "FAILED":
        raise ValueError(f"File processing failed: {uploaded_file.error}")
    
    # Prompt to extract metadata
    prompt = """
    Analyze this PDF document and extract the following information:
    1. Title of the document/paper/study
    2. Any ID present (it could be labeled as Control ID, Abstract ID, Problem Statement ID, Paper ID, Study ID, or any similar identifier)
    
    Return your response ONLY as a valid JSON object in this exact format:
    {"title": "extracted title here", "id": "extracted id here"}
    
    If you cannot find the title, use the first heading or main topic.
    If you cannot find an ID, use "N/A".
    Do not include any other text, just the JSON.
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
            prompt
        ]
    )
    
    # Parse the JSON response
    try:
        # Clean the response text
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        metadata = json.loads(response_text)
        title = metadata.get("title", "Unknown Title")
        doc_id = metadata.get("id", "N/A")
        
        print(f"✓ Extracted - Title: {title}")
        print(f"✓ Extracted - ID: {doc_id}")
        
        # Delete the temporary file after extraction
        client.files.delete(name=uploaded_file.name)
        
        return title, doc_id
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        print(f"Response received: {response.text}")
        # Clean up file
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
        # Fallback to filename
        return os.path.basename(file_path).replace(".pdf", ""), "N/A"

def upload_doc(file_path):
    """Upload a document to the file search store with metadata"""
    file_name = os.path.basename(file_path)
    
    # Extract metadata using Gemini
    title, doc_id = extract_metadata(file_path)
    
    print(f"\nUploading to file search store...")
    
    # Import the file into the file search store with metadata
    operation = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=file_search_store.name,
        file=file_path,
        config={
            'display_name': title,
            'chunking_config': chunking_config["chunking_config"],
            'custom_metadata': [
                {"key": "title", "string_value": title},
                {"key": "ID", "string_value": doc_id},
                {"key": "file_name", "string_value": file_name},
            ],
        }
    )
    
    # Wait until import is complete
    while not operation.done:
        time.sleep(5)
        print("Indexing...")
        operation = client.operations.get(operation)
    
    print(f"✓ {file_name} is uploaded and indexed with metadata\n")

def query_documents(prompt):
    """Query the uploaded documents"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""{prompt}\n(return your answer in markdown as concise bullet points)\nANSWER:\n""",
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
    return response.text

# Main loop for uploading PDFs
print("=" * 60)
print("PDF UPLOAD AND QUERY SYSTEM WITH AUTO-METADATA EXTRACTION")
print("=" * 60)

uploaded_files = []

while True:
    file_path = input("\nEnter the PDF file path (or press Enter to finish uploading): ").strip()
    
    if not file_path:
        if len(uploaded_files) == 0:
            print("No files uploaded. Please upload at least one PDF.")
            continue
        else:
            break
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        continue
    
    # Check if it's a PDF
    if not file_path.lower().endswith('.pdf'):
        print("Error: Please upload a PDF file")
        continue
    
    # Upload the document with metadata extraction
    try:
        print("\n" + "─" * 60)
        upload_doc(file_path)
        uploaded_files.append(file_path)
        print(f"Total files uploaded: {len(uploaded_files)}")
        print("─" * 60)
    except Exception as e:
        print(f"Error uploading file: {e}")
        continue
    
    # Ask if user wants to upload another file
    another = input("\nDo you want to upload another PDF? (yes/no): ").strip().lower()
    if another not in ['yes', 'y']:
        break

# Query section
print("\n" + "=" * 60)
print(f"All files uploaded! Total: {len(uploaded_files)} PDFs")
print("=" * 60)

while True:
    print("\n" + "─" * 60)
    prompt = input("\nEnter your question (or 'quit' to exit): ").strip()
    
    if prompt.lower() in ['quit', 'exit', 'q']:
        print("Exiting...")
        break
    
    if not prompt:
        print("Please enter a valid question")
        continue
    
    print("\nSearching documents...\n")
    try:
        result = query_documents(prompt)
        print("ANSWER:")
        print("─" * 60)
        print(result)
        print("─" * 60)
    except Exception as e:
        print(f"Error querying documents: {e}")
    
    # Ask if user wants to ask another question
    another_q = input("\nDo you want to ask another question? (yes/no): ").strip().lower()
    if another_q not in ['yes', 'y']:
        print("Exiting...")
        break

print("\nThank you for using the PDF Query System!")