import requests
import os
import pickle
import logging
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from tqdm import tqdm
from pypdf import PdfMerger

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients and models
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
MODEL_SERVER_URL = "http://localhost:8000"  # Replace with actual server URL

# Constants
FAISS_INDEX_FILE = "./faiss_index.pkl"
MASTER_PDF = "./OpenROAD_Master_Document.pdf"
OPENROAD_DOC = "./OpenROAD_Doc"

# Global variable for the vector store
vectorstore = None

# Initialise PDF_FILES list
PDF_FILES = []

# Functions for RAG system
def emb_text(text):
    return embedding_model.encode(text, normalize_embeddings=True)

# Prompt template
PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

def create_pdf_array(directory):
    # Iterate through the directory
    for filename in os.listdir(directory):
        # Check if the file is a PDF
        if filename.endswith('.pdf'):
            # Add the full path of the PDF file to the list
            PDF_FILES.append(os.path.join(directory, filename))

    # Print the list of PDF files
    print(PDF_FILES)

def get_rag_response(question, top_k=5):
    global vectorstore
    # Embed the question
    question_embedding = emb_text(question)
    # Use the embedding for similarity search
    docs = vectorstore.similarity_search_by_vector(question_embedding, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = PROMPT.format(context=context, question=question)
    
    # Send a request to the model server
    response = requests.post(
        f"{MODEL_SERVER_URL}/generate",
        json={"prompt": prompt, "max_new_tokens": 1000}
    )
    
    if response.status_code == 200:
        answer = response.json().get("generated_text", "").strip()
    else:
        answer = "Error: Unable to get response from model server."

    return answer

def save_index_to_hf_space(index_path):
    try:
        token = os.environ.get("HF_TOKEN")
        space_id = "connorgilchrist/RAGPOCMilvus"

        if not token:
            logger.error("HF_TOKEN environment variable is not set. Unable to authenticate.")
            return

        if not space_id:
            logger.error("SPACE_ID environment variable is not set. Unable to determine the target Space.")
            return

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=index_path,
            path_in_repo="faiss_index.pkl",
            repo_id=space_id,
            repo_type="space",
        )
        logger.info(f"Uploaded {index_path} to Hugging Face Space {space_id}")
    except Exception as e:
        logger.error(f"Failed to upload index to Hugging Face Space: {str(e)}")

def initialize_rag_system():
    global vectorstore
    try:
        if not os.path.exists(FAISS_INDEX_FILE):
            logger.info("FAISS index file not found. Creating new index.")
            # Remove old master pdf 
            if os.path.exists(MASTER_PDF):
                os.remove(MASTER_PDF)
            #Populate PDF_FILES
            create_pdf_array(OPENROAD_DOC)
                
            # Create master pdf doc
            merger = PdfMerger()

            for pdf in PDF_FILES:
                merger.append(pdf)

            merger.write(MASTER_PDF)
            merger.close()
            
            # Load and process documents
            loader = PyPDFLoader(MASTER_PDF)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            text_lines = [chunk.page_content for chunk in chunks]
            
            # Create embeddings and FAISS index
            embeddings = [emb_text(line) for line in tqdm(text_lines, desc="Creating embeddings")]
            vectorstore = FAISS.from_embeddings(list(zip(text_lines, embeddings)), embedding_model)
            
            # Save the index
            try:
                with open(FAISS_INDEX_FILE, "wb") as f:
                    pickle.dump(vectorstore, f)
                logger.info(f"RAG system initialized and saved to {FAISS_INDEX_FILE}")
                save_index_to_hf_space(FAISS_INDEX_FILE)
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {str(e)}")
        else:
            logger.info(f"Loading existing FAISS index from {FAISS_INDEX_FILE}")
            with open(FAISS_INDEX_FILE, "rb") as f:
                vectorstore = pickle.load(f)
            logger.info("RAG system loaded from disk.")
        if vectorstore is None:
            raise ValueError("vectorstore is None after initialization")
    except Exception as e:
        logger.error(f"Error in initialize_rag_system: {str(e)}", exc_info=True)
        raise

    # Print the current working directory and list its contents
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    logger.info(f"Contents of the working directory: {os.listdir(cwd)}")
    logger.info(f"Master PDF Path: {os.path.abspath(MASTER_PDF)}")

# Function for generating responses from the model server
def generate_response(prompt, max_new_tokens=1000):
    response = requests.post(
        f"{MODEL_SERVER_URL}/generate",
        json={"prompt": prompt, "max_new_tokens": max_new_tokens}
    )
    
    if response.status_code == 200:
        return response.json().get("generated_text", "").strip()
    else:
        return "Error: Unable to get response from model server."

# Example usage
if __name__ == "__main__":
    try:
        # Initialize RAG system
        initialize_rag_system()
        # Example call to generate a response
        example_question = "What is the purpose of the OpenROAD project?"
        answer = get_rag_response(example_question)
        print(f"Answer: {answer}")
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
