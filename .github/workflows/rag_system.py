import os
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from tqdm import tqdm
import pickle
import logging
from huggingface_hub import HfApi
from pypdf import PdfMerger

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Constants
HF_TOKEN = os.getenv("HF_TOKEN")
FAISS_INDEX_FILE = "./faiss_index.pkl"
MASTER_PDF = "./OpenROAD_Master_Document.pdf"
OPENROAD_DOC = "./OpenROAD_Doc"

# Global variable for the vector store
vectorstore = None

# Initialize PDF_FILES list
PDF_FILES = []

def emb_text(text):
    return embedding_model.encode(text, normalize_embeddings=True)

def create_pdf_array(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            PDF_FILES.append(os.path.join(directory, filename))
    logger.info(f"PDF files found: {PDF_FILES}")

def get_rag_response(question, top_k=5):
    global vectorstore
    question_embedding = emb_text(question)
    docs = vectorstore.similarity_search_by_vector(question_embedding, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    return context

def save_index_to_hf_space(index_path):
    try:
        token = os.environ.get("HF_TOKEN")
        space_id = "connorgilchrist/RAGPOCMilvus"

        if not token:
            logger.error("HF_TOKEN environment variable is not set. Unable to authenticate.")
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
            if os.path.exists(MASTER_PDF):
                os.remove(MASTER_PDF)
            create_pdf_array(OPENROAD_DOC)
            
            merger = PdfMerger()
            for pdf in PDF_FILES:
                merger.append(pdf)
            merger.write(MASTER_PDF)
            merger.close()
            
            loader = PyPDFLoader(MASTER_PDF)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            text_lines = [chunk.page_content for chunk in chunks]
            
            embeddings = [emb_text(line) for line in tqdm(text_lines, desc="Creating embeddings")]
            vectorstore = FAISS.from_embeddings(list(zip(text_lines, embeddings)), embedding_model)
            
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

    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    logger.info(f"Contents of the working directory: {os.listdir(cwd)}")
    logger.info(f"Master PDF Path: {os.path.abspath(MASTER_PDF)}")

if __name__ == "__main__":
    initialize_rag_system()
