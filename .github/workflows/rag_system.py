import gradio as gr
from huggingface_hub import InferenceClient
import os
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from tqdm import tqdm
import pickle
import logging
import huggingface_hub
from huggingface_hub import HfApi
from pypdf import PdfMerger

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients and models
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
llm_client = InferenceClient(model="mistralai/Mistral-Nemo-Instruct-2407", timeout=120)

# Constants
HF_TOKEN = "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"
FAISS_INDEX_FILE = ".github/workflows/faiss_index.pkl"
MASTER_PDF = ".github/workflows/OpenROAD_Doc/OpenROAD_Master_Document.pdf"
OPENROAD_DOC = ".github/workflows/OpenROAD_Doc"

# Global variable for the vector store
vectorstore = None

# Initialise PDF_FILES list
PDF_FILES = []

# Functions for RAG system
def emb_text(text):
    return embedding_model.encode(text, normalize_embeddings=True)

# Prompt template
PROMPT =  """
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
    
    answer = llm_client.text_generation(
        prompt,
        max_new_tokens=1000
    ).strip()
    
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

# Chatbot response function
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    top_k  # Add top_k here
):
    # First, get the RAG response with the specified top_k
    rag_response = get_rag_response(message, top_k)
    
    # Prepare messages for the chatbot
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # Add the current message and RAG response
    messages.append({"role": "user", "content": message})
    messages.append({"role": "assistant", "content": f"Based on the information I have: {rag_response}"})
    
    # Now, let the chatbot generate a response based on the conversation including the RAG response
    response = ""
    for message in llm_client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Gradio interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant. You are proficient in Actian 4GL OpenROAD.", label="System message"),
        gr.Slider(minimum=1, maximum=128000, value=30000, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.4, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
        gr.Slider(minimum=1, maximum=20, value=6, step=1, label="Top-k")  # Add top_k slider
    ],
)

if __name__ == "__main__":
    try:
        # Initialize RAG system
        initialize_rag_system()
        # Launch the demo
        demo.launch()
    except Exception as e:
        logger.error(f"Failed to launch the app: {str(e)}", exc_info=True)
