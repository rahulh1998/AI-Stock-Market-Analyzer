import os
import glob
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables (API keys)
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOOKS_DIR = os.path.join(os.getcwd(), "data", "books")
CHROMA_PATH = os.path.join(os.getcwd(), "data", "chromadb")

def build_vector_store():
    """Reads documents, creates embeddings, and stores them in ChromaDB."""
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found in .env file.")
        return

    # 1. Find all PDFs and TXTs in the books directory
    files = glob.glob(os.path.join(BOOKS_DIR, "*.pdf")) + glob.glob(os.path.join(BOOKS_DIR, "*.txt"))
    
    if not files:
        logger.warning(f"No documents found in {BOOKS_DIR}. Please add PDFs.")
        return

    documents = []
    for file_path in files:
        logger.info(f"Loading document: {os.path.basename(file_path)}")
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        documents.extend(loader.load())

    # 2. Split text into manageable chunks for the LLM
    logger.info(f"Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    # 3. Create Embeddings and Store in ChromaDB
    logger.info("Generating embeddings and saving to ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    logger.info(f"Successfully indexed documents into {CHROMA_PATH}")
    return vector_store

if __name__ == "__main__":
    build_vector_store()