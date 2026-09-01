import os
import glob
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOOKS_DIR = os.path.join(os.getcwd(), "data", "books")
CHROMA_PATH = os.path.join(os.getcwd(), "data", "chromadb")

def build_vector_store():
    """Reads documents, creates local Ollama embeddings, and stores them in batches."""
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

    logger.info(f"Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    logger.info(f"Created {total_chunks} chunks.")

    logger.info("Initializing ChromaDB and Ollama Embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Initialize an empty Chroma vector store
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Process in safe batches of 50 to prevent Ollama from crashing
    batch_size = 50
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        current_batch_num = (i // batch_size) + 1
        total_batches = (total_chunks // batch_size) + 1
        
        logger.info(f"Processing batch {current_batch_num}/{total_batches} ({len(batch)} chunks)")
        
        # Add the small batch to the database
        vector_store.add_documents(documents=batch)
        
    logger.info(f"Successfully indexed all {total_chunks} chunks into {CHROMA_PATH}")

if __name__ == "__main__":
    build_vector_store()