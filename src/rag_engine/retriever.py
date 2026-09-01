import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
logger = logging.getLogger(__name__)

class StrategyRetriever:
    def __init__(self):
        self.chroma_path = os.path.join(os.getcwd(), "data", "chromadb")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load the existing database
        self.vector_store = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        
    def get_trading_rules(self, query: str, k: int = 3) -> str:
        """
        Searches the trading books for rules matching the query.
        Returns a concatenated string of the most relevant text chunks.
        """
        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            return "No specific rules found in the trading literature for this setup."
        
        # Combine the retrieved text chunks
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        return context

if __name__ == "__main__":
    # Quick test execution
    retriever = StrategyRetriever()
    test_query = "What are the confirmation rules for a Bullish Engulfing pattern?"
    print(f"Querying: {test_query}\n")
    
    answer = retriever.get_trading_rules(test_query)
    print("Retrieved Context:\n")
    print(answer)