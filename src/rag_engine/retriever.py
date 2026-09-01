import os
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class StrategyRetriever:
    def __init__(self):
        self.chroma_path = os.path.join(os.getcwd(), "data", "chromadb")
        # Must match the model used in indexer.py exactly
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        self.vector_store = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        
    def get_trading_rules(self, query: str, k: int = 3) -> str:
        """
        Searches the local ChromaDB for rules matching the query.
        """
        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            return "No specific rules found in the trading literature for this setup."
        
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        return context

if __name__ == "__main__":
    retriever = StrategyRetriever()
    test_query = "What are the rules for a Cup with Handle pattern breakout?"
    print(f"Querying local DB for: {test_query}\n")
    
    answer = retriever.get_trading_rules(test_query)
    print("Retrieved Context:\n")
    print(answer)