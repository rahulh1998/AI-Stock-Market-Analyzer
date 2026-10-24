import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class StrategyRetriever:
    def __init__(self):
        from src.paths import CHROMA_DIR
        self.chroma_path = str(CHROMA_DIR)
        # Must match the model used in indexer.py exactly
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        self.vector_store = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        
    def retrieve(self, query: str, k: int = 3) -> list:
        """Structured passages with their source book, cached on disk.

        Embedding a query through Ollama takes about 20 seconds on this hardware, which is
        far too slow to repeat per rerun, and the book corpus never changes - so a hit is
        permanent rather than time-limited.
        """
        import hashlib, json, sqlite3
        from src.paths import DB_PATH

        key = hashlib.sha1(f'{query}|{k}'.encode('utf-8')).hexdigest()
        conn = None
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
            conn.execute('CREATE TABLE IF NOT EXISTS rag_cache ('
                         'cache_key TEXT PRIMARY KEY, query TEXT, passages_json TEXT)')
            row = conn.execute('SELECT passages_json FROM rag_cache WHERE cache_key = ?',
                               (key,)).fetchone()
            if row:
                return json.loads(row[0])
        except Exception as exc:
            logger.warning('RAG cache read failed: %s', exc)

        try:
            documents = self.vector_store.similarity_search(query, k=k)
        except Exception as exc:
            logger.warning('RAG retrieval failed: %s', exc)
            if conn is not None:
                conn.close()
            return []
        passages = [{'text': d.page_content,
                     'source': os.path.basename(str(d.metadata.get('source', 'unknown')))}
                    for d in documents]
        try:
            if conn is not None:
                conn.execute('INSERT OR REPLACE INTO rag_cache VALUES (?, ?, ?)',
                             (key, query, json.dumps(passages)))
                conn.commit()
        except Exception as exc:
            logger.warning('RAG cache write failed: %s', exc)
        finally:
            if conn is not None:
                conn.close()
        return passages

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