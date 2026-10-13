"""
ChromaDB search for a character dialogue index
"""
import chromadb
from typing import List, Dict, Optional
import os


class DialogueVectorStore:
    """Interface to existing ChromaDB collection"""
    
    def __init__(self, db_path: str):
        """
        Initialize connection to existing ChromaDB
        
        Args:
            db_path: Path to chroma_db_nick_valentine folder
        """
        self.db_path = db_path
        
        print(f"🔄 Connecting to existing ChromaDB at {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get the collection (should already exist)
        collections = self.client.list_collections()
        if not collections:
            raise Exception("No collections found in ChromaDB!")
        
        self.collection = collections[0]  # Use first collection
        print(f"✅ Connected to collection: {self.collection.name} ({self.collection.count()} documents)")
    
    def semantic_search(
        self, 
        query: str, 
        n_results: int = 5,
        context_filter: Optional[str] = None,
        emotion_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform semantic search for relevant dialogue
        
        Args:
            query: User input or search query
            n_results: Number of results to return
            context_filter: Optional context filter
            emotion_filter: Optional emotion filter
            
        Returns:
            List of relevant dialogue examples with metadata
        """
        # Build metadata filter
        where_filter = {}
        if context_filter:
            where_filter['context'] = context_filter
        if emotion_filter:
            where_filter['emotion'] = emotion_filter
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter if where_filter else None
            )
            
            # Format results
            formatted_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'text': doc,
                        'context': results['metadatas'][0][i].get('context', 'casual') if results.get('metadatas') else 'casual',
                        'emotion': results['metadatas'][0][i].get('emotion', 'neutral') if results.get('metadatas') else 'neutral',
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_contextual_examples(
        self,
        user_input: str,
        context: str = "casual",
        emotion: Optional[str] = None,
        n_results: int = 3
    ) -> str:
        """
        Get contextual dialogue examples for prompt engineering
        
        Args:
            user_input: User's message
            context: Conversation context
            emotion: Desired emotion/tone
            n_results: Number of examples
            
        Returns:
            Formatted string of example dialogues
        """
        results = self.semantic_search(
            query=user_input,
            n_results=n_results,
            context_filter=context,
            emotion_filter=emotion
        )
        # A strict context/emotion filter often matches nothing on loosely labeled
        # data. Fall back to unfiltered semantic search instead of dropping examples.
        if not results and (context or emotion):
            results = self.semantic_search(
                query=user_input,
                n_results=n_results,
            )

        if not results:
            return ""
        
        # Format as examples
        examples = []
        for result in results:
            examples.append(f'"{result["text"]}"')
        
        return "\n".join(examples)


_stores = {}


def default_chroma_path() -> str:
    """Legacy single-character path. Prefer discover_characters()."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db_nick_valentine')


def open_vector_store(db_path: str) -> DialogueVectorStore:
    """Open a Chroma directory, reusing the client if that path is already open."""
    key = os.path.abspath(db_path)
    store = _stores.get(key)
    if store is None:
        store = DialogueVectorStore(db_path)
        _stores[key] = store
    return store


def get_vector_store(db_path: str = None) -> DialogueVectorStore:
    """Get a vector store. Each database path keeps its own client."""
    if db_path is None:
        db_path = default_chroma_path()
    return open_vector_store(db_path)
