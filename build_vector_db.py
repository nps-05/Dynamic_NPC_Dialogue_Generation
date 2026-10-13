"""
Build ChromaDB Vector Store from CSV dialogue files
Creates embeddings and stores them in ChromaDB for semantic search
"""
import os
import pandas as pd
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dialogue_processor import infer_retrieval_labels


def create_vector_db(
    csv_path: str,
    db_path: str,
    collection_name: str,
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """
    Create ChromaDB vector store from CSV dialogue file
    
    Args:
        csv_path: Path to CSV file with dialogue data
        db_path: Path where ChromaDB will be stored
        collection_name: Name for the ChromaDB collection
        embedding_model: Sentence transformer model for embeddings
    """
    print(f"Building vector database from: {csv_path}")
    print(f"Database path: {db_path}")
    print(f"Collection name: {collection_name}")
    print(f"Embedding model: {embedding_model}")
    print("-" * 60)
    
    # Load CSV data
    print("\n1. Loading CSV data...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} dialogue entries")
    
    # Initialize embedding model
    print(f"\n2. Loading embedding model ({embedding_model})...")
    embedder = SentenceTransformer(embedding_model)
    print("   Model loaded")
    
    # Initialize ChromaDB client
    print(f"\n3. Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)
    
    # Delete collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"   Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": f"Dialogue embeddings for {collection_name}"}
    )
    print(f"   Created collection: {collection_name}")
    
    # Process dialogues and create embeddings
    print(f"\n4. Creating embeddings and storing in ChromaDB...")
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing dialogues"):
        # Get response text
        response_text = str(row.get('RESPONSE TEXT', '')).strip()
        
        # Skip empty responses
        if not response_text or response_text == 'nan':
            continue
        
        category = str(row.get('CATEGORY', '')).lower()
        dialogue_type = str(row.get('TYPE', '')).lower()
        labels = infer_retrieval_labels(row.to_dict())
        context = labels['context']
        emotion = labels['emotion']
        
        metadata = {
            'context': context,
            'emotion': emotion,
            'category': category if category != 'nan' else 'unknown',
            'type': dialogue_type if dialogue_type != 'nan' else 'unknown',
            'scene': str(row.get('SCENE', 'unknown')).lower() if pd.notna(row.get('SCENE')) else 'unknown',
        }
        
        # Clean up metadata
        for key, value in metadata.items():
            if value == 'nan' or value == '':
                metadata[key] = 'unknown'
        
        # Store dialogue
        documents.append(response_text)
        metadatas.append(metadata)
        ids.append(f"{collection_name}_{idx}")
    
    print(f"\n5. Generating embeddings for {len(documents)} dialogues...")
    
    # Generate embeddings in batches
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Generating embeddings"):
        batch = documents[i:i + batch_size]
        batch_embeddings = embedder.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings.tolist())
    
    print(f"\n6. Storing embeddings in ChromaDB...")
    
    # Add to ChromaDB in batches
    for i in tqdm(range(0, len(documents), batch_size), desc="Storing in ChromaDB"):
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
            embeddings=batch_embeddings
        )
    
    print(f"\n✅ Vector database created successfully!")
    print(f"   Collection: {collection_name}")
    print(f"   Documents: {collection.count()}")
    print(f"   Database path: {db_path}")


def main():
    """Main function to build vector database"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ChromaDB vector store from CSV dialogue file")
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with dialogue data'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path where ChromaDB will be stored (default: chroma_db_<character_name>)'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default=None,
        help='Name for ChromaDB collection (default: character name from CSV path)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    # Determine collection name from CSV filename if not provided
    if args.collection is None:
        from characters import character_id_from_filename
        collection_name = character_id_from_filename(args.csv)
    else:
        collection_name = args.collection
    
    # Determine DB path if not provided
    if args.db_path is None:
        args.db_path = f"chroma_db_{collection_name}"
    
    # Create vector database
    create_vector_db(
        csv_path=args.csv,
        db_path=args.db_path,
        collection_name=collection_name,
        embedding_model=args.model
    )


if __name__ == "__main__":
    main()

