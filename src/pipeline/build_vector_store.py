import sys
import io
import json
from pathlib import Path

# Windows encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def get_collection():
    """Returns the ChromaDB collection for the amazon support data."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="amazon_support")
    return collection

def main():
    print(f"Loading data from {PROCESSED_DATA}")
    with open(PROCESSED_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} records")
    
    print(f"Loading embedding model {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    collection = get_collection()
    
    batch_size = 100
    
    print("Embedding and storing in ChromaDB")
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        
        ids = []
        documents_to_embed = []
        plain_documents = []
        metadatas = []
        
        for item in batch:
            item_id = str(item['id'])
            ids.append(item_id)
            
            customer_text = item['customer_text']
            # Add prefix for bge-small-en-v1.5
            text_to_embed = f"Represent this sentence: {customer_text}"
            documents_to_embed.append(text_to_embed)
            plain_documents.append(customer_text)
            
            # Prepare metadata
            metadata = {
                'id': item_id,
                'customer_text': customer_text,
                'brand_reply': item.get('brand_reply', ''),
                'thread_context': json.dumps(item.get('thread_context', [])),
                'escalation_is_escalation': bool(item.get('escalation_signals', {}).get('is_escalation', False))
            }
            metadatas.append(metadata)
            
        # Compute embeddings in batch
        embeddings = model.encode(documents_to_embed).tolist()
        
        # Store in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=plain_documents
        )
        
    print("Vector store creation complete!")

if __name__ == "__main__":
    main()
