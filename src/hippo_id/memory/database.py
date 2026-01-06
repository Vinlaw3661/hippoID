import chromadb
from hippo_id.memory.constants import IndexNames, APIKeys

# Set up ChromaDB for persisting embeddings
chromadb_client = chromadb.PersistentClient()
chromadb_faces_collection = chromadb_client.get_or_create_collection(IndexNames.FACES)
chromadb_voices_collection = chromadb_client.get_or_create_collection(IndexNames.VOICES)

class LocalVectorCollection:
    voice: chromadb.Collection = chromadb_voices_collection
    face: chromadb.Collection = chromadb_faces_collection

class HostedVectorCollection:
    voice: chromadb.Collection = chromadb_voices_collection
    face: chromadb.Collection = chromadb_faces_collection

class DefaultDatabase:
    local: LocalVectorCollection =  LocalVectorCollection
    hosted: HostedVectorCollection = HostedVectorCollection

