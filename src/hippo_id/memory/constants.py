from enum import Enum 
import os 
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class IndexNames():
    FACES = "Faces"
    VOICES = "Voices"

class APIKeys:
    PINECONE = PINECONE_API_KEY
