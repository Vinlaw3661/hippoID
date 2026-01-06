from enum import StrEnum 
from dotenv import load_dotenv

load_dotenv()

class IndexNames(StrEnum):
    FACES = "Faces"
    VOICES = "Voices"
