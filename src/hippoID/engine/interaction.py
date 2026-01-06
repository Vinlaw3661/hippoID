import base64
import os
import assemblyai as aai
from elevenlabs.client import VoiceSettings, stream
from hippoID.models.language.llms import DefaultLLM
from hippoID.models.language.schemas import PersonName
from hippoID.io.utils import record_audio
from hippoID.models.speech.stt import DefaultSTT
from hippoID.models.speech.tts import DefaultTTS
from hippoID.models.language.llms import DefaultLLM
from langchain_core.messages import HumanMessage

class InteractionEngine:
    def __init__(self, tts_model = DefaultTTS.elevenlabs, llm_model = DefaultLLM.claude, stt_model = DefaultSTT.assemblyai):
        self.tts = tts_model
        self.llm = llm_model
        self.stt = stt_model
    
    def describe_person(self, image_path:str) -> str:

        if not os.path.exists(image_path):
            raise FileNotFoundError("Image not found")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = '''Describe what is in the image in a way that a person could understand. Do not include a description of the black background. Frame your response as a question 
        asking who they are. Here are some examples:
        
        Who is the person with the black hair and hazel eyes?

        Who is the person with the blonde hair, blue eyes, and green hoop earrings?

        NOTE: Respond only with the question and nothing else. Do not add any additional text to your response.
        
        '''

        message = HumanMessage(
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url":f"data:image/png;base64,{image_data}"}}
            ]
        )

        response = "Hello! " + self.llm.invoke([message]).content

        return response

    def ask_for_name(self, person_image_path: str) -> bool:
        description = self.describe_person(person_image_path)

        audio_stream = self.tts.text_to_speech.convert_as_stream(
            text=description,
            voice_id="pqHfZKP75CvOlQylNhV4",
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                use_speaker_boost=True
            )
        )
        stream(audio_stream)
        return True 

    def listen_for_name(self, audio_path: str = "audio.wav", save_directory: str = "./outputs/audio") -> str:
        structured_llm = self.llm.with_structured_output(PersonName)
    
        FILE_URL = record_audio(file_name=audio_path, save_directory=save_directory)
        transcript = self.stt.transcribe(FILE_URL)
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Unable to transcribe audio: {transcript.error}")
        else:
            prompt = "Extract the person's name from the following text: " + transcript.text
            name = structured_llm.invoke(prompt).name
            return name.lower()

    def acknowledge_person(self, name:str) -> None:

        text = f"Thank you for introducing me to {name}! It's a pleasure to meet you {name}. I will make sure to remember you the next time we meet."
        audio_stream = self.tts.text_to_speech.convert_as_stream(
            text=text,
            voice_id="pqHfZKP75CvOlQylNhV4",
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                use_speaker_boost=True
            )
        )

        stream(audio_stream)
        return 

 

