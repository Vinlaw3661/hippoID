import assemblyai as aai
from elevenlabs import VoiceSettings, stream
from src.hippoID.models.language.llms import DefaultLLM
from src.hippoID.models.language.schemas import PersonName
from src.hippoID.io.utils import record_audio
from src.hippoID.models.speech.stt import DefaultSTT
from src.hippoID.models.speech.tts import DefaultTTS
from src.hippoID.models.language.llms import DefaultLLM
from src.hippoID.models.language.prompts import PromptBuilder
from src.hippoID.engine.constants import ElevenLabsConfig
from src.hippoID.io.constants import AudioRecordingFileNames

class InteractionEngine:
    def __init__(
            self, 
            tts_model = DefaultTTS.elevenlabs, 
            llm_model = DefaultLLM.claude, 
            stt_model = DefaultSTT.assemblyai,
            prompt_builder: PromptBuilder = PromptBuilder()
            ):
        self.tts = tts_model
        self.llm = llm_model
        self.stt = stt_model
        self.prompt_builder = prompt_builder
    
    def describe_person(self, image_path:str) -> str:
        physical_description_prompt = self.prompt_builder.physical_description(image_path)
        response = self.llm.invoke(physical_description_prompt).content
        return response

    def ask_for_name(self, description: str) -> bool:

        # There is repeated code here for TTS. I will refactor later into a separate class
        # with common TTS functionality and finegrained control over voice settings.
        audio_stream = self.tts.text_to_speech.stream(
            text=description,
            voice_id=ElevenLabsConfig.VOICE_ID.value,
            model_id=ElevenLabsConfig.TTS_MODEL_ID.value,
            voice_settings=VoiceSettings(
                stability=ElevenLabsConfig.STABILITY.value,
                similarity_boost=ElevenLabsConfig.SIMILARITY_BOOST.value,
                use_speaker_boost=ElevenLabsConfig.USE_SPEAKER_BOOST.value
            )
        )
        stream(audio_stream)
        return True 

    def listen_for_name(self, audio_path: str = AudioRecordingFileNames.FILE_NAME.value, save_directory: str = AudioRecordingFileNames.SAVE_DIRECTORY.value) -> str:
        structured_llm = self.llm.with_structured_output(PersonName)
    
        FILE_URL = record_audio(file_name=audio_path, save_directory=save_directory)
        transcript = self.stt.transcribe(FILE_URL)
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Unable to transcribe audio: {transcript.error}")
        else:
            prompt = self.prompt_builder.name_extraction(transcript.text)
            name = structured_llm.invoke(prompt).name
            return name.lower()

    def acknowledge_person(self, name:str) -> None:
        acknowledgment_prompt = self.prompt_builder.acknowledge_name(name)   
        acknowledgment_text = self.llm.invoke(acknowledgment_prompt).content

        # There is repeated code here for TTS. I will refactor later into a separate class
        # with common TTS functionality and finegrained control over voice settings.
        audio_stream = self.tts.text_to_speech.stream(
            text=acknowledgment_text,
            voice_id=ElevenLabsConfig.VOICE_ID.value,
            model_id=ElevenLabsConfig.TTS_MODEL_ID.value,
            voice_settings=VoiceSettings(
                stability=ElevenLabsConfig.STABILITY.value,
                similarity_boost=ElevenLabsConfig.SIMILARITY_BOOST.value,
                use_speaker_boost=ElevenLabsConfig.USE_SPEAKER_BOOST.value
            )
        )
        stream(audio_stream)
        return 

 

