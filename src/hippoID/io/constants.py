from enum import Enum
import os

# IO hyperparameters
class AudioRecordingDurationsInSeconds(Enum):
    DEFAULT = 5  
    SHORT = 2    
    LONG = 10

class VideoRecordingDurationsInSeconds(Enum):
    DEFAULT = 10  
    SHORT = 5    
    LONG = 30
    
class AudioRecordingSettings(Enum):
    SAMPLING_FREQUENCY = 44100
    FILE_FORMAT = "wav"

class VideoRecordingSettings(Enum):
    FRAMES_PER_SECOND = 30.0
    FOUR_CHARACTER_CODEC_CODE = 'XVID'
    FILE_FORMAT = "avi"
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480 

class ImageCaptureSettings(Enum):
    FILE_FORMAT = "png"

# IO File and Directory Names   
class AudioRecordingFileNames(Enum):
    FILE_NAME = "audio" + '.' + AudioRecordingSettings.FILE_FORMAT.value
    SAVE_DIRECTORY = os.path.abspath("runtime/audio")

class VideoRecordingFileNames(Enum):
    FILE_NAME = "video" + '.' + VideoRecordingSettings.FILE_FORMAT.value
    SAVE_DIRECTORY = os.path.abspath("runtime/video")

class ImageCaptureFileNames(Enum):
    FILE_NAME = "image" + '.' +  ImageCaptureSettings.FILE_FORMAT.value
    MASKED_FILE_NAME = "masked_image" + '.' +  ImageCaptureSettings.FILE_FORMAT.value
    SAVE_DIRECTORY = os.path.abspath("runtime/image")
    FACE_DETECTION_SAVE_DIRECTORY = os.path.abspath("runtime/image/detected_faces")