# src/hippo_id/io/constants.py


# IO hyperparameters
AUDIO_RECORDING_DURATIONS_IN_SECONDS = {
    "default": 5, 
    "short": 2,   
    "long": 10,  
}  

VIDEO_RECORDING_DURATIONS_IN_SECONDS = {
    "default": 10, 
    "short": 5,   
    "long": 30,  
}

AUDIO_SAMPLING_FREQUENCY = 44100

VIDEO_FRAMES_PER_SECOND = 30.0

VIDEO_FOUR_CHARACTER_CODEC_CODE = 'XVID'

VIDEO_FRAME_WIDTH = 640

VIDEO_FRAME_HEIGHT = 480

# Constants for audio recording
AUDIO_RECORDING_FILE_FORMAT = "wav"

AUDIO_RECORDING_FILE_NAME = "recording" + '.' + AUDIO_RECORDING_FILE_FORMAT

AUDIO_RECORDING_SAVE_DIRECTORY = "runtime/audio"

AUDIO_ASK_NAME_RESPONSE_DIRECTORY = AUDIO_RECORDING_SAVE_DIRECTORY + "/ask_name_responses"

# Constants for image capture
IMAGE_CAPTURE_FILE_FORMAT = "png"

IMAGE_CAPTURE_FILE_NAME = "captured_image" + '.' + IMAGE_CAPTURE_FILE_FORMAT

IMAGE_CAPTURE_SAVE_DIRECTORY = "runtime/images"

IMAGE_FACE_DETECTION_SAVE_DIRECTORY = "runtime/images/detected_faces"

# Constants for video recording
VIDEO_RECORDING_FILE_FORMAT = "avi"

VIDEO_RECORDING_FILE_NAME = "recorded_video" + '.' + VIDEO_RECORDING_FILE_FORMAT

VIDEO_RECORDING_SAVE_DIRECTORY = "runtime/videos"

VIDEO_STREAMING_SAVE_DIRECTORY = "runtime/streaming_videos"