from hippo_id.io.constants import (AUDIO_RECORDING_FILE_NAME, AUDIO_SAMPLING_FREQUENCY, AUDIO_RECORDING_SAVE_DIRECTORY, AUDIO_RECORDING_DURATIONS_IN_SECONDS, AUDIO_ASK_NAME_RESPONSE_DIRECTORY,
                                   VIDEO_STREAMING_SAVE_DIRECTORY, VIDEO_RECORDING_FILE_NAME, VIDEO_RECORDING_SAVE_DIRECTORY, VIDEO_RECORDING_DURATIONS_IN_SECONDS, 
                                   IMAGE_CAPTURE_FILE_NAME, IMAGE_CAPTURE_SAVE_DIRECTORY, IMAGE_FACE_DETECTION_SAVE_DIRECTORY, VIDEO_FRAMES_PER_SECOND, VIDEO_FOUR_CHARACTER_CODEC_CODE, VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT  )
from hippo_id.utils.processing import ensure_directory_exists
from hippo_id.io.decorators import DecoratedUtils
import sounddevice as sd
import cv2
from scipy.io.wavfile import write
from enum import Enum
from dataclasses import dataclass


def record_audio(duration: int = AUDIO_RECORDING_DURATIONS_IN_SECONDS['default'], file_name: str = AUDIO_RECORDING_FILE_NAME, save_directory: str = AUDIO_RECORDING_SAVE_DIRECTORY) -> str:
    ensure_directory_exists(save_directory)
    audio = sd.rec(int(duration * AUDIO_SAMPLING_FREQUENCY), samplerate=AUDIO_SAMPLING_FREQUENCY, channels=1, dtype='int16')
    sd.wait()
    file_path = f"{save_directory}/{file_name}"
    write(file_path, AUDIO_SAMPLING_FREQUENCY, audio)
    return file_path

def capture_image(file_name: str = IMAGE_CAPTURE_FILE_NAME, save_directory: str = IMAGE_CAPTURE_SAVE_DIRECTORY) -> str:
    ensure_directory_exists(save_directory)
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if ret:
        file_path = f"{save_directory}/{file_name}"
        cv2.imwrite(file_path, frame)
        camera.release()
        cv2.destroyAllWindows()
        return file_path
    else:
        camera.release()
        cv2.destroyAllWindows()
        raise Exception("Failed to capture image")

def record_video(file_name: str = VIDEO_RECORDING_FILE_NAME, save_directory: str = VIDEO_RECORDING_SAVE_DIRECTORY) -> str:
    ensure_directory_exists(save_directory)
    camera = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOUR_CHARACTER_CODEC_CODE)
    out = cv2.VideoWriter(f"{save_directory}/{file_name}", fourcc, VIDEO_FRAMES_PER_SECOND, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))

    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return f"{save_directory}/{file_name}"
        else:
            raise Exception("Failed to record video")
    camera.release()
    out.release()
    cv2.destroyAllWindows()

def stream_video() -> None:
    ensure_directory_exists(VIDEO_STREAMING_SAVE_DIRECTORY)
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if ret:
            cv2.imshow("Video Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            raise Exception("Failed to stream video")
    
    camera.release()
    cv2.destroyAllWindows()

@dataclass
class IOUtils:
    record_audio: DecoratedUtils = DecoratedUtils(record_audio)
    capture_image: DecoratedUtils = DecoratedUtils(capture_image)
    record_video: DecoratedUtils = DecoratedUtils(record_video)
    stream_video: DecoratedUtils = DecoratedUtils(stream_video)





