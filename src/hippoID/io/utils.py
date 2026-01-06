from hippoID.io.constants import (
    AudioRecordingDurationsInSeconds, AudioRecordingSettings, 
    AudioRecordingFileNames, VideoRecordingFileNames,
    VideoRecordingSettings, ImageCaptureFileNames
)
from hippoID.utils.processing import ensure_directory_exists
from hippoID.io.decorators import DecoratedUtils
import sounddevice as sd
import cv2
from scipy.io.wavfile import write
from dataclasses import dataclass


def record_audio(duration: int = AudioRecordingDurationsInSeconds.DEFAULT.value, file_name: str = AudioRecordingFileNames.FILE_NAME.value, save_directory: str = AudioRecordingFileNames.SAVE_DIRECTORY.value) -> str:
    ensure_directory_exists(save_directory)
    audio = sd.rec(int(duration * AudioRecordingSettings.SAMPLING_FREQUENCY.value), samplerate=AudioRecordingSettings.SAMPLING_FREQUENCY.value, channels=1, dtype='int16')
    sd.wait()
    file_path = f"{save_directory}/{file_name}"
    write(file_path, AudioRecordingSettings.SAMPLING_FREQUENCY.value, audio)
    return file_path

def capture_image(file_name: str = ImageCaptureFileNames.FILE_NAME.value, save_directory: str = ImageCaptureFileNames.SAVE_DIRECTORY.value) -> str:
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

def record_video(file_name: str = VideoRecordingFileNames.FILE_NAME.value, save_directory: str = VideoRecordingFileNames.SAVE_DIRECTORY.value) -> str:
    ensure_directory_exists(save_directory)
    camera = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*VideoRecordingSettings.FOUR_CHARACTER_CODEC_CODE.value)
    out = cv2.VideoWriter(f"{save_directory}/{file_name}", fourcc, VideoRecordingSettings.FRAMES_PER_SECOND.value, (VideoRecordingSettings.FRAME_WIDTH.value, VideoRecordingSettings.FRAME_HEIGHT.value))
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
    ensure_directory_exists(VideoRecordingFileNames.SAVE_DIRECTORY.value)
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





