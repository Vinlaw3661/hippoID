import os
import cv2
import time
from typing import Callable 
from src.hippoID.engine.hippo import Hippo

def do_not_print(text: str) -> None:
    return 

def verbose_print(verbose: bool = False) -> Callable[[str], None]:
    if verbose:
        return print
    return do_not_print

def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def video_stream(hippo: Hippo, print_fn: Callable[[str], None]) -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print_fn("Cannot open camera")
        return 
    
    start_time = time.time()
    frame_count = 0
    
    while True:

        ret, frame = cap.read()

        if not ret:
            print_fn("Could not read frame")
            break

        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        else:
            fps = 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)
        result, message = hippo.identify(frame, verbose=True)
        print_fn((result, message))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()