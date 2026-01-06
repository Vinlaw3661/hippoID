from src.hippoID.engine.hippo import Hippo
from src.hippoID.utils.processing import video_stream

def main():
    try:
        hippo: Hippo = Hippo(verbose=True)
        video_stream(hippo=hippo, print_fn=hippo.print)
    except Exception as e:
        print(f"An error occurred while running the video stream: {e}")

if __name__ == "__main__":
    main()