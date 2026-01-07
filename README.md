# hippoID

An AI-powered facial recognition system that identifies people through webcam video streams and learns new faces through interactive voice conversations.

## Features

- **Real-time Face Detection**: Uses MediaPipe for live face detection from webcam feed
- **Face Recognition**: Employs DeepFace with FaceNet embeddings for accurate face matching
- **Interactive Learning**: When encountering unknown faces, the system:
  - Generates a physical description using vision AI
  - Asks the person for their name via text-to-speech
  - Records and transcribes the audio response
  - Stores the face embedding with the person's name
- **Vector Database**: Uses ChromaDB for efficient face embedding storage and similarity search
- **Multi-modal AI**: Integrates speech-to-text, text-to-speech, and vision models

## System Dependencies

### macOS

```bash
brew install mpv
```

### Linux

```bash
# Ubuntu/Debian
sudo apt install mpv

# Fedora
sudo dnf install mpv
```

### Windows

Download and install from https://mpv.io/

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Vinlaw3661/hippoID
cd hippoID
```

2. Install Python dependencies:

```bash
pip install -e .
```

3. Set up environment variables in `.env`:

```bash
ELEVENLABS_API_KEY=your_elevenlabs_key
ASSEMBLYAI_API_KEY=your_assemblyai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

Run the main application:

```bash
python main.py
```

The system will:

1. Open your webcam feed
2. Display real-time video with FPS counter
3. Automatically detect and identify faces
4. For unknown faces, initiate an interactive learning session
5. Store learned identities for future recognition

Press 'q' to quit the application.

## Architecture

- **Engine**: Core facial recognition and interaction logic
- **Models**: AI model integrations (LLM, TTS, STT, Vision)
- **Memory**: Vector database for face embeddings
- **IO**: Audio/video recording and file management utilities

## AI Models Used

- **Vision**: MediaPipe (face detection), DeepFace/FaceNet (face recognition)
- **Language**: Anthropic Claude (description generation)
- **Speech**: ElevenLabs (TTS), AssemblyAI (STT)

## Author

Vinlaw Mudehwe [me@vinlawmudehwe.com](mailto:me@vinlawmudehwe.com)
