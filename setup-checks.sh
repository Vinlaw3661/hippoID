# !/bin/bash
# This script verifiees that the set up of the runtime environment for the Hippo engine is correct.
# Script will be automatically run when the docker container starts.

llm_api_keys = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")
stt_api_keys = ("ASSEMBLYAI_API_KEY")
tts_api_keys = ("ELEVENLABS_API_KEY")

if [[ -n "$ASSEMBLYAI_API_KEY" ]]; then
    echo "at least one required SST API key is set"
else 
    echo "no SST API keys are set, please set at least one of the following: $stt_api_keys"
    exit 1

if [[ -n "$ELEVENLABS_API_KEY" ]]; then
    echo "at least one required TTS API key is set"
else 
    echo "no TTS API keys are set, please set at least one of the following: $tts_api_keys"
    exit 1

if [[ -n "$OPENAI_API_KEY" || -n "$GOOGLE_API_KEY" || -n "$ANTHROPIC_API_KEY" ]]; then
    echo "at least one required LLM  API key is set"
else 
    echo "no LLM API keys are set, please set at least one of the following: $llm_api_keys"
    exit 1

if [[-d runtime/audio && -d runtime/faces]]; then
    echo "runtime directories exist"
else 
    echo "creating runtime directories"
    mkdir -p runtime/audio runtime/faces
fi
    echo "runtime directories set up"

