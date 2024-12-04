# AI Home Assistant: A Comprehensive Guide

Welcome to your AI Home Assistant project! This README serves as a roadmap and detailed instruction manual to guide you through building a local AI assistant that can:

- Capture video and audio from your mobile phone.
- Analyze the video to understand ongoing activities (e.g., cooking).
- Provide context-aware suggestions and instructions.
- Convert text responses into speech.
- Accept voice commands as input.
- Run entirely on your local AI server with 2 x NVIDIA RTX 3060 12GB GPUs.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Step-by-Step Instructions](#step-by-step-instructions)
  - [1. Set Up the AI Server Environment](#1-set-up-the-ai-server-environment)
  - [2. Develop the Mobile Application](#2-develop-the-mobile-application)
  - [3. Establish Network Communication](#3-establish-network-communication)
  - [4. Receive and Process Video Stream on the Server](#4-receive-and-process-video-stream-on-the-server)
  - [5. Implement Computer Vision Processing](#5-implement-computer-vision-processing)
  - [6. Set Up the Language Model (LLM)](#6-set-up-the-language-model-llm)
  - [7. Integrate Text-to-Speech (TTS)](#7-integrate-text-to-speech-tts)
  - [8. Implement Speech-to-Text (STT)](#8-implement-speech-to-text-stt)
  - [9. Develop Server APIs for Communication](#9-develop-server-apis-for-communication)
  - [10. Integrate Components and Build the Main Loop](#10-integrate-components-and-build-the-main-loop)
  - [11. Optimize Performance and Test the System](#11-optimize-performance-and-test-the-system)
- [Additional Resources](#additional-resources)
- [Final Remarks](#final-remarks)
- [License](#license)

---

## Project Overview

The goal is to create a local AI assistant that enhances your home cooking experience by providing real-time suggestions, instructions, and interactions using your mobile phone as the camera and user interface while all processing happens on your AI server.

---

## System Architecture

![System Architecture Diagram](architecture_diagram.png)

**Components:**

1. **Mobile Phone Application**: Captures video and audio, displays responses, and serves as the user interface.
2. **Network Communication**: Streams data between the mobile app and the AI server over Wi-Fi.
3. **AI Server**: Processes video and audio streams, performs computer vision, natural language processing, text-to-speech, and speech-to-text.
4. **Response Delivery**: Sends processed responses back to the mobile app for display or playback.

---

## Prerequisites

### Hardware

- **AI Server**: A computer with 2 x NVIDIA RTX 3060 GPUs (12GB VRAM each).
- **Mobile Phone**: Android or iOS smartphone with camera and microphone capabilities.
- **Network**: A stable Wi-Fi network connecting both devices.

### Software

- **Operating System**: Ubuntu 20.04 LTS (recommended for the AI server).
- **Python**: Version 3.8 or higher.
- **CUDA and cuDNN**: Compatible with your GPUs (CUDA 11.7 recommended).
- **Development Tools**: Git, Python virtual environment tools.
- **Mobile Development Platforms**:
  - **Android**: Android Studio (if developing a custom app).
  - **iOS**: Xcode (if developing a custom app).
  - **Cross-Platform**: React Native or Flutter (optional).

---

## Step-by-Step Instructions

### 1. Set Up the AI Server Environment

#### 1.1. Update and Upgrade the System


sudo apt-get update
sudo apt-get upgrade
#### 1.2. Install NVIDIA Drivers, CUDA, and cuDNN
Follow NVIDIA's official installation guides to install the appropriate drivers, CUDA toolkit, and cuDNN libraries compatible with your GPUs.

#### 1.3. Install Python and Create a Virtual Environment
bash
Copy code
sudo apt-get install python3 python3-pip python3-venv
python3 -m venv ai_assistant_env
source ai_assistant_env/bin/activate
pip install --upgrade pip
#### 1.4. Install PyTorch with CUDA Support
bash
Copy code
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
#### 1.5. Install Essential Python Libraries
bash
Copy code
pip install opencv-python numpy pillow transformers flask flask-cors flask-socketio sounddevice simpleaudio TTS git+https://github.com/openai/whisper.git
### 2. Develop the Mobile Application
Option A: Use an Existing Streaming App
Android: Install IP Webcam or DroidCam.
iOS: Install Larix Broadcaster.
Configure the app to stream video (and audio if possible) to your AI server via RTSP or RTMP.

Option B: Develop a Custom Mobile App
#### 2.1. Choose a Development Framework
React Native: For cross-platform development.
Flutter: Another cross-platform option.
Native Development: Use Android Studio (Java/Kotlin) or Xcode (Swift).
#### 2.2. Implement App Features
Camera Streaming: Capture and stream video to the server.
Audio Streaming: Capture and stream audio to the server.
Display Responses: Show text responses or play audio received from the server.
User Interface: Buttons to start/stop the assistant, display ongoing activities, etc.
#### 2.3. Handle Permissions
Ensure the app requests and handles camera and microphone permissions appropriately.

3. Establish Network Communication
#### 3.1. Network Setup
Connect both the mobile phone and the AI server to the same Wi-Fi network.
Assign a static IP address to the AI server for consistent communication.
#### 3.2. Configure Firewall Settings
Open necessary ports on the AI server for incoming streams and API requests (e.g., ports 5000, 8000).
4. Receive and Process Video Stream on the Server
#### 4.1. Using OpenCV with RTSP Stream
python
Copy code
import cv2

stream_url = 'rtsp://your_phone_ip:port/stream'

cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame")
        break
    # Process frame (to be implemented in later steps)
#### 4.2. Using WebRTC (Optional)
Implement a WebRTC server using libraries like aiortc if you prefer lower latency.

5. Implement Computer Vision Processing
#### 5.1. Clone the YOLOv5 Repository
bash
Copy code
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
#### 5.2. Load the YOLOv5 Model in Your Script
python
Copy code
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to('cuda')
#### 5.3. Perform Object Detection on Each Frame
python
Copy code
results = model(frame)
detections = results.pandas().xyxy[0]
5.4. Infer Activities from Detections
Create a function to interpret detections and infer the current activity.

python
Copy code
def infer_activity(detections):
    # Simplified example
    if 'person' in detections['name'].values and 'knife' in detections['name'].values:
        return "You are preparing ingredients."
    elif 'person' in detections['name'].values and 'stove' in detections['name'].values:
        return "You are cooking on the stove."
    else:
        return "Activity not recognized."

activity_description = infer_activity(detections)
6. Set Up the Language Model (LLM)
6.1. Choose an Appropriate Model
Given hardware constraints, use a smaller model like GPT-Neo 1.3B.

6.2. Install Transformers Library
bash
Copy code
pip install transformers
#### 6.3. Load the Model
python
Copy code
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").half().to('cuda')
#### 6.4. Generate Suggestions
python
Copy code
def generate_suggestion(activity_description, user_input=None):
    prompt = activity_description
    if user_input:
        prompt += f"\nUser: {user_input}\nAssistant:"
    else:
        prompt += "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)
    suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return suggestion.split("Assistant:")[-1].strip()
7. Integrate Text-to-Speech (TTS)
#### 7.1. Install Coqui TTS
bash
Copy code
pip install TTS
#### 7.2. Initialize TTS
python
Copy code
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)
7.3. Convert Text to Speech
python
Copy code
def text_to_speech(text, filename):
    tts.tts_to_file(text=text, file_path=filename)
8. Implement Speech-to-Text (STT)
8.1. Install Whisper (Already Installed)
Ensure that whisper is installed as per the previous steps.

#### 8.2. Load the Whisper Model
python
Copy code
import whisper

stt_model = whisper.load_model("base").to('cuda')
8.3. Receive Audio Stream
Implement code to receive audio from the mobile app. This could be via an HTTP POST request containing audio data.

#### 8.4. Transcribe Audio
python
Copy code
def transcribe_audio(audio_file):
    result = stt_model.transcribe(audio_file)
    return result['text']
9. Develop Server APIs for Communication
#### 9.1. Install Flask and Related Libraries (Already Installed)
#### 9.2. Create Flask App
python
Copy code
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#### 9.3. Define API Endpoints
Endpoint to Receive User Input and Return Assistant Response
python
Copy code
@app.route('/assistant', methods=['POST'])
def assistant():
    data = request.json
    user_input = data.get('user_input')
    activity_description = data.get('activity_description')

    # Generate response using the LLM
    response = generate_suggestion(activity_description, user_input)

    # Convert response to speech and save as a file
    audio_filename = 'response.wav'
    text_to_speech(response, audio_filename)

    # Send back the response text and a link to the audio file
    return jsonify({'response_text': response, 'audio_file': audio_filename})
Endpoint to Receive Audio and Transcribe
python
Copy code
@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    # Save the audio file temporarily
    audio_file.save('user_input.wav')

    # Transcribe the audio
    user_text = transcribe_audio('user_input.wav')

    return jsonify({'transcription': user_text})
10. Integrate Components and Build the Main Loop
10.1. Main Server Script
Combine all the components into a main script.

python
Copy code
def main():
    # Initialize models and server
    # Load YOLOv5 model
    # Load LLM model
    # Load TTS model
    # Load STT model
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
10.2. Handle Incoming Video Stream
Within the Flask app or as a separate thread, continuously read frames from the video stream, perform object detection, and infer activities.

10.3. Handle User Requests
When a request comes in from the mobile app:

Retrieve the current activity_description.
Get the user_input (if any).
Generate a response using the LLM.
Convert the response to speech.
Send the response back to the mobile app.
11. Optimize Performance and Test the System
11.1. Performance Optimization
Use Half-Precision Models: Load models in half-precision to reduce VRAM usage.
Model Quantization: Apply 8-bit or 4-bit quantization where possible.
Efficient Processing: Process frames at intervals (e.g., every second) instead of every frame.
GPU Utilization: Assign different models to different GPUs.
11.2. Testing
Unit Tests: Test each component individually.
Integration Tests: Test the communication between the mobile app and the server.
Real-World Scenarios: Use the system during actual cooking sessions to evaluate performance and usability.
11.3. Debugging
Logging: Implement logging at each stage to monitor system performance and catch errors.
Resource Monitoring: Use tools like nvidia-smi to monitor GPU usage.
Additional Resources
YOLOv5 Documentation: https://github.com/ultralytics/yolov5
Transformers Documentation: https://huggingface.co/docs/transformers/index
Coqui TTS Documentation: https://tts.readthedocs.io/en/latest/
OpenAI Whisper Documentation: https://github.com/openai/whisper
Flask Documentation: https://flask.palletsprojects.com/en/2.0.x/
aiortc (WebRTC in Python): https://aiortc.readthedocs.io/en/latest/
React Native: https://reactnative.dev/
Flutter: https://flutter.dev/
Final Remarks
This README provides a comprehensive roadmap to build your AI Home Assistant. Here are some final tips:

Iterative Development: Tackle one component at a time to manage complexity.
Version Control: Use Git to track changes and manage your codebase effectively.
Community Support: Don't hesitate to seek help from online communities and forums if you encounter challenges.
Documentation: Keep updating this README and comment your code for future reference.
Good luck with your project!

License
This project is intended for personal, non-commercial use. Please ensure compliance with the licenses of the models and libraries used in this project.

