# TransSubs-AI

## Introduction
Welcome to TransSubs-AI, a powerful and intuitive tool designed to transform the way we work with video subtitles. This repository holds a Python application capable of extracting audio from video files, transcribing it, translating the subtitles into various languages, and generating a title and description for the video in the translated language. It's a perfect blend of technology and convenience for content creators, translators, and anyone working with multilingual video content.

## Technologies Used
TransSubs-AI leverages several cutting-edge technologies and tools:

- **Whisper.cpp**: An efficient tool for audio transcription, which can be found at [Whisper.cpp GitHub Repository](https://github.com/ggerganov/whisper.cpp).
- **ffmpeg**: A complete, cross-platform solution to record, convert and stream audio and video. More details can be found at [ffmpeg Official Website](https://ffmpeg.org/).
- **OpenAI Whisper Model**: Utilized for accurate transcription of audio into text.
- **OpenAI ChatGPT Completion API**: Powers the translation of subtitles and the generation of video titles and descriptions.

## Usage
To use TransSubs-AI, follow these steps:

1. **Environment Setup**:
   - Create a Conda virtual environment using Python 3.10 for consistent performance.
   - Install necessary libraries by running:
     ```
     pip install openai python-dotenv
     ```

2. **Running the Application**:
   - Use the command line interface (CLI) to easily process your videos. Here's an example command to get you started:
     ```
     python Main.py ~/Downloads/RPReplay_Final1702137844.MP4
     ```
   - Replace the file path with the path to your own video file.

We hope TransSubs-AI makes your work with subtitles more efficient and enjoyable. Feel free to contribute to this project by submitting pull requests or reporting issues.

Happy subtitling!
