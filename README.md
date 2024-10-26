# TransSubs-AI

## Introduction
Welcome to TransSubs-AI, a powerful and intuitive tool designed to transform the way we work with video subtitles. This repository holds a Python application capable of extracting audio from video files, transcribing it, translating the subtitles into various languages, and generating a title and description for the video in the translated language. It's a perfect blend of technology and convenience for content creators, translators, and anyone working with multilingual video content.
<img width="2487" alt="AI Assistant Workflow" src="https://github.com/user-attachments/assets/68c34238-8fb7-41db-92bf-8e33001dd191">

## Technologies
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
     pip install -r requirements.txt
     ```
   - Rename `example.env` to `.env` and fill in the missing information
   - pip install .
2. **Running the Application**:
   - Use the command line interface (CLI) to easily process your videos. Here's an example command to get you started:
     ```
     > video_subs -h
      usage: video_subs [-h] [--output_path OUTPUT_PATH] [--blur_area {kron4,abc7}] [--trim_end {abc7,inside,nbc}] [--cn_only CN_ONLY] [--llm {ollama,openai,claude}] video_url
      
      Process a video file to generate translated subtitles.
      
      positional arguments:
        video_url             The path to the input video file.
      
      options:
        -h, --help            show this help message and exit
        --output_path OUTPUT_PATH
                              The path to the output video files.
        --blur_area {kron4,abc7}
                              Specify the key for a preset blur area. If not provided, no blur area is applied.
        --trim_end {abc7,inside,nbc}
                              Specify the key for a preset trim end. If not provided, no trim end is applied.
        --cn_only CN_ONLY     Chinese subtitle only.
        --llm {ollama,openai,claude}
                              Specify the language model to use. Default is ollama.
     ```

We hope TransSubs-AI makes your work with subtitles more efficient and enjoyable. Feel free to contribute to this project by submitting pull requests or reporting issues.

Happy subtitling!
