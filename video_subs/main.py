import time
import argparse
import logging
import subprocess
import re
import os

from ollama import Client
from openai import OpenAI
from anthropic import Anthropic

from shlex import quote
from .ffmpeg_utils import get_video_length, process_input_video

from dotenv import load_dotenv
load_dotenv('/Users/jianliao/Work/git/transsubs-ai/video_subs/.env')

client = OpenAI()

client_claude = client = Anthropic()

client_ollama = Client(host=os.getenv('OLLAMA_HOST'))

current_directory = os.getcwd()
log_file_path = os.path.join(current_directory, 'app.log')
# Configure logging to save in the current directory
logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def normalize_title(title):
    # Remove single and double quotes and commas
    title = title.replace("'", "").replace(
        '"', "").replace(',', '').replace(';', '')

    # Replace disallowed characters with an underscore
    disallowed_chars = r'[<>:"/\\|?*]+'
    title = re.sub(disallowed_chars, '_', title)

    # Replace spaces with underscores (optional)
    title = title.replace(' ', '_')

    return title


def download_youtube_video(url, directory, trim_end):
    # Validate YouTube URL
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube\.com/watch\?v=|youtu\.be/)'
        r'[\w-]+'
    )

    if not re.match(youtube_regex, url):
        raise ValueError("Invalid YouTube URL")

    # Extract title for output file path
    title = subprocess.check_output(
        ['yt-dlp', '--print', 'filename', '-o', '%(title)s.%(ext)s', url], text=True).strip().replace('.webm', '')

    # Normalize title
    title = normalize_title(title)

    # Resolve directory path
    directory = os.path.abspath(directory) if directory == '.' else directory
    output_file = os.path.join(directory, f"{title}.mp4")
    output_description = os.path.join(directory, f"{title}.description")

    # Download command
    command = [
        'yt-dlp', url,
        '-f', 'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]',
        '--output', output_file,
        '--write-description'
    ]

    # Execute download command
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError("yt-dlp command failed")

    if os.path.exists(output_file) and os.path.exists(output_description):
        # Trim the end of the video if needed
        if trim_end:
            video_length = get_video_length(output_file)
            command = [
                'ffmpeg', '-nostdin',
                '-i', output_file,
                '-t', f"{video_length - trim_end}",
                '-c', 'copy',
                f"{title}-trimmed.mp4"
            ]
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise RuntimeError("ffmpeg command failed")
            # Remove the intermediate file
            os.remove(output_file)
            return f"{title}-trimmed.mp4", output_description
        else:
            return output_file, output_description
    else:
        raise FileNotFoundError("Downloaded video file not found")


def extract_and_transcribe_audio(input_video_path, prompt=None):
    # Load WHISPER_CPP_HOME environment variable
    whisper_cpp_home = os.getenv('WHISPER_CPP_HOME')
    if not whisper_cpp_home:
        raise Exception("WHISPER_CPP_HOME environment variable is not set")

    # Construct the paths for Whisper.cpp executable and model
    whisper_cpp_executable = os.path.join(whisper_cpp_home, 'main')
    whisper_cpp_model = os.path.join(
        whisper_cpp_home, 'models', 'ggml-large-v3.bin')

    # Construct the base command
    command = f"ffmpeg -nostdin -threads 0 -i {quote(input_video_path)} -f wav -ac 1 -acodec pcm_s16le -ar 16000 - | {quote(whisper_cpp_executable)} -m {quote(whisper_cpp_model)} {('--prompt ' + quote(prompt)) if prompt else ''} --output-srt --logprob-thold 3 -f -"

    print(command)

    # Run the command
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if process.returncode != 0:
        raise Exception(
            f"Error in audio extraction and transcription: {error.decode()}")

    return output.decode()


def translate_subtitle(srt_en_content, target_language, temperature=0.2, context=None, llm='ollama'):
    system_prompt = f"""
Translate the English subtitle into a {target_language} subtitle while maintaining the SRT format. Ensure that specified elements remain in their original English form.

- **Elements to Maintain in English**:
  - Proper names (people, places, organizations)
  - Brand names and trademarks
  - Specific technical terms
  - Acronyms and abbreviations
  - Cultural references, idioms, and sayings
  - Titles of works (books, movies, songs)
  - Units of measurement
  - Email addresses and URLs
  - Direct quotes
  - Certain legal terms
  - Location names (cities, states, countries, etc.)

# Steps

1. Carefully read the English subtitle content for context.
2. Translate the main content into {target_language}, ensuring fluency and accuracy.
3. Identify and preserve the specified elements in their original English form.
4. Maintain the SRT timing and numbering structure exactly.

# Output Format

- Provide the translated subtitle in SRT format.

# Examples

**Example SRT Input:**
1
00:00:01,000 --> 00:00:04,000
Hello, my name is John Smith. I work at Microsoft.

2
00:00:05,000 --> 00:00:08,000
Visit www.example.com for more info.

**Translated SRT Output:**
1
00:00:01,000 --> 00:00:04,000
你好，我的名字是John Smith。我在Microsoft工作。

2
00:00:05,000 --> 00:00:08,000
请访问www.example.com获取更多信息。

(Actual examples should follow the correct timing and formatting, longer examples will naturally align with full subtitle scripts)

# Notes

- Ensure each translated subtitle is meaningful and contextually appropriate.
- Review for any cultural nuances that may affect translation.
- Double-check the consistency of kept English elements across the subtitle file.
- Maintain the original timing and numbering structure of the English subtitles.
"""
    if llm == 'openai':
        ########################
        # Openai               #
        ########################

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=temperature,
            max_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": srt_en_content
                }
            ]
        )
        return response.choices[0].message.content.strip()
    elif llm == 'ollama':
        ########################
        # Local Ollama LLM     #
        ########################
        response = client_ollama.chat(model=os.getenv('LOCAL_LLM'), messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                'role': 'user',
                'content': srt_en_content
            }
        ])
        return response['message']['content']

    elif llm == 'claude':
        ########################
        # Anthropics           #
        ########################

        response = client.messages.create(
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": srt_en_content
                }
            ],
            model="claude-3-sonnet-20240229",
        )
        return response.content[0].text


def save_subtitle_file(translated_content, output_path, language=None):
    if language == 'Chinese':
        translated_content = translated_content.replace(
            '，', ' ').replace('。', ' ')

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(translated_content)


def format_transcription_to_srt(transcript_content):
    srt_formatted = ""
    counter = 1

    for line in transcript_content.strip().split('\n'):
        if line.strip():
            # Add the sequential number
            srt_formatted += f"{counter}\n"
            counter += 1

            # Remove the opening bracket, replace the closing bracket, and format the timecode
            line = line.lstrip("[").replace("]", "", 1)
            timecode, text = line.replace('.', ',', 2).split(
                "   ", 1)  # Splitting timecode and text

            # Add the formatted timecode
            srt_formatted += timecode + "\n"

            # Add the subtitle text on a new line, starting with a space
            srt_formatted += f" {text.strip()}\n\n"

    return srt_formatted


def generate_video_metadata(translated_content, language):
    prompt = (f"Based on the following subtitles, suggest a concise and engaging title and a brief description for the video. "
              f"Subtitles:\n{translated_content}")

    # response = client.chat.completions.create(
    #     model="gpt-4-1106-preview",  # or whichever model you're using
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": f"You are a helpful assistant designed to be very good at {language}."
    #         },
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    #     temperature=0.9,  # Adjust as needed for creativity vs. specificity
    #     max_tokens=4096  # Adjust based on how long you expect the title and description to be
    # )

    response = client_ollama.chat(model=os.getenv('LOCAL_LLM'), messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])

    generated_text = response['message']['content']

    # Assuming the first line is the title and the rest is the description
    title, description = generated_text.split('\n', 1)

    return title.strip(), description.strip()


def main():
    start_time = time.time()

    # Preset blur areas mapped to keys
    blur_area_presets = {
        'kron4': [185, 182, 105, 105],
        'abc7': [190, 190, 160, 155],
        # Add more presets here as needed
    }

    llms = [
        'ollama',
        'openai',
        'claude'
    ]

    # Preset trim ends mapped to keys
    trim_end_presets = {
        'abc7': 17,
        'inside': 8,
        'nbc': 10,
        # Add more presets here as needed
    }

    parser = argparse.ArgumentParser(
        description="Process a video file to generate translated subtitles.")
    parser.add_argument('video_url', type=str,
                        help="The path to the input video file.")
    parser.add_argument('--output_path', type=str, default='.',
                        help="The path to the output video files.")
    parser.add_argument('--blur_area', type=str, choices=blur_area_presets.keys(),
                        help="Specify the key for a preset blur area. If not provided, no blur area is applied.")
    parser.add_argument('--trim_end', type=str, choices=trim_end_presets.keys(),
                        help='Specify the key for a preset trim end. If not provided, no trim end is applied.')
    parser.add_argument('--cn_only', type=bool, default=False,
                        help="Chinese subtitle only.")
    parser.add_argument('--llm', type=str, choices=llms,
                        default='ollama', help="Specify the language model to use. Default is ollama.")

    args = parser.parse_args()

    try:
        print("Step 0: Downloading video...\n")
        step_start_time = time.time()
        input_video, input_video_description = download_youtube_video(
            args.video_url, args.output_path, trim_end_presets[args.trim_end] if args.trim_end else None)
        # Open the file with UTF-8 encoding
        with open(input_video_description, 'r', encoding='utf-8') as file:
            # Read the content of the file
            prompt = file.read()
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print("Step 1: Setting up paths and variables...\n")
        video_dir, video_filename = os.path.split(input_video)
        video_basename, _ = os.path.splitext(video_filename)
        srt_en_path = os.path.join(video_dir, f"{video_basename}.en.srt")
        translated_srt_path = os.path.join(
            video_dir, f"{video_basename}.cn.srt")

        print("Step 2: Extracting and transcribing audio...\n")
        step_start_time = time.time()
        raw_srt_content = extract_and_transcribe_audio(
            input_video, prompt=prompt)
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print("Step 3: Formatting transcription to SRT...\n")
        step_start_time = time.time()
        formatted_srt = format_transcription_to_srt(raw_srt_content)
        save_subtitle_file(formatted_srt, srt_en_path)
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print("Step 4: Translating subtitles...\n")
        step_start_time = time.time()
        translated_srt = translate_subtitle(
            formatted_srt, "Chinese", context=prompt, llm=args.llm)
        save_subtitle_file(
            translated_srt, translated_srt_path, language="Chinese")
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        if args.cn_only == False:
            print("Step 4a: Combine two subtitles into one. \n")
            combined_sub = translated_srt + '\n\n' + formatted_srt
            combined_sub_path = os.path.join(
                video_dir, f"{video_basename}.srt")
            save_subtitle_file(combined_sub, combined_sub_path)
            translated_srt_path = combined_sub_path

        print("Step 5: Processing video...\n")
        step_start_time = time.time()
        blur_area = None
        if args.blur_area in blur_area_presets:
            blur_area = blur_area_presets[args.blur_area]
        process_input_video(
            input_video, translated_srt_path, blur_area=blur_area)
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print("Step 6: Generating video metadata...\n")
        step_start_time = time.time()
        title, description = generate_video_metadata(translated_srt, "Chinese")
        print(title + "\n\n" + description)
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print(
            f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes.")
        print(
            f"Find the final output video at: {os.path.join(video_dir, video_basename + '_output.mp4')}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
