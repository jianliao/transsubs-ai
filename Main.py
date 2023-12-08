from openai import OpenAI
from shlex import quote
import argparse
import subprocess
import os

from dotenv import load_dotenv
load_dotenv()


client = OpenAI()


def extract_and_transcribe(input_video_path):
    # Load WHISPER_CPP_HOME environment variable
    whisper_cpp_home = os.getenv('WHISPER_CPP_HOME')
    if not whisper_cpp_home:
        raise Exception("WHISPER_CPP_HOME environment variable is not set")

    # Construct the paths for Whisper.cpp executable and model
    whisper_cpp_executable = os.path.join(whisper_cpp_home, 'main')
    whisper_cpp_model = os.path.join(
        whisper_cpp_home, 'models', 'ggml-large-v3.bin')

    # Construct and run the command
    command = f"ffmpeg -nostdin -threads 0 -i {quote(input_video_path)} -f wav -ac 1 -acodec pcm_s16le -ar 16000 - | {quote(whisper_cpp_executable)} -m {quote(whisper_cpp_model)} --output-srt -f -"
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if process.returncode != 0:
        raise Exception(
            f"Error in audio extraction and transcription: {error.decode()}")

    return output.decode()


def translate_srt(srt_en_content, target_language, temperature=0.7):
    prompt = f"""First, please correct any grammar or wording issues in the following English subtitles. After correcting, translate the subtitles into {target_language} and only return the translated subtitles, but keep the following elements in their original English form:
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

Here are the subtitles to be corrected and translated:

{srt_en_content}"""

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=temperature,
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant designed to translate English subtitles to {target_language} subtitles."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content.strip()


def save_translated_srt(translated_content, output_path):
    with open(output_path, 'w') as file:
        file.write(translated_content)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Process a video file to generate translated subtitles.")

    # Add the arguments
    parser.add_argument('input_video', type=str,
                        help="The path to the input video file.")

    # Execute the parse_args() method
    args = parser.parse_args()

    input_video_path = args.input_video

    # Determine the directory and filename of the input video
    video_dir, video_filename = os.path.split(input_video_path)
    video_basename, _ = os.path.splitext(video_filename)

    # Path for the original English SRT file
    srt_en_path = os.path.join(video_dir, f"{video_basename}.en.srt")

    # Path for the translated SRT file
    translated_srt_path = os.path.join(video_dir, f"{video_basename}.cn.srt")

    # Extract audio and transcribe to SRT
    srt_content = extract_and_transcribe(input_video_path)

    print(f"English:\n {srt_content}")

    # Save the original English subtitles
    save_translated_srt(srt_content, srt_en_path)

    # # Translate
    # translated_content = translate_srt(srt_content, "Chinese")

    # print(f"Chinese:\n {translated_content}")

    # # Save the translated subtitles
    # save_translated_srt(translated_content, translated_srt_path)


if __name__ == "__main__":
    main()
