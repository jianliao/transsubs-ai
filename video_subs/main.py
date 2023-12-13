import time
import argparse
import subprocess
import os

from openai import OpenAI
from shlex import quote
from .ffmpeg_utils import process_input_video

from dotenv import load_dotenv
load_dotenv('/Users/jianliao/Work/git/transsubs-ai/video_subs/.env')

client = OpenAI()


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
    command = f"ffmpeg -nostdin -threads 0 -i {quote(input_video_path)} -f wav -ac 1 -acodec pcm_s16le -ar 16000 - | {quote(whisper_cpp_executable)} -m {quote(whisper_cpp_model)} --output-srt --logprob-thold 10 -f -"

    # Add the prompt option to the command if provided
    if prompt:
        command += f" --prompt {quote(prompt)}"

    print(command)

    # Run the command
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if process.returncode != 0:
        raise Exception(
            f"Error in audio extraction and transcription: {error.decode()}")

    return output.decode()


def translate_subtitle(srt_en_content, target_language, temperature=0.2):
    prompt = f"""First, please correct any grammar or wording issues in the following English subtitles. After correcting, translate the subtitles into {target_language}, but keep the following elements in their original English form:
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
- Location names (cities, state, countries, etc.)

Third, please insert special \\N characters to indicate line breaks if the translated subtitles are too long to fit on one line. For example, if the translated subtitles are:
"距离旧金山举办Apex峰会已经过去三周多, 世界领导人和一些社区居民开始发出警示, 指出为该峰会而进行的大规模清理工作正在迅速恶化。". You should insert \\N characters to approximately middle of the line to break the line into two lines:
"距离旧金山举办Apex峰会已经过去三周多, 世界领导人和一些社区居民开始发出警示, \\N指出为该峰会而进行的大规模清理工作正在迅速恶化。"

Edit the subtitle content within an SRT file, ensuring that the original structure of the SRT format is maintained. Specifically, do not make any changes to the time range stamps (the timestamps that dictate when each subtitle appears and disappears on screen). Focus only on correcting, modifying and translating the text of the subtitles, leaving the timing and sequence of each subtitle entry as it is.
Only return the translated subtitles, do not include the original English subtitles.
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


def save_subtitle_file(translated_content, output_path):
    with open(output_path, 'w') as file:
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
    prompt = (f"Based on the following translated video subtitles in {language}, suggest a concise and engaging title and a brief description for the video. "
              f"Both the title and description should be in {language}.\n\nSubtitles:\n{translated_content}")

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # or whichever model you're using
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant designed to be very good at {language}."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.9,  # Adjust as needed for creativity vs. specificity
        max_tokens=4096  # Adjust based on how long you expect the title and description to be
    )

    generated_text = response.choices[0].message.content.strip()

    # Assuming the first line is the title and the rest is the description
    title, description = generated_text.split('\n', 1)

    return title.strip(), description.strip()


def main():
    start_time = time.time()

    # Preset blur areas mapped to keys
    blur_area_presets = {
        'kron4': [185, 182, 105, 105],
        # Add more presets here as needed
    }

    parser = argparse.ArgumentParser(
        description="Process a video file to generate translated subtitles.")
    parser.add_argument('input_video', type=str,
                        help="The path to the input video file.")
    parser.add_argument('--blur_area_key', type=str, choices=blur_area_presets.keys(),
                        help="Specify the key for a preset blur area. If not provided, no blur area is applied.")
    parser.add_argument('--prompt', type=str,
                        help="Optional prompt to use for audio transcription.")

    args = parser.parse_args()

    try:
        print("Step 1: Setting up paths and variables...")
        video_dir, video_filename = os.path.split(args.input_video)
        video_basename, _ = os.path.splitext(video_filename)
        srt_en_path = os.path.join(video_dir, f"{video_basename}.en.srt")
        translated_srt_path = os.path.join(
            video_dir, f"{video_basename}.cn.srt")

        print("Step 2: Extracting and transcribing audio...")
        step_start_time = time.time()
        raw_srt_content = extract_and_transcribe_audio(
            args.input_video, prompt=args.prompt)
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print("Step 3: Formatting transcription to SRT...")
        step_start_time = time.time()
        formatted_srt = format_transcription_to_srt(raw_srt_content)
        save_subtitle_file(formatted_srt, srt_en_path)
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print("Step 4: Translating subtitles...")
        step_start_time = time.time()
        translated_srt = translate_subtitle(formatted_srt, "Chinese")
        save_subtitle_file(translated_srt, translated_srt_path)
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print("Step 5: Processing video...")
        step_start_time = time.time()
        blur_area = None
        if args.blur_area_key in blur_area_presets:
            blur_area = blur_area_presets[args.blur_area_key]
        process_input_video(
            args.input_video, translated_srt_path, blur_area=blur_area)
        print(f"Completed in {time.time() - step_start_time:.2f} seconds.")

        print("Step 6: Generating video metadata...")
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
