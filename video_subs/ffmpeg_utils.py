import subprocess
import os
import ffmpeg
from typing import Tuple
from shlex import quote


def get_video_dimensions(input_path: str) -> Tuple[int, int]:
    # Get the metadata for the input video file.
    probe = ffmpeg.probe(input_path)
    # Select the first video stream.
    video_streams = [stream for stream in probe['streams']
                     if stream['codec_type'] == 'video']
    # Extract the width and height of the video.
    width = int(video_streams[0]['width'])
    height = int(video_streams[0]['height'])
    return width, height


def get_video_length(video_path):
    """Get the length of a video in seconds using ffmpeg-python."""
    try:
        # Use ffprobe to get video info
        probe = ffmpeg.probe(video_path)

        # Extract duration from the first video stream
        duration = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)['duration']

        return float(duration)
    except ffmpeg.Error as e:
        raise RuntimeError(f"An error occurred while probing the video: {e}")
    except StopIteration:
        raise ValueError("No video stream found in the file")


def process_input_video(input_path, subtitles_file, font_name='Adobe Clean Han', font_size=12, blur_area=None, width=None, height=None, output_quality='high'):
    try:
        output_path = os.path.splitext(input_path)[0] + '_output.mp4'

        # Remove output file if it already exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Initialize command and filters
        cmd = ['ffmpeg', '-i', input_path]
        complex_filters = []

        # Check if resizing is needed
        if width is not None and height is not None:
            complex_filters.append(f'scale={width}:{height}')

        # Check if blur filter is needed
        if blur_area is not None:
            video_width, video_height = get_video_dimensions(input_path)
            x, y, blur_width, blur_height = blur_area
            x = video_width - x  # Position the blur area at the bottom right corner
            y = video_height - y
            blur_filter = f'split=2[base][blur];[blur]crop={blur_width}:{blur_height}:{x}:{y},boxblur=luma_radius=min(h\\,w)/20:luma_power=1[blurred];[base][blurred]overlay={x}:{y}'
            complex_filters.append(blur_filter)

        # Encoding settings for output quality
        if output_quality == 'high':
            quality_settings = ['-crf', '10', '-preset', 'slower']
        else:
            quality_settings = ['-crf', '23', '-preset', 'medium']

        # Apply complex filters if needed
        if complex_filters:
            cmd.extend(['-filter_complex', ';'.join(complex_filters)])
            intermediate_output = os.path.splitext(input_path)[0] + '_temp.mp4'
            cmd.extend(quality_settings + [intermediate_output])

            # Execute the first part of the command
            subprocess.run(cmd, check=True)

            # Apply subtitles in a second step
            subtitles_cmd = ['ffmpeg', '-i', intermediate_output, '-vf',
                             f"subtitles={quote(subtitles_file)}:force_style='FontName={quote(font_name)},FontSize={font_size},Shadow=0,BackColour=&H80000000,BorderStyle=4'", output_path]
            subprocess.run(subtitles_cmd, check=True)

            # Remove the intermediate file
            os.remove(intermediate_output)
        else:
            # Only apply subtitles if no resizing or blur is needed
            cmd.extend(
                ['-vf', f"subtitles={subtitles_file}:force_style='FontName={font_name},FontSize={font_size},Shadow=0,BackColour=&H80000000,BorderStyle=4'"] + quality_settings + [output_path])
            subprocess.run(cmd, check=True)

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


# Test the function
# result = process_input_video(input_path='/Users/jianliao/Desktop/SocialNetwork/47_KRON4/input.mp4',
#                        subtitles_file='/Users/jianliao/Desktop/SocialNetwork/47_KRON4/cn.srt', blur_area=(185, 182, 105, 105))

# print('Processing result:', result)
