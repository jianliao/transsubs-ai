import subprocess
import os
import ffmpeg


def get_video_dimensions(input_path):
    """Get the dimensions of the video."""
    probe = ffmpeg.probe(input_path)
    video_streams = [stream for stream in probe['streams']
                     if stream['codec_type'] == 'video']
    width = int(video_streams[0]['width'])
    height = int(video_streams[0]['height'])
    return width, height


def process_input_video(input_path, subtitles_file, font_name='Adobe Clean Han', font_size=20, blur_area=None, width=None, height=None, output_quality='high'):
    try:
        output_path = os.path.splitext(input_path)[0] + '_output.mp4'

        # Remove output file if it already exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Get the dimensions of the video
        video_width, video_height = get_video_dimensions(input_path)

        # Base ffmpeg command
        cmd = ['ffmpeg', '-i', input_path]

        # Construct the complex filter string
        complex_filters = []

        # Add scaling if specified
        if width is not None and height is not None:
            scale_filter = f'scale={width}:{height}'
            complex_filters.append(scale_filter)
            video_width = width
            video_height = height

        # Add blur filter if blur_area is specified
        if blur_area is not None:
            x, y, blur_width, blur_height = blur_area
            x = video_width - x  # Position the blur area at the bottom right corner
            y = video_height - y
            blur_filter = f'split=2[base][blur];[blur]crop={blur_width}:{blur_height}:{x}:{y},boxblur=luma_radius=min(h\\,w)/20:luma_power=1[blurred];[base][blurred]overlay={x}:{y}'
            complex_filters.append(blur_filter)

        # Combine complex filters
        if complex_filters:
            cmd.extend(['-filter_complex', ';'.join(complex_filters)])

        # Encoding settings for output quality
        if output_quality == 'high':
            cmd.extend(['-crf', '10', '-preset', 'slower'])
        else:
            cmd.extend(['-crf', '23', '-preset', 'medium'])

        # Output file for intermediate step
        intermediate_output = os.path.splitext(input_path)[0] + '_temp.mp4'
        cmd.extend([intermediate_output])

        # Execute the first part of the command
        subprocess.run(cmd, check=True)

        # Apply subtitles in a second step
        subtitles_cmd = ['ffmpeg', '-i', intermediate_output, '-vf',
                         f"subtitles={subtitles_file}:force_style='FontName={font_name},FontSize={font_size},Shadow=0,BackColour=&H80000000,BorderStyle=4'", output_path]
        subprocess.run(subtitles_cmd, check=True)

        # Remove the intermediate file
        os.remove(intermediate_output)

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


# Test the function
# result = process_input_video(input_path='/Users/jianliao/Desktop/SocialNetwork/18_KRON4/input.mov',
#                        subtitles_file='/Users/jianliao/Desktop/SocialNetwork/18_KRON4/cn.srt', blur_area=(185, 182, 105, 105))

# print('Processing result:', result)
