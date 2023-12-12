import ffmpeg
import os


def process_video(input_path, subtitles_file, font_name='Adobe Clean Han', font_size=20, width=None, height=None):
    try:
        output_path = os.path.splitext(input_path)[0] + '_output.mp4'

        video_stream = ffmpeg.input(input_path)

        # Apply resizing if both width and height are specified
        if width and height:
            video_stream = video_stream.filter('scale', width, height)

        (
            video_stream.output(
                output_path, vf=f"subtitles={subtitles_file}:force_style='FontName={font_name},FontSize={font_size},Shadow=0,BackColour=&H80000000,BorderStyle=4'", format='mp4').run()
        )

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


# Example usage
result = process_video('/Users/jianliao/Desktop/SocialNetwork/16_NBC_1/input.mov',
                       '/Users/jianliao/Desktop/SocialNetwork/16_NBC_1/input.cn.srt')
print('Processing successful:', result)
