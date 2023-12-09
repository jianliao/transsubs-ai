ffmpeg -threads 0 -i test.mov -f wav -ac 1 -acodec pcm_s16le -ar 16000 test.wav

./main -m models/ggml-large-v3.bin -f ~/Downloads/test.wav --output-srt --logprob-thold 3