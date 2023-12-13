ffmpeg -threads 0 -i test.mov -f wav -ac 1 -acodec pcm_s16le -ar 16000 test.wav

./main -m models/ggml-large-v3.bin -f ~/Downloads/test.wav --output-srt --logprob-thold 3

/Users/jianliao/Work/git/whisper.cpp/main -m /Users/jianliao/Work/git/whisper.cpp/models/ggml-large-v3.bin \
    --output-srt \
    --logprob-thold 3 \
    -pc \
    -debug \
    --prompt "KRON 4 news"  \
    video.wav