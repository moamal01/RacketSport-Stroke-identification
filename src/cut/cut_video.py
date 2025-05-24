import subprocess

video = 2
input_file = f"../../videos/game_{video}.mp4"
output_file = f"../../videos/game_{video}_cut.mp4"
frames_to_skip = 75306
frame_rate = 120

# Calculate time to skip in seconds
seconds_to_skip = frames_to_skip / frame_rate

# Use ffmpeg to trim the video
subprocess.run([
    "ffmpeg",
    "-ss", str(seconds_to_skip),
    "-i", input_file,
    "-c", "copy",
    output_file
])
