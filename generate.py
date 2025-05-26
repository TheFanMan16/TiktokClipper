
import os
import subprocess
import json
import tempfile
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
import openai

# ------------------ CONFIGURATION ------------------
PART_DURATION = 45  # seconds per candidate clip
MAIN_DIR    = "main_video"
GAMEPLAY_DIR= "gameplay"
OUTPUT_DIR  = "output"
OPENAI_KEY  = os.getenv("Eneter API Key here")  # set this in your env
MAX_CLIPS   = 5     # maximum number of viral clips to produce

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper to locate a single video file in a directory
def get_single_video_file(directory):
    files = [f for f in os.listdir(directory)
             if f.lower().endswith((".mp4", ".webm", ".mkv"))]
    if not files:
        raise FileNotFoundError(f"No video found in '{directory}'")
    return os.path.join(directory, files[0])

# Slice the main video into fixed-length parts (start, end)
def slice_main_video(path):
    base = VideoFileClip(path)
    duration = int(base.duration)
    parts = []
    for idx, start in enumerate(range(0, duration, PART_DURATION), start=1):
        end = min(start + PART_DURATION, duration)
        parts.append((idx, start, end))
    return parts

# Transcribe a clip segment to text using new OpenAI v1 API
def transcribe_clip(video_path, start, end):
    # extract audio segment
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start), "-to", str(end),
        "-ar", "16000", "-ac", "1", audio_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # use new client.audio.transcriptions.create
    with open(audio_path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    text = resp.strip() if isinstance(resp, str) else resp.get("text", "")
    os.remove(audio_path)
    return text

# Transcribe the whole video once, returning verbose JSON
def transcribe_whole(video_path):
    wav_path = tempfile.mktemp(suffix=".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(wav_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
    os.remove(wav_path)
    return transcript

# Score segments and pick top N viral parts
def select_viral_parts(main_video, parts):
    scored = []
    for idx, start, end in parts:
        text = transcribe_clip(main_video, start, end)
        prompt = (
            f"Transcript ({end-start:.0f}s):\n" +
            text +
            "\nRate this clip from 1 to 10 on its viral potential and return just the number."
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        try:
            score = int(content.split()[0])
        except:
            score = 0
        scored.append((score, idx, start, end))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:MAX_CLIPS]
    return [(i, s, e) for (_, i, s, e) in top]

# Stack top clip + gameplay and export
    
def make_clip(main_video, gameplay, idx, start, end):
    print(f"🎬 Generating viral clip {idx}: {start}s–{end}s")
    top = VideoFileClip(main_video).subclip(start, end).resize(width=1080)
    duration = top.duration

    gp = VideoFileClip(gameplay).without_audio()
    if gp.duration < duration:
        loops = int(duration // gp.duration) + 1
        gp = concatenate_videoclips([gp] * loops)
    gp = gp.subclip(0, duration).resize(width=1080)

    top_h = top.resize(height=960)
    gp_h  = gp.resize(height=960)
    final = CompositeVideoClip([
        top_h.set_position((0,0)),
        gp_h.set_position((0,960))
    ], size=(1080,1920)).set_duration(duration).set_audio(top.audio)

    out = os.path.join(OUTPUT_DIR, f"viral_clip_{idx}.mp4")
    final.write_videofile(out, fps=30, codec="libx264", audio_codec="aac")
    print(f"✅ Saved viral_clip_{idx}.mp4")

# Main orchestration
if __name__ == "__main__":
    main_video = get_single_video_file(MAIN_DIR)
    gameplay   = get_single_video_file(GAMEPLAY_DIR)
    parts      = slice_main_video(main_video)
    selected   = select_viral_parts(main_video, parts)
    for idx, start, end in selected:
        make_clip(main_video, gameplay, idx, start, end)
    print("🎉 Done! Generated top viral clips in", OUTPUT_DIR)
