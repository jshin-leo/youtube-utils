# ---------------------------------------------------------------------
#   - Title: YouTube_transcript_chunks.py
#   - Author: JS
#   - Date: June 4th, 2025
#   - What: Extract chunks from YouTube Video
#   - How: 
#       1. Prepare the list of YouTube Links in youtube_links.txt
#       2. Run the program: python YouTube_transcript_chunks.py
#       3. In Data folder, transcripts and chunks will be generated. 
# ---------------------------------------------------------------------
import os
os.environ["PATH"] = os.path.expanduser("~/ffmpeg-7.0.2-amd64-static") + ":" + os.environ["PATH"]
import subprocess
import whisper
import pandas as pd
from datetime import datetime
import re 
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
import json

# --- SETTINGS ---
URL_LIST_FILE = "youtube_links.txt"  # one URL per line
# VIDEO_URL = "https://www.youtube.com/watch?v=mAvyLobFdoE"
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
MAX_CHARS_PER_CHUNK = 1000
OUTPUT_DIR = "data"

# --- Get clean title from YouTube ---
def get_video_title(video_url):
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", video_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        title = result.stdout.strip()
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).replace(" ", "_")
        return safe_title
    except subprocess.CalledProcessError as e:
        print(f"Failed to get title: {e.stderr}")
        return "unknown_title"

# --- STEP 1: Download YouTube audio using yt-dlp ---
def download_audio(video_url, output_path):
    print(f"Downloading audio from: {video_url}")
    ffmpeg_path = os.path.expanduser("~/ffmpeg-7.0.2-amd64-static")
    command = [
        "yt-dlp",
        "--ffmpeg-location", ffmpeg_path,
        "-f", "bestaudio",
        "-x", "--audio-format", "mp3",
        "-o", output_path,
        video_url
    ]
    subprocess.run(command, check=True)
    print(f"Audio saved to {output_path}")
    return output_path

# --- STEP 2: Transcribe using Whisper ---
def transcribe_audio(audio_path, model_name, output_name):
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    print("Transcribing...")
    result = model.transcribe(audio_path)
    print("Transcription completed.")

    full_path = f"{output_name}_transcript.txt"
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Full transcript saved to {full_path}")
    return result["text"]

# --- STEP 3: Chunk transcript text ---
# [first try - just using . ]
def chunk_text(text, max_chars=1000):
    sentences = text.split(". ")
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence.strip() + ". "
        else:
            chunks.append(current.strip())
            current = sentence.strip() + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def semantic_chunking(text, max_chars=1000, min_chars=200, similarity_threshold=0.65):
    # 1. Break text into rough blocks (e.g., every ~50 words)
    words = text.split()
    raw_blocks = [" ".join(words[i:i+50]) for i in range(0, len(words), 50)]

    # 2. Encode each block using SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(raw_blocks, convert_to_tensor=True)

    # 3. Group semantically coherent blocks
    chunks, current_chunk, current_len = [], [], 0

    for i in range(len(raw_blocks)):
        current_chunk.append(raw_blocks[i])
        current_len += len(raw_blocks[i])

        # Get similarity to next block, if any
        is_last = (i == len(raw_blocks) - 1)
        if not is_last:
            similarity = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
        else:
            similarity = 1.0  # end of text

        # Decide whether to break the chunk
        if current_len >= max_chars or similarity < similarity_threshold or is_last:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= min_chars:
                chunks.append(chunk_text)
            current_chunk = []
            current_len = 0

    return chunks

# --- STEP 4: Save chunks as .csv and .txt ---
def save_chunks(chunks, output_name):
    csv_file = f"{output_name}_chunks.csv"
    txt_file = f"{output_name}_chunks.txt"

    df = pd.DataFrame({"chunk": chunks})
    df.to_csv(csv_file, index=False)
    print(f"Chunks saved to {csv_file}")

    # Save as TXT with '|' as chunk separator
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(" | ".join(chunks))
    print(f"Chunks also saved to {txt_file}")

def save_chunks_rag_format(chunks, title, url, output_path):
    records = []
    for i, chunk in enumerate(chunks, 1):
        record = {
            "id": f"{title}_chunk_{str(i).zfill(3)}",
            "chunk": chunk,
            "title": title.replace("_", " "),
            "url": url,
            "source": "YouTube"
        }
        records.append(record)

    # Save as JSONL
    jsonl_path = f"{output_path}_rag.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"RAG-formatted chunks saved to {jsonl_path}")

# --- MAIN ---
if __name__ == "__main__":
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        for VIDEO_URL in urls:
            try:
                print(f"\n--- Processing: {VIDEO_URL} ---")
                title = get_video_title(VIDEO_URL)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                base_filename = f"{title}_{timestamp}"
                base_path = os.path.join(OUTPUT_DIR, base_filename)

                audio_file = f"{base_path}.mp3"
                download_audio(VIDEO_URL, audio_file)

                full_text = transcribe_audio(audio_file, WHISPER_MODEL, base_path)

                chunks = semantic_chunking(full_text, max_chars=1000, min_chars=200, similarity_threshold=0.65)

                save_chunks(chunks, base_path)
                save_chunks_rag_format(chunks, title, VIDEO_URL, base_path)

                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    print(f"Deleted audio file: {audio_file}")

            except Exception as e:
                print(f"Failed to process {VIDEO_URL}: {e}")
            
        print(f"\n--- Finished. ---")

    except Exception as e:
        print(f"Error: {e}")