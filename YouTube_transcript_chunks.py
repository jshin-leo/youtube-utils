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
import spacy
from nltk.tokenize import word_tokenize
import json
from collections import Counter

# --- SETTINGS ---
nlp = spacy.load("en_core_web_md")
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

MAX_CHARS_PER_CHUNK = 1000
URL_LIST_FILE = "youtube_links.txt"  # one URL per line
OUTPUT_DIR = "data"


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)
    
# --- STEP 0: Get meta data from YouTube ---
def get_video_metadata(video_url):
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", video_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        metadata = json.loads(result.stdout)
        return metadata

    except subprocess.CalledProcessError as e:
        print(f"Failed to get metadata: {e.stderr}")
        return {}

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

# --- STEP 3: transcript text into raw blocks of sentence ---
def generate_reference_text(transcript: str, top_n: int = 10) -> str:
    """
    Extracts a rough summary or topic anchor from transcript using spaCy.
    Focuses on most frequent noun chunks to approximate topic relevance.
    """
    doc = nlp(transcript)

    # Filter noun chunks that aren't stop words or very short
    noun_chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks
                   if not chunk.root.is_stop and len(chunk.text.strip()) > 3]

    # Count most common noun chunks
    top_chunks = [chunk for chunk, _ in Counter(noun_chunks).most_common(top_n)]

    # Return combined as a simple "topic reference"
    return " ".join(top_chunks).capitalize()

# --- STEP 4: Divide transcript into sentences (raw blocks) ---
def transcript_to_raw_blocks(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# --- STEP 5: Filter the raw blocks - remove unnecessary part (e.g., small talks, greetings) --- 
def filter_blocks_by_similarity(raw_blocks, reference_text, audio_file_nm, threshold=0.00):
    reference_doc = nlp(reference_text) # Process the reference_text using spaCy to get its semantic representation.
    filtered_blocks = []
    removed_blocks = []

    for block in raw_blocks:
        block_doc = nlp(block)
        similarity = block_doc.similarity(reference_doc)

        if similarity >= threshold:
            filtered_blocks.append(block)
        else:
            removed_blocks.append((block, similarity))
    
    # Log removed blocks - for checking and adjusting the threshold
    if removed_blocks:
        full_path = f"{audio_file_nm}_remove_log.txt"
        with open(full_path, "w", encoding="utf-8") as f:
            for text, sim in removed_blocks:
                f.write(f"[SIMILARITY: {sim:.4f}]\n{text}\n\n{'='*40}\n\n")

        print(f"Filtered {len(removed_blocks)} blocks. Logged to {full_path}.")

    else: 
        print("No blocks were filtered out based on similarity threshold.")

    return filtered_blocks

# --- STEP 6: Make chunks from the raw_blocks --- 
def similarity_guided_chunking(raw_blocks, min_chars=200, max_chars=600, similarity_threshold=0.75):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(raw_blocks, convert_to_tensor=True)

    chunks = []
    current_chunk = raw_blocks[0]
    current_len = len(current_chunk)

    for i in range(1, len(raw_blocks)):
        next_block = raw_blocks[i]
        next_len = len(next_block)

        similarity = util.cos_sim(embeddings[i - 1], embeddings[i]).item()

        # Decide whether to add next_block
        # if still less then max, add it when (similarity is high or current length is less than min).
        if current_len + next_len + 1 <= max_chars and (similarity >= similarity_threshold or current_len < min_chars):
            current_chunk += " " + next_block
            current_len += next_len + 1
        else:   # similarity is low and current chunk is already big enough
            chunks.append(current_chunk.strip())
            current_chunk = next_block
            current_len = next_len

    # Final chunk
    if current_len >= min_chars:
        chunks.append(current_chunk.strip())

    return chunks

# --- STEP 7: Save chunks as .jsonl ---
def save_chunks_rag_format(chunks, title, url, output_path, tags=None, description=None):
    records = []
    for i, chunk in enumerate(chunks, 1):
        record = {
            "id": f"{title}_chunk_{str(i).zfill(3)}",
            "chunk": chunk,
            "title": title.replace("_", " "),
            "url": url,
            "source": "YouTube", 
            "tags": tags or [],
            "description": description or ""
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
        count = 0

        with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        for VIDEO_URL in urls:
            try:
                count += 1
                print(f"\n---[{count}/{len(urls)}] Processing: {VIDEO_URL} ---")
                
                metadata = get_video_metadata(VIDEO_URL)
                title = metadata.get("title", "")
                tags = metadata.get("tags", [])
                description = metadata.get("description", "")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                base_filename = f"{sanitize_filename(title)}_{timestamp}"
                base_path = os.path.join(OUTPUT_DIR, base_filename)

                audio_file = f"{base_path}.mp3"
                download_audio(VIDEO_URL, audio_file)

                # Get full transcript
                full_text = transcribe_audio(audio_file, WHISPER_MODEL, base_path)

                # Get refernce text for filtering meaningless talks.
                reference_text = generate_reference_text(full_text)
                print(f"Auto-generated reference text: {reference_text}")
                
                # Get the list of sentences for processing
                raw_blocks = transcript_to_raw_blocks(full_text)
                 
                # filter with ref sentenece. 
                filtered_blocks = filter_blocks_by_similarity(raw_blocks, reference_text, audio_file, threshold=0.70)
                
                # Make chunks.
                chunks = similarity_guided_chunking(filtered_blocks, min_chars=200, max_chars=600, similarity_threshold=0.75)

                # Save the chunks in jsonl.
                save_chunks_rag_format(chunks, title, VIDEO_URL, base_path, tags, description)

                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    print(f"Deleted audio file: {audio_file}")

            except Exception as e:
                print(f"Failed to process {VIDEO_URL}: {e}")
            
        print(f"\n--- Finished. ---")

    except Exception as e:
        print(f"Error: {e}")
