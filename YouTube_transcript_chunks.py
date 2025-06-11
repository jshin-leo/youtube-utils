# ---------------------------------------------------------------------
#   - Title: YouTube_transcript_chunks.py
#   - Author: JS
#   - Date: June 11th, 2025
#   - What: YouTube Transcript Chunking
#   - How: 
#       1. [OPTION 1] Provide YouTube channel or playlist URLs in 'youtube_channels.txt'.
#          - The script will extract all video links and save them in 'youtube_links.txt'.
#
#       2. [OPTION 2] Skip channel extraction and directly use pre-collected video links
#          listed in 'youtube_links.txt' (e.g., from an external scraper).
#
#       3. For each video:
#          - Download audio using yt-dlp
#          - Transcribe speech using OpenAI Whisper
#          - Filter irrelevant sentences using spaCy-based similarity
#          - Segment the transcript into coherent chunks using SentenceTransformer
#
#       4. Save all processed outputs (transcripts and RAG-ready chunks) in the 'data/' folder.
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

# ----- SETTINGS -----
URL_LIST_CHANNELS = "youtube_channels.txt"  # [Input] YouTube Channel Links
URL_LIST_FILE = "youtube_links.txt"  # one URL per line
OUTPUT_DIR = "data"

nlp = spacy.load("en_core_web_md")
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
MAX_CHARS_PER_CHUNK = 1000

######################################################################
# [Get a list of youtube videos from Youtube Channels]
def get_video_links_from_youtube_url(url):
    # Handle both channel URLs (e.g., /@handle) and playlist URLs (/playlist?list=...).
    if "youtube.com/@" in url or "youtube.com/user/" in url:
        if not url.endswith("/videos"):
            url += "/videos"

    command = [
        "yt-dlp",
        "--flat-playlist",  # Only get metadata
        "-j",               # Output JSON lines
        url
    ]
    
    video_links = []
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        for line in result.stdout.strip().split("\n"):
            if line:
                data = json.loads(line)
                video_id = data.get("id")
                if video_id:
                    video_links.append(f"https://www.youtube.com/watch?v={video_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing URL: {url}")
        print(e.stderr)

    return video_links

######################################################################
# [Extract Transcripts and Make into Chunks] 
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
######################################################################

if __name__ == "__main__":
    try:
        # Check if 'channel&playlist' list exists and is non-empty.
        if os.path.exists(URL_LIST_CHANNELS) and os.path.getsize(URL_LIST_CHANNELS) > 0:
            print("Detected channel/playlist input. Extracting video links...")
            # URL_LIST_CHANNELS 
            with open(URL_LIST_CHANNELS, "r", encoding="utf-8") as f:
                channel_urls = [line.strip() for line in f if line.strip()]    # get the list of channels.
            print(f"Number of sources: {len(channel_urls)}")
            all_links = {}

            # From the Channels list, Extract the list of all video that has.
            for url in channel_urls:
                print(f"\nProcessing {url}...")
                links = get_video_links_from_youtube_url(url)
                all_links[url] = links
                print(f"  Found {len(links)} videos")

            # Print or save results
            print(f"\n[Summary]")
            for channel, videos in all_links.items():
                print(f"    - Source: {channel} - {len(videos)} links")
            
            # save the video links to txtfile.
            with open(URL_LIST_FILE, "w", encoding="utf-8") as f:
                for channel, videos in all_links.items():
                    for link in videos:
                        f.write(link + "\n")
            print(f"Lists are saved in {URL_LIST_FILE} file.")

            # [log for checking the title] Save video titles and URLs (tab-separated)
            # ↓↓↓ You can uncomment this part if you need the log of titles. (This part takes some time!)

            # with open("youtube_titles_and_links.txt", "w", encoding="utf-8") as out_file:
            #     for source_url, video_links in all_links.items():
            #         for video_url in video_links:
            #             try:
            #                 metadata = get_video_metadata(video_url)
            #                 title = metadata.get("title", "").strip().replace("\n", " ")
            #                 out_file.write(f"{title}\t{video_url}\n")
            #             except Exception as e:
            #                 print(f"  Failed to get metadata for {video_url}: {e}")
        else: 
            print("Skipping channel extraction. Using existing video links in youtube_links.txt.")

        # From the youtube video lists, Extract Transcript & Make them into Chunks.
        if not os.path.exists(URL_LIST_FILE):
            raise FileNotFoundError(f"{URL_LIST_FILE} not found. Provide it or specify channels to extract from.")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        count = 0

        with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        print(f"\n ----- Starting transcript extraction for {len(urls)} videos. ----- ")

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