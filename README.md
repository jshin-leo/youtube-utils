# YouTube Utilities

This repository contains two Python scripts for working with YouTube links and transcripts:

## 1. YouTube_Scraper.py
A web crawler that:
- Accepts a list of starting URLs
- Crawls up to 10,000 internal pages
- Extracts YouTube video links
- Saves them to `youtube_links.txt`

## 2. YouTube_transcript_chunks.py
A transcript extractor that:
- Loads YouTube links from `youtube_links.txt`
- Downloads audio and generates transcripts using OpenAI Whisper
- Breaks transcripts into semantically meaningful chunks for RAG
- Saves output into the `data/` directory

## Requirements
- git+https://github.com/openai/whisper.git
- yt-dlp
- pandas
- sentence_transformers
- nltk
- requests==2.31.0
- beautifulsoup4==4.12.3
- selenium==4.19.0

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```
## Run
To crawl websites and collect YouTube video links:
- Add a list of starting URLs in the code, and execute the code.
```bash
python YouTube_Scraper.py
```
To generate transcript chunks from YouTube videos:
```bash
python YouTube_transcript_chunks.py
```
