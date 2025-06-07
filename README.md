# YouTube Utilities

This repository contains two Python scripts for working with YouTube links and transcripts:

## Purpose

This project is intended to help researchers and developers make publicly available YouTube content more accessible for use in machine learning (ML) workflows, such as:

- **Retrieval-Augmented Generation (RAG):** The transcript chunks can serve as context for question-answering or summarization models.
- **Natural Language Processing (NLP):** Enables building datasets for topic modeling, classification, or semantic search.
- **Speech-to-Text Evaluation:** Allows testing or comparing different transcription models against real-world content.

It is especially useful when working with educational, research, or tutorial videos that are publicly accessible and intended for reuse.

⚠️ **Use responsibly**: This tool is not intended for scraping copyrighted or private content without permission.

## 1. YouTube_Scraper.py
A web crawler that:
- Accepts a list of starting URLs
- Crawls up to 10,000 internal pages
- Extracts YouTube video links
- Saves them to `youtube_links.txt`

## 2. YouTube_transcript_chunks.py
A transcript and chunk generator for YouTube videos, designed for use in RAG pipelines:
- Reads YouTube URLs from youtube_links.txt
- Downloads audio using yt-dlp and transcribes with OpenAI Whisper
- Extracts and filters meaningful sentences using spaCy and semantic similarity
- Chunks the transcript into coherent blocks using SentenceTransformers
- Saves transcripts, logs, and RAG-formatted .jsonl files in the data/ directory

## Requirements
- git+https://github.com/openai/whisper.git
- yt-dlp
- pandas
- sentence_transformers==2.5.1
- nltk
- requests==2.31.0
- beautifulsoup4==4.12.3
- selenium==4.19.0
- spacy==3.7.4
- torch>=1.11.0
- transformers>=4.40.0,<4.41.0
- scipy<1.12.0
- numpy>=1.23.5,<2.0.0
- tensorflow==2.15.0
- keras==2.15.0

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
