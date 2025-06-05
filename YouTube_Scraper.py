# ---------------------------------------------------------------------
#   - Title: YouTube_Scraper.py
#   - Author: JS
#   - Date: June 5th, 2025
#   - What: Crawl websites to extract YouTube links from all subpages
#   - How: 
#       1. Provide starting URLs (e.g., resource center)
#       2. This script crawls internal links and gathers YouTube URLs
#       3. Saves results to youtube_links.txt
# ---------------------------------------------------------------------
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

# === Settings ===
TARGETS = [
    ("https://nexusipe.org/informing/resource-center/", "/informing/resource-center/"),
    # Add more (start_url, allowed_path_prefix) pairs here
]

MAX_PAGES = 10000
OUTPUT_FILE = "youtube_links.txt"

visited_urls = set()
youtube_links = set()

def is_allowed_path(link, allowed_prefix):
    parsed = urlparse(link)
    return parsed.path.startswith(allowed_prefix)

def clean_url(url):
    return url.split('#')[0].strip('/')

def extract_internal_links(soup, base_url, allowed_prefix):
    links = set()
    for tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, tag["href"])
        if full_url.startswith("http") and is_allowed_path(full_url, allowed_prefix):
            links.add(clean_url(full_url))
    return links

def extract_youtube_links(soup):
    links = set()
    for a in soup.find_all("a", href=True):
        if "youtube.com/watch" in a["href"] or "youtu.be/" in a["href"]:
            links.add(a["href"])
    for iframe in soup.find_all("iframe", src=True):
        if "youtube.com/embed/" in iframe["src"] or "youtube.com/watch" in iframe["src"]:
            links.add(iframe["src"])
    return links

def crawl(start_url, allowed_prefix):
    queue = [start_url]
    count = 0
    youtubelink_cnt = 0

    while queue and count < MAX_PAGES:
        current_url = queue.pop(0)
        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        count += 1

        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract YouTube links
            found_links = extract_youtube_links(soup)
            if len(found_links) >= 1:
                youtubelink_cnt += 1
                print(f"Crawling ({count} pages): {current_url}")
                print(f"   ➜ Found {len(found_links)} YouTube links (current total: {youtubelink_cnt})")
                youtube_links.update(found_links)

            # Add crawlable internal links
            new_links = extract_internal_links(soup, current_url, allowed_prefix)
            for link in new_links:
                if link not in visited_urls:
                    queue.append(link)

        except Exception as e:
            print(f"[ERROR] Could not process {current_url}: {e}")

if __name__ == "__main__":
    print(" Starting full crawl...\n")
    for start_url, path_prefix in TARGETS:
        print(f"\n Starting crawl from: {start_url} (Allowed path: {path_prefix})")
        crawl(start_url, path_prefix)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for link in sorted(youtube_links):
            f.write(link + "\n")

    print(f"\n Done! Saved {len(youtube_links)} YouTube links to '{OUTPUT_FILE}'")