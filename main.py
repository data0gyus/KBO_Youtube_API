from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

SEARCH_QUERY = '2025 KBO 리그 H/L'
CHANNEL_ID = 'UCoVz66yWHzVsXAFG8WhJK9g'
MAX_RESULTS = 10

PUBLISHED_AFTER = '2025-05-05T00:00:00Z'
PUBLISHED_BEFORE = '2025-05-12T00:00:00Z'

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION, developerKey=API_KEY)

request = youtube.search().list(
    part='snippet',
    q=SEARCH_QUERY,
    channelId=CHANNEL_ID,
    maxResults=MAX_RESULTS,
    publishedAfter=PUBLISHED_AFTER,
    publishedBefore=PUBLISHED_BEFORE,
    type='video'
)

search_response = request.execute()


with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(search_response, f, ensure_ascii=False, indent=2)

print("✅ test.json 저장 완료!")
