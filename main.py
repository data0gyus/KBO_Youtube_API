from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


import json
from dotenv import load_dotenv
import os

# .env 파일에서 API키 로드
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 타겟 재생 목록 ID = (KBO 2025 H/L 재생목록)
PLAYLIST_ID = "PLQSdHthiYH-_E5ElMhXJxm4a9YgDrF3ET"
MAX_RESULTS = 50


# Youtube API 객체 생성
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION, developerKey=API_KEY)

# test.json이 존재하면 API 호출 생략
if os.path.exists("test.json"):
    print("기존 test.json이 존재합니다.")
    with open("test.json", 'r', encoding='utf-8')as f:
        results = json.load(f)
# test.json이 없어서 API 호출
else:
    print("Youtube API를 호출합니다.")
    results = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=PLAYLIST_ID,
            maxResults=MAX_RESULTS,
            pageToken=next_page_token
        )
        response = request.execute()
        results.extend(response["items"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ test.json 저장 완료!")

# 잠실 경기만 추출
Jansil = [
    {
        "title": item["snippet"]["title"],
        "videoId": item["snippet"]["resourceId"]["videoId"],
        "publishedAt": item["snippet"]["publishedAt"]
    }
    for item in results
    if "vs 두산" in item["snippet"]["title"] or "vs LG" in item["snippet"]["title"]
]

# 필터링 결과 저장
with open("jansil.json", 'w', encoding="utf-8") as f:
    json.dump(Jansil, f, ensure_ascii=False, indent=2)

print(f"잠실 경기 저장 완료 {len(Jansil)}개")
