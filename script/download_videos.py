import os
import json
import subprocess
import re


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
JSON_PATH = os.path.join(BASE_DIR, "data", "jansil.json")

# vidoes 다운로드 폴더 유무 확인 및 생성
os.makedirs(VIDEO_DIR, exist_ok=True)

# Json 로딩
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    video_data = json.load(f)

for item in video_data:
    video_id = item["videoId"]
    title = item["title"]
    published_at = item["publishedAt"][:10]  # 날짜만 출력 (2025-05-05)

    # 특수문자 삭제
    clean_title = re.sub(r'[\\/*?:"<>|]', "", title).strip()

    # 파일 이름 날짜_제목.mp4로 출력
    filename = f"{published_at}_{clean_title}.mp4"
    video_path = os.path.join(VIDEO_DIR, filename)

    # 이미 다운 완료시 건너뛰기
    if os.path.exists(video_path):
        print(f"✅ 이미 다운로드됨: {filename}")
        continue

    # 다운로드 시작
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"⬇️  다운로드 중: {filename}")

    subprocess.run([
        "yt-dlp",
        "-f", "bv[height<=720]+ba/b[height<=720]",
        "-o", video_path,
        url
    ])

print("🎉잠실 경기 하이라이트 영상 다운로드 완료")
