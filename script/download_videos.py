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

    # 제목 전처리
    clean_title = re.sub(r'[\\/*?:"<>|]', "", title).strip()

    # [팀 vs 팀] 추출
    match_vs = re.search(r'[^ ]+ vs [^ ]+', clean_title)

    # "더블헤더 1차전", "더블헤더 2차전", "경기" 등의 구분 정보 추출
    match_suffix = re.search(r'(더블헤더\s*\d차전|경기)', clean_title)

    if match_vs:
        vs_title = match_vs.group(0)
        suffix = match_suffix.group(0) if match_suffix else "영상"
        filename = f"{published_at}_[{vs_title}]_{suffix}"
    else:
        filename = f"{published_at}_[{clean_title[:20]}]"

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
        "--no-write-thumbnail",
        "--no-write-info-json",
        "--no-write-comments",
        "--no-write-sub",
        "--no-part",
        "-f", "best[ext=mp4][height<=720]",
        "-o", f"{video_path}.%(ext)s",
        url
    ])

print("🎉잠실 경기 하이라이트 영상 다운로드 완료")
