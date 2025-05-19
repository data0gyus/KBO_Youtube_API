import subprocess
import os

url = "https://www.youtube.com/watch?v=xjfMRJG0-MU"

# 저장 경로 폴더 생성
os.makedirs("downloads", exist_ok=True)

# yt-dlp 실행
subprocess.run([
    "yt-dlp",
    "-f", "best[ext=mp4][height<=720]",  # mp4 중 720p 이하 최고 화질
    "-o", "downloads/%(upload_date)s_%(title).50s.%(ext)s",  # 저장 경로 및 파일명
    url
])
