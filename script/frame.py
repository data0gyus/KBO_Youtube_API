import cv2
import face_recognition
import os
import dlib
import time
import datetime
from tqdm import tqdm


def process_video_in_memory(video_path: str, reference_img_path: str, interval: int = 1, tolerance: float = 0.6):
    # 1. 기준 얼굴 로딩
    print(f"[INFO] 기준 얼굴 로딩 중: {reference_img_path}")
    reference_img = cv2.imread(reference_img_path)
    if reference_img is None:
        print("[ERROR] 기준 이미지 로딩 실패 (None 반환)")
        return

    rgb_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    print("[DEBUG] 기준 이미지 shape:", rgb_img.shape)

    # 2. 얼굴 감지
    print("[INFO] 기준 얼굴 감지 중...")
    start = time.time()
    face_locations = face_recognition.face_locations(
        rgb_img, model="hog", number_of_times_to_upsample=0)
    end = time.time()
    print(f"[DEBUG] 감지 완료 - 소요 시간: {round(end - start, 2)}초")
    print("[DEBUG] 감지된 얼굴 개수:", len(face_locations))
    if not face_locations:
        print("❌ 기준 얼굴에서 얼굴을 감지하지 못했습니다.")
        return

    # 3. 얼굴 인코딩
    print("[INFO] 얼굴 인코딩 중...")
    try:
        reference_encoding = face_recognition.face_encodings(
            rgb_img, face_locations)[0]
        print("[INFO] 기준 얼굴 인코딩 완료")
    except Exception as e:
        print("[ERROR] 얼굴 인코딩 실패:", e)
        return

    # 4. 영상 열기
    print(f"[INFO] 영상 로딩 중: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 영상 열기 실패:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)

    os.makedirs("matched_faces", exist_ok=True)

    frame_count = 0
    match_count = 0

    progress = tqdm(total=total_frames, desc="🎞 영상 분석 중", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_frame, model="hog", number_of_times_to_upsample=0)
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations)

            for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                distance = face_recognition.face_distance(
                    [reference_encoding], encoding)[0]

                if distance < tolerance:
                    time_position_sec = frame_count / fps
                    time_str = str(datetime.timedelta(
                        seconds=int(time_position_sec)))
                    print(f"\n✅ 얼굴 매칭 성공: {time_str} 위치 (거리: {distance:.4f})")

                    save_path = os.path.join(
                        "matched_faces", f"frame_{frame_count}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(f"[저장 완료] {save_path}")
                    match_count += 1
                else:
                    print(f"\n⚠️ 유사하지만 기준 거리 초과 (거리: {distance:.4f}) → 무시")

        frame_count += 1
        progress.update(1)

    cap.release()
    progress.close()
    print(f"\n[완료] 총 매칭된 프레임 수: {match_count}장")


if __name__ == "__main__":
    # video_path = "C:/mini/videos/2025-04-11_[[두산 vs LG]]_경기.mp4"
    video_path = "C:/mini/downloads/20250512.mp4"
    reference_img_path = "C:/mini/sample.jpg"

    process_video_in_memory(
        video_path=video_path,
        reference_img_path=reference_img_path,
        interval=1,  # 초당 1프레임 검사
        tolerance=0.45
    )
