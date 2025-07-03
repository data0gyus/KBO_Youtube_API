import cv2
import face_recognition
import os
import time
import datetime
from tqdm import tqdm
import numpy as np
import pickle


def enhance_image_quality(image):
    """
    이미지 품질 향상으로 얼굴 감지율 개선
    1. 히스토그램 평활화를 통해 이미지의 명암 대비를 개선합니다.
    2. 샤프닝 필터를 적용하여 이미지의 선명도를 높입니다.
    """
    # 1. 히스토그램 평활화 (LAB 색 공간에서 L 채널에 적용)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 2. 샤프닝 필터 적용 (이미지 선명도 향상)
    # 이 커널은 이미지의 가장자리를 강조하여 선명하게 만듭니다.
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def augment_reference_image(image, face_location):
    """
    기준 이미지에서 다양한 변형(회전, 밝기, 반전)을 생성하여
    얼굴 인식의 강건성을 높입니다.
    dlib의 얼굴 랜드마크 감지가 안정적으로 이루어지도록 얼굴 크롭 시 약간의 마진을 추가합니다.
    """
    top, right, bottom, left = face_location

    # 얼굴 크롭 시 여유 공간 추가 (랜드마크 감지에 도움)
    margin = 20  # 픽셀 단위로 여유 공간 추가 (상하좌우에 20픽셀씩 추가)
    y1 = max(0, top - margin)
    y2 = min(image.shape[0], bottom + margin)
    x1 = max(0, left - margin)
    x2 = min(image.shape[1], right + margin)

    face_crop = image[y1:y2, x1:x2]  # 마진이 적용된 얼굴 영역 크롭

    augmented_faces = []

    # 1. 원본 크롭 이미지
    augmented_faces.append(face_crop)

    # 2. 좌우 반전 (프로필 대칭성 활용, 옆 얼굴 인식에 도움)
    flipped = cv2.flip(face_crop, 1)  # 1은 좌우 반전을 의미
    augmented_faces.append(flipped)

    # 3. 밝기 조정 (다양한 조명 조건에 대응)
    # alpha는 대비, beta는 밝기를 조절합니다.
    bright = cv2.convertScaleAbs(face_crop, alpha=1.2, beta=10)  # 더 밝게
    dark = cv2.convertScaleAbs(face_crop, alpha=0.8, beta=-10)  # 더 어둡게
    augmented_faces.extend([bright, dark])

    # 4. 약간의 회전 (-15도, +15도) (얼굴 각도 변화에 대응)
    height, width = face_crop.shape[:2]
    center = (width // 2, height // 2)

    for angle in [-15, 15]:  # -15도와 +15도로 각각 회전
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 회전 변환 행렬 생성
        # 회전 시 이미지 밖으로 나가는 부분을 검은색(BORDER_CONSTANT) 대신
        # 가장자리 픽셀을 복제(BORDER_REPLICATE)하여 채웁니다.
        # 이는 dlib의 랜드마크 감지기가 더 안정적으로 작동하도록 돕습니다.
        rotated = cv2.warpAffine(face_crop, matrix, (width, height),
                                 borderMode=cv2.BORDER_REPLICATE)
        augmented_faces.append(rotated)

    return augmented_faces


def create_multiple_encodings_from_single_image(net, image_path):
    """
    단일 기준 이미지에서 여러 개의 증강된 얼굴 인코딩을 생성합니다.
    계산된 인코딩은 캐시 파일로 저장하여 다음 실행 시 재사용함으로써
    긴 처리 시간을 단축합니다.
    """
    # 인코딩 저장 경로 및 파일명 설정
    cache_dir = "face_encodings_cache"  # 캐시 파일을 저장할 폴더명
    os.makedirs(cache_dir, exist_ok=True)  # 폴더가 없으면 생성

    # 이미지 파일명을 기반으로 캐시 파일명 생성 (확장자 제거)
    # 예: "sample2.jpg" -> "sample2_augmented_encodings.pkl"
    cache_filename = os.path.splitext(os.path.basename(image_path))[
        0] + "_augmented_encodings.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 캐시 파일이 존재하는지 확인하고, 있다면 로드하여 반환
    if os.path.exists(cache_path):
        print(f"[INFO] 캐시된 인코딩 로드 중: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                encodings = pickle.load(f)
            print(f"[INFO] 캐시 로드 완료. 총 {len(encodings)}개의 인코딩.")
            return encodings
        except Exception as e:
            # 캐시 파일이 손상되었거나 로드 실패 시, 경고 메시지 출력 후 다시 계산
            print(f"[WARNING] 캐시 로드 실패: {e}. 인코딩을 다시 계산합니다.")
            # 손상된 캐시 파일은 삭제하여 다음 실행 시 다시 생성되도록 함
            os.remove(cache_path)

    # 캐시 파일이 없거나 로드 실패 시 새로 계산
    print(f"[INFO] 기준 얼굴 로딩 및 증강 중: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] 이미지 로딩 실패")
        return []

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_image = enhance_image_quality(rgb_image)

    # 기준 이미지에서 얼굴 감지 (OpenCV DNN 사용)
    print("[INFO] 기준 이미지에서 얼굴 감지 중 (OpenCV DNN 모델 사용)...")
    start_time_detection = time.time()  # 감지 시작 시간 기록
    face_locations = detect_faces_opencv_dnn(
        net, enhanced_image, confidence_threshold=0.7)
    end_time_detection = time.time()  # 감지 종료 시간 기록
    print(
        f"[INFO] 기준 이미지 얼굴 감지 완료 (소요 시간: {end_time_detection - start_time_detection:.2f}초)")

    if not face_locations:
        print("❌ 기준 얼굴에서 얼굴을 감지하지 못했습니다. 정확한 기준 이미지를 사용해주세요.")
        return []

    # 감지된 첫 번째 얼굴의 위치 사용 (단일 인물 기준으로 가정)
    reference_face_location = face_locations[0]
    print(f"[INFO] 기준 이미지에서 얼굴 1개 감지 완료: {reference_face_location}")

    # 이미지 증강 (원본 얼굴 위치를 기반으로 크롭 및 변형)
    augmented_faces = augment_reference_image(
        enhanced_image, reference_face_location)
    print(f"[INFO] 총 {len(augmented_faces)}개의 증강 얼굴 이미지 생성 완료.")

    encodings = []
    # tqdm을 사용하여 증강 얼굴 인코딩 진행 상황을 시각적으로 표시
    for i, face_aug in enumerate(tqdm(augmented_faces, desc="✨ 증강 얼굴 인코딩 중")):
        try:
            # 증강된 이미지(face_aug)에서 얼굴을 다시 찾고 인코딩을 수행합니다.
            # 회전 시 검은 배경 대신 BORDER_REPLICATE를 사용하여 dlib의 안정성을 높였으므로,
            # 라이브러리가 스스로 얼굴을 찾도록 하는 것이 가장 강건한 방법입니다.

            h, w, _ = face_aug.shape
            face_aug_rgb = cv2.cvtColor(face_aug, cv2.COLOR_BGR2RGB)
            face_locations_for_aug = [(0, w, h, 0)]

            encs = face_recognition.face_encodings(
                face_aug_rgb,
                known_face_locations=face_locations_for_aug
            )

            if encs:  # 인코딩이 성공적으로 생성되었다면
                encodings.append(encs[0])  # 첫 번째 인코딩만 사용 (얼굴이 하나라고 가정)
            else:
                # 얼굴을 찾지 못하여 인코딩이 생성되지 않은 경우 경고 메시지 출력
                print(
                    f"\n[WARNING] 증강 얼굴 #{i+1} 에서 얼굴을 찾지 못하여 인코딩 실패 (인코딩 없음).")

        except Exception as e:
            # 인코딩 중 예외 발생 시 오류 메시지 출력 및 디버그 정보 제공
            print(f"\n[ERROR] 증강 얼굴 #{i+1} 인코딩 중 예외 발생: {e}")
            print(
                f"[DEBUG] 해당 이미지 shape: {face_aug.shape}, dtype: {face_aug.dtype}")
            # 디버깅을 위해 문제된 이미지 저장 (필요 시 주석 해제)
            # cv2.imwrite(f"debug_augmented_face_{i+1}_error.jpg", cv2.cvtColor(face_aug, cv2.COLOR_RGB2BGR))

    print(f"[INFO] 총 {len(encodings)}개의 인코딩 생성 완료")

    # 계산된 인코딩을 캐시 파일로 저장
    if encodings:
        try:
            with open(cache_path, 'wb') as f:  # 바이너리 쓰기 모드 ('wb')
                pickle.dump(encodings, f)  # 인코딩 리스트를 파일에 덤프
            print(f"[INFO] 인코딩 캐시 저장 완료: {cache_path}")
        except Exception as e:
            print(f"[ERROR] 인코딩 캐시 저장 실패: {e}")
    else:
        print("[WARNING] 생성된 인코딩이 없어 캐시 파일을 저장하지 않습니다.")
    return encodings


def iou(boxA, boxB):
    """
    IoU (Intersection over Union)를 계산하여 두 개의 바운딩 박스(얼굴 위치)가
    얼마나 겹치는지 측정합니다. 중복 얼굴 감지 제거에 사용됩니다.
    boxA와 boxB는 (top, right, bottom, left) 형식의 튜플입니다.
    """
    # 겹치는 영역의 좌표를 계산합니다.
    xA = max(boxA[3], boxB[3])  # 겹치는 사각형의 왼쪽 상단 x 좌표
    yA = max(boxA[0], boxB[0])  # 겹치는 사각형의 왼쪽 상단 y 좌표
    xB = min(boxA[1], boxB[1])  # 겹치는 사각형의 오른쪽 하단 x 좌표
    yB = min(boxA[2], boxB[2])  # 겹치는 사각형의 오른쪽 하단 y 좌표

    # 겹치는 영역의 너비와 높이를 계산합니다.
    inter_width = xB - xA
    inter_height = yB - yA

    # 겹치는 영역이 없으면 (너비 또는 높이가 0 이하) IoU는 0.0
    if inter_width <= 0 or inter_height <= 0:
        return 0.0

    # 겹치는 영역의 면적
    inter_area = inter_width * inter_height

    # 각 바운딩 박스의 면적을 계산합니다.
    boxA_area = (boxA[1] - boxA[3]) * (boxA[2] - boxA[0])
    boxB_area = (boxB[1] - boxB[3]) * (boxB[2] - boxB[0])

    # IoU를 계산합니다: 겹치는 면적 / (박스 A 면적 + 박스 B 면적 - 겹치는 면적)
    iou_val = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou_val


def detect_faces_opencv_dnn(net, frame, confidence_threshold=0.5):
    """OpenCV의 DNN 모델을 사용하여 프레임에서 얼굴 위치를 감지합니다."""
    (h, w) = frame.shape[:2]
    # 모델이 기대하는 300x300 크기로 이미지 변환 및 블롭 생성
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    face_locations = []

    # 감지 결과 반복
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 신뢰도가 임계값보다 높은 경우에만 처리
        if confidence > confidence_threshold:
            # 바운딩 박스 좌표 계산
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # face_recognition 라이브러리가 사용하는 (top, right, bottom, left) 형식으로 변환
            face_locations.append((startY, endX, endY, startX))

    return face_locations


def advanced_face_detection(net, frame):
    """
    OpenCV의 DNN 모델을 사용하여 프레임에서 얼굴을 감지하고,
    face_recognition을 사용하여 인코딩합니다.
    """
    # OpenCV DNN을 사용하여 얼굴 위치 감지
    # BGR 프레임을 그대로 사용해도 무방합니다.
    face_locations = detect_faces_opencv_dnn(
        net, frame, confidence_threshold=0.7)

    # face_recognition은 RGB 이미지를 기대하므로 인코딩 전에 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 감지된 위치를 기반으로 얼굴 인코딩
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    return face_locations, face_encodings


def process_video_enhanced_single_ref(video_path, reference_img_path, interval=1, tolerance=0.55):
    """
    단일 기준 인물 이미지와 영상 파일을 비교하여,
    영상 내에서 기준 인물과 일치하는 얼굴을 찾아 저장합니다.
    향상된 얼굴 감지 및 중복 저장 방지 로직을 포함합니다.
    """

    # 0. OpenCV DNN 모델 로드
    proto_path = "C:/mini/models/deploy.prototxt"
    model_path = "C:/mini/models/res10_300x300_ssd_iter_140000.caffemodel"
    print("[INFO] OpenCV DNN 얼굴 감지 모델 로딩 중...")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    print("[INFO] 모델 로딩 완료.")

    # 1. 기준 얼굴에서 다중 인코딩 생성 (캐싱 기능 활용)
    reference_encodings = create_multiple_encodings_from_single_image(
        net, reference_img_path)
    if not reference_encodings:
        print("[ERROR] 기준 얼굴 처리 실패. 프로그램 종료.")
        return

    # 2. 영상 열기
    print(f"[INFO] 영상 로딩 중: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():  # 영상 파일이 제대로 열리지 않으면 오류 메시지 출력
        print("[ERROR] 영상 열기 실패:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # 영상의 초당 프레임 수(FPS) 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 영상의 총 프레임 수 가져오기
    # 검사할 프레임 간격 (예: 1초에 한 번 검사하려면 fps 만큼)
    frame_interval = int(fps * interval)

    # 중복 저장 방지 로직: 최근 매칭 프레임과 쿨다운 프레임 수 설정
    cooldown_frames = int(fps * 2)  # 2초에 한 번만 저장 (동일 인물 중복 저장 방지)
    recent_match_frame = -cooldown_frames  # 초기값을 음수로 설정하여 첫 매칭 허용

    # 매칭된 얼굴 이미지를 저장할 폴더 생성
    os.makedirs("matched_faces", exist_ok=True)

    frame_count = 0  # 현재 처리 중인 프레임 번호
    match_count = 0  # 매칭된 얼굴 수

    # tqdm을 사용하여 전체 영상 처리 진행 상황 표시
    progress = tqdm(total=total_frames, desc="🎞 DNN 영상 분석 중", unit="frame")

    # 영상 프레임별로 처리
    while cap.isOpened():
        ret, frame = cap.read()  # 한 프레임 읽기
        if not ret:  # 프레임을 더 이상 읽을 수 없으면 루프 종료
            break

        # 지정된 간격마다 프레임 검사
        if frame_count % frame_interval == 0:
            # 고급 얼굴 감지 적용
            face_locations, face_encodings = advanced_face_detection(
                net, frame)

            # 현재 프레임에서 감지된 모든 얼굴에 대해 반복
            for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                # 모든 기준 인코딩(증강된 인코딩 포함)과 현재 감지된 얼굴 인코딩 간의 거리 계산
                distances = face_recognition.face_distance(
                    reference_encodings, encoding)
                min_distance = min(distances)  # 가장 짧은 거리 (가장 유사한 인코딩)
                best_match_idx = np.argmin(distances)  # 가장 유사한 기준 인코딩의 인덱스

                # 매칭 기준 충족 (거리 임계값 tolerance보다 작고, 쿨다운 기간이 지났는지)
                if min_distance < tolerance and frame_count - recent_match_frame >= cooldown_frames:
                    recent_match_frame = frame_count  # 마지막 매칭 프레임 업데이트
                    time_position_sec = frame_count / fps  # 현재 프레임의 시간(초) 계산
                    time_str = str(datetime.timedelta(  # 시:분:초 형식으로 변환
                        seconds=int(time_position_sec)))
                    print(f"\n✅ 얼굴 매칭 성공: {time_str}")
                    print(
                        f"   거리: {min_distance:.4f} (기준 변형 #{best_match_idx + 1})")

                    # 얼굴 영역 확대해서 저장
                    # 매칭된 얼굴 주변에 여백을 추가하여 저장 (얼굴 전체가 잘리지 않도록)
                    margin = 50  # 50 픽셀 여백
                    y1 = max(0, top - margin)
                    y2 = min(frame.shape[0], bottom + margin)
                    x1 = max(0, left - margin)
                    x2 = min(frame.shape[1], right + margin)

                    face_crop_to_save = frame[y1:y2, x1:x2]  # 얼굴 영역 크롭
                    # 저장할 파일명 생성 (프레임 번호, 시간, 거리 포함)
                    save_path = os.path.join("matched_faces",
                                             f"frame_{frame_count}_t{time_str.replace(':', '')}_d{min_distance:.3f}.jpg")
                    cv2.imwrite(save_path, face_crop_to_save)  # 이미지 저장
                    match_count += 1  # 매칭된 얼굴 수 증가

                    # 매칭 성공 시 해당 프레임 전체를 저장하고 얼굴에 사각형 그리기 (선택사항)
                    full_frame_path = os.path.join("matched_faces",
                                                   f"full_frame_{frame_count}.jpg")
                    # 원본 프레임에 얼굴 위치 표시 (초록색 사각형)
                    cv2.rectangle(frame, (left - margin, top - margin),  # 마진이 적용된 좌표로 사각형 그리기
                                  (right + margin, bottom + margin), (0, 255, 0), 3)  # (B,G,R), 두께 3
                    cv2.imwrite(full_frame_path, frame)  # 전체 프레임 저장
                    break  # 해당 프레임에서 얼굴 매칭이 성공하면 더 이상 다른 얼굴을 찾지 않고 다음 프레임으로 이동

        frame_count += 1  # 다음 프레임으로 이동
        progress.update(1)  # tqdm 진행률 업데이트

    cap.release()  # 영상 파일 해제
    progress.close()  # tqdm 진행률 바 닫기
    print(f"\n[완료] 총 매칭된 얼굴: {match_count}개")
    print(f"결과 저장 위치: matched_faces/ 폴더")


# 사용 예시
if __name__ == "__main__":
    # 분석할 영상 파일 경로
    # video_path = "C:/mini/videos/2025-05-11_[[NC vs 두산]]_더블헤더 2차전.mp4"
    video_path = "C:/mini/downloads/20250512.mp4"  # 다른 영상 테스트 시 주석 해제

    # 기준 인물 얼굴 이미지 파일 경로
    reference_img_path = "C:/mini/sample2.jpg"

    # 얼굴 인식 프로세스 실행
    process_video_enhanced_single_ref(
        video_path=video_path,
        reference_img_path=reference_img_path,
        interval=0.5,  # 0.5초(약 15프레임)마다 영상 프레임 검사 (간격을 줄이면 더 꼼꼼히 검사하지만 속도 저하)
        tolerance=0.55  # 얼굴 유사도 허용치 (낮을수록 더 엄격, 높을수록 더 관대)
        # 옆얼굴이나 다양한 각도/표정에 대응하기 위해 0.6에서 약간 낮춤 (더 엄격해짐)
    )
