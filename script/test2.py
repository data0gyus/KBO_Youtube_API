import face_recognition
import time

img_path = "C:/mini/sample2.jpg"
img = face_recognition.load_image_file(img_path)

print("[TEST] 이미지 로드 완료")
start = time.time()
faces = face_recognition.face_locations(
    img, model="hog", number_of_times_to_upsample=0)
print("[TEST] 얼굴 개수:", len(faces))
print("[TEST] 처리 시간:", round(time.time() - start, 2), "초")
