import face_recognition
import cv2

img_path = "C:/mini/sample2.jpg"
image = face_recognition.load_image_file(img_path)

face_locations = face_recognition.face_locations(image)
print("[INFO] 얼굴 감지 개수:", len(face_locations))

# 얼굴 박스 시각화
for top, right, bottom, left in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# 이미지 띄우기
cv2.imshow("Detected Face", image[:, :, ::-1])  # BGR 변환
cv2.waitKey(0)
cv2.destroyAllWindows()
