import json

with open("jansil.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"총 영상 개수: {len(data)}개")
