from PIL import Image, ImageDraw
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# ===============================
# 1. 환경변수 & Gemini 설정
# ===============================
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

# ===============================
# 2. 이미지 경로
# ===============================
input_image_path = "images/sample.jpg"
output_image_path = "images/grided_sample.jpg"

# ===============================
# 3. 이미지 로드
# ===============================
img = Image.open(input_image_path).convert("RGB")
draw = ImageDraw.Draw(img)

width, height = img.size

# ===============================
# 4. 격자 설정 (매크로용 핵심)
# ===============================
grid_size = 50   # 격자 한 칸 크기(px) ← 여기 중요

# 세로선
for x in range(0, width, grid_size):
    draw.line((x, 0, x, height), fill=(255, 0, 0), width=1)

# 가로선
for y in range(0, height, grid_size):
    draw.line((0, y, width, y), fill=(255, 0, 0), width=1)

# ===============================
# 5. 가공 이미지 저장
# ===============================
img.save(output_image_path)
print(f"격자 이미지 저장 완료: {output_image_path}")

# ===============================
# 6. Gemini에 전달
# ===============================
with open(output_image_path, "rb") as f:
    image_bytes = f.read()

image_part = types.Part.from_bytes(
    data=image_bytes,
    mime_type="image/jpeg"
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        """
        이 이미지에는 격자가 표시되어 있다.
        특정 버튼이나 UI 요소가 위치한 격자 좌표를 설명해줘.
        (좌상단 기준)
        """,
        image_part
    ]
)

print(response.text)
