from PIL import Image, ImageDraw, ImageFont
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

try:
    font = ImageFont.truetype("arial.ttf", 12)
except:
    font = ImageFont.load_default()

for y_idx, y in enumerate(range(0, height, grid_size)):
    for x_idx, x in enumerate(range(0, width, grid_size)):
        # 격자 좌표 텍스트
        text = f"({x_idx},{y_idx})"

        # 텍스트 위치 (칸 안쪽)
        text_x = x + 3
        text_y = y + 3

        draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)

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
        이 이미지는 격자 기반 매크로용 화면이다.

        규칙:
        1. 격자 좌표는 반드시 (x, y) 정수 형태로만 답한다.
        2. 각 UI 요소당 좌표는 정확히 하나만 제시한다.
        3. 기준은 해당 UI 요소의 '중심이 위치한 격자 칸'이다.
        4. 추정, 대략, 설명 표현을 사용하지 마라.

        아래 JSON 형식으로만 출력하라:
        {
        "id_field": [x, y],
        "password_field": [x, y],
        "login_button": [x, y]
        }
        """,
        image_part
    ],
    config=types.GenerateContentConfig(
        temperature=0.0
    )
)

print(response.text)
