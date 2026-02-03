from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# 1. 환경변수 로드
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# 2. Gemini 클라이언트
client = genai.Client(api_key=gemini_api_key)

# 3. 이미지 경로
image_path = "images/sample.jpg"

# 4. 이미지 읽기
with open(image_path, "rb") as f:
    image_bytes = f.read()

# 5. 이미지 Part 생성 (⭐ 핵심)
image_part = types.Part.from_bytes(
    data=image_bytes,
    mime_type="image/jpeg"
)

# 6. 이미지 분석 요청
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        "이 이미지에 대해 자세히 설명해줘(한글로)",
        image_part
    ]
)

# 7. 결과 출력
print(response.text)
