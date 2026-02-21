from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")

t0 = time.perf_counter()
client = genai.Client(api_key = gemini_api_key)
t1 = time.perf_counter()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="상대 탑 모데카이저, 정글 잭스, 미드 요네, 원딜 진, 서폿 소라카" \
    "우리 탑 럼블, 정글 바이, 원딜 케이틀린, 서폿 럭스" \
    "이때 내가 미드 5픽인데 뽑아야 될 픽은?"
    "[오리아나, 말자하, 갈리오, 말파이트, 문도, 가렌, 초가스, 사이온, 카사딘, 트린다미어, 아칼리, 모르가나, 스웨인, 워윅, 트런들] 중" \
    "1,2,3순위 추천 & 점수부여, 이유는 한문장으로 간단하게."
)

t2 = time.perf_counter()

print("genai 클라이언트 로드:", t1 - t0)
print("Gemini API 호출:", t2 - t1)
print("전체:", t2 - t0)
print(response.text)