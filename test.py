from google import genai
from google.genai import types
from dotenv import load_dotenv
import os, time

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        # ms 단위. 10초 “완료 보장”이 아니라, 10초 넘으면 끊고 에러 내게 하는 안전장치
        timeout=10_000,
        # 기본 재시도로 지연이 늘어날 수 있어서(최대 4회+백오프) attempts를 줄여 둠 :contentReference[oaicite:1]{index=1}
        retry_options=types.HttpRetryOptions(
            attempts=1,
            initial_delay=0.2,
            max_delay=0.5,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        ),
    ),
)

prompt = """너는 LoL 밴픽 코치다. 아래 정보로 미드 5픽을 추천한다.

출력 규칙:
- 추천 3개만 출력한다.
- 각 추천은 정확히 2줄로만 출력한다:
  1) <챔피언> - <점수>/10
  2) 라인전: <핵심근거 1개> | 팀가치: <핵심근거 1개>
- 추가 설명, 요약, 머리말 금지.

평가 원칙:
- 라인전 평가는 "웨이브 주도권 및 딜교환 주도권" 기준으로 판단한다. 단순히 죽지 않는 픽은 유리로 평가하지 않는다.
- 팀가치(운영/교전/한타)는 라인전 판단 이후에 반영한다.
- 확신이 없으면 점수는 8/10을 넘기지 않는다.
- 후보 풀은 포지션 고정이 아니며, 미드 탱/서포트형 픽도 허용한다.

상대:
탑 모데카이저, 정글 잭스, 미드 요네, 원딜 진, 서폿 소라카
우리:
탑 럼블, 정글 바이, 원딜 케이틀린, 서폿 럭스
나는 미드 5픽.

후보 풀:
[오리아나, 말자하, 갈리오, 말파이트, 문도, 가렌, 초가스, 사이온, 카사딘, 트린다미어, 아칼리, 모르가나, 스웨인, 워윅, 트런들]
"""
t0 = time.perf_counter()
first_token_t = None
buf = []

try:
    for chunk in client.models.generate_content_stream(  # 스트리밍 API :contentReference[oaicite:2]{index=2}
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300,  # 짧게 강제
            thinking_config=types.ThinkingConfig(
                thinking_budget=128  # 0 = thinking 비활성 :contentReference[oaicite:3]{index=3}
            ),
        ),
    ):
        if not chunk.text:
            continue

        if first_token_t is None:
            first_token_t = time.perf_counter()
            print(f"\n⏱ 첫 토큰: {first_token_t - t0:.2f}s\n")

        print(chunk.text, end="", flush=True)
        buf.append(chunk.text)

    t1 = time.perf_counter()
    print(f"\n\n⏱ 전체: {t1 - t0:.2f}s")

except Exception as e:
    t1 = time.perf_counter()
    print(f"\n❌ 실패(총 {t1 - t0:.2f}s): {e!r}")