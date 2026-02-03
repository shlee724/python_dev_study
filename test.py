from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = gemini_api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="안녕! 한 문장으로 자기소개 해줘"
)

print(response.text)