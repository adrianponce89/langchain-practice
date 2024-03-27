from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
  openai_api_key=api_key
)

result = llm('translate English to French: Hello, how are you?')

print(result)