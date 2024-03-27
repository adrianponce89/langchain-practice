import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, help="Programming language", default="Python")
parser.add_argument("--task", type=str, help="Task to perform", default="print 'Hello, World!'")
args = parser.parse_args()

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
  openai_api_key=api_key
)

code_prompt = PromptTemplate(
  template="Write a very short {language} program that will {task}",
  input_variables=["language", "task"]
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt
)

result = code_chain({
  "language": args.language,
  "task": args.task
})

print(result["text"])