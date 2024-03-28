import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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

test_prompt = PromptTemplate(
  template="Write a unit test for the following {language} code:\n{code}",
  input_variables=["language", "code"]
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt,
  output_key="code"
)

test_chain = LLMChain(
  llm=llm,
  prompt=test_prompt,
  output_key="test"
)

chain = SequentialChain(
  chains=[code_chain, test_chain],
  input_variables=["language", "task"],
  output_variables=["code", "test"]
)

result = chain({
  "language": args.language,
  "task": args.task
})

print("# Code:")
print(result["code"])
print("# Test:")
print(result["test"])
