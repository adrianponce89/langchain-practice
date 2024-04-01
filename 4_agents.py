from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv

from tools.slq import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool

load_dotenv()

chat = ChatOpenAI()

tables = list_tables()


chat_prompt = ChatPromptTemplate(
  messages=[
    SystemMessage(content=(
      "You are an AI that have access to a SQLite database.\n"
      f"The database contains the following tables: {tables}\n"
      "Do not make any assumptions about the schema of the tables. "
      "Intead, use the 'describe_tables' tool to get the schema of a table.\n"
      )),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
  ]
)

tools = [
  run_query_tool,
  describe_tables_tool,
  write_report_tool
]

agent = OpenAIFunctionsAgent(
  llm=chat,
  prompt=chat_prompt,
  tools=tools
)

agent_executor = AgentExecutor(
  agent=agent,
  verbose=True,
  tools=tools
)

#agent_executor("How many users have provided their shiping address?")
agent_executor("Summarize the top 5 most popular products with their names. Write the report to 'top_products.html'")