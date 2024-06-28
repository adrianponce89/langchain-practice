from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from tools.slq import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()

chat = ChatOpenAI(
  callbacks=[handler]
)

tables = list_tables()


chat_prompt = ChatPromptTemplate(
  messages=[
    SystemMessage(content=(
      "You are an AI that have access to a SQLite database.\n"
      f"The database contains the following tables: {tables}\n"
      "Do not make any assumptions about the schema of the tables. "
      "Intead, use the 'describe_tables' tool to get the schema of a table.\n"
      )),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
  ]
)

memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True # Return the messages to the agent as objects instead of strings
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
  # verbose=True,
  tools=tools,
  memory=memory
)

# agent_executor("How many users have provided their shiping address?")
# agent_executor("Summarize the top 5 most popular products with their names. Write the report to 'top_products.html'")
agent_executor("How many orders there are?")
# agent_executor("Repeat the same procces for users")

# while True:
#   content = input(">> ")
#   agent_executor(content)