from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Define the language model
chat = ChatOpenAI()

# Define the memory
memory = ConversationSummaryMemory(
  # chat_memory=FileChatMessageHistory("chat_memory.json"),
  memory_key="messages",
  return_messages=True,
  llm=chat
)

# Define a prompt
prompt = ChatPromptTemplate(
  input_variables=["content", "messages"],
  messages=[
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

# Define the chain
chain = LLMChain(
  llm=chat,
  prompt=prompt,
  memory=memory,
  verbose=True # Set to True to see the prompt and response
)

while True:
  content = input(">> ")

  # Run the chain
  result = chain.run({"content": content})

  print(result)