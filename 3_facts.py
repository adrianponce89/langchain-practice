from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Split the text into characters
text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=200,
  chunk_overlap=0
)

# Load the text file
loader = TextLoader("assets/facts.txt")

# Load the documents
docs = loader.load_and_split(
  text_splitter=text_splitter
)

# Print the documents
for doc in docs:
  print(doc.page_content)
  print("\n")
