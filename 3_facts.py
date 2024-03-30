from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

embeddings = OpenAIEmbeddings()

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

db = Chroma.from_documents(
  docs,
  embedding=embeddings,
  persist_directory="emb"
)

results = db.similarity_search(
  "What is a interesting fact about the English language?"
)

for result in results:
  print("\n")
  print(result.page_content)
