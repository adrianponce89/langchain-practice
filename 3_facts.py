from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Load the text file
loader = TextLoader("assets/facts.txt")

# Load the documents
docs = loader.load()

# Print the documents
print(docs)