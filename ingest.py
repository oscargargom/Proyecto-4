import os, toml, sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def load_api_key(toml_file_path="secrets.toml"):
    try:
        with open(toml_file_path, 'r') as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {toml_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {toml_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)

load_api_key()
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_KEY_OPENAI = os.environ.get("OPENAI_API_KEY")
model = "openai/gpt-4-turbo"


loader = TextLoader("libro/El_camino_de_los_reyes_Brandon_Sanderson.md")
data = loader.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=300, add_start_index=True
)
all_splits = text_splitter.split_documents(data)


print(len(all_splits))

print(all_splits[0])



vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(model="text-embedding-3-large"), persist_directory="./chroma_db")
