import os, toml, sys
import time
from typing import Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.chat_models import ChatOpenAI

from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st

from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import OpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import streamlit as st




# Initialize Langfuse handler
from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    secret_key="sk-lf-8a05bde2-3bb3-4618-a68c-0d670ad6fbf6",
    public_key="pk-lf-65af33df-677d-4081-a0ac-e50ba6ddc4d7",
    host="https://cloud.langfuse.com", # 🇪🇺 EU region
)

# Your Langchain code
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

database = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))

retriever = database.as_retriever(search_type="similarity", search_kwargs={"k": 6})


llm = ChatOpenAI(model="openai/gpt-4o", base_url="https://openrouter.ai/api/v1", openai_api_key=API_KEY)



#--
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#for chunk in rag_chain.stream("¿Que poderes tiene Kaladin?", config={"callbacks":[langfuse_handler]}):
#  print(chunk, end="", flush=True)
    

# STREAMLIT

st.title("Streamlit - ChatBook")
st.write("Streamlit app sobre El camino de los reyes")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


with st.sidebar:
    openai_api_key = API_KEY

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="¿Qué duda tienes sobre el libro?")]

# Busca el último mensaje con el rol 'assistant' y modifica su contenido
for i in range(len(st.session_state["messages"]) - 1, -1, -1):
    if st.session_state["messages"][i].role == 'assistant' and st.session_state["messages"][i].content != "¿Qué duda tienes sobre el libro?":
        st.session_state["messages"][i].content = ''.join(st.session_state["messages"][i].content)
        break

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = rag_chain.stream(prompt, config={"callbacks":[stream_handler, langfuse_handler]})
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))