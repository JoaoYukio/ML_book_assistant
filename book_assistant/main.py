import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool
from langchain.serpapi import SerpAPIWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub

from langchain.document_loaders import PyPDFLoader

import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

load_dotenv()

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENV"),
)

import streamlit as st
from PyPDF2 import PdfReader

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from templates.htmltemplate import css, bot_template, user_template

from langchain.chains import RetrievalQA


def get_pdf_docs(pdf_docs: list) -> str:
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text: str) -> list:
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    chunks = splitter.split_text(text)
    return chunks


def embed_and_store(text_chunks: list) -> Pinecone:
    embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vector_store = Pinecone.from_texts(
        texts=text_chunks, embedding=embedding, index_name="langchain-doc-index"
    )
    st.session_state.vector_store = vector_store
    return vector_store


def handle_user_input(user_question):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        # llm=HuggingFaceHub(
        #     repo_id="google/flan-t5-xxl",
        #     model_kwargs={
        #         "temperature": 0.3,
        #     },
        # ),
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever(),
        return_source_documents=True,  # Mostra aonde a LLM pegou as respostas
    )

    query = user_question
    result = qa({"query": query})

    st.write(user_template.replace("{{MSG}}", result["query"]), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", result["result"]), unsafe_allow_html=True)
    # st.write(
    #     bot_template.replace("{{MSG}}", str(result["source_documents"])),
    #     unsafe_allow_html=True,
    # )


def read_the_book_and_save() -> Pinecone:
    embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    loader = PyPDFLoader("data\DL_book_Goodfellow.pdf")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    chunks = splitter.split_documents(documents)

    vector_store = Pinecone.from_documents(
        chunks, embedding=embedding, index_name="langchain-doc-index"
    )

    st.session_state.vector_store = vector_store

    return vector_store


def main():
    st.set_page_config(page_title="Assistente de Deep Learning", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # read_the_book_and_save() #! Apenas na primera vez
    st.session_state.vector_store = Pinecone.from_existing_index(
        "langchain-doc-index",
        OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")),
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "history" not in st.session_state:
        st.session_state.history = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Digite sua pergunta sobre os pdfs")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader(
            "Carregar documento",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Carregar"):
            with st.spinner("Processando dados..."):
                raw_text = get_pdf_docs(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                embed_and_store(text_chunks)


if __name__ == "__main__":
    # llm = OpenAI(temperature=0.3, openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # search = SerpAPIWrapper(serpapi_api_key=os.environ.get("SERPAPI_API_KEY"))

    # print(search.run("Obama's first name?"))

    # https://python.langchain.com/docs/integrations/tools/arxiv
    # tools = load_tools(
    #     ["arxiv"],
    # )

    # agent_chain = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    # )

    # res = agent_chain.run(
    #     "What's the paper 1605.08386 about?",
    # )
    main()
