import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool
from langchain.serpapi import SerpAPIWrapper
from langchain.text_splitter import CharacterTextSplitter

import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

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


def embed_and_store(text_chunks: list) -> FAISS:
    embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    vector_store.save_local("faiss_index_react")
    new_vector_store = FAISS.load_local("faiss_index_react", embeddings=embedding)
    return new_vector_store


def get_conversation_chain(vector_store: FAISS):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    conversasion_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversasion_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})

    st.session_state.history = response["chat_history"]

    for i, msg in enumerate(st.session_state.history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True
            )


def main():
    st.set_page_config(page_title="Assistente de Deep Learning", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    st.text_input("Digite sua pergunta sobre deep learning")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "history" not in st.session_state:
        st.session_state.history = None

    st.write(user_template.replace("{{MSG}}", "Olá"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Olá"), unsafe_allow_html=True)

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

                vector_store = embed_and_store(text_chunks)

                st.session_state.conversation = get_conversation_chain(vector_store)


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
