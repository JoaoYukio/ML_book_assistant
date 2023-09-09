import os
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool
from langchain.serpapi import SerpAPIWrapper
from langchain.text_splitter import CharacterTextSplitter

import streamlit as st
from PyPDF2 import PdfReader


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


def main():
    st.set_page_config(page_title="Assistente de Deep Learning", page_icon="📚")
    st.text_input("Digite sua pergunta sobre deep learning")

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader(
            "Carregar documento",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )
        if st.button("Carregar"):
            with st.spinner("Processando dados..."):
                raw_text = get_pdf_docs(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)


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
