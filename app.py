import streamlit as st
from dotenv import load_dotenv
import os
# from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings


def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')


def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf_file in pdf_docs:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            pdf_text += page_text
    return pdf_text


def get_chunks_rec(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_chunks_tiktoken(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

# Instructor Embeddings (FREE!)


def get_vectorstore_instructor(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


def main():

    load_dotenv()

    st.set_page_config(page_title="Multi-pdf-chat", page_icon=":sharks:")

    st.subheader("Chat with multiple pdf documents :sunglasses:")
    st.text_input("Ask a question regarding your docs")

    with st.sidebar:

        st.subheader("Please upload your docs")
        pdf_docs = st.file_uploader("Click Process after uploading", type=[
                                    "pdf"], accept_multiple_files=True)
        if st.button("Process"):

            with st.spinner("Processing..."):
                # get pdf text
                if pdf_docs is not None:
                    pdf_text = get_pdf_text(pdf_docs)
                # divide in chunks
                text_chunks = get_chunks_tiktoken(pdf_text)
                # create embeddings -> create vectorstore
                # vectorstore = get_vectorstore(text_chunks)
                vectorstore = get_vectorstore_instructor(text_chunks)

                st.write(vectorstore)
        #  Markdown
            # Add vertical space
        v_spacer(height=3, sb=True)
        st.markdown('''
     ## About
     This app is designed to chat with your personal pdf articles and books.
     Tools used:
     - [Streamlit](https://streamlit.io/)
     - [Langchain](https://python.langchain.com/)
     - [OpenAI](https://platform.openai.com/)
     - [Llama2](https://ai.meta.com/llama/)
     ''')


if __name__ == '__main__':
    main()
