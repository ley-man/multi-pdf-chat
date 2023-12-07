import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


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


def get_vectorstore_instructor(text_chunks):
    # Free Embeddings- takes much longer though
    embeddings = HuggingFaceInstructEmbeddings(
        model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


def get_conv_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for idx, message in enumerate(st.session_state.chat_history):
        if idx % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    load_dotenv()

    st.set_page_config(page_title="Multi-pdf-chat", page_icon=":sharks:")
    st.write(css, unsafe_allow_html=True)
    st.subheader("Chat with multiple pdf documents :sunglasses:")
    user_ques = st.text_input("Ask a question regarding your docs")
    if user_ques:
        handle_user_input(user_ques)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

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
                vectorstore = get_vectorstore(text_chunks)

                # vectorstore = get_vectorstore_instructor(text_chunks)
                st.write(":green[Documents processed]")

                # Create conversation chain
                st.session_state.conversation = get_conv_chain(vectorstore)

        #  Add Markdown
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
