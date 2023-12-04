import streamlit as st
from dotenv import load_dotenv  
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader
import PyPDF2


def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf_doc in pdf_docs:
      with open(pdf_doc, 'rb') as f:
          pdf = PyPDF2.PdfFileReader(f)
          for page in range(pdf.getNumPages()):
              page_text = pdf.getPage(page).extractText()
              pdf_text += page_text
      return pdf_text

def main():
  load_dotenv()

  st.set_page_config(page_title="Multi-pdf-chat", page_icon=":sharks:")

  st.subheader("Chat with multiple pdf documents :sunglasses:")  
  st.text_input("Ask a question regarding your docs")

  with st.sidebar:
     
     st.subheader ("Please upload your docs")  
     pdf_docs = st.file_uploader("Click Process after uploading", type=["pdf"]
                      ,accept_multiple_files=True)
     if st.button("Process"):
        with st.spinner("Processing..."):
            for pdf_doc in pdf_docs:
               print (pdf_doc)
                # get pdf text
                # for pdf_doc in pdf

                # divide in chunks

                # create embeddings

     #  Markdown
     add_vertical_space(num_lines=3)
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