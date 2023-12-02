import streamlit as st


def main():
  st.set_page_config(page_title="Multi-pdf-chat", page_icon=":sharks:")

  st.subheader("Chat with multiple pdf documents :sunglasses:")  
  st.text_input("Ask a question regarding your docs")

  with st.sidebar:
     st.subheader ("Please upload your docs")  
     st.file_uploader("Click Process after uploading", type=["pdf"]
                      ,accept_multiple_files=True)
     st.button("Process")








 




if __name__ == '__main__':
    main()