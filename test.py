import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,chunk_overlap=200)
    chunks =text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore=FAISS.from_texts(text_chunks,embedding=embeddings)
    vectorstore.save_local("faiss_index")
   # return vectorstore
def get_conversational_chain():
        prompt_template="""Answer the question as possible from the provided context, make sure to provide all the details ,if the answer is not in provided context just say, "answer is not in the context", don't provide any wrong information\n\n
        Context:\n{context}?\nQuestion:\n{question}\n
        Answer:
        """
        model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
        prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
        chain =load_qa_chain(model,chain_type="stuff",prompt=prompt)
        return chain    
def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    FAISS.allow_dangerous_deserialization = True   
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    response=chain({"input_documents":docs,"question":user_question},return_only_outputs=True)
    print(response)

    st.write("Reply::",response["output_text"])
def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":speech_balloon:", layout="wide")
    st.header("Chat PDF")
    user_question=st.text_input("Ask a question about the PDF")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st .file_uploader("Upload PDF", type=["pdf"],accept_multiple_files=True)
        if st.button("Process PDF"):
            with st.spinner("Processing PDF"):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                vector_store=get_vectorstore(text_chunks)
                st.success("PDF Processed")
if __name__ == '__main__':   
        main()
    