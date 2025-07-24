from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
