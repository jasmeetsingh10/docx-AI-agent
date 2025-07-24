import os
from dotenv import load_dotenv
from doc_reader import read_docx
from summarizer import summarize_text
from embedder import create_vector_store
from qa_agent import create_qa_chain


def run():
    load_dotenv()
    file_path = input("📄 Enter DOCX file path: ").strip()
    if not os.path.exists(file_path):
        print("❌ File does not exist.")
        return

    print("\n🔍 Reading document...")
    text = read_docx(file_path)

    print("\n📝 Summarizing...")
    summary = summarize_text(text)
    print("\n📄 Summary:\n", summary)

    print("\n📚 Creating vector store...")
    vectorstore = create_vector_store(text)

    print("\n🤖 Ask your questions (type 'exit' to quit):")
    qa_chain = create_qa_chain(vectorstore)
    while True:
        query = input("❓> ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = qa_chain.run(query)
        print("💡", answer)

if __name__ == "__main__":
    run()
