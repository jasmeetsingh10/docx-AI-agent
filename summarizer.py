import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Split text into smaller chunks (around 2000 tokens each)
def split_text(text, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# Step 2: Summarize each chunk using Gemini
def summarize_text(text):
    chunks = split_text(text)
    print(f"🔍 Total chunks to summarize: {len(chunks)}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Use faster model
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

    summarized_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"⏳ Summarizing chunk {i}...")
        prompt = f"Please summarize the following text:\n\n{chunk}"
        result = llm.invoke(prompt)
        summarized_chunks.append(result.content)

    return "\n\n".join(summarized_chunks)

