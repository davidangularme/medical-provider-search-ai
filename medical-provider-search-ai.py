import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama.from_pretrained(
    repo_id="davidfred/Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    filename="Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    log_level="error"
)

# Load the Excel data into a DataFrame
df = pd.read_excel("ServiceBook.xlsx")

# Preprocess and clean the data
df = df.fillna("")  # Handle missing values
df["search_text"] = df.apply(lambda row: " ".join(row.astype(str)), axis=1)  # Combine all fields into a single search text

# Extract phone numbers and addresses from the XML-like structures
df["טלפון"] = df["טלפון"].apply(lambda x: ", ".join(re.findall(r'<Number>(\d+)</Number>', x)))
df["כתובת"] = df["כתובת"].apply(lambda x: " ".join(re.findall(r'<city>(.+?)</city>', x) + re.findall(r'<street>(.+?)</street>', x) + re.findall(r'<streetNumber>(.+?)</streetNumber>', x)))

# Load the multilingual sentence transformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Check if the embeddings file exists
try:
    with open("doc_embeddings.pkl", "rb") as f:
        doc_embeddings = pickle.load(f)
    print("Loaded document embeddings from file.")
except FileNotFoundError:
    # Generate embeddings for the doctor search texts
    doc_embeddings = model.encode(df["search_text"].tolist())
    
    # Save the embeddings to a file
    with open("doc_embeddings.pkl", "wb") as f:
        pickle.dump(doc_embeddings, f)
    print("Generated and saved document embeddings.")

def search_doctors(query):
    # Generate embedding for the query
    query_embedding = model.encode([query])
    
    # Find the most similar doctor embeddings to the query
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    top_indices = similarities.argsort()[-5:][::-1]  # Get the top 5 most similar indices
    
    # Return the top matching doctors
    return df.iloc[top_indices]

# Initialize the Llama model
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="qwen2.5-3b-instruct-q4_0.gguf",
    log_level="error"
)

# Interactive search loop
while True:
    user_input = input("Enter your search query (or 'q' to quit): ")
    
    if user_input.lower() == 'q':
        break
    
    results = search_doctors(user_input)
    
    # Prepare the context for the LLM
    context = "Top matching doctors/clinics:\n"
    for _, row in results.iterrows():
        context += f"Name: {row['שם פרטי']} {row['שם משפחה']}\n"
        context += f"Title: {row['תואר']}\n"
        context += f"Specialty: {row['התמחות']}\n"
        context += f"Sub-specialty: {row['תת-התמחות']}\n"
        context += f"Phone: {row['טלפון']}\n"
        context += f"Address: {row['כתובת']}\n"
        context += "---\n"
    
    print('context')
    print(context)
    # Generate a response using the LLM
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"Based on the following information, provide a summary of the top matching doctors for the query '{user_input}':\n\n{context}"}
    ]
    
    response = llm.create_chat_completion(
        messages=messages,
        stream=True
    )
    
    print("Assistant: ", end="")
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
    print("\n")
