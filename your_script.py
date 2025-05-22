import gradio as gr
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.schema import HumanMessage
import os
import sys
import io
import random
import time
from groq import Client
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Hardcode the API key (replace with your actual key or keep in .env for security)
os.environ["GROQ_API_KEY"] = "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"
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not defined.")

# Initialize Groq client
try:
    client = Client(api_key=api_key)
    models = client.models.list()  # Test API key
    print("Groq API key is valid.")
    use_groq = True
    from langchain_groq import ChatGroq
except Exception as e:
    print(f"Error with Groq API key: {str(e)}")
    use_groq = False

# Set encoding for Python
if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')
else:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Function to create vector store from Excel
def create_new_vectorstore(excel_path, embedding_function, persist_directory):
    try:
        df = pd.read_excel(excel_path, sheet_name="liste_amm", engine="xlrd")
        df = df.fillna("")  # Replace NaN with empty strings
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # Strip spaces

        docs = []
        for _, row in df.iterrows():
            doc_text = (
                f"Drug: {row.get('Nom', '')}\n"
                f"Dosage: {row.get('Dosage', '')}\n"
                f"Form: {row.get('Forme', '')}\n"
                f"Active Ingredient: {row.get('DCI', '')}\n"
                f"Class: {row.get('Classe', '')}\n"
                f"Sub-Class: {row.get('Sous Classe', '')}\n"
                f"Indications: {row.get('Indications', '')}\n"
                f"Manufacturer: {row.get('Laboratoire', '')}\n"
                f"Type: {row.get('G/P/B', '')}\n"
                f"Essentiality: {row.get('VEIC', '')}\n"
                f"Packaging: {row.get('Conditionnement primaire', '')}\n"
                f"Shelf Life: {row.get('Durée de conservation', '')} months"
            )
            metadata = {
                "Nom": str(row.get("Nom", "")),
                "Dosage": str(row.get("Dosage", "")),
                "Forme": str(row.get("Forme", "")),
                "DCI": str(row.get("DCI", "")),
                "Classe": str(row.get("Classe", "")),
                "Laboratoire": str(row.get("Laboratoire", "")),
                "G/P/B": str(row.get("G/P/B", "")),
                "VEIC": str(row.get("VEIC", ""))
            }
            docs.append(Document(page_content=doc_text, metadata=metadata))

        vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
        print(f"ChromaDB vector store created with {len(docs)} documents")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return Chroma.from_texts(
            ["Drug: Paracetamol\nIndications: Pain relief, fever reduction.",
             "Drug: Amoxicillin\nIndications: Bacterial infections."],
            embedding_function,
            persist_directory=persist_directory
        )

# Function to set up or load vector store
def setup_vectorstore():
    excel_path = r"C:\Users\KENZA\Desktop\Projet AI\liste_amm.xls"
    persist_directory = "./chroma_db_medications"
    os.makedirs(persist_directory, exist_ok=True)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        collection_count = vectorstore._collection.count()
        if collection_count > 0:
            print(f"ChromaDB loaded successfully: {collection_count} documents")
            return vectorstore
    except Exception as e:
        print(f"Error loading ChromaDB or empty database: {str(e)}")

    print("Creating new ChromaDB...")
    return create_new_vectorstore(excel_path, embedding_function, persist_directory)

# Initialize LLM
if use_groq:
    try:
        groq_llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            api_key=api_key
        )
        print("Groq model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Groq model: {str(e)}")
        use_groq = False

if not use_groq:
    print("Falling back to HuggingFace model...")
    from langchain_huggingface import HuggingFaceEndpoint
    os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN", "")
    try:
        groq_llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            temperature=0.5,
            max_length=512
        )
        print("HuggingFace fallback model initialized.")
    except Exception as e:
        print(f"Error initializing fallback model: {str(e)}")
        from langchain_community.llms.fake import FakeListLLM
        groq_llm = FakeListLLM(responses=[
            "Sorry, I cannot respond to your question at this time as the language model is unavailable. Please check your Groq API key or internet connection."
        ])
        print("Using fake LLM with error message.")

# Load or create vector store
try:
    vectorstore = setup_vectorstore()
except Exception as e:
    print(f"Critical error setting up vector store: {str(e)}")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(
        ["Drug: Paracetamol\nIndications: Pain relief, fever reduction.",
         "Drug: Amoxicillin\nIndications: Bacterial infections."],
        embedding_function,
        persist_directory="./chroma_db_medications"
    )

# Initialize conversation memory
memory = ConversationBufferMemory(
    return_messages=True,
    input_key="question",
    output_key="answer",
    memory_key="chat_history"
)

# Define custom prompt template
qa_prompt = PromptTemplate(
    template=(
        "You are a pharmacist expert in medications available in Tunisia. "
        "Use the information from the context below to answer accurately and relevantly. "
        "Provide details about the drug, its indications, form, dosage, and other relevant information. "
        "If the question concerns the need for a prescription, indicate whether the drug is likely prescription-based based on its VEIC (Vital/Essential) status or therapeutic class.\n\n"
        "Context: {context}\n"
        "Client question: {question}\n\n"
        "Response as a pharmacist:"
    ),
    input_variables=["context", "question"]
)

# Create conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=groq_llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

# Predefined questions
sample_questions = [
    "What are the indications for Paracetamol in Tunisia?",
    "Do I need a prescription for Aciclovir?",
    "Which medications are used for hypertension?",
    "ما هو استخدام دواء Zithromax؟",
    "What antibiotics are available in Tunisia?"
]

# Chatbot response function
def chatbot(question):
    if not question.strip():
        return "Please enter a question."

    try:
        response = qa_chain({"question": question})
        answer = response['answer']
        sources = "\n\n".join([f"**Document {i+1}:** {doc.page_content}" for i, doc in enumerate(response.get('source_documents', []))])
        full_response = f"**Response:** {answer}\n\n**Sources:**\n{sources}"
        save_conversation_history()
        return full_response
    except Exception as e:
        error_msg = str(e)
        save_conversation_history()
        if "invalid_api_key" in error_msg:
            return "**Error:** Invalid API key. Please check the key in the code."
        return f"**Error:** {error_msg}"

# Show conversation history
def show_history():
    if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
        try:
            history_items = []
            for msg in memory.chat_memory.messages:
                role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                content = msg.content if msg.content else "[Empty content]"
                history_items.append(f"**{role}:** {content}")
            return "\n\n".join(history_items)
        except Exception as e:
            return f"Error displaying history: {str(e)}"
    return "No history available."

# Clear conversation history
def clear_history():
    save_conversation_history()
    memory.clear()
    return "History cleared and saved."

# Save conversation history
def save_conversation_history():
    if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
        with open("conversation_history.txt", "w", encoding="utf-8") as f:
            for msg in memory.chat_memory.messages:
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                f.write(f"{role}: {msg.content}\n")
        print("Conversation history saved to conversation_history.txt")
        return "History saved."

# Main Gradio Interface
# Main Gradio Interface
# Main Gradio Interface

# Main Gradio Interface
# Interface Gradio
with gr.Blocks() as interface:
    gr.Markdown("## Chatbot IA - LangChain & Groq\nPosez votre question ou choisissez-en une prédéfinie.")

    with gr.Row():
        question_input = gr.Textbox(placeholder="Posez votre question ici...")
        submit_btn = gr.Button("Envoyer")
    
    output = gr.Markdown()

    # Boutons pour les questions prédéfinies
    gr.Markdown("### Questions Prédéfinies")
    for q in sample_questions:
        gr.Button(q).click(fn=chatbot, inputs=gr.Textbox(value=q, visible=False), outputs=output)

    # Lien entre l'entrée manuelle et le chatbot
    submit_btn.click(chatbot, inputs=question_input, outputs=output)

# Lancement de l'interface
interface.launch(share=True)

