import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

class Chatbot:
    def __init__(self, config: dict):
        self.db_path = config.get("db_path", "db")
        self.embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.model_name = config.get("llm_model", "llama3-8b-8192")
        self.top_k = config.get("top_k", 3)
        self.temperature = config.get("temperature", 0.1)
        
        # Initialiser les embeddings et la base vectorielle
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vector_store = Chroma(
            persist_directory=self.db_path, 
            embedding_function=self.embedding_model
        )
        
        # Vérifier la clé API Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY n'est pas définie dans les variables d'environnement")
        
        # Initialiser le modèle Groq
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # Template de prompt avec un espace pour l'historique
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template="""
Historique de conversation :
{history}

Contexte extrait des documents :
{context}

Question : {question}

Réponds de manière claire et structurée en t’appuyant sur le contexte et l’historique.
            """
        )
        
        # Chaîne LLM
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        # Mémoire pour l’historique de la conversation
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    def retrieve_documents(self, query: str):
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        contexts = []
        for i, (doc, score) in enumerate(results):
            metadata_str = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata_str = f"Source: {doc.metadata.get('source', 'Inconnue')}"
                if 'page' in doc.metadata:
                    metadata_str += f", Page: {doc.metadata['page']}"
            contexts.append(f"--- Document {i+1} (score: {score:.4f}) ---\n{metadata_str}\n{doc.page_content}")
        return "\n\n".join(contexts)
    
    def chat(self, question: str) -> str:
        # Récupérer le contexte depuis la base vectorielle
        context = self.retrieve_documents(question)
        # Récupérer l'historique de conversation
        history = self.memory.load_memory_variables({}).get("history", "")
        # Générer la réponse
        response = self.chain.invoke({
            "history": history,
            "context": context,
            "question": question
        })
        answer = response.get('text', '')
        # Sauvegarder l'échange dans la mémoire
        self.memory.save_context({"question": question}, {"answer": answer})
        return answer

if __name__ == "__main__":
    config = {
        "db_path": "db",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "llama3-8b-8192",
        "top_k": 3,
        "temperature": 0.1
    }
    
    chatbot = Chatbot(config)
    print("Bienvenue dans le chatbot RAG ! (Tapez 'exit' pour quitter)")
    while True:
        question = input("Vous: ")
        if question.lower() in ['exit', 'quit']:
            break
        answer = chatbot.chat(question)
        print("Chatbot:", answer)
