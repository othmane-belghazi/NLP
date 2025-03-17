import os
import os
from typing import List, Tuple, Any, Dict
import transformers
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import yaml

load_dotenv()



class QuestionAnsweringSystem:
    def __init__(self, config_path_or_dict="config.yaml"):
        """Initialise le système de question-réponse avec un fichier de configuration ou un dictionnaire."""
        if isinstance(config_path_or_dict, str):
            # Load configuration from file
            with open(config_path_or_dict, "r") as file:
                config = yaml.safe_load(file)
        elif isinstance(config_path_or_dict, dict):
            # Use the provided dictionary directly
            config = config_path_or_dict
        else:
            raise TypeError("config_path_or_dict must be a file path (str) or a dictionary (dict)")

        self.db_path = config["db_path"]
        self.embedding_model_name = config["embedding_model"]
        self.model_name = config["llm_model"]
        self.top_k = config["top_k"]
        self.temperature = config["temperature"]
        
        # Initialiser les embeddings
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Initialiser la base vectorielle
        self.vector_store = Chroma(
            persist_directory=self.db_path, 
            embedding_function=self.embedding_model
        )
        
        # Vérification de la clé API Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY n'est pas définie dans les variables d'environnement")
        
        # Initialiser le modèle Groq
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # Création du template de prompt
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Tu es un assistant intelligent spécialisé dans l'analyse précise de documents.
            
            En te basant exclusivement sur les informations suivantes :
            
            {context}
            
            Réponds à la question suivante de manière claire, structurée et concise :
            
            {question}
            
            Si les informations fournies ne sont pas suffisantes pour répondre à la question, 
            indique-le clairement au lieu de faire des suppositions.
            Cite les parties des documents qui appuient ta réponse.
            """
        )
        
        # Initialisation de la chaîne LLM
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def retrieve_documents(self, query: str) -> List[Tuple[Document, float]]:
        """
        Recherche les documents les plus pertinents pour la requête.
        
        Args:
            query: La requête de l'utilisateur
            
        Returns:
            Liste de tuples (document, score)
        """
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        return results

    def generate_answer(self, query: str) -> str:
        """
        Génère une réponse en utilisant les documents récupérés et le modèle Groq.
        
        Args:
            query: La question posée par l'utilisateur
            
        Returns:
            Réponse générée par le modèle
        """
        # Récupération des documents pertinents
        results = self.retrieve_documents(query)
        
        # Afficher les documents récupérés (facultatif, pour le débogage)
        print(f"Documents récupérés: {len(results)}")
        
        # Construction du contexte
        contexts = []
        for i, (doc, score) in enumerate(results):
            # Inclure les métadonnées si disponibles
            metadata_str = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata_str = f"Source: {doc.metadata.get('source', 'Inconnue')}"
                if 'page' in doc.metadata:
                    metadata_str += f", Page: {doc.metadata['page']}"
            
            contexts.append(f"--- Document {i+1} (score: {score:.4f}) ---\n{metadata_str}\n{doc.page_content}")
        
        context = "\n\n".join(contexts)
        
        # Générer la réponse
        response = self.chain.invoke({"context": context, "question": query})
        
        return response['text']

    def get_document_sources(self, query: str) -> List[str]:
        """
        Récupère les sources des documents utilisés pour répondre à la requête.
        
        Args:
            query: La requête de l'utilisateur
            
        Returns:
            Liste des sources des documents
        """
        results = self.retrieve_documents(query)
        sources = []
        
        for doc, score in results:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if source not in sources:
                    sources.append(source)
        
        return sources

if __name__ == "__main__":
    # Configuration par défaut (à remplacer par le chargement depuis un fichier de config)
    default_config = {
        "db_path": "db",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "llama3-8b-8192",
        "top_k": 3,
        "temperature": 0.1
    }
    
    qa_system = QuestionAnsweringSystem(default_config)
    
    query = input("Posez votre question : ")
    response = qa_system.generate_answer(query)
    
    print("\nRéponse générée :")
    print(response)
    
    print("\nSources utilisées :")
    sources = qa_system.get_document_sources(query)
    for source in sources:
        print(f"- {source}")