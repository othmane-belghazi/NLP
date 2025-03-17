import os
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
class DocumentIndexer:
    def __init__(self, db_path="db", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialise l'indexeur avec un modèle d'embeddings et une base de données vectorielle.
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db_path = db_path
        self.vector_store = None

    def load_documents(self, directory="data/"):
        """
        Charge et extrait le texte des fichiers PDF d'un dossier.
        """
        documents = []
        for file in os.listdir(directory):
            if file.endswith(".pdf"):
                pdf_loader = PyPDFLoader(os.path.join(directory, file))
                documents.extend(pdf_loader.load())
        return documents

    def split_documents(self, documents, chunk_size=500, chunk_overlap=100):
        """
        Divise les documents en petits segments (chunks) pour l'indexation.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    def index_documents(self, documents):
        """
        Convertit les documents en embeddings et les stocke dans une base vectorielle.
        """
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        self.vector_store = Chroma.from_documents(
            documents,
            self.embedding_model,
            persist_directory=self.db_path
        )
        print("Indexation terminée et sauvegardée.")

    def run(self, config_path="config.yaml"):
    """Charge la configuration et indexe les documents."""
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        print("Chargement des documents...")
        documents = self.load_documents(config["data_directory"])

        print("Segmentation des documents...")
        split_docs = self.split_documents(documents)

        print("Indexation des documents...")
        self.index_documents(split_docs)

if __name__ == "__main__":
    indexer = DocumentIndexer()
    indexer.run("data/")  # Assurez-vous que le dossier "data/" contient des fichiers PDF
