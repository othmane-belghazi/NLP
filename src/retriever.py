from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentRetriever:
    def __init__(self, config_path="config.yaml"):
        """Initialise le système de recherche documentaire avec un fichier de configuration."""
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.embedding_model = HuggingFaceEmbeddings(model_name=config["embedding_model"])
        self.db_path = config["db_path"]
        self.vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_model)
        
    def search(self, query, top_k=5):
        """
        Recherche les documents les plus pertinents pour une requête utilisateur.
        """
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return results

if __name__ == "__main__":
    retriever = DocumentRetriever()
    
    query = input("Entrez votre requête : ")
    results = retriever.search(query)

    print("\nRésultats pertinents :")
    for i, (doc, score) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        print(doc.page_content)
        print("-" * 50)
