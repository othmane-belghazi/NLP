import argparse
import yaml
from indexer import DocumentIndexer
from retriever import DocumentRetriever
from llm import QuestionAnsweringSystem

def load_config(config_path="config.yaml"):
    """Charge la configuration à partir d'un fichier YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def index_documents():
    """Exécute l'indexation des documents."""
    config = load_config()
    indexer = DocumentIndexer(db_path=config["db_path"], embedding_model=config["embedding_model"])
    indexer.run(directory=config["data_directory"])
    print("Indexation terminée.")

def search_documents(query):
    """Effectue une recherche simple dans la base vectorielle."""
    config = load_config()
    retriever = DocumentRetriever(db_path=config["db_path"], embedding_model=config["embedding_model"])
    results = retriever.search(query, top_k=config["top_k"])
    
    print("\nRésultats trouvés :")
    for i, (doc, score) in enumerate(results):
        print(f"{i+1}. (Score: {score:.4f})\n{doc.page_content}\n{'-'*50}")

def answer_question(query):
    """Génère une réponse à partir des documents indexés."""
    config = load_config()
    qa_system = QuestionAnsweringSystem(config)
    response = qa_system.generate_answer(query)
    
    print("\nRéponse générée :")
    print(response)
    
    sources = qa_system.get_document_sources(query)
    if sources:
        print("\nSources utilisées :")
        for source in sources:
            print(f"- {source}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface CLI pour le système de recherche documentaire et de question-réponse.")
    parser.add_argument("--index", action="store_true", help="Indexe les documents PDF.")
    parser.add_argument("--search", type=str, help="Effectue une recherche dans les documents indexés.")
    parser.add_argument("--ask", type=str, help="Pose une question et obtient une réponse.")

    args = parser.parse_args()

    if args.index:
        index_documents()
    elif args.search:
        search_documents(args.search)
    elif args.ask:
        answer_question(args.ask)
    else:
        print("Utilisation : cli.py --index | --search 'query' | --ask 'question'")
