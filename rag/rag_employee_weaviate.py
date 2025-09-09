import json
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Configure


class WeaviateRAGDemo:
    def __init__(self):
        # Load embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Connect to local Weaviate v4 - Updated connection method
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080
        )

        # Delete existing collection if exists
        if self.client.collections.exists("Employee"):
            self.client.collections.delete("Employee")

        # Create Employee collection with external vectors - Updated configuration
        self.client.collections.create(
            name="Employee",
            vectorizer_config=None,  # disable internal vectorizer
            properties=[
                weaviate.classes.config.Property(name="name", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="department", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="position", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="skills", data_type=weaviate.classes.config.DataType.TEXT_ARRAY),
                weaviate.classes.config.Property(name="projects",
                                                 data_type=weaviate.classes.config.DataType.TEXT_ARRAY),
                weaviate.classes.config.Property(name="notes", data_type=weaviate.classes.config.DataType.TEXT),
            ]
        )
        self.collection = self.client.collections.get("Employee")

    def __del__(self):
        # Close connection when object is destroyed
        if hasattr(self, 'client'):
            self.client.close()

    def load_employee_data(self, json_file_path: str):
        with open(json_file_path, 'r') as f:
            return json.load(f)

    def create_employee_embeddings(self, employees):
        for emp in employees:
            # Create text for embedding
            text = f"{emp['name']} {emp['department']} {emp['position']} {', '.join(emp['skills'])} {', '.join(emp['projects'])} {emp['notes']}"
            embedding = self.encoder.encode(text).tolist()

            # Insert with explicit vector
            self.collection.data.insert(
                properties={
                    "name": emp["name"],
                    "department": emp["department"],
                    "position": emp["position"],
                    "skills": emp["skills"],
                    "projects": emp["projects"],
                    "notes": emp["notes"],
                },
                vector=embedding
            )

    def search_employees(self, query, top_k=2):
        query_vec = self.encoder.encode(query).tolist()
        results = self.collection.query.near_vector(
            near_vector=query_vec,
            limit=top_k
        )
        return [r.properties for r in results.objects]

    def respond_without_rag(self, question: str) -> str:
        return "I don't have access to specific employee details. Please contact HR or check the internal system."

    def respond_with_rag(self, question: str) -> str:
        results = self.search_employees(question, top_k=1)
        if not results:
            return "No relevant employee information found."
        emp = results[0]
        return f"{emp['name']} works in {emp['department']} as {emp['position']} and is skilled in {', '.join(emp['skills'][:3])}."


def main():
    try:
        rag = WeaviateRAGDemo()
        employees = rag.load_employee_data("employee_records.json")
        rag.create_employee_embeddings(employees)

        questions = [
            "Who has Python and machine learning skills?",
            "Find someone with sales experience",
            "Who works in HR?",
            "Do we have any finance analysts?"
        ]

        for q in questions:
            print("\nQ:", q)
            print("Without RAG:", rag.respond_without_rag(q))
            print("With RAG:", rag.respond_with_rag(q))

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Weaviate is running locally on port 8080")
    finally:
        # Ensure connection is closed
        if 'rag' in locals():
            del rag


if __name__ == "__main__":
    main()