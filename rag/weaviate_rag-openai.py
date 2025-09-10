import json
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Configure
from openai import OpenAI
import os


class WeaviateRAGDemo:
    def __init__(self):
        # Load embedding model
        # Load embedding model
        # Initialize the encoder model using SentenceTransformer.
        # 'all-MiniLM-L6-v2' is a lightweight, efficient pre-trained model
        # that converts text into embeddings (dense vector representations).
        # This particular model is widely used because it balances speed and accuracy,
        # making it well-suited for tasks like semantic search, clustering, and
        # comparing the similarity between sentences or documents.
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
        """Generate response using only OpenAI without RAG context"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful HR assistant. You don't have access to specific employee databases or records."},
                    {"role": "user", "content": question}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"

    def respond_with_rag(self, question: str) -> str:
        """Generate response using OpenAI with RAG context from Weaviate"""
        try:
            # Get only top 1 result to minimize context size and cost
            results = self.search_employees(question, top_k=1)

            if not results:
                context = "No employee found."
            else:
                # Minimized context format to reduce token count
                emp = results[0]
                context = f"{emp['name']}: {emp['position']}, {emp['department']}, skills: {', '.join(emp['skills'][:3])}"  # Limit to top 3 skills

            # Generate response with minimal context
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "HR assistant. Answer using provided employee data."},
                    {"role": "user", "content": f"{context}\n\nQ: {question}"}  # Compact format
                ],
                max_tokens=80,  # Reduced from 250 to minimize cost
                temperature=0.1,  # Very low for consistent, focused responses
                top_p=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with RAG or OpenAI call: {str(e)}"


def main():
    try:
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("Warning: OPENAI_API_KEY environment variable not set")
            print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
            return

        rag = WeaviateRAGDemo()
        employees = rag.load_employee_data("employee_records.json")
        rag.create_employee_embeddings(employees)

        # questions = [
        #     "Who has Python and machine learning skills?",
        #     "Find someone with sales experience",
        #     "Who works in HR?",
        #     "Do we have any finance analysts?"
        # ]

        questions = [
            "Who has Python and machine learning skills?"
        ]

        for q in questions:
            print(f"\n{'=' * 50}")
            print(f"Q: {q}")
            print(f"{'=' * 50}")
            print("Without RAG:", rag.respond_without_rag(q))
            print("\nWith RAG:", rag.respond_with_rag(q))

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Weaviate is running locally on port 8080")
        print("And that your OpenAI API key is set correctly")
    finally:
        # Ensure connection is closed
        if 'rag' in locals():
            del rag


if __name__ == "__main__":
    main()