import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class SimpleRAGDemo:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = {}
        self.documents = {}

    def load_employee_data(self, json_file_path: str):
        with open(json_file_path, 'r') as file:
            return json.load(file)

    def create_employee_embeddings(self, employees: List[Dict]):
        for employee in employees:
            text = f"""
            Name: {employee['name']}
            Department: {employee['department']}
            Position: {employee['position']}
            Skills: {', '.join(employee['skills'])}
            Projects: {', '.join(employee['projects'])}
            Location: {employee['location']}
            Notes: {employee['notes']}
            Performance: {employee['performance_rating']}
            Certifications: {', '.join(employee['certifications'])}
            """
            embedding = self.encoder.encode(text.strip())
            emp_id = employee['employee_id']
            self.vector_store[emp_id] = embedding
            self.documents[emp_id] = employee

    def search_employees(self, query: str, top_k: int = 2) -> List[Dict]:
        query_embedding = self.encoder.encode(query)
        similarities = {}
        for emp_id, emp_embedding in self.vector_store.items():
            similarity = np.dot(query_embedding, emp_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emp_embedding)
            )
            similarities[emp_id] = similarity
        top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for emp_id, score in top_matches:
            employee = self.documents[emp_id].copy()
            employee['similarity_score'] = score
            results.append(employee)
        return results

    def respond_with_rag(self, question: str) -> str | dict:
        relevant_employees = self.search_employees(question, top_k=1)
        if not relevant_employees:
            return "No relevant employee information found."
        employee = relevant_employees[0]
        return employee

def main():
    rag = SimpleRAGDemo()
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
        print(rag.respond_with_rag(q))


if __name__ == "__main__":
    main()
