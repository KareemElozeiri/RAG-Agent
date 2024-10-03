from preprocessing import preprocess_document
from retrieval_system import RetrievalSystem
from generation_model import GenerationModel

class RAGAgent:
    def __init__(self):
        self._retrieval_system = RetrievalSystem()
        self._generation_model = GenerationModel()

    def add_document(self, file_path):
        paragraphs = preprocess_document(file_path)
        self._retrieval_system.add_documents(paragraphs)

    def process_query(self, query, max_context_length=1000):
        # get relevant documents
        retrieved_docs = self._retrieval_system.search(query, k=3)
        print(retrieved_docs)
        
        #Combining retrieved documents into a context
        context = " ".join([doc['document'] for doc in retrieved_docs])
        
        # Truncate context if it's too long
        if len(context) > max_context_length:
            context = context[:max_context_length]
        
        # Generate response
        response = self._generation_model.generate_response(query, context)
        
        return response


