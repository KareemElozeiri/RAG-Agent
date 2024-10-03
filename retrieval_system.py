import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RetrievalSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        embeddings = self.encoder.encode(documents)
        
        #if the vector store does not exist create one
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(embeddings.astype('float32'))

    def search(self, query, k=5):
        '''
            searches through the vector store and returns the top K matching pieces of documents to the input query
        '''
        query_vector = self.encoder.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = [
            {'document': self.documents[i], 'distance': distances[0][j]}
            for j, i in enumerate(indices[0])
        ]
        
        return results



