import os
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, dotenv_values 

class GenerationModel:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        
        #setting the generation model
        self.llm = HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={
                "temperature": 0.5,
                "top_p": 0.95,
                "max_new_tokens": 100,
            }
        )

        # Creating a prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""[INST] You are a helpful assistant. Answer the question based on the context. Keep the answer within 2 sentences and concise. If you don't know the answer, just say you don't know.

Context: {context}

Question: {query}

Answer: """
        )

    def generate_response(self, query, context):
        prompt = self.prompt_template.format(context=context, query=query)

        response = self.llm(prompt)

        # Extract only the generated answer
        answer = response.split("Answer:")[-1].strip()
        return answer

if __name__ == "__main__":
    load_dotenv()
    generation_model = GenerationModel()
    
    
    
    query = "How many sellers in total do we have?"
    context = "We have 100 sellers in Egypt, 500 in Germany and 600 in USA"

    response = generation_model.generate_response(query, context)
    print(f"Query: {query}")
    print(f"Context: {context}")
    print(f"Generated Response: {response}")