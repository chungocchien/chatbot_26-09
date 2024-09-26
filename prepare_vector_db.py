import json
from langchain.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
    
def extract_text_and_description(data):
    texts = []
    descriptions = []
    
    # Duyệt qua JSON
    def recursive_extract(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == "text":
                    texts.append(value)
                if key == "description":
                    descriptions.append(value)
                recursive_extract(value)
        elif isinstance(d, list):
            for item in d:
                recursive_extract(item)
    
    recursive_extract(data)
    return texts, descriptions

def get_vector_store(text_chunks):
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                       cohere_api_key='jgfGLUxBFOGVtZzQyNLcorveyREv7i8ENFRTQADk')
    vector_stores = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_stores.save_local('vectors')

file_path = 'dataset_test-https-www-gcu-ac-uk_2024-09-16_04-06-11-317.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

texts, descriptions = extract_text_and_description(data)
filtered_texts = [s for s in texts if s is not None and s.strip() != ""]
get_vector_store(filtered_texts)

