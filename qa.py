from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere, CohereEmbeddings
import os
import getpass

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY") or getpass.getpass()

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatCohere(model="command-r", cohere_api_key=cohere_api_key)
    # model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.1)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def response(user_question):
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                       cohere_api_key=cohere_api_key)
    db = FAISS.load_local("vectors", embedding_model)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k":3})
    docs = retriever.invoke(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    return response

res = response("what about the university's ranking ?")
print(res['output_text'])