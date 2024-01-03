import pinecone
import cohere 
from dotenv import dotenv_values


env_name = "credentials.env"
config = dotenv_values(env_name)

co = cohere.Client(config['cohere_api_key'])

index_name = config['index_name']

pinecone.init(
    api_key=config['pinecone_api_key'],
    environment=config['pinecone_environment']
)

index = pinecone.Index(index_name)

def vector_search(query):
    rag_data = ""

    xq = co.embed(
        texts=[query],
        model='embed-english-v3.0',
        input_type='search_query'
    )

    res = index.query([xq.embeddings[0]], top_k=2, include_metadata=True)

    for match in res['matches']:
        if match['score'] < 0.8:
            continue 

        rag_data += match['metadata']['text']

    return rag_data 


def cohere_completion_with_vector_search(prompt, rag):
    DEFAULT_SYSTEM_PROMPT = '''You are a helpful, respectful and honest INTP-T AI Assistant named Gathnex AI. You are talking to a human User.
    Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    You also have access to RAG vectore database access which has Indian Law data. Be careful when giving response, sometime irrelevent Rag content will be there so give response effectivly to user based on the prompt.
    You can speak fluently in English.
    Note: Sometimes the Context is not relevant to Question, so give Answer according to that sutiation.
    '''

    response = co.chat(
    chat_history=[
        {f"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {f"role": "user", "content": rag},
    ],
    message=prompt
    )

    return response.text



