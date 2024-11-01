from dotenv import load_dotenv
import os


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


class JSONRetriever:
    def __init__(self, collection_name, chroma_persist_directory):
        
        chroma_vector_store = Chroma(embedding_function = OllamaEmbeddings(model="mxbai-embed-large"),
                            collection_name = collection_name, 
                            persist_directory = chroma_persist_directory,
                            collection_metadata={"hnsw:space": "cosine"})

        # Keep temperature at 0, since we want deterministic responses.
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        # Use the retrieval-qa-chat prompt from Langchain Hub, this prompt ensures that only the retrieved context is used to generate the answer.
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        # Use the chat-langchain-rephrase prompt from Langchain Hub, this prompt ensures that the question is rephrased to be more relevant to the context. This allows the customer to ask natural follow up questions. 
        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        history_aware_retriever = create_history_aware_retriever(llm, chroma_vector_store.as_retriever(), rephrase_prompt)
        
        self.retrieval_chain_chroma = create_retrieval_chain(
            retriever=history_aware_retriever, 
            combine_docs_chain=combine_documents_chain,
        )
       
    
    def chat(self, query, return_context=False, chat_history=[]):
        res = self.retrieval_chain_chroma.invoke(input= {"input": query, "chat_history": chat_history})
        if return_context:
            return res['answer'], res['context']
        else:
            return res['answer']
    
    
    @staticmethod
    def print_answer(query, answer):
        print("Question: ", query)
        print("Answer: ", answer)
        print("-"*100, "\n")
    

if __name__ == "__main__":
    print("Retrieving...")
    load_dotenv()
    
    print("Responses using RAG \n ", "-"*100, "\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_retriever = JSONRetriever(collection_name="customer_support", chroma_persist_directory=os.path.join(script_dir, "./chroma_db"))

    query1 = "Tell me about Thoughtful AI?"
    JSONRetriever.print_answer(query1, json_retriever.chat(query1))

    query2 = "Why should i use Thoughtful AI?"
    JSONRetriever.print_answer(query2, json_retriever.chat(query2))

    query3 = "How does this service help with automation ?"
    JSONRetriever.print_answer(query3, json_retriever.chat(query3 ))

    
    