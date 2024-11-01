from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import os 


def ingest_json(file_path, json_schema = '.',name = "default_collection"):
    """ Given a JSON file located under the file_path, 
    ingest the data into a local Chroma vector store. 
    Embeddings are generated using Ollama deployed locally.

    Pass in a name to create a new collection, or leave blank to add to the default collection.
    The vector store is persisted locally in ./chroma_db
    
    Returns the number of chunks embedded, the name of the collection, and the path to the local vector store.
    """
    
    # load the JSON file
    loader = JSONLoader(file_path, 
                        jq_schema=json_schema,
                        text_content=False)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    
    #Ideally update the metadata fields in the texts documents to give more info to the customer. Since 
    #the SampleDataset.json does not have metadata, we will not update it.

    # embed the chunks using Ollama mxbai-embed-large embeddings model (state of the art embeddings model)
    # model is hosted on Ollama server deployed locally. The embeddings are stored in Chroma vector store
    # locally in ./chroma_db, so that propritary data is not sent to the cloud either for embedding generation
    # or for vector similarity search. Alternatively a pinecone vector database could be used to host the embeddings
    # and the data would be stored in the cloud. One thing that I dont like about Pinecone is that pincone by default in addition to storingt the embeddings, also stores the raw chunks. Better methods such as only using Pinecone only to storing the embeddings, and re-directing to client for retrieval (using obfuscated paths/line ranges) are possible, but all need to be implemented manually.  

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_persist_directory = os.path.join(script_dir, "chroma_db")
    vector_store = Chroma.from_documents(texts, 
                                         embeddings, 
                                         collection_name = name, 
                                         persist_directory = chroma_persist_directory,
                                         collection_metadata={"hnsw:space": "cosine"}) # Note that we use cosine similarity, since the model mxbai-embed-large used cosine similarity during training.
    
    chroma_client = vector_store._client
    collection = chroma_client.get_collection(name)
    return collection.count(), name, chroma_persist_directory

if __name__ == "__main__":
    
    print("Ingesting data...")
    
    load_dotenv()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../data/SampleDataset.json")
    
    count, name, chroma_persist_directory = ingest_json(file_path, 
                                                        json_schema= '.questions[] | {question: .question, answer: .answer}', 
                                                        name="customer_support")
    print(f"Ingested {count} chunks into collection {name}")
    print(f"Chroma vector store persisted in {chroma_persist_directory}")
    
    
    