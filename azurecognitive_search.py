
import os
from langchain_openai import AzureOpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import AzureSearch
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

model: str = "text-embedding-ada-002"

#Azure Open AI Endpoint
azure_openai_endpoint = os.environ.get('AZURE_OPENAI_RESOURCE_NAME')
azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")

embeddings = AzureOpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_key = azure_openai_api_key, 
    azure_endpoint = azure_openai_endpoint,
    deployment=model, 
    chunk_size=1
    )

vector_store_address: str = f"https://{os.environ.get('AZURE_SEARCH_SERVICE_NAME')}.search.windows.net"
index_name: str = "langchain-vector-demo"

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=os.environ.get("AZURE_SEARCH_API_KEY"),
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

loader = AzureBlobStorageContainerLoader(
    conn_str=os.environ.get("AZURE_CONN_STRING"),
    container=os.environ.get("CONTAINER_NAME"),
)

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
vector_store.add_documents(documents=docs)

print("Data loaded into vectorstore successfully")
