mykey ="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"
import os


from langchain.document_loaders import TextLoader
loader = TextLoader(file_path = "mypersonal.txt")
document = loader.load()
document



from langchain.text_splitter import CharacterTextSplitter
textchunk = CharacterTextSplitter(chunk_size=1000)
texts = textchunk.split_documents(document)
len(texts)
texts




from langchain.embeddings import OpenAIEmbeddings
myembedmodel = OpenAIEmbeddings(openai_api_key = mykey)
from langchain.vectorstores import Pinecone
import pinecone
os.environ["OPENAI_API_KEY"] = mykey

pinecone.init(
    api_key="6036647d-a00b-43d7-8171-9b9cc5b89b46",
    environment = "gcp-starter"
)

docsearch = Pinecone.from_documents(
                documents=texts,
                embedding=myembedmodel,
                index_name="myspindex"
)



from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm = OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

myquery = "who will be perform jauhar"
qa({"query":myquery})