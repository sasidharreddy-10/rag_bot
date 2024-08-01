# import
import os
import time
import PyPDF2
from typing import Any
from openai import OpenAI
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    def _extract_text(self, file_path):
        try:
            reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() if page.extract_text() else ""
            return text
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            raise Exception("Error in PDF reading")
    def split_text(self, file_path, file_name):
        try:
            text=self._extract_text(file_path)
            docs=self.text_splitter.create_documents([text], metadatas=[{"filename":file_name}])
            return docs
        except Exception as e:
            print(f"Error splitting PDF text: {e}")
            raise Exception("Error in PDF text splitting")
        
class RAG:
    def __init__(self, index_name, text_processor : TextProcessor, PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type) -> None:
        self.index_name = index_name
        self.docsearch=None
        self.answer=''
        self.gpt_engine_name=gpt_engine_name
        self.doc_processing=text_processor
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        if openai_type=='azure_openai':
            self.openai_client = AzureOpenAI(
                    azure_endpoint = azure_endpoint,
                    api_key=api_key,
                    api_version= api_version
                )
            self.embeddings = AzureOpenAIEmbeddings(
                model=embedding_model_name, 
                api_key=api_key, 
                azure_endpoint=azure_endpoint, 
                openai_api_version=api_version)
        else:
            self.openai_client = OpenAI(api_key=api_key)
            self.embeddings = OpenAIEmbeddings(model=embedding_model_name, api_key=api_key)
        self.__call__()
        
    def __call__(self) -> None:
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        self.docsearch = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)

    def insert_doc(self, file_path, file_name):
        try:
            docs=self.doc_processing.split_text(file_path, file_name)
            self.docsearch.add_documents(docs)
            return 'Successfully indexed the document'
        except Exception as e:
            print(f"Error splitting PDF text: {e}")

    def _qna_helper(self, query, context):
        res = self.openai_client.chat.completions.create(model=self.gpt_engine_name,
                                                messages=[
                                                        {'role': 'system','content':f'''You are a expert qna bot. Your task is to strictly answer to user query based on the given context. If answer can't be fount in the given context then reply "Sorry, this information is out of my uploaded knowledge base, Please ask queries from Uploaded Documents.".
Only use information in context as ground truth to answer user query. Strictly answer need to be extracted and sourced from only given above context.

Follow the below steps to answer the user query:

Step 1: Analyse the given user query in accordance with the given below context.

Step 2: Based on the above analysis, give concise and precise answer to the given user query based on the given context. Ensure that the answer must be exclusively sourced from the given context.   

context: 
{context}

++++

Based on the above context, answer the user query.
'''},  
                                                        {'role': 'user',  'content':query}],
                                                        stream=True
                                                )
        
        return res
    
    def qna(self, query):
        docs = self.docsearch.similarity_search(query)
        context=''
        for i in docs:
            doc_name=i.metadata['filename']
            text=i.page_content
            context+=f'document_name: {doc_name} \n\n' + f'text: {text} \n\n'
        answer=self._qna_helper(query, context)
        final_answer=''
        for chunk in answer:
            if len(chunk.choices)>0 and chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                final_answer+=text
                yield text
                time.sleep(0.02)
        self.answer=final_answer