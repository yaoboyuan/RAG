import os
os.environ['USER_AGENT'] = 'myagent'

import re
import sys
import bs4
import time
import argparse
import tiktoken
import warnings
from tqdm.auto import tqdm, trange
from langchain import hub
from langchain.load import dumps, loads
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredExcelLoader, UnstructuredCSVLoader, WebBaseLoader

class RAG:
    def __init__(self, model_path, docs_path, embedding_model, embedding_device, db_name):
        print('[System Info] Load LLM model')
        self.llm = LlamaCpp(model_path=model_path, n_gpu_layers=100, n_batch=512, n_ctx=2048, f16_kv=True,
                            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=False)
        print('[System Info] Load embedding model')
        embedding = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': embedding_device})
        print('[System Info] Build RAG process')
        docs = self.__docs_loader(docs_path)
        splits = self.__spliter(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=db_name)
        retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 1})
        template = """Answer the question based only on the following context:{context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = (
            {'context': retriever, 'question': RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
    def convert_tokens(self, s, encoding_name='cl100k_base'):
        encoding = tiktoken.get_encoding(encoding_name)
        res = encoding.encode(s)
        return res
        
    def __docs_loader(self, path):
        if os.path.isfile(path):
            file_name = os.path.basename(path)
            extension = file_name.split('.')[1]
            if extension == 'pdf':
                loader = PyMuPDFLoader(path)
                res = loader.load()
            elif extension == 'xlsx':
                loader = UnstructuredExcelLoader(path, mode="elements")
                res = loader.load()
            elif extension == 'csv':
                loader = UnstructuredCSVLoader(path, mode="elements")
                res = loader.load()
            else:
                print('Error: Not pdf file.')
        elif path.startswith('http') or path.startswith('https'):
            bs4_strainer = bs4.SoupStrainer(class_=('post-content', 'post-title', 'post-header')) # Only keep post title, headers, and content
            loader = WebBaseLoader(web_paths=(path,), bs_kwargs={"parse_only": bs4_strainer})
            res = loader.load()
        else:
            raise Exception('Error: Not pdf or website start with http or https')
        return res

    def __spliter(self, docs):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name='cl100k_base', chunk_size=20, chunk_overlap=0)
        splits = text_splitter.split_documents(docs)
        if 'languages' in splits[0].metadata and type(splits[0].metadata['languages'] == list):
            for s in splits:
                s.metadata['languages'] = s.metadata['languages'][0]
        return splits


def main():
    parser = argparse.ArgumentParser(
        prog = 'Intel RAG Demo',
        description = 'Intel RAG demo developed by Intel Taiwan SAE Yao Bo Yuan',
        epilog = 'Copyright(r), 2024'
    )
    parser.add_argument('-mo', '--mode', default='rag')
    parser.add_argument('-m', '--model', default='./Llama-2-7b-chat-hf.gguf')
    parser.add_argument('-em', '--embed_model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-dp', '--docs_path', default='./sae_story.pdf')
    parser.add_argument('-dn', '--database_name', default = 'db_sae_story')

    args = parser.parse_args()
    
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    rag = RAG(args.model, args.docs_path, args.embed_model, args.device, args.database_name)

    if args.mode == 'rag':
        print('[System Info] RAG inference mode')
        while True:
            user_input = input('Enter a question (insert q to leave): ')
            if user_input == 'q':
                break
            ans = rag.chain.invoke(user_input)
            print('\n')
    elif args.mode == 'llm':
        print('[System Info] LLM inference mode')
        while True:
            user_input = input('Enter a question (insert q to leave): ')
            if user_input == 'q':
                break
            ans = rag.llm.invoke(user_input)
            print('\n')

if __name__ == '__main__':
    main()