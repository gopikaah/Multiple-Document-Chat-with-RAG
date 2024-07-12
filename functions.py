
import os 
import glob
import json
import shutil
import PyPDF2
import camelot
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List,  Dict
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

os.environ.setdefault('OPENAI_API_KEY', 'Your-OpenAIAPIkey')
CHROMA_PATH = "wikipedia_chroma"
DATA_PATH = "uploads"  # Make sure this path contains your PDF files
ADDITIONAL_DATA_PATH = "wikipedia_table.csv"  # Path to your additional data
EXTENSIONS = ["pdf"]  
GROUND_TRUTH_FILE= 'ground_truth.json'
LOADER_MAPPING = {".pdf": (PyPDFLoader, {}),}

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_file_paths(source_dir: str, extensions: list):

    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True))
        all_files.extend(glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True))
    
    return all_files

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(all_files: List[str]) -> List[Document]:
    results = []
    with tqdm(total=len(all_files), desc='Loading new documents', ncols=80) as pbar:
        for file_path in all_files:
            try:
                docs = load_single_document(file_path)
                results.extend(docs)
            except Exception as e:
                print(f"Error loading document {file_path}: {e}")
            pbar.update(1)
    return results

def generate_tables_csv(documents):
    rows = []
    for file_path in documents:
        
        file = open(file_path,'rb')
        pdfReader = PyPDF2.PdfReader(file)
        pages_length = len(pdfReader.pages)
        listtt = np.arange(0, pages_length).tolist()

        for page in listtt:
            try:
                tables = camelot.read_pdf(file_path,str(page))            
                for table in tables:
                    df = table.df                
                    table_as_dict_list = []
                    for i, row in df.iterrows():
                        row_dict = row.to_dict()
                        table_as_dict_list.append(row_dict)                
                    rows.append(table_as_dict_list)
            except Exception as e:
                print(f"Error processing  : {e}")

            # print('Document',file_path,'completed')

    final_df = pd.DataFrame({"Table_Data": rows})
    final_df.drop_duplicates()
    
    final_df["Table_Data"] = final_df["Table_Data"].apply(lambda x: str(x))
    final_df.to_csv('wikipedia_table.csv', index=False)
    print("Tables have been saved to wikipedia_table.csv")

def load_additional_data():
    df = pd.read_csv(ADDITIONAL_DATA_PATH)
    return df

def merge_additional_data(documents: list[Document], additional_data: pd.DataFrame):
    # Assume additional_data has a 'document_id' column to match with document metadata
    for document in documents:
        doc_id = document.metadata.get('Table_Data')
        if doc_id in additional_data['Table_Data'].values:
            additional_info = additional_data[additional_data['Table_Data'] == doc_id].to_dict('records')[0]
            document.metadata.update(additional_info)
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    file_paths = get_file_paths(DATA_PATH, EXTENSIONS)
    generate_tables_csv(file_paths)
    documents = load_documents(file_paths)
    additional_data = load_additional_data()
    documents_with_additional_data = merge_additional_data(documents, additional_data)
    chunks = split_text(documents_with_additional_data)
    save_to_chroma(chunks)

def ask(query):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = ChatOpenAI()
    results = db.similarity_search_with_relevance_scores(query, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    return formatted_response

def get_pdf_files():
    pdf_files = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.pdf'):
            pdf_files.append(filename)
    return pdf_files
