# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 00:42:16 2025

@author: User
"""


import time
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# Step 1: Load  Documents
def load_documents(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


# Step 2: Split Documents into Chunks (Add "passage:" prefix)
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Add "passage:" prefix to each chunk for E5
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
    return chunks


# Step 3: Create Embeddings with Microsoft E5
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"}  # Use "cuda" if GPU available
)


def create_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


# Step 4: Set Up RAG Pipeline
def setup_rag_pipeline(vector_store,model_name):
    llm = Ollama(model=model_name,temperature=0,num_thread=30)  # Ensure the model is running via Ollama
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain


# Step 5: Query the RAG System (Add "query:" prefix)
def query_rag_system(qa_chain, query):
    query_with_prefix = f"query: {query}"
    result = qa_chain({"query": query_with_prefix})
    return result["result"], result["source_documents"]


# Step 6: Evaluate Response Using Precision, Recall, F1
def calculate_precision_recall_f1(system_answer, reference_answer):

    # Ensure inputs are strings
    system_answer = str(system_answer).lower()  # Convert to lowercase
    reference_answer = str(reference_answer).lower()  # Convert to lowercase
     #remove punctuation 
    system_answer=system_answer.translate(str.maketrans('', '', string.punctuation))
    reference_answer=reference_answer.translate(str.maketrans('', '', string.punctuation))
    
    
    reference_answer=re.sub(r'[آأإ]', 'ا', reference_answer)
    reference_answer = re.sub(r'ة', 'ه', reference_answer)
    
    system_answer=re.sub(r'[آأإ]', 'ا', system_answer)
    system_answer = re.sub(r'ة', 'ه', system_answer)
    
    
    reference_tokens = set(reference_answer.split())
    system_tokens = set(system_answer.split())

    common_tokens = reference_tokens.intersection(system_tokens)
    precision = len(common_tokens) / len(system_tokens) if system_tokens else 0
    recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print ("Precision = ",precision, "Recall = ", recall, "F1 = ",f1_score )
    return precision, recall, f1_score


# Step 7: Calculate Cosine Similarity
def calculate_cosine_sim(system_answer, reference_answer):
    # Ensure inputs are strings
    system_answer = str(system_answer).lower()  # Convert to lowercase
    reference_answer = str(reference_answer).lower()  # Convert to lowercase
     #remove punctuation 
    system_answer=system_answer.translate(str.maketrans('', '', string.punctuation))
    reference_answer=reference_answer.translate(str.maketrans('', '', string.punctuation))
    
    
      
    reference_answer=re.sub(r'[آأإ]', 'ا', reference_answer)
    reference_answer = re.sub(r'ة', 'ه', reference_answer)
    
    system_answer=re.sub(r'[آأإ]', 'ا', system_answer)
    system_answer = re.sub(r'ة', 'ه', system_answer)
    
    
    system_answer_v = embedding_model.embed_documents([system_answer])
    reference_answer_v = embedding_model.embed_documents([reference_answer])

    cosine_sim = cosine_similarity(system_answer_v, reference_answer_v)[0][0]
    #print (cosine_sim)
    
    return cosine_sim


# Main Execution
if __name__ == "__main__":
    
    # Input and Output Excel file paths
    
    models=["command-r7b","qwen2.5","llama3.1","llama3.3","command-r7b"]
    #theModel="command-r7b"
    topics=["law","science","management","history"]
    languages=["Arabic","English"]
    for  topic in topics :
        for language in languages:
            for theModel in models:
                
       
                input_excel_path = "D:\\Research\\GEN-AI\\PH2\\"+topic +"\\"+language+"-Questions.xlsx"
                output_excel_path = "D:\\Research\\GEN-AI\\PH2\\"+topic +"\\"+language+"\\"+theModel+"-"+language+"_resultsNPE.xlsx"
                ouput_txt_file= "D:\\Research\\GEN-AI\\PH2\\"+topic +"\\"+language+"\\"+theModel+"-"+language+"_resultsNPE.txt"
                # Load and process documents
                file_path = "D:\\Research\\GEN-AI\\PH2\\"+topic +"\\"+language+"-document.pdf"  # Replace with your Arabic document
                documents = load_documents(file_path) 
                chunks = split_documents(documents)
                vector_store = create_vector_store(chunks)
                qa_chain = setup_rag_pipeline(vector_store,theModel)
            
                # Read input Excel file
                df_input = pd.read_excel(input_excel_path)
            
                # Prepare a list to store results
                results = []
                file=open(ouput_txt_file, 'w', encoding='utf-8') 
                # Process each question
                for index, row in df_input.iterrows():
                   # if (index==57):
                    #    continue
                    question = row["question"]
                    reference_answer = row["reference_answer"]
                    answer_length=len(str(reference_answer).split())
                    if(language=="Arabic"):
                       prompt=" اجب على السؤال التالي باللغة العربية فقط ومن خلال النص المقدم فقط وليس بناءا على معرفتك  اعط الاجابة النهائية فقط بدون شرح اجابتك يجب ان تكون     "+str( answer_length)+" كلمات فقط. السؤال هو: "+question
                     #  prompt=" اجب على السؤال التالي باللغة العربية فقط ومن خلال النص المقدم فقط وليس بناءا على معرفتك  اعط الاجابة النهائية فقط بدون شرح. السؤال هو: "+question
                    else:
                        prompt="You are an assistant that only provides direct answers without showing reasoning steps. answer this question in English based on the provided text only and your answer should contain exaclty "+str( answer_length)+" words. the question is:" + question
                        #prompt="You are an assistant that only provides direct answers without showing reasoning steps. answer this question in English based on the provided text only. the question is:" + question
                    print ("\nModel: "+theModel+" is running now.")
                    print(f"\nProcessing Question {index + 1}: {question}")
            
                    # Measure elapsed time
                    start_time = time.time()
            
                    # Query the RAG system
                    response, source_documents = query_rag_system(qa_chain, prompt)
                    elapsed_time = time.time() - start_time
                    
                    
                    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
                    response = response.strip()
                    
                    print ("The answer is: "+ response)
                    # Evaluate metrics
                    precision, recall, f1 = calculate_precision_recall_f1(response, reference_answer)
                    cosine_sim = calculate_cosine_sim(response, reference_answer)
            
                    # Calculate elapsed time
                    
                   
                        # Remove non-Arabic characters and normalize whitespace
                        #cleaned_text = re.sub(r".?[^\u0600-\u06FF\s]", "", text)
                       
                    file.writelines("#############################\n\n")  
                    file.writelines(", ".join([doc.page_content for doc in source_documents]))
                    
                    # Append results
                    results.append({
                        "question": question,
                        "reference_answer": reference_answer,
                        "response": response,
                        #"source_documents": ", ".join([doc.page_content for doc in source_documents]),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "cosine_similarity": cosine_sim,
                        "elapsed_time": elapsed_time
                    })
            
                    print(f"Completed Question {index + 1}")
                
                # Calculate averages
                average_precision = sum(result["precision"] for result in results) / len(results)
                average_recall = sum(result["recall"] for result in results) / len(results)
                average_f1 = sum(result["f1"] for result in results) / len(results)
                average_cosine_sim = sum(result["cosine_similarity"] for result in results) / len(results)
                average_elapsed_time = sum(result["elapsed_time"] for result in results) / len(results)
                
                # Create a DataFrame from results
                df_output = pd.DataFrame(results)
                
                # Add a new row at the beginning with the averages
                averages_row = pd.DataFrame([{
                    "question": "Averages",
                    "reference_answer": "",
                    "response": "",
                    #"source_documents": "",
                    "precision": average_precision,
                    "recall": average_recall,
                    "f1": average_f1,
                    "cosine_similarity": average_cosine_sim,
                    "elapsed_time": average_elapsed_time
                }])
                
                # Concatenate the averages row with the rest of the DataFrame
                df_output = pd.concat([averages_row, df_output], ignore_index=True)
                
                # Save results to an output Excel file
                df_output.to_excel(output_excel_path, index=False)
                file.close
                print(f"\nResults saved to: {output_excel_path}")   
             