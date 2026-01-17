from langchain_ollama import OllamaLLM

import pandas as pd

llm = OllamaLLM(model="llama3.3",num_thread=24)



def process_excel_file(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Ensure required columns exist
    required_columns = ["question", "reference_answer", "response"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing: {missing_columns}")
    
    # Initialize a list to store scores
    scores = []
    
    # Iterate over each row and get the comparison score
    for index, row in df.iterrows():
       
        reference_answer = row["reference_answer"]
        response = row["response"]
        
        # Get the score from Llama3.1 via Ollama
        print (index,"--->")
        score = get_comparison_score(reference_answer, response)
        scores.append(score)
       
    # Add the scores to the DataFrame
    df["comparison_score"] = scores
    
    # Save the updated DataFrame back to Excel
    output_file_path = file_path.replace(".xlsx", "_updated.xlsx")
    df.to_excel(output_file_path, index=False)
    print(f"Updated Excel file saved as: {output_file_path}")


def get_comparison_score( reference_answer, response):
    # Define the prompt for Llama3.1 to evaluate the response
    prompt = f"""
    I want you to analyze and compare the meanings of the following two sentences they represents answers, scentence 1 is the reference answer and scentence 2 is the student answer . After analyzing them, calculate and provide the percentage of correctness of the student answer based on the reference answer. The percentage should be a number between 0% (completely incorrect) and 100% (identical in meaning and completely correct).
    
    Sentence 1: {reference_answer} 
    Sentence 2: {response}
    Please provide only the numerical score (e.g., 0.8) without any additional text. and if  Sentence 2 contains any word in language other than Arabic or english give it a score of 0
    """
    response = llm.invoke(prompt)
    print(response)
    return response



if __name__ == "__main__":
    models=["command-r7b" ,"qwen2.5","llama3.3","llama3.1","deepseek-r1"]#"
    topics=["law","management","history","science"]
    for topic in topics:
        for model in models:
      
            excel_file_path = "D:\\Research\\GEN-AI\\PH2\\"+topic+"\\Arabic\\"+model+"-Arabic_results.xlsx"
            process_excel_file(excel_file_path)
            
            excel_file_path = "D:\\Research\\GEN-AI\\PH2\\"+topic+"\\English\\"+model+"-English_results.xlsx"
            process_excel_file(excel_file_path)
   
