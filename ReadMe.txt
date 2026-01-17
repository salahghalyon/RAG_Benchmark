This README file is to accompany code for  multilingual benchmark of Retrieval-Augmented Generation (RAG) systems applied to Arabic and English documents,
produced by Bassam J. Mohd, Khalil M. Ahmad Yousef, and Salah Abu Ghalyon as a companion to their paper:
Benchmarking and AI-Assisted Human-Like Evaluation of Retrieval-Augmented Generation for Arabic and English Documents.

This work has been done at the computer engineering departmnet at the Hashemite University, Zarqa, Jordan.

If you use this code in project that results in a publication, please cite the paper above. 

This README covers the type of attacks or threats that can be perofrmed using this tool on Adept robotic platforms.

January 17, 2026

Comments/Bugs/Problems: khalil@hu.edu.jo,Bassam@hu.edu.jo, and salah.g.ghalyon@hu.edu.jo 

Required library:

time
pandas
langchain
sklearn


Made Assumption(s): 

We evaluate five open-source Large Language Models (LLMs): Llama3.3, Llama3.1, Qwen2.5, Command R7b, and DeepSeek R1, across four knowledge domains: History, Law, Management, and Science. Our hybrid evaluation framework integrates standard lexical metrics (Precision, Recall, F1-Score) with a novel semantic assessment using an LLM-based evaluator to simulate human judgment of answer correctness