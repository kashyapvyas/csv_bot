#importing all the required dependencies

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages.ai import AIMessage
import re

# Load the CSV file
file_path = './data/ai412020.csv'  
data = pd.read_csv(file_path)

# Initialize the LLM
groq_api_key = "####"  # I have removed my API key for security purpose
llm_groq = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=0.2)

# Initialize HuggingFaceEmbeddings with the model name
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load CSV into document loader
loader = CSVLoader(file_path)
documents = loader.load()

# Create FAISS vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Enhanced prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
    You are a highly knowledgeable data analyst and machine learning statistics expert. You have access to a dataset from a manufacturing company containing information about predictive maintenance.
    Based on the following dataset context, please provide a comprehensive and intelligent response to the query.

    Dataset Context:
    {context}

    Query:
    {query}
    
    Answer:
    Provide the Python pandas code to filter and display the relevant data. The output should be stored in a variable named 'result'. For example, if the query was "show Air temperature [K] when Torque [Nm] is greater than 42", the answer should be:

    result = data[data["Torque [Nm]"] > 42][["Air temperature [K]"]]
    """
)

#handle query function handles the incoming queries
def handle_query(query):
    try:
        # Retrieve relevant documents
        retriever = vector_store.as_retriever()
        relevant_docs = retriever.invoke(input=query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        if not context:
            return "I couldn't find any relevant data in the CSV to answer your question."

        # Format the prompt with the retrieved context
        formatted_prompt = prompt_template.format(context=context, query=query)

        # Log the formatted prompt for debugging
        print("Formatted Prompt:\n", formatted_prompt)

        # Run the LLM with the formatted prompt
        response = llm_groq.invoke(input=formatted_prompt)

        # Print the entire response for debugging
        print("LLM Response:\n", response)
            # Print the response type and structure
        print("Response Type:", type(response))
        print("Response Structure:", response)

        if response and isinstance(response,AIMessage):
            llm_output = response.content.strip()
          #  llm_output = response['content'].strip()
            print("Generated Code:\n", llm_output)
            
            # Extract the code part from the response
            code_match = re.search(r'(result\s*=\s*data\[.*\]\[.*\])', llm_output)
            if code_match:
                code = code_match.group(1)
                print("Extracted Code:\n", code)

       
                
                # Execute the extracted code
                exec(code)
                return locals().get('result', 'The LLM did not generate a result variable.') 
            else:
                return ""
        else:
            return "Sorry, Please improve the query for accurate results"

    except Exception as e:
        return f"An error occurred: {str(e)}"

#this function performs calculations like mean,average on the data columns
def perform_calculations(operation, column):
    try:
        column = next((col for col in data.columns if col.lower() == column.lower()), None)
        if column is None:
            return f"Column '{column}' does not exist in the dataset. Available columns are: {', '.join(data.columns)}"

        if operation == "mean":
            result = data[column].mean()
        elif operation == "average":
            result = data[column].mean()
        else:
            return f"Unsupported operation: {operation}"
        return f"The {operation} of {column} is {result}"
    except Exception as e:
        return f"An error occurred during calculations: {str(e)}"

#here we return output as a table
def return_table(query):
    try:
        query = query.lower()
        if "top" in query or "bottom" in query:
            parts = query.split()
            num = int(parts[parts.index("top") + 1] if "top" in parts else parts[parts.index("bottom") + 1])
            column_start_index = parts.index("of") + 1 if "of" in parts else -1
            if column_start_index != -1 and column_start_index < len(parts):
                column_name_parts = parts[column_start_index:]
                column = next((col for col in data.columns if " ".join(column_name_parts).strip() in col.lower()), None)
                if column is None:
                    return f"Column '{' '.join(column_name_parts)}' does not exist in the dataset. Available columns are: {', '.join(data.columns)}"
                if "top" in query:
                    filtered_data = data.nlargest(num, column)
                else:
                    filtered_data = data.nsmallest(num, column)
                return filtered_data
            else:
                return "Please provide a valid query for retrieving top or bottom values."
        else:
            return "Please provide a valid query for retrieving top or bottom values."
    except Exception as e:
        return f"An error occurred while processing the query: {str(e)}"


#this function helps implement data visualizations
def plot_data(data, x_column, y_columns, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    for y_column in y_columns:
        sns.lineplot(x=data[x_column], y=data[y_column], label=y_column)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    st.pyplot(plt)

# Streamlit App
st.title("AI4I 2020 Predictive Maintenance Chatbot")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        if "mean" in query.lower() or "average" in query.lower():
            parts = query.lower().split()
            if "mean" in parts:
                column_start_index = parts.index("mean") + 2 if "of" in parts else parts.index("mean") + 1
                if column_start_index < len(parts):
                    column_name_parts = parts[column_start_index:]
                    column = next((col for col in data.columns if " ".join(column_name_parts).strip() in col.lower()), None)
                    if column:
                        calculation_result = perform_calculations("mean", column.strip())
                    else:
                        calculation_result = f"Column '{' '.join(column_name_parts)}' does not exist in the dataset. Available columns are: {', '.join(data.columns)}"
                else:
                    calculation_result = "Please specify a column to calculate the mean."
            elif "average" in parts:
                column_start_index = parts.index("average") + 2 if "of" in parts else parts.index("average") + 1
                if column_start_index < len(parts):
                    column_name_parts = parts[column_start_index:]
                    column = next((col for col in data.columns if " ".join(column_name_parts).strip() in col.lower()), None)
                    if column:
                        calculation_result = perform_calculations("average", column.strip())
                    else:
                        calculation_result = f"Column '{' '.join(column_name_parts)}' does not exist in the dataset. Available columns are: {', '.join(data.columns)}"
                else:
                    calculation_result = "Please specify a column to calculate the average."
            st.write(f"Calculation Result: {calculation_result}")
        elif "show" in query.lower() and ("top" in query.lower() or "bottom" in query.lower()):
            table_result = return_table(query)
            if isinstance(table_result, pd.DataFrame):
                st.dataframe(table_result)
            else:
                st.write(table_result)
        else:
            response = handle_query(query)
            st.write(response)
        
        # Check if the query is about a specific column and plot relevant data
        if "temperature" in query.lower():
            plot_data(data, 'UDI', ['Air temperature [K]', 'Process temperature [K]'], 'Temperature Over Time', 'Index', 'Temperature [K]')
        elif "speed" in query.lower() or "torque" in query.lower():
            plot_data(data, 'UDI', ['Rotational speed [rpm]', 'Torque [Nm]'], 'Speed and Torque Over Time', 'Index', 'Value')
        elif "tool wear" in query.lower():
            plot_data(data, 'UDI', ['Tool wear [min]'], 'Tool Wear Over Time', 'Index', 'Tool Wear [min]')
        elif "failure" in query.lower():
            failure_columns = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            plot_data(data, 'UDI', failure_columns, 'Failures Over Time', 'Index', 'Failure Count')
    else:
        st.write("Please enter a query.")