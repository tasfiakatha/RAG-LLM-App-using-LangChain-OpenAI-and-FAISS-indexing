# DataScout AI: A Retrieval-Augmented Generation (RAG) LLM Application for Website and PDF Querying Using LangChain, OpenAI, and FAISS Indexing
![image](https://github.com/user-attachments/assets/ed6effcb-340d-4a53-b476-dc0635784d44)


## Authors
[@tasfiakatha](https://github.com/tasfiakatha)

## Overview
DataScout AI is a Streamlit-based chatbot application that leverages Retrieval-Augmented Generation (RAG) to provide real-time, accurate, and source-backed answers. Users can upload documents and URLs, which are processed to build a custom knowledge base. The chatbot retrieves relevant information using FAISS indexing, allowing users to interact with the data seamlessly and receive precise answers quickly.

View the application: (https://datascoutai.streamlit.app/)

## Key Advantages
DataScout AI simplifies the data retrieval process by allowing users to query multiple sources of information—such as documents and websites—without manual searching. It combines these sources into a single, searchable database and uses natural language processing to deliver context-aware insights. Built with RAG, OpenAI, LangChain, and FAISS, the platform is scalable and adaptable for future enhancements.

## Method: A Step-by-Step Walkthrough
**1. Setting Up the Environment:**

- The code begins by importing necessary libraries like nltk, requests, streamlit, and openai, among others.
- Environment variables are loaded using the dotenv package, which includes the OpenAI API key for accessing OpenAI's models.


**2. Fetching and Processing Data:**

When the "Get content" button is clicked, the app starts retrieving data:
- URLs: It sends a request to each URL and fetches the raw content (HTML or text).
- Uploaded Files: It reads the content from uploaded files (PDF, DOCX, or TXT).
- The content from both sources is then stored in a list as dictionaries with either a URL or file name.


**3. Text Preprocessing and Splitting:**

- After loading the content, the app checks if the data is valid.
- It then uses LangChain’s RecursiveCharacterTextSplitter to split the content into smaller chunks. This helps in better indexing and retrieval later. These chunks are manageable pieces of text (around 1000 characters each) that are easier for the model to handle.


**4. Creating Embeddings and Storing in FAISS Index:**

- OpenAI embeddings are generated for the split text chunks. These embeddings represent the meaning of the content in a format that allows easy comparison.
- The embeddings are then stored in a FAISS index (a highly efficient vector storage method), which allows for fast similarity searches.
- The FAISS index is stored in Streamlit’s session state, making it accessible throughout the app.


**5. User Queries and Information Retrieval:**

- Once the data is processed and indexed, users can ask questions through a text input box in the Streamlit app.
- When a user submits a question, the app retrieves the most relevant chunks from the FAISS index based on similarity to the user’s query.
- The app then uses OpenAI’s GPT model to generate an answer by analyzing the retrieved content, providing the user with a context-aware response.


**6. Building the Web Interface:**

- A simple web interface is created using Streamlit. The interface includes:
A section for entering URLs to fetch content from.
An option for users to upload text, PDF, or DOCX files.
A button to trigger the data processing from the provided sources.


**7. Deployment and User Interaction:**

- The entire application is deployed on Streamlit Cloud, which allows users to interact with it via a simple web interface.
- Users can upload documents, input URLs, and ask questions about the content—receiving answers backed by the data they provided.


## Installation

### Prerequisites

- Python 3.8+

- Streamlit

- OpenAI API key

### Steps

1. Clone the repository:

![image](https://github.com/user-attachments/assets/648802bf-82be-4e43-8219-80c7fe07c650)


2. Install the required dependencies:

  pip install -r requirements.txt

3. Set up your OpenAI API key:

  Create a .env file in the root directory.

![image](https://github.com/user-attachments/assets/505131ed-471e-4041-9e50-c36c26c4294d)


4. Run the application:

![image](https://github.com/user-attachments/assets/27451a44-f731-413c-96f3-fb5370a7d30b)


  Open the local development URL provided by Streamlit in your browser.


## How to use application?

**1. Input Data**

URLs: Enter up to 3 URLs in the sidebar.

File Upload: Upload .txt, .pdf, or .docx files from your computer.


**2. Process Data**

Click the "Get content" button to start processing.

Monitor the progress and view any errors or warnings in real-time.


**3. Ask Questions**

Once the data is processed, enter a question in the query box.

Click "Enter" to retrieve the answer from the data.


## Project Structure
![image](https://github.com/user-attachments/assets/da5d2956-3d02-46d3-b84e-a74f8f77755a)


## Technologies Used
1. Retrieval Augmented Generation (RAG): Framework that combines large language models (LLMs) with information retrieval systems. RAG improves the quality of responses by giving LLMs access to information outside of their training data

2. OpenAI GPT-3.5-turbo: Language model for conversational responses.

3. LangChain: Framework for building retrieval-based conversational AI.

4. FAISS: Efficient similarity search for embedding-based retrieval.

5. Streamlit: Web interface for user interaction.

## Limitations
File Size: Large files may impact performance during text extraction and embedding creation.  
Scanned PDFs: Does not support text extraction from scanned documents.

## Contribution
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or contribute.

## License
MIT License

Copyright (c) 2025 Tasfia Katha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Learn more about [MIT](https://choosealicense.com/licenses/mit/) license

