# Academix – Your Smartest Study Companion! 🚀📚

**Academix** is a powerful study assistant tool built with Python, Streamlit, and AI-powered models like Google's Gemini and sentence-transformers. This tool allows you to upload study materials, summarize documents, create quizzes, generate notes, and query documents using natural language processing (NLP). It also provides intelligent question-answering capabilities based on the content of your documents and external web searches.

## Features

- **File Upload**: Upload PDFs and text files for processing.
- **Document Summarization**: Automatically generate concise summaries of documents.
- **Quiz Generation**: Create quizzes based on the content of uploaded documents.
- **Document Q&A**: Ask questions related to the uploaded documents and get answers using a mix of document content and web search results.
- **Notes Generation**: Generate key insights and notes from documents.
- **Search Integration**: Fetch relevant information from the web when the context from the document is insufficient.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- Streamlit
- TensorFlow
- `faiss`
- `sentence-transformers`
- `reportlab`
- `google-generativeai`
- `sqlite3`
- `PyPDF2`
- `numpy`
- `langchain` and `tavily_search`

## Setup and Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/udhaybhat00/academixxx
    cd academixxx
    ```

2. **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Required Packages**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up API Keys**:

    You will need the following API keys for this project:

    - **Tavily API Key**: For web search functionality.
    - **Google Gemini API Key**: For content generation and NLP tasks.

    You can store these keys in a `.streamlit/secrets.toml` file like this:

    ```toml
    TAVILY_API_KEY = "your-tavily-api-key"
    GEMINI_API_KEY = "your-gemini-api-key"
    ```

## How to Run the App

Once everything is set up, run the following command to launch the app:

```bash
streamlit run Academix.py
```
## Contributors

- **Aharnish**
- **Ayushmaan**
- **Udhay**
- **Shreyansh**

## Acknowledgments

Thank you to all the contributors for their efforts in making Academix a smarter and better study companion. Special thanks to the open-source community for providing valuable tools and resources.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.



