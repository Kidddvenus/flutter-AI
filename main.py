# Import necessary libraries for document loading and processing
from langchain_community.document_loaders import PyPDFLoader  # For loading PDF documents
from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings
from langchain.indexes import VectorstoreIndexCreator  # For creating vector store indexes
from langchain.chains import RetrievalQA  # For question-answering chains
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text
from langchain_community.vectorstores import FAISS  # Vector store implementation
from langchain_core.runnables import RunnableLambda  # For creating runnable pipelines
from langchain_core.prompt_values import StringPromptValue  # For handling prompt values
import streamlit as st  # For building the web interface
import asyncio  # For asynchronous operations

# IBM Watsonx imports for AI model integration
from ibm_watsonx_ai import Credentials  # For authentication to IBMwatsonx account
from ibm_watsonx_ai.foundation_models import ModelInference  # For model interaction
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params  # For parameter names

# Initialize Watsonx credentials with API key and service URL
creds = Credentials(
    api_key="",  # My API key
    url="https://eu-de.ml.cloud.ibm.com"  # IBM Cloud service URL
)

# Initialize the Watsonx language model with specific parameters
watsonx_llm = ModelInference(
    model_id='meta-llama/llama-3-3-70b-instruct',  # Model identifier
    credentials=creds,  # Authentication credentials
    project_id="",  # IBM Cloud project ID
    params={
        Params.DECODING_METHOD: 'sample',  # Text generation method
        Params.MAX_NEW_TOKENS: 200,  # Maximum length of generated response
        Params.TEMPERATURE: 0.5  # Creativity/randomness control (0-1)
    }
)


def watsonx_generate(prompt, **kwargs):
    """
    Custom wrapper function to handle Watsonx API calls with different prompt types.

    Args:
        prompt: Input prompt which can be StringPromptValue, str, or list
        **kwargs: Additional arguments (ignored in this implementation)

    Returns:
        str: Generated text response from Watsonx

    Raises:
        ValueError: If prompt type is not recognized
    """
    # Handle different prompt types from LangChain
    if isinstance(prompt, StringPromptValue):
        prompt_text = prompt.to_string()  # Convert LangChain prompt to string
    elif isinstance(prompt, str):
        prompt_text = prompt  # Use string directly
    elif isinstance(prompt, list):
        prompt_text = prompt[0] if len(prompt) > 0 else ""  # Use first item if list
    else:
        raise ValueError(f"Unexpected prompt type: {type(prompt)}")

    # Get response from Watsonx API
    response = watsonx_llm.generate(prompt=prompt_text)

    # Extract text from different possible response formats
    if isinstance(response, dict):
        if 'results' in response and len(response['results']) > 0:
            return response['results'][0].get('generated_text', '')  # Standard response format
        elif 'generated_text' in response:
            return response['generated_text']  # Alternative response format
        else:
            return str(response)  # Fallback for unexpected dictionary format
    return str(response)  # Fallback for non-dictionary responses


# Create LangChain runnable with our custom function
llm = RunnableLambda(watsonx_generate)

# Initialize Streamlit application
st.title('Ask me about flutter')  # Set app title

# Initialize chat message history if not exists
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages in the chat interface
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


@st.cache_resource  # Cache the PDF loading to improve performance
def load_pdf():
    """
    Load and process PDF document, creating a vector store index.

    Returns:
        VectorstoreIndexCreator: Index containing document embeddings
    """
    # Create new event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        pdf_name = "flutter_tutorial.pdf"  # PDF document to load
        loader = PyPDFLoader(pdf_name)  # Create PDF loader

        # Create vector store index with specific configuration
        index = VectorstoreIndexCreator(
            vectorstore_cls=FAISS,  # Use FAISS for vector storage
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),  # Embedding model
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=500,  # Size of text chunks
                chunk_overlap=50  # Overlap between chunks
            )
        ).from_loaders([loader])  # Create index from loader
        return index
    finally:
        loop.close()  # Clean up event loop


def main():
    """
    Main execution function for the Streamlit application.
    Handles user input, generates responses, and manages chat interface.
    """
    # Load PDF and create vector index
    index = load_pdf()

    # Create question-answering chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,  # Our custom LLM wrapper
        chain_type='stuff',  # Chain type for document processing
        retriever=index.vectorstore.as_retriever(),  # Document retriever
        input_key="question"  # Input key for the question
    )

    # Get user input from chat interface
    user_input = st.chat_input('Enter your query here')
    if user_input:
        # Display user message and add to history
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        try:
            # Generate response using the QA chain
            response = chain.invoke({"question": user_input})

            # Ensure response is in string format for display
            result = str(response.get('result', '')) if isinstance(response, dict) else str(response)

            # Display assistant response and add to history
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        except Exception as e:
            # Display error message if something goes wrong
            st.error(f"Error generating response: {str(e)}")


# Standard Python entry point
if __name__ == '__main__':
    main()  # Run the application
