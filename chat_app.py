import os
import time
from io import BytesIO
import streamlit as st
from rag_pipeline import RAG, TextProcessor

#Initialize TextProcessor class
textprocessor=TextProcessor(chunk_size=1000, chunk_overlap=20)

# Set an environment variable
os.environ['PINECONE_API_KEY'] = '193bfd5b-1a4a-4c8b-bc06-7ec7c4cfc66a'

PINECONE_API_KEY="193bfd5b-1a4a-4c8b-bc06-7ec7c4cfc66a"

# List of recommended questions
recommended_questions = [
    "Tell about baroda home loans",
    "benifits of Baroda Home Loan",
    "what is statement of confidentiality?"
]


# Set the page title and layout
st.set_page_config(page_title="Chatbot Interface", layout="wide")

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'endpoint' not in st.session_state:
    st.session_state.endpoint = ''
if 'version' not in st.session_state:
    st.session_state.version = ''
if 'model_name' not in st.session_state:
    st.session_state.model_name = ''
if 'embedding_model_name' not in st.session_state:
    st.session_state.embedding_model_name = ''
if 'openai_type' not in st.session_state:
    st.session_state.openai_type = ''
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Function to display the modal
def display_modal():
    with st.expander("Please enter your Model Details:", expanded=False):
        # User selection for OpenAI or Azure OpenAI
        service = st.selectbox("Select Service", ["OpenAI", "Azure OpenAI"])
        with st.form(key='model_details_form'):
            if service == "OpenAI":
                st.session_state.openai_type = "openai"
                st.session_state.api_key = st.text_input("API Key", type="password")
                st.session_state.model_name = st.text_input("Model Name")
                st.session_state.embedding_model_name = st.text_input("Embedding Model Name")
                st.warning("Use text-embedding-002 model as VectorDB is optimized for it.")
            elif service == "Azure OpenAI":
                st.session_state.openai_type = "azure_openai"
                st.session_state.api_key = st.text_input("API Key", type="password")
                st.session_state.endpoint = st.text_input("Endpoint")
                st.session_state.version = st.text_input("Version")
                st.session_state.model_name = st.text_input("Model Name")
                st.session_state.embedding_model_name = st.text_input("Embedding Model Name")
                st.warning("Use text-embedding-002 model as VectorDB is optimized for it.")

            if st.form_submit_button("Submit"):
                # Collect and display the input values
                st.success("Details submitted successfully!")

# Display the modal when the app starts
display_modal()

# Function to show notification
def show_notification(message, message_type="success"):
    # Define CSS styles for the notification
    css = f"""
    <style>
    .notification {{
        position: fixed;
        top: 0;
        right: 0;
        margin: 20px;
        padding: 10px 20px;
        color: white; /* Text color */
        border-radius: 5px;
        z-index: 1000; /* Ensure it's on top */
        background-color: {"#4CAF50" if message_type == "success" else "#f44336"}; /* Green for success, red for error */
    }}
    </style>
    """
    notification_placeholder = st.empty()
    notification_placeholder.markdown(f'{css}<div class="notification">{message}</div>', unsafe_allow_html=True)
    time.sleep(5)
    notification_placeholder.empty()


if st.session_state.api_key:
    openai_type=st.session_state.openai_type
    gpt_engine_name=st.session_state.model_name
    embedding_model_name=st.session_state.embedding_model_name
    azure_endpoint = st.session_state.endpoint
    api_key=st.session_state.api_key
    api_version= st.session_state.version
    rag_obj=RAG('indextext', textprocessor, PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type)


# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

uploaded_files = None
selected_file = None


# Handle file uploads only when user interacts with the sidebar
with st.sidebar:
    uploaded_files = st.file_uploader('Upload files', type=['pdf'], accept_multiple_files=True, label_visibility="hidden")

    # condition=st.session_state.uploaded_files==uploaded_files
    # if condition:
    for uploaded_file in uploaded_files:
        st.write("Select files:")
        if uploaded_file not in st.session_state.uploaded_files:
            bytes_data = uploaded_file.read()
            file_like_object = BytesIO(bytes_data)
            file_name=uploaded_file.name
            try:
                if st.session_state.api_key:
                    show_notification(f"{uploaded_file.name} preprocessing started, Please wait for a while!")
                    rag_obj.insert_doc(file_like_object, file_name)
                    show_notification(f"{uploaded_file.name} inserted successfully!")
                    if st.checkbox(uploaded_file.name):
                        st.session_state.selected_files.append(uploaded_file)
                    st.session_state.uploaded_files.append(uploaded_file)
                else:
                    show_notification("Please enter your openai credentails before uploading documents!", message_type='error')
            except Exception as e:
                show_notification("Something went wrong while preprocessing, please try again!", message_type='error')
        else:
            if st.checkbox(uploaded_file.name):
                st.session_state.selected_files.append(uploaded_file)


# Main content area
st.markdown("### What can I help you with today?")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state for the selected question
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = "What is up?"

def handle_button_click(question):
    st.session_state.selected_question = question

# Function to handle button click
def handle_button_click(question):
    st.session_state.selected_question = question
    # Trigger bot query
    process_input(question)

def process_input(prompt):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        if api_key:
            answer=rag_obj.qna(prompt)
            with st.chat_message("BOT"):
                st.write_stream(answer)
            # Add user message to chat history
            final_answer=rag_obj.answer
            st.session_state.messages.append({"role": "Bot", "content": final_answer})
            rag_obj.answer=''
        else:
            st.error("Please enter your API Key and other details at the top.")
    except Exception as e:
        answer="Something went wrong, please check your openai credentials"
        st.error(answer)

# Display recommended questions as buttons
if not st.session_state.messages:
    st.write("**Recommended Questions:**")
    cols = st.columns(len(recommended_questions))
    for i, question in enumerate(recommended_questions):
        if cols[i].button(question):
            handle_button_click(question)

# Prefill input bar with the selected question if available
if prompt := st.chat_input(st.session_state.selected_question):
    process_input(prompt)


# Dummy element to trigger auto-scrolling
auto_scroll = st.empty()

# Trigger auto-scroll by adding an empty message after all other messages
with auto_scroll:
    st.write("")  # This empty write forces the page to render and auto-scroll
