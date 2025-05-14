import streamlit as st
import os
import subprocess
import tempfile
import uuid
from typing import List, Dict, Any, Annotated, TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define agent state
class AgentState(TypedDict):
    messages: List[Any]
    user_input: str
    files_info: Dict[str, Any]
    current_agent: str
    next_agent: Optional[str]

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.set_page_config(page_title="File Analysis Agent Team", page_icon="🤖", layout="wide")

# Initialize Gemini model
def initialize_model():
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            convert_system_message_to_human=True
        )
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# File tools
@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

@tool
def list_files(directory: str) -> str:
    """List all files in a directory."""
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

@tool
def execute_python(code: str) -> str:
    """Execute Python code in a safe environment."""
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return f"Execution successful. Output: {local_vars}"
    except Exception as e:
        return f"Execution failed: {str(e)}"

@tool
def analyze_dependencies(directory: str) -> str:
    """Analyze Python dependencies in a directory."""
    try:
        imports = set()
        for file in os.listdir(directory):
            if file.endswith('.py'):
                with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('import') or line.startswith('from'):
                            imports.add(line.strip())
        return "\n".join(imports)
    except Exception as e:
        return f"Error analyzing dependencies: {str(e)}"

@tool
def code_quality_check(file_path: str) -> str:
    """Run pylint code quality check on a Python file."""
    try:
        result = subprocess.run(["pylint", file_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error running pylint: {str(e)}"

# Define specialized agents

def create_coordinator():
    """Create a coordinator agent that decides which specialized agent should handle a request."""
    model = initialize_model()
    
    coordinator_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Coordinator, responsible for routing user requests to the appropriate specialized agent.
        
Available agents:
1. File Manager: Handles file operations like reading, writing, and listing files
2. Code Analyzer: Analyzes code quality, dependencies, and structure
3. Data Processor: Executes Python code and processes data
4. Documentation Agent: Creates documentation and explanations

Based on the user's request, determine which agent is best suited to handle it. If multiple agents are needed, 
select the primary agent that should handle the request first.

Your response should be just the name of the agent: "File Manager", "Code Analyzer", "Data Processor", or "Documentation Agent".
If you're unsure, respond with "Documentation Agent".
"""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}")
    ])
    
    def coordinator_response(state):
        messages = state["messages"]
        user_input = state["user_input"]
        files_info = state["files_info"]
        
        # Add file information to the context
        file_context = f"Available files: {', '.join(files_info.keys())}"
        
        # Invoke the model with the coordinator prompt
        response = model.invoke(coordinator_prompt.format(
            messages=messages,
            user_input=f"{file_context}\n\nUser request: {user_input}"
        ))
        
        # Determine the next agent based on the model's response
        agent_name = response.content.strip()
        
        # Update state with the next agent
        return {
            "messages": messages + [AIMessage(content=f"Coordinator: Routing to {agent_name}")],
            "next_agent": agent_name
        }
    
    return coordinator_response

def create_file_manager():
    """Create a file manager agent that handles file operations."""
    model = initialize_model()
    
    file_manager_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the File Manager, specialized in handling file operations. 
You can read files, write files, and list files in directories.
Provide helpful and concise responses about file operations.
"""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}")
    ])
    
    def file_manager_response(state):
        messages = state["messages"]
        user_input = state["user_input"]
        files_info = state["files_info"]
        
        # Add file information to the context
        file_context = ""
        for file_name, file_path in files_info.items():
            file_context += f"File: {file_name} (Path: {file_path})\n"

        # Invoke the model with the file manager prompt
        response = model.invoke(file_manager_prompt.format(
            messages=messages,
            user_input=f"{file_context}\n\nUser request: {user_input}"
        ))
        
        # Update state with the response
        return {
            "messages": messages + [AIMessage(content=response.content)],
            "next_agent": None  # End the chain after this agent
        }
    
    return file_manager_response

def create_code_analyzer():
    """Create a code analyzer agent that analyzes code quality and structure."""
    model = initialize_model()
    
    code_analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Code Analyzer, specialized in analyzing code quality, dependencies, and structure.
You can analyze Python dependencies and run code quality checks.
Provide detailed analysis and recommendations for improving code.
"""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}")
    ])
    
    def code_analyzer_response(state):
        messages = state["messages"]
        user_input = state["user_input"]
        files_info = state["files_info"]
        
        # Add file information to the context
        file_context = ""
        for file_name, file_path in files_info.items():
            if file_name.endswith('.py'):
                file_content = read_file(file_path)
                file_context += f"Python file: {file_name}\nContent:\n{file_content}\n\n"
        
        # Invoke the model with the code analyzer prompt
        response = model.invoke(code_analyzer_prompt.format(
            messages=messages,
            user_input=f"{file_context}\n\nUser request: {user_input}"
        ))
        
        # Update state with the response
        return {
            "messages": messages + [AIMessage(content=response.content)],
            "next_agent": None  # End the chain after this agent
        }
    
    return code_analyzer_response

def create_data_processor():
    """Create a data processor agent that executes Python code and processes data."""
    model = initialize_model()
    
    data_processor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Data Processor, specialized in executing Python code and processing data.
You can execute Python code to analyze and transform data.
Provide clear explanations of your data processing steps and results.
"""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}")
    ])
    
    def data_processor_response(state):
        messages = state["messages"]
        user_input = state["user_input"]
        files_info = state["files_info"]
        
        # Add file information to the context
        file_context = ""
        for file_name, file_path in files_info.items():
            if file_name.endswith(('.csv', '.txt', '.json', '.py')):
                file_content = read_file(file_path)
                file_context += f"File: {file_name}\nPreview:\n{file_content[:500]}...\n\n"
        
        # Invoke the model with the data processor prompt
        response = model.invoke(data_processor_prompt.format(
            messages=messages,
            user_input=f"{file_context}\n\nUser request: {user_input}"
        ))
        
        # Update state with the response
        return {
            "messages": messages + [AIMessage(content=response.content)],
            "next_agent": None  # End the chain after this agent
        }
    
    return data_processor_response

def create_documentation_agent():
    """Create a documentation agent that creates documentation and explanations."""
    model = initialize_model()
    
    documentation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Documentation Agent, specialized in creating documentation and explanations.
You can explain code, create user guides, and document processes.
Provide clear, comprehensive, and well-structured documentation.
"""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}")
    ])
    
    def documentation_response(state):
        messages = state["messages"]
        user_input = state["user_input"]
        files_info = state["files_info"]
        
        # Add file information to the context
        file_context = ""
        for file_name, file_path in files_info.items():
            if file_name.endswith('.py'):
                file_content = read_file(file_path)
                file_context += f"Python file: {file_name}\nContent:\n{file_content}\n\n"
            else:
                file_context += f"File: {file_name} (Path: {file_path})\n"
        
        # Invoke the model with the documentation prompt
        response = model.invoke(documentation_prompt.format(
            messages=messages,
            user_input=f"{file_context}\n\nUser request: {user_input}"
        ))
        
        # Update state with the response
        return {
            "messages": messages + [AIMessage(content=response.content)],
            "next_agent": None  # End the chain after this agent
        }
    
    return documentation_response

# Create agent routing logic
def route_to_agent(state):
    next_agent = state["next_agent"]
    
    if next_agent == "File Manager":
        return "file_manager"
    elif next_agent == "Code Analyzer":
        return "code_analyzer"
    elif next_agent == "Data Processor":
        return "data_processor"
    elif next_agent == "Documentation Agent":
        return "documentation_agent"
    else:
        # Default to documentation agent if unknown
        return "documentation_agent"

# Create the graph
def create_agent_graph():
    # Create a graph with the agent state
    graph = StateGraph(AgentState)
    
    # Add nodes for each agent
    graph.add_node("coordinator", create_coordinator())
    graph.add_node("file_manager", create_file_manager())
    graph.add_node("code_analyzer", create_code_analyzer())
    graph.add_node("data_processor", create_data_processor())
    graph.add_node("documentation_agent", create_documentation_agent())
    
    # Add conditional edges from coordinator to each agent
    graph.add_conditional_edges(
        "coordinator",
        route_to_agent,
        {
            "file_manager": "file_manager",
            "code_analyzer": "code_analyzer",
            "data_processor": "data_processor",
            "documentation_agent": "documentation_agent"
        }
    )
    
    # Add edges to END for each agent
    graph.add_edge("file_manager", END)
    graph.add_edge("code_analyzer", END)
    graph.add_edge("data_processor", END)
    graph.add_edge("documentation_agent", END)
    
    # Set the entry point
    graph.set_entry_point("coordinator")
    
    # Compile the graph without checkpointing
    return graph.compile()

# Function to save uploaded files
def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    saved_files = {}
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files[uploaded_file.name] = file_path
    return temp_dir, saved_files

# Function to analyze uploaded files
def analyze_uploaded_files(saved_files):
    analysis = []
    for file_name, file_path in saved_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                analysis.append(f"File: {file_name}\nContent Preview: {content[:200]}...\n")
        except Exception as e:
            analysis.append(f"File: {file_name}\nError reading file: {str(e)}\n")
    return "\n".join(analysis)

# Process user input with the agent graph
def process_with_agent_graph(graph, user_input, files_info, messages):
    try:
        # Create the initial state
        initial_state = {
            "messages": messages,
            "user_input": user_input,
            "files_info": files_info,
            "current_agent": "coordinator",
            "next_agent": None
        }
        
        # Process with the graph
        result = graph.invoke(initial_state)
        
        # Return the final messages
        return result["messages"]
    
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        return messages + [AIMessage(content=f"Error processing request: {str(e)}")]

def main():
    st.title("🤖 File Analysis Agent Team")
    
    with st.sidebar:
        st.header("Upload Files")
        uploaded_files = st.file_uploader("Choose files to analyze", accept_multiple_files=True)
        
        if uploaded_files:
            temp_dir, saved_files = save_uploaded_files(uploaded_files)
            st.session_state.uploaded_files = saved_files
            st.success(f"Uploaded {len(uploaded_files)} files")
            
            st.subheader("Uploaded Files:")
            for file_name in saved_files.keys():
                st.write(f"- {file_name}")
            
            if 'initial_analysis' not in st.session_state:
                st.session_state.initial_analysis = analyze_uploaded_files(saved_files)
                st.info("Initial file analysis completed. You can now ask questions about the files.")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.thread_id = str(uuid.uuid4())  # Generate new thread ID
            st.success("Chat history cleared!")
    
    st.header("Chat with the Agent Team")
    
    # Check if model is available
    model = initialize_model()
    if model:
        # Create the agent graph
        graph = create_agent_graph()
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Get user input
        user_input = st.chat_input("Ask about your files or request advanced tasks...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            # Convert chat history to messages format for the agent
            messages = []
            for msg in st.session_state.chat_history[:-1]:  # Exclude the current user message
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Process with agent graph
            result_messages = process_with_agent_graph(
                graph,
                user_input,
                st.session_state.uploaded_files,
                messages
            )
            
            # Extract the final agent response
            final_response = result_messages[-1].content
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": final_response})
            with st.chat_message("assistant"):
                st.write(final_response)

if __name__ == "__main__":
    main()