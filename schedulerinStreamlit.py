import streamlit as st

# Move set_page_config to the top, right after the streamlit import
st.set_page_config(page_title="Calendar Assistant", page_icon="📅")

from datetime import datetime, timedelta
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
# from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
import json
import os.path
import pickle
from dotenv import load_dotenv
import time
from pydantic import BaseModel, Field
load_dotenv()

# ... existing code ...

# Add this after the create_calendar_event tool definition
class AskQuestionArgs(BaseModel):
    """Arguments for the ask_question tool."""
    question: str = Field(description="The question to ask the AI assistant")

@tool(args_schema=AskQuestionArgs)
def ask_question(question: str):
    """Ask a question to the AI assistant based on the loaded documents."""
    documents = []
    data_folder = "data"
    
    # Load documents
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            documents.extend(PyPDFLoader(pdf_path).load())
    
    if not documents:
        print("No documents loaded.")  # Debugging statement
        return "No documents available to answer the question."
    
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embedding_model)
    retriever = vector_store.as_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = '''
    You are a friendly and helpful AI assistant. Your task is to answer questions based on the given context and chat history. If the question is not related to the context, engage in a friendly conversation.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}
    Answer:
    '''

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    if not context:
        print("No relevant documents found for the question.")  # Debugging statement
        return "I couldn't find any relevant documents to answer your question."

    prompt_input = {
        "context": context,
        "question": question,
        "chat_history": memory.chat_memory.messages
    }
    formatted_prompt = prompt.format(**prompt_input)
    response = llm.invoke(formatted_prompt)
    
    if response and hasattr(response, 'content'):
        return response.content
    else:
        print("No response generated from the model.")  # Debugging statement
        return "I couldn't generate a response."
# Google Calendar Setup
SCOPES = ['openid', 'https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']

def authenticate_google_calendar():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Initialize the flow
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            
            flow.redirect_uri = 'http://localhost:3000/callback'
            
            creds = flow.run_local_server(port=3001)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('calendar', 'v3', credentials=creds)

class CreateCalendarEventArgs(BaseModel):
    """Arguments for the create_calendar_event tool."""
    title: str = Field(description="The title of the event")
    start_time: str = Field(description="The start time of the event in ISO 8601 format")
    duration_minutes: int = Field(description="The duration of the event in minutes, defaults to 60")
    attendees: str = Field(description="The attendees of the event, comma-separated email addresses")

@tool(args_schema=CreateCalendarEventArgs)
def create_calendar_event(title: str, start_time: str, duration_minutes: int = 60, attendees: str = None):
    """Create a calendar event with given parameters."""
    service = authenticate_google_calendar()
    start = datetime.fromisoformat(start_time)
    end = start + timedelta(minutes=duration_minutes)
    
    event = {
        'summary': title,
        'start': {'dateTime': start.isoformat(), 'timeZone': 'Asia/Kolkata'},
        'end': {'dateTime': end.isoformat(), 'timeZone': 'Asia/Kolkata'},
    }
    
    if attendees:
        event['attendees'] = [{'email': email.strip()} for email in attendees.split(',')]
    
    event = service.events().insert(calendarId='primary', body=event).execute()
    return f"Event created: {event.get('htmlLink')}"

# LangGraph Setup
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

model = ChatOpenAI(model="gpt-4o-mini")
tools = [create_calendar_event, ask_question]
model = model.bind_tools(tools)

def tool_node(state: AgentState):
    print("TOOL NODE", state)  # Debugging statement
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = {tool.name: tool for tool in tools}[tool_call["name"]].invoke(
            tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def call_model(state: AgentState):
    system_prompt = SystemMessage(
        """You are a helpful assistant that can create calendar events and answer questions based on loaded documents. 
        For calendar events, parse user requests and create events accordingly. Make sure dates are in ISO 8601 format.

        You have access to two tools:

        1. create_calendar_event: Creates a Google Calendar event with specified details.
           Arguments:
           - `title`: (Required) Name/subject of the event
           - `start_time`: (Required) ISO format datetime (YYYY-MM-DDTHH:MM:SS)
           - `duration_minutes`: (Optional) Length of event in minutes, defaults to 60
           - `attendees`: (Optional) Comma-separated email addresses

        2. ask_question: Asks a question to the AI assistant based on loaded documents. This tool is designed to provide information about any content loaded in the documents, including but not limited to meeting details, project information, and general knowledge. 
           Arguments:
           - `question`: (Required) The question to ask the AI assistant

        Example Prompts:
        1. "Schedule a meeting called 'Team Sync' for tomorrow at 3pm"
        2. "Create a 30-minute event titled 'Quick Check-in' with bob@company.com at 2pm today"
        3. "Set up a 2-hour planning session next Monday at 10am with the team: alice@company.com, charlie@company.com"
        4. "What information do we have about project deadlines?"
        5. "Can you summarize the main points from the latest meeting notes?"
        6. "What are the key findings from the recent market research report?"

        Choose the appropriate tool based on the user's request. If the user asks a question about the content of the documents, use the ask_question tool to provide a response."""
    )
    human_prompt = HumanMessage("The time is now: " + datetime.now().isoformat())
    # print(system_prompt, human_prompt, state["messages"])
    response = model.invoke([system_prompt, human_prompt] + state["messages"])
    print("CALL MODEL: ",response)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    print("Last message:", last_message)  # Debugging statement
    print("Tool calls:", last_message.tool_calls)  # Debugging statement
    print("continue" if last_message.tool_calls else "end")
    return "continue" if last_message.tool_calls else "end"

# Initialize Graph with Memory
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Streamlit App
st.title("Calendar Assistant")

# Initialize session ID and messages in Streamlit state
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"

if "messages" not in st.session_state:
    st.session_state.messages = []

config = {"configurable": {"thread_id": st.session_state.session_id}}

# Create a scrollable container for chat history
st.markdown("### Chat History")
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you schedule events?"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process with LangGraph agent
    with st.spinner("Thinking..."):
        # Create input message
        input_message = HumanMessage(content=prompt)
        state_snapshot = graph.get_state(config=config)
        
        conversation_state = {"messages": []}
        if state_snapshot and hasattr(state_snapshot, 'values'):
            conversation_state = state_snapshot.values.get("messages", {"messages": []})
        
        conversation_state["messages"].append(input_message)
        
        for response in graph.stream(conversation_state, config=config):
            if 'agent' in response and 'messages' in response['agent']:
                for message in response['agent']['messages']:
                    if isinstance(message, (AIMessage, ToolMessage)) and (hasattr(message, 'content') and message.content):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": message.content
                        })
                        st.rerun()

# Custom CSS to make the chat container scrollable
st.markdown("""
    <style>
    .stChatMessage {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #e6f3ff;
    }
    .stChatMessage[data-testid="stChatMessageAI"] {
        background-color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)
