import streamlit as st

# Move set_page_config to the top, right after the streamlit import
st.set_page_config(page_title="Calendar Assistant", page_icon="📅")

from datetime import datetime, timedelta
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import json
import os.path
import pickle
from dotenv import load_dotenv
import time
from pydantic import BaseModel, Field
load_dotenv()

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
        'start': {'dateTime': start.isoformat(), 'timeZone': 'UTC'},
        'end': {'dateTime': end.isoformat(), 'timeZone': 'UTC'},
    }
    
    if attendees:
        event['attendees'] = [{'email': email.strip()} for email in attendees.split(',')]
    
    event = service.events().insert(calendarId='primary', body=event).execute()
    return f"Event created: {event.get('htmlLink')}"

# LangGraph Setup
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

model = ChatOpenAI(model="gpt-4o-mini")
tools = [create_calendar_event]
model = model.bind_tools(tools)

def tool_node(state: AgentState):
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
        """You are a helpful assistant that creates calendar events. Parse user requests and create events accordingly.
        Make sure that its in ISO 8601 format.

        Creates a Google Calendar event with specified details.

        ## Usage Example
        "Schedule a team meeting titled 'Q4 Planning' tomorrow at 2pm for 90 minutes with john@example.com and mary@example.com"

        ## Arguments

        - `title`: (Required) Name/subject of the event
        - `start_time`: (Required) ISO format datetime (YYYY-MM-DDTHH:MM:SS)
        - `duration_minutes`: (Optional) Length of event in minutes, defaults to 60
        - `attendees`: (Optional) Comma-separated email addresses

    ## Example Prompts

    1. "Schedule a meeting called 'Team Sync' for tomorrow at 3pm"
    2. "Create a 30-minute event titled 'Quick Check-in' with bob@company.com at 2pm today"
        3. "Set up a 2-hour planning session next Monday at 10am with the team: alice@company.com, charlie@company.com"""
    )
    human_prompt = HumanMessage("The time is now: " + datetime.now().isoformat())
    response = model.invoke([system_prompt, human_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
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

# Initialize session ID in Streamlit state
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"

config = {"configurable": {"thread_id": st.session_state.session_id}}


# Chat input
if prompt := st.chat_input("How can I help you schedule events?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with LangGraph agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create input message
            input_message = HumanMessage(content=prompt)
            
            # Get conversation state from memory or start new
            state_snapshot = graph.get_state(config=config)
            print("State snapshot:", state_snapshot)
            
            conversation_state = {"messages": []}
            if state_snapshot and hasattr(state_snapshot, 'values'):
                conversation_state = state_snapshot.values.get("messages", {"messages": []})
            
            conversation_state["messages"].append(input_message)
            
            # Run agent
            for response in graph.stream(conversation_state, config=config):
                print("Response:", response)
                if 'agent' in response and 'messages' in response['agent']:
                    last_message = response['agent']['messages'][-1]
                    if isinstance(last_message, AIMessage) and hasattr(last_message, 'content'):
                        st.markdown(last_message.content)

# Display conversation history from memory
state_snapshot = graph.get_state(config=config)
print("Final state snapshot:", state_snapshot)

if state_snapshot and hasattr(state_snapshot, 'values'):
    conversation_state = state_snapshot.values.get("messages", {})
    if conversation_state and isinstance(conversation_state, dict) and "messages" in conversation_state:
        messages = conversation_state["messages"]
        if messages and isinstance(messages, (list, tuple)):
            for message in messages:
                role = "user" if isinstance(message, HumanMessage) else "assistant"
                with st.chat_message(role):
                    if hasattr(message, 'content'):
                        st.markdown(message.content)
