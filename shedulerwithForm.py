import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Annotated, Sequence, TypedDict
from datetime import datetime, timedelta
import json
import uuid
import os
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

load_dotenv()
SCOPES = ['openid', 'https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']

# Pydantic models for question types
class BaseQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question_text: str
    required: bool = True
    description: Optional[str] = None

class TextQuestion(BaseQuestion):
    type: Literal["text"]
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    placeholder: Optional[str] = None

class SliderQuestion(BaseQuestion):
    '''Use this for questions that require a numerical rating or budget or other range'''
    type: Literal["slider"]
    min_value: float 
    max_value: float
    step: float = 1.0
    unit: Optional[str] = None

class MultipleChoiceQuestion(BaseQuestion):
    '''Use this for questions that require a multiple choice selection'''
    type: Literal["multiple_choice"]
    options: List[str] = Field(..., description="List of options for the multiple choice question")
    allow_multiple: bool = False

class RGBQuestion(BaseQuestion):
    '''Use this for questions that require a color selection'''
    type: Literal["rgb"]
    default_color: Optional[str] 

class DateTimeQuestion(BaseQuestion):
    '''Use this for questions that require a date and time selection'''
    type: Literal["datetime"]
    start_time: Optional[str] = Field(None, description="Start time in ISO format")
# Define the Question type using Union
Question = TextQuestion | SliderQuestion | MultipleChoiceQuestion | RGBQuestion | DateTimeQuestion

# Main form model
class Form(BaseModel):
    id: str = Field(default_factory=lambda: f"survey_{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}")
    title: str = Field(..., description="Title of the survey")
    description: Optional[str] = Field(None, description="Description of the survey purpose")
    questions: List[Question] = Field(..., description="List of survey questions")
    version: str = Field(default="1.0", description="Form version")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp in ISO format")

    class Config:
        json_schema_extra = {
            "examples": [{
                "id": "customer-satisfaction-20240101",
                "title": "Customer Satisfaction Survey",
                "description": "Help us improve our services",
                "questions": [
                    {
                        "type": "multiple_choice",
                        "id": "q1",
                        "question_text": "How satisfied are you with our service?",
                        "options": ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"],
                        "required": True
                    },
                    {
                        "type": "text",
                        "id": "q2",
                        "question_text": "What could we improve?",
                        "max_length": 500,
                        "required": True
                    }
                ]
            }]
        }

class MeetingDetails(BaseModel):
    title: str = Field(..., description="Title of the Meeting")
    date: str = Field(..., description="Date of the meeting")
    time: str = Field(..., description="Time of the meeting")
    participants: List[str] = Field(..., description="List of participants")




def generate_survey(topic: str, num_questions: int, requirements: str, api_key: str) -> Form:
    """Generate a survey using OpenAI"""
    template = """Generate a survey form based on the following specifications:
    Topic: {topic}
    Number of questions: {num_questions}
    Additional requirements: {requirements}
    
    Create a form with a mix of question types (text, slider, multiple choice, rgb) that are appropriate for the topic.
    Make sure the questions flow logically and cover the topic comprehensively.
    Use clear, concise language for questions and descriptions.
    Guidelines for creating questions:
    1. Only create questions that are absolutely necessary to help the user
    2. Use appropriate question types:
       - Text: for open-ended responses and detailed information
       - Slider: for ratings, budgets, or numerical ranges
       - Multiple choice: for specific options or preferences
       - RGB: only for color-related queries
       - DateTime: for scheduling meetings or events
       
    
    Examples of when to create questions:
    - User wants product recommendations (ask about preferences, budget, requirements)
    - User needs technical advice (ask about specifications, current setup)
    - User seeks personalized suggestions (ask about preferences, constraints)
    - User wants to schedule a meeting or create a calendar event or add something to the calendar schedule (ask for meeting title, date, start time, duration and participants details)
    
    Do NOT create questions when:
    - User asks for factual information
    - User's query is clear and complete
    - Information can be provided directly
    
    Each question should:
    1. Have a clear purpose related to helping the user
    2. Be concise and specific
    3. Include helpful descriptions when needed
    4. Use appropriate question types
    
    The form should be professional and focused on gathering only essential information."""


    try:
        llm = ChatOpenAI(model="gpt-4o-mini",temperature=0, api_key=api_key)
        chain = llm.with_structured_output(Form)
        
        formatted_prompt = template.format(
            topic=topic,
            num_questions=num_questions,
            requirements=requirements
        )
        
        form = chain.invoke(formatted_prompt)
        
        # Ensure unique IDs
        used_ids = set()
        for i, question in enumerate(form.questions, 1):
            if not question.id or question.id in used_ids:
                question.id = f"q{i}"
            used_ids.add(question.id)
        
        return form
        
    except Exception as e:
        raise Exception(f"Error generating survey: {str(e)}")


# Render a question based on its type
def render_question(question: Question):
    """Render a single question based on its type"""
    # print(question['type'])
    try:
        # Display question text and description
        st.markdown(f"{question['question_text']}")
        # if question.description:
        #     st.markdown(f'*{question["description"]}*')
            
        st.markdown('') # Add some spacing
        
        # Handle different question types based on type attribute
        if question['type'] == 'text':
            return st.text_input(
                label=question['question_text'],
                key=f'text_{question["id"]}',
                max_chars=question['max_length'],
                placeholder=question['placeholder'] or 'Type your answer here...',
                label_visibility='collapsed'
            )
            
        elif question['type'] == 'multiple_choice':
            if question["allow_multiple"]:
                return st.multiselect(
                    label=question['question_text'],
                    options=question['options'],
                    key=f'multi_{question["id"]}',
                    label_visibility='collapsed'
                )
            else:
                return st.selectbox(
                    label=question['question_text'],
                    options=question['options'],
                    key=f'single_{question["id"]}',
                    label_visibility='collapsed'
                )
        
        elif question['type'] == 'slider':
            # Format string for currency if unit is provided
            format_str = '${:,}' if question['unit'] == '$' else None
            
            return st.slider(
                label=question['question_text'],
                min_value=question['min_value'],
                max_value=question['max_value'],
                step=question['step'],
                key=f'slider_{question["id"]}',
                label_visibility='collapsed',
                format=format_str
            )
            
        elif question['type'] == 'rgb':
            return st.color_picker(
                label=question['question_text'],
                value=question['default_color'] if question['default_color'] else '#FFFFFF',
                key=f'rgb_{question["id"]}',
                label_visibility='collapsed'
            )
        elif question['type'] == 'datetime':
            # Use Streamlit's date and time input widgets
            date = st.date_input(
                label=f'{question["question_text"]} - Date',
                key=f'date_{question["id"]}',
                label_visibility='collapsed'
            )   
            time = st.time_input(
                label=f'{question["question_text"]} - Time',
                key=f'time_{question["id"]}',
                label_visibility='collapsed'
            )
            # Combine date and time into a single datetime object
            return datetime.combine(date, time).isoformat()


        st.markdown("---")  # Add separator between questions
        
    except Exception as e:
        st.error(f"Error rendering question: {str(e)}")
        return None
    
# Process responses from form submission
def process_form_responses(responses: dict, form: Form, llm: ChatOpenAI) -> str:
    """Process form responses using LLM"""
    process_prompt = """You are a helpful AI assistant analyzing user responses to provide personalized recommendations and insights. If the user has provided details for a meeting, you will prioritize creating the event using the create_calendar_event function.

    Context:
    Form Title: {title}
    Form Purpose: {description}

    User's Responses:
    {responses}

    Please provide a detailed analysis and recommendations based on these responses. Your response should include:

    1. A brief summary of the user's requirements/preferences
    2. Specific real and accurate recommendations based on their responses (at least 2-3 options if applicable else tell the user that the given scenario of the values entered may not exist)
    3. Detailed justification for each recommendation
    4. Additional insights or considerations
    5. Any follow-up suggestions or questions if needed

    If the user has provided details for a meeting, such as title, start time, duration, and attendees, please prioritize creating the event using the create_calendar_event function.

    Guidelines for your response:
    - If the details are for a meeting ignore all the following else make them the priority
    - Be specific and actionative in your recommendations
    - Explain your reasoning clearly
    - Consider all aspects of their responses holistically
    - If any response needs clarification, mention it
    - Provide practical next steps or actions they can take
    - If budget was mentioned, ensure recommendations stay within their range
    - Consider any specific preferences or requirements they've mentioned
    
    Make your response conversational and helpful, focusing on providing valuable insights and clear recommendations that directly address their needs."""

    # Format responses in a clear way
    formatted_responses = "\n".join([
        f"- {data['question']}: {data['response']}"
        for _, data in responses.items()
    ])
    
    # Create the full prompt
    prompt = process_prompt.format(
        title=form['title'],
        description=form['description'],
        responses=formatted_responses
    )
    
    # Get response from LLM
    try:
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
            if 'agent' in response and 'messages' in response['agent']:
                last_message = response['agent']['messages'][-1]
                if isinstance(last_message, AIMessage) and hasattr(last_message, 'content'):
                    # Add AI message to session state
                    response_text = last_message.content + "\n\nFeel free to ask any follow-up questions or request more specific details about any of these recommendations."
        
        # Add a note about follow-up questions
        
        
        return response_text
    except Exception as e:
        return f"I apologize, but I encountered an error processing your responses. Please try again or rephrase your questions. Error: {str(e)}"

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
# Tool for creating calendar events
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

# Tool for answering questions
class AskQuestionArgs(BaseModel):
    """Arguments for the ask_question tool."""
    question: str = Field(description="The question to ask the AI assistant")

@tool(args_schema=AskQuestionArgs)
def ask_question(question: str):
    """Ask a question to the AI assistant based on the loaded documents."""
    documents = []
    data_folder = "data"
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            documents.extend(PyPDFLoader(pdf_path).load())
    if not documents:
        return "No documents available to answer the question."
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embedding_model)
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt_template = '''
    Context:
    {context}

    Question: {question}
    Answer:
    '''
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    if not context:
        return "I couldn't find any relevant documents to answer your question."
    prompt_input = {
        "context": context,
        "question": question,
        "chat_history": memory.chat_memory.messages
    }
    formatted_prompt = prompt.format(**prompt_input)
    response = llm.invoke(formatted_prompt)
    return response.content if response and hasattr(response, 'content') else "I couldn't generate a response."

# Tool for creating dynamic forms
class CreateFormArgs(BaseModel):
    prompt: str = Field(description="The user's query or prompt that may require additional information.")
    api_key: str = Field(description="The API key for accessing the language model or other services.")
    
def create_dynamic_form(form_config: dict, api_key: str) -> Form:

    try:
        form = generate_survey(
            topic=form_config["title"],  # Use the title as the topic
            num_questions=form_config["num_questions"],
            requirements=form_config["description"],  # Use description as requirements
            api_key=api_key
        )
        return form
    except Exception as e:
        raise Exception(f"Error creating form: {str(e)}")
    
@tool(args_schema=CreateFormArgs)
def create_dynamic_form_tool(prompt: str, api_key: str):
    """
    This function dynamically creates a form based on the user's prompt. It first analyzes the prompt to determine if a form is needed to gather more information. If a form is required, it generates the form with the specified configuration. The form configuration includes the title, description, and the number of questions. The function returns the form as a dictionary if a form is needed, otherwise, it returns a message response.

    Parameters:
    - prompt (str): The user's query or prompt that may require additional information.
    - api_key (str): The API key for accessing the language model or other services.

    Returns:
    - dict: A dictionary containing the form configuration if a form is needed, otherwise, a message response.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    response_data = generate_bot_response(prompt, llm)
    print("CREATE DYNAMIC FORM: ", response_data)
    if response_data["needs_form"] and response_data.get("form_config"):
        form = create_dynamic_form(response_data["form_config"], api_key)
        return form.dict()
    return {"message": response_data["response"]}

def generate_bot_response(prompt: str, llm: ChatOpenAI) -> dict:
    """Generate bot response and determine if a form is needed"""
    
    analysis_prompt = """Analyze the user's query and determine if additional information is needed through a questionnaire.
    
    User Query: {prompt}
    
    Guidelines:
    1. Only request a form if you need specific details to provide a helpful response
    2. Consider creating a form when:
       - User needs personalized recommendations
       - Query requires understanding user preferences
       - Detailed technical information is needed
       - User wants to schedule a meeting or create a calendar event or anything related to scheduling (ask for meeting title, date, start and end times, and participants)

    3. Do NOT create a form when:
       - Query is clear and complete
       - Information can be provided directly
       - User is asking for factual information
    
    If a form is needed, provide:
    - A clear title for the form
    - A description explaining why these questions are needed
    - The number of questions required (keep it minimal)
    
    Response format:
    {{
        "needs_form": boolean,
        "response": "Your initial response to the user",
        "form_config": {{
            "title": "Clear title for the form",
            "description": "Why you need this information",
            "num_questions": number (only what's absolutely necessary)
        }} if needs_form else null
    }}
    
    If the user wants to schedule a meeting or create a calendar event or anything related to scheduling the Response can be set as follows:
    {{
        "needs_form": True,
        "response": "Your initial response to the user",
        "form_config": {{
            "title": "Meeting Scheduling Form Enter the Details",   
            "description": "Please provide the details to schedule your meeting Title, Date, Start Time, Duration, Participants",
            "num_questions": 5
        }} 
    }}
    """
    try:
        input_message = HumanMessage(content=prompt)
        
        # Get conversation state from memory or start new
        state_snapshot = graph.get_state(config=config)
        # print("State snapshot:", state_snapshot)
        
        conversation_state = {"messages": []}
        if state_snapshot and hasattr(state_snapshot, 'values'):
            conversation_state = state_snapshot.values.get("messages", {"messages": []})
        conversation_state['messages'].append(analysis_prompt)
        conversation_state["messages"].append(input_message)
        
        # Run agent
        for response in graph.stream(conversation_state, config=config):
            if 'agent' in response and 'messages' in response['agent']:
                last_message = response['agent']['messages'][-1]
                if isinstance(last_message, AIMessage) and hasattr(last_message, 'content'):
                    # Add AI message to session state
                    response_text = last_message.content
        print("Graph: ",response_text)
        print("Normal LLM: ", response)
        # response = llm.invoke(analysis_prompt.format(prompt=prompt))
        response = response_text
        return json.loads(response)
    except Exception as e:
        raise Exception(f"Error analyzing query: {str(e)}")


# LangGraph Setup
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

model = ChatOpenAI(model="gpt-4o-mini")
tools = [create_calendar_event, ask_question, create_dynamic_form_tool]
model = model.bind_tools(tools)

def call_model(state: AgentState):
    system_prompt = SystemMessage(
        """You are a helpful assistant that can create calendar events, answer questions based on loaded documents, and create forms when needed. 
        For calendar events, parse user requests and create events accordingly. Make sure dates are in ISO 8601 format.

        You have access to three tools:

        1. create_calendar_event: Creates a Google Calendar event with specified details.
           Arguments:
           - `title`: (Required) Name/subject of the event
           - `start_time`: (Required) ISO format datetime (YYYY-MM-DDTHH:MM:SS)
           - `duration_minutes`: (Optional) Length of event in minutes, defaults to 60
           - `attendees`: (Optional) Comma-separated email addresses

        2. ask_question: Asks a question to the AI assistant based on loaded documents. This tool is designed to provide information about any content loaded in the documents, including but not limited to meeting details, project information, and general knowledge. 
           Arguments:
           - `question`: (Required) The question to ask the AI assistant

        3. create_dynamic_form: Creates a form for additional details if required.
           Arguments:
           - `prompt`: (Required) The user's query or prompt that may require additional information.
           - `api_key`: (Required) The API key for accessing the language model or other services.

        Example Prompts:
        1. "Schedule a meeting called 'Team Sync' for tomorrow at 3pm"
        2. "Create a 30-minute event titled 'Quick Check-in' with bob@company.com at 2pm today"
        3. "Set up a 2-hour planning session next Monday at 10am with the team: alice@company.com, charlie@company.com"
        4. "What information do we have about project deadlines?"
        5. "Can you summarize the main points from the latest meeting notes?"
        6. "What are the key findings from the recent market research report?"
        7. "Please provide more details about your request."

        Choose the appropriate tool based on the user's request. If the user asks a question about the content of the documents, use the ask_question tool to provide a response. If the user's request requires additional information, use the create_dynamic_form tool to gather more details."""
    )
    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        state["messages"].insert(0, system_prompt)
    response = model.invoke(state["messages"])
    print("Call model",response)
    return {"messages": [response]}


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

# Streamlit Rendering for Forms
st.title("Enhanced Calendar and Form Assistant")

# Initialize session state for the form
if "current_form" not in st.session_state:
    st.session_state.current_form = None
    
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"

if "messages" not in st.session_state:
    st.session_state.messages = []

config = {"configurable": {"thread_id": st.session_state.session_id}}

# Display chat history
st.markdown("### Chat History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Render current form if available
if st.session_state.current_form:
    form_data = st.session_state.current_form
    with st.form(key="dynamic_form"):
        st.header(form_data["title"])
        if form_data["description"]:
            st.write(form_data["description"])
        st.write("---")
        responses = {}
        for question_data in form_data["questions"]:
            response = render_question(question_data)
            if response:
                responses[question_data['id']] = {"question": question_data["question_text"], "response": response}
        submitted = st.form_submit_button("Submit")
        if submitted:
            llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            result = process_form_responses(responses, form_data, llm)
            print("Process fomr Response: ", result)
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.session_state.current_form = None
            st.rerun()

# Chat input
elif prompt := st.chat_input("How can I help you schedule events or provide suggestions?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(prompt)
        form_analysis = generate_bot_response(prompt, llm)
        print("FORMANALYSIS : ", form_analysis)
        st.session_state.messages.append({"role": "assistant", "content": form_analysis["response"]})
        if form_analysis["needs_form"]:
            form = create_dynamic_form(form_analysis["form_config"], os.getenv("OPENAI_API_KEY"))
            st.session_state.current_form = form.dict()
        st.rerun()

# Custom CSS
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

