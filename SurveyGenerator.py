#SurverRenderer
import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Optional, Literal, Union
from datetime import datetime
import json
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

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
    - User wants to schedule a meeting (ask for meeting title, date, start and end times, and participants)
    
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

def render_question(question: Question):
    """Render a single question based on its type"""
    try:
        # Display question text and description
        st.markdown(f"**{question.question_text}**")
        if question.description:
            st.markdown(f"*{question.description}*")
            
        st.markdown("") # Add some spacing
        
        # Handle different question types based on type attribute
        if question.type == "text":
            return st.text_input(
                label=question.question_text,
                key=f"text_{question.id}",
                max_chars=question.max_length,
                placeholder=question.placeholder or "Type your answer here...",
                label_visibility="collapsed"
            )
            
        elif question.type == "multiple_choice":
            if question.allow_multiple:
                return st.multiselect(
                    label=question.question_text,
                    options=question.options,
                    key=f"multi_{question.id}",
                    label_visibility="collapsed"
                )
            else:
                return st.selectbox(
                    label=question.question_text,
                    options=question.options,
                    key=f"single_{question.id}",
                    label_visibility="collapsed"
                )
        
        elif question.type == "slider":
            # Format string for currency if unit is provided
            format_str = "${:,}" if question.unit == "$" else None
            
            return st.slider(
                label=question.question_text,
                min_value=question.min_value,
                max_value=question.max_value,
                step=question.step,
                key=f"slider_{question.id}",
                label_visibility="collapsed",
                format=format_str
            )
            
        elif question.type == "rgb":
            return st.color_picker(
                label=question.question_text,
                value=question.default_color if question.default_color else "#FFFFFF",
                key=f"rgb_{question.id}",
                label_visibility="collapsed"
            )
        elif question.type == "datetime":
            # Use Streamlit's date and time input widgets
            date = st.date_input(
                label=f"{question.question_text} - Date",
                key=f"date_{question.id}",
                label_visibility="collapsed"
            )
            time = st.time_input(
                label=f"{question.question_text} - Time",
                key=f"time_{question.id}",
                label_visibility="collapsed"
            )
            # Combine date and time into a single datetime object
            return datetime.combine(date, time).isoformat()


        st.markdown("---")  # Add separator between questions
        
    except Exception as e:
        st.error(f"Error rendering question: {str(e)}")
        return None

def process_form_responses(responses: dict, form: Form, llm: ChatOpenAI) -> str:
    """Process form responses using LLM"""
    process_prompt = """You are a helpful AI assistant analyzing user responses to provide personalized recommendations and insights.

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

    Guidelines for your response:
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
        title=form.title,
        description=form.description,
        responses=formatted_responses
    )
    
    # Get response from LLM
    try:
        response = llm.invoke(prompt)
        
        # Add a note about follow-up questions
        response_text = response.content + "\n\nFeel free to ask any follow-up questions or request more specific details about any of these recommendations."
        
        return response_text
    except Exception as e:
        return f"I apologize, but I encountered an error processing your responses. Please try again or rephrase your questions. Error: {str(e)}"

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
            "description": "Please provide the details to schedule your meeting Title, Date, Start Time, Participants",
            "num_questions": 4
        }} 
    }}
    
    """
    
    try:
        response = llm.invoke(analysis_prompt.format(prompt=prompt))
        return json.loads(response.content)
    except Exception as e:
        raise Exception(f"Error analyzing query: {str(e)}")

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

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_form" not in st.session_state:
        st.session_state.current_form = None

    st.title("Interactive AI Assistant")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Display current form if it exists
    if st.session_state.current_form:
        form = st.session_state.current_form
        
        with st.form(key="survey_form"):
            st.title(form.title)
            if form.description:
                st.write(form.description)
            
            st.write("---")
            
            responses = {}
            for question in form.questions:
                response = render_question(question)
                if response is not None:
                    responses[question.id] = {
                        'question': question.question_text,
                        'response': response
                    }

            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

            if submitted:
                if len(responses) == len(form.questions):
                    llm = ChatOpenAI(temperature=0.7, api_key=api_key)
                    result = process_form_responses(responses, form, llm)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    st.session_state.current_form = None
                    st.rerun()
                else:
                    st.error("Please answer all required questions before submitting.")
    
    # Show chat input if no form is displayed
    else:
        if prompt := st.chat_input("How can I help you?"):
            if not api_key:
                st.warning("Please enter your OpenAI API key first.")
                return

            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                llm = ChatOpenAI(temperature=0.7, api_key=api_key)
                
                # First, analyze if form is needed and get form requirements
                response_data = generate_bot_response(prompt, llm)
                
                # Add bot's initial response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_data["response"]})
                
                if response_data["needs_form"] and response_data.get("form_config"):
                    form = create_dynamic_form(response_data["form_config"], api_key)
                    st.session_state.current_form = form
                
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()