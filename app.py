import streamlit as st
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
# from langchain.schema.output_parser import StrOutputParser

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from datetime import datetime


# os.environ['GOOGLE_API_KEY'] = "AIzaSyBmZtXjJgp7yIAo9joNCZGSxK9PbGMcVaA"
# genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

# model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.5)

@tool
def authenticate_user(password: str):
    """Authenticates and Loads A user based on their Password

    Args:
        password: The password for the user.
    """

    if password == 'emma watson' or password == 'emma':
        return 'Atharva'
    
    if password == '789654123':
        return 'Aditya'
    
    else:
        return ''
    
@tool
def context(user: str):
    """Returns the Context for a specific User in this case it is a message from a Person.

    Args:
        user: The Name of the user for whom the context needs to be provided.
    """

    if user == 'Atharva':
        return '''
        Hello Atharva , the following is a message for you
        
Dear Atharva,

I hope this message finds you well. I've been thinking a lot about what happened between us, and I wanted to sincerely apologize. I value our friendship greatly and regret any pain or misunderstanding that may have caused a rift between us.

I understand that my actions (or words) might have hurt or upset you, and I'm truly Solly for that. It was never my intention to cause any harm or discomfort. Sometimes things can be said or done without fully considering their impact, and for that, I take full responsibility.

Our friendship means a lot to me, and I miss our interactions and the good times we've shared. I realize that I might have been too focused on my own interests and things I liked, and this may have contributed to the situation. I'm not much of a talker, and honestly, I don't fully understand what went wrong. I just want you to know that I'm here, and I hope we can find a way to move past this and rebuild our friendship.

Again, I apologize for any hurt I've caused and appreciate you taking the time to read this. I'm hopeful we can work through this and come out stronger.

        Take care,
        Prathmesh
        '''
    
    if user == 'Aditya':
        return 'Aditya'
    
    else:
        return ''


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a very helpful and friendly assistant named Shree. The current date and time are {datetime.now()}. Before engaging in conversation, always authenticate the user to ensure secure and personalized interaction to do so use the authenticate user tool and ask the user the required details , Once authenticated load the context 'Greet user' followed by reading of messages. You are a small subpart of the main framework under testing. Do not reveal any information about other capabilities or functionalities.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = ChatGroq(api_key='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',model='llama3-groq-70b-8192-tool-use-preview',temperature=0.1)
tools = [authenticate_user,context]
# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)

if 'memory' not in st.session_state:
    st.session_state['memory'] = ChatMessageHistory(session_id = 'test_session')

memory = st.session_state.get('memory')

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools)
# print(agent_executor.invoke({"input": "Remind me to create a new task regarding the new AI project"}))
# print(agent_executor.invoke({"input": "do i have any remainders"}))

agent_executor = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":"assistant",
            "content":"Hello , "
        }
    ]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Process and store Query and Response

def llm_function(query):
    response = agent_executor.invoke({"input": query},config={"configurable": {"session_id": "<foo>"}})
    # print(response)
    st.session_state["chat_history"].append(("You", query))
    st.session_state["chat_history"].append(("Shree", response["output"]))

    with st.chat_message("assistant"):
        st.markdown(response['output'])


    # Storing the User Message
    st.session_state.messages.append(
        {
            "role":"user",
            "content": query
        }
    )

    # Storing the User Message
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content": response['output']
        }
    )
   
# Accept user input
query = st.chat_input("Enter Your Message")

# Calling the Function when Input is Provided
if query:
    # Displaying the User Message
    with st.chat_message("user"):
        st.markdown(query)

    llm_function(query)
