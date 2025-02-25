from turtle import st
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display , Image
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from utils._admin_util import create_rag

class State(TypedDict):
  query: str
  category: str
  sentiment: str
  response: str
  
def check_api_key():
    load_dotenv()
    """Verify that the API key is set and valid"""
    api_key = os.getenv("OPENAI_API_KEY")
    print("api_key", api_key)
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return api_key
   
api_key = check_api_key()

llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            temperature=0.7
        )

def rag(state: State)->State:
    rag_chain = create_rag()
     # Extract just the query string from the state
    query = state["query"]
    print("query", query)
    response = rag_chain.invoke(query)  # Pass the string directly, not a dict
    print("response", response)
    return {"response": response}

def categorize(state: State) -> State:
  "HR, IT, Transportation"
  prompt = ChatPromptTemplate.from_template(
      "Categorize the following  query into one of these categories: "
      "HR, IT, Transportation, Other. Query: {query}"
  )
  chain = prompt | llm
  category = chain.invoke({"query": state["query"]}).content
  return {"category": category}

def analyze_sentiment(state: State) -> State:
  prompt = ChatPromptTemplate.from_template(
      "Analyze the sentiment of the following customer query"
      "Response with either 'Position', 'Neutral' , or 'Negative'. Query: {query}"
  )
  chain = prompt | llm
  sentiment = chain.invoke({"query": state["query"]}).content
  return {"sentiment": sentiment}


def handle_hr(state: State)->State:
  prompt = ChatPromptTemplate.from_template(
      "Provide a HR support response to the following query : {query}"
  )
  chain = prompt | llm
  response = chain.invoke({"query": state["query"]}).content
  return {"response": response}

def handle_it(state: State)->State:
  prompt = ChatPromptTemplate.from_template(
      "Provide a IT support response to the following query : {query}"
  )
  chain = prompt | llm
  response = chain.invoke({"query": state["query"]}).content
  return {"response": response}

def handle_transportation(state: State)->State:
  prompt = ChatPromptTemplate.from_template(
      "Provide a transportation support response to the following query : {query}"
  )
  chain = prompt | llm
  response = chain.invoke({"query": state["query"]}).content
  return {"response": response}

def handle_general(state: State)->State:
  prompt = ChatPromptTemplate.from_template(
      "Provide a general support response to the following query : {query}"
  )
  chain = prompt | llm
  response = chain.invoke({"query": state["query"]}).content
  return {"response": response}

def escalate(state: State)->State:
  return {"response": "This query has been escalate to a human agent due to its negative sentiment"}

def route_query(state: State)->State:
  if state["sentiment"] == "Negative":
    return "escalate"
  elif state["category"] == "HR":
    return "handle_hr"
  elif state["category"] == "IT":
    return "handle_it"
  elif state["category"] == "Transportation":
    return "handle_transportation"
  else:
    return "handle_general"

def rout_to_agent(state: State)->State:
    if "i don't know" in state["response"].lower():
        print(state["response"])
        print("return analyze_sentiment")
        return "analyze_sentiment"
    else:
        return "END"


def run_customer_support(query: str)->Dict[str, str]:
    workflow = StateGraph(State)
    workflow.add_node("categorize", categorize)
    workflow.add_node("rag", rag) 
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("handle_hr", handle_hr)
    workflow.add_node("handle_it", handle_it)
    workflow.add_node("handle_transportation", handle_transportation)
    workflow.add_node("escalate", escalate)
    
    workflow.add_edge("categorize", "rag")
    workflow.add_conditional_edges("rag", rout_to_agent, {"analyze_sentiment": "analyze_sentiment", "END": END})  
    workflow.add_conditional_edges(
        "analyze_sentiment",
        route_query,
        {
            "handle_hr" : "handle_hr",    
            "handle_it" :  "handle_it",
            "handle_transportation" : "handle_transportation",
            "escalate": "escalate"
        }
    )

    workflow.add_edge("handle_hr", END)
    workflow.add_edge("handle_it", END)
    workflow.add_edge("handle_transportation", END)
    workflow.add_edge("escalate", END)

    workflow.set_entry_point("categorize")

    app  = workflow.compile()
    results = app.invoke({"query": query})
    return {
        "category": results.get('category', ''),  # Returns empty string if key missing
        "sentiment": results.get('sentiment', ''), 
        "response": results['response']
    }