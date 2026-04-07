import logging
from typing import AsyncGenerator, Dict, Any
import os

# Import ADK components
from google.adk.agents import LlmAgent, BaseAgent, Agent, SequentialAgent, ParallelAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools import ToolContext, google_search
from google.adk.tools.agent_tool import AgentTool
from typing_extensions import override
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define the Session-based Memory Tools ---
def save_user_preferences(tool_context: ToolContext, new_preferences: Dict[str, Any]) -> str:
    """Saves or updates user preferences in the persistent session storage."""
    current_preferences = tool_context.state.get('user_preferences') or {}
    current_preferences.update(new_preferences)
    tool_context.state['user_preferences'] = current_preferences
    return f"Preferences updated successfully: {new_preferences}"

def recall_user_preferences(tool_context: ToolContext) -> Dict[str, Any]:
    """Recalls all saved preferences for the current user from the session."""
    preferences = tool_context.state.get('user_preferences')
    if preferences:
        return preferences
    else:
        return {"message": "No preferences found for this user."}


# --- 2. Define the General Planner Specialist Agents ---
planner_tool_agent = LlmAgent(
    name="PlannerToolAgent",
    model="gemini-2.5-flash",
    description="A specialist that finds activities and restaurants based on a user's request and preferences.",
    instruction="""
    You are a planning assistant. Based on the user's request, weather conditions, and provided preferences, find one activity and one restaurant.
    Output the plan as a simple JSON object.
    """,
    tools=[google_search]
)

weather_tool_agent = LlmAgent(
    name="WeatherToolAgent",
    model="gemini-2.5-flash",
    description="A specialist that checks the current weather and forecast for a given location to advise on activities.",
    instruction="""
    You are a meteorological assistant. Find the current weather and a brief forecast for the requested destination.
    Provide actionable advice for a trip planner.
    """,
    tools=[google_search]
)

budget_tool_agent = LlmAgent(
    name="BudgetToolAgent",
    model="gemini-2.5-flash",
    description="A specialist that estimates the rough cost of a planned itinerary.",
    instruction="""
    You are a financial planning assistant. Given a set of planned activities, restaurants, and a location, estimate a rough budget range.
    Break down the costs into categories: Food, Activities, and Miscellaneous.
    """
)


# --- 3. Define the Sequential Dining & Navigation Workflow ---
foodie_agent = Agent(
    name="foodie_agent",
    model="gemini-2.5-flash",
    tools=[google_search],
    instruction="""You are an expert food critic. Your goal is to find the best restaurant based on a user's request.
    When you recommend a place, you must output *only* the name of the establishment and nothing else.
    """,
    output_key="destination"
)

transportation_agent = Agent(
    name="transportation_agent",
    model="gemini-2.5-flash",
    tools=[google_search],
    instruction="""You are a navigation assistant. Given a destination, provide clear directions.
    The user wants to go to: {destination}.
    Analyze the user's full original query to find their starting point.
    Then, provide clear directions from that starting point to {destination}.
    """,
)

find_and_navigate_agent = SequentialAgent(
    name="find_and_navigate_agent",
    sub_agents=[foodie_agent, transportation_agent],
    description="A specialized workflow that first finds a highly-rated restaurant, then provides directions to it from the user's starting point."
)


# --- 4. Define the Parallel Multi-Category Research Workflow ---
museum_finder_agent = Agent(
    name="museum_finder_agent", model="gemini-2.5-flash", tools=[google_search],
    instruction="You are a museum expert. Find the best museum based on the user's query. Output only the museum's name.",
    output_key="museum_result"
)

concert_finder_agent = Agent(
    name="concert_finder_agent", model="gemini-2.5-flash", tools=[google_search],
    instruction="You are an events guide. Find a concert based on the user's query. Output only the concert name and artist.",
    output_key="concert_result"
)

restaurant_finder_agent = Agent(
    name="restaurant_finder_agent",
    model="gemini-2.5-flash",
    tools=[google_search],
    instruction="""You are an expert food critic. Your goal is to find the best restaurant based on a user's request.
    When you recommend a place, you must output *only* the name of the establishment.
    """,
    output_key="restaurant_result" 
)

parallel_research_agent = ParallelAgent(
    name="parallel_research_agent",
    sub_agents=[museum_finder_agent, concert_finder_agent, restaurant_finder_agent]
)

synthesis_agent = Agent(
    name="synthesis_agent", model="gemini-2.5-flash",
    instruction="""You are a helpful assistant. Combine the following research results into a clear, bulleted list for the user.
    - Museum: {museum_result}
    - Concert: {concert_result}
    - Restaurant: {restaurant_result}
    """
)

parallel_planner_agent = SequentialAgent(
    name="parallel_planner_agent",
    sub_agents=[parallel_research_agent, synthesis_agent],
    description="A workflow that finds a museum, concert, and restaurant in parallel, then summarizes the results."
)


# --- 5. Define the Supreme Coordinator Agent ---
root_agent = LlmAgent(
    name="MemoryCoordinatorAgent",
    model="gemini-2.5-flash",
    instruction="""
    You are a highly intelligent, personalized trip planner orchestrating a team of specialist agents and complex workflows.
    
    1. RECALL: Always call `recall_user_preferences` first to check for dietary restrictions, favorite cuisines, or general likes.
    2. ROUTE THE REQUEST:
       - If the user asks for a complex itinerary involving multiple distinct categories at once (like a museum, a concert, AND a restaurant), hand off the task by calling the `parallel_planner_agent`.
       - If the user specifically asks for the *best* food/restaurant AND directions from a starting point, hand off the task by calling the `find_and_navigate_agent`.
       - If the user asks for a general day trip (activities + food), use the `WeatherToolAgent`, then the `PlannerToolAgent`, and finally the `BudgetToolAgent`.
    3. PRESENT & LEARN: Present the final results clearly. Ask if they want to save any new preferences.
    4. SAVE: If the user provides a new preference, call `save_user_preferences`.
    """,
    tools=[
        recall_user_preferences,
        save_user_preferences,
        AgentTool(agent=planner_tool_agent),
        AgentTool(agent=weather_tool_agent),
        AgentTool(agent=budget_tool_agent),
        AgentTool(agent=find_and_navigate_agent),
        AgentTool(agent=parallel_planner_agent) # The new Parallel workflow is now a tool!
    ]
)

print("🤖 Ultimate Agent Orchestrator (Memory, Specialists, Sequential & Parallel Workflows) is ready.")
