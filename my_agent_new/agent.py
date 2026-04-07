from google.adk.agents.llm_agent import Agent


root_agent = Agent(
    model='gemini-2.5-flash',
    name='planner_agent',
    description='Specialist planner agent',
    instruction="""
        Specialist planner agent
    """
)
