from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='An expert otaku assistant for discovering anime.',
    instruction="""
        You are an Anime Show Finder. Help users find anime based on 
        genre, mood, or similar shows they enjoy. 
        Always provide the Title, a brief synopsis, and where to watch.
    """,
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url="https://aeo-mcp-server.amdal-dev.workers.dev/mcp",
            ),
        )
    ],
)