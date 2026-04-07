import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from google.adk.cli.fast_api import get_fast_api_app

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Modify sys.path to allow imports from my_agent_new to work seamlessly
sys.path.append(os.path.join(AGENT_DIR, "my_agent_new"))
from executor import AnimeAgentExecutor

# Example session service URI (e.g., SQLite)
SESSION_SERVICE_URI = "sqlite+aiosqlite:///./sessions.db"
# Example allowed origins for CORS
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]
# Set web=True if you intend to serve a web interface, False otherwise
SERVE_WEB_INTERFACE = True

# Call the function to get the FastAPI app instance
app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=SESSION_SERVICE_URI,
    allow_origins=ALLOWED_ORIGINS,
    web=SERVE_WEB_INTERFACE,
)

app.router.redirect_slashes = False
# === A2A Agent Configuration ===

AGENT_URL = os.getenv(
    "AGENT_PUBLIC_URL",
    "https://capital-agent-service-git-475756125529.us-central1.run.app/anime/",
)

agent_card = AgentCard(
    name="Anime Finder Agent",
    description="An expert otaku assistant for discovering anime by genre, mood, or similar shows.",
    url=AGENT_URL,
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="find_anime",
            name="Find anime",
            description="Recommends anime based on genre, mood, or shows the user already likes.",
            inputModes=["text"],
            outputModes=["text"],
            tags=["anime", "recommendation"],
        )
    ],
)

a2a_app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=DefaultRequestHandler(
        agent_executor=AnimeAgentExecutor(),
        task_store=None,  # uses in-memory task store by default
    ),
)

# Mount the A2A app onto our main FastAPI app at the /anime path
app.mount("/anime", a2a_app.build())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))