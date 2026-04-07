import logging

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from agent import root_agent

logger = logging.getLogger(__name__)


class AnimeAgentExecutor(AgentExecutor):
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=root_agent,
            app_name="anime_agent",
            session_service=self.session_service,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        session_id = context.context_id or "default"

        logger.info(f"[AnimeAgent] execute | session_id={session_id} | input={user_input!r}")

        session = await self.session_service.get_session(
            app_name="anime_agent",
            user_id="a2a_user",
            session_id=session_id,
        )

        if session is None:
            logger.info(f"[AnimeAgent] creating new session | session_id={session_id}")
            session = await self.session_service.create_session(
                app_name="anime_agent",
                user_id="a2a_user",
                session_id=session_id,
            )

        adk_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_input)],
        )

        chunk_count = 0
        async for event in self.runner.run_async(
            user_id="a2a_user",
            session_id=session.id,
            new_message=adk_message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        chunk_count += 1
                        logger.info(f"[AnimeAgent] chunk #{chunk_count}: {part.text!r}")
                        await event_queue.enqueue_event(
                            new_agent_text_message(part.text)
                        )

        logger.info(f"[AnimeAgent] done | total chunks={chunk_count}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("[AnimeAgent] cancel called")