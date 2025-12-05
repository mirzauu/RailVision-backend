from typing import AsyncGenerator
from crewai import Agent as CrewAgent, Crew, Task

from .base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig


class CrewAIChatAgent(ChatAgent):
    def __init__(self, config: AgentConfig):
        self.tasks = config.tasks
        self.agent = CrewAgent(role=config.role, goal=config.goal, backstory=config.backstory)

    def _create_task(self, task_config: TaskConfig, ctx: ChatContext) -> Task:
        description = (
            f"User Query: {ctx.query}\nProject ID: {ctx.project_id}\n{ctx.additional_context}"
        )
        return Task(description=f"{task_config.description}\n\n{description}")

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        task = self._create_task(self.tasks[0], ctx)
        crew = Crew(agents=[self.agent], tasks=[task])
        result = crew.kickoff()
        return ChatAgentResponse(response=str(result), tool_calls=[], citations=[])

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        resp = await self.run(ctx)
        yield resp

