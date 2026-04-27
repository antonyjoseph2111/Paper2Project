from __future__ import annotations

from app.models.schemas import AgentTurn, JobRecord


class SharedAgentMemory:
    def append(self, job: JobRecord, stage: str, provider: str, model: str, role: str, content: str) -> None:
        job.agent_memory.append(
            AgentTurn(stage=stage, provider=provider, model=model, role=role, content=content)
        )

    def render(self, job: JobRecord, stage: str, max_turns: int = 6, max_chars_per_turn: int = 900) -> str:
        preferred_stages = {
            "paper_analyst": {"paper_analyst"},
            "planner": {"paper_analyst", "planner"},
            "decision": {"planner", "decision"},
            "code_generator": {"planner", "decision", "code_generator"},
            "notebook_builder": {"decision", "code_generator", "notebook_builder"},
        }
        allowed = preferred_stages.get(stage, {stage})
        relevant = [turn for turn in job.agent_memory if turn.stage in allowed][-max_turns:]
        if not relevant:
            return "No prior shared memory."
        lines = []
        for turn in relevant:
            content = turn.content
            if len(content) > max_chars_per_turn:
                content = content[:max_chars_per_turn] + "\n...[truncated for memory budget]..."
            lines.append(
                f"[stage={turn.stage} provider={turn.provider} model={turn.model} role={turn.role}] {content}"
            )
        return "\n".join(lines)
