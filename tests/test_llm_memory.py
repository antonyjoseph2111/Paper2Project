from app.models.schemas import AgentTurn, JobRecord, JobStatus
from app.services.llm_memory import SharedAgentMemory


def test_memory_render_filters_and_truncates() -> None:
    job = JobRecord(job_id="job-memory", filename="paper.pdf", status=JobStatus.QUEUED)
    memory = SharedAgentMemory()
    memory.append(job, "paper_analyst", "provider-a", "model-a", "assistant", "A" * 1200)
    memory.append(job, "planner", "provider-b", "model-b", "assistant", "planner output")
    memory.append(job, "notebook_builder", "provider-c", "model-c", "assistant", "notebook output")

    rendered_for_decision = memory.render(job, "decision")
    assert "planner output" in rendered_for_decision
    assert "notebook output" not in rendered_for_decision
    assert "[truncated for memory budget]" not in rendered_for_decision

    rendered_for_planner = memory.render(job, "planner")
    assert "[truncated for memory budget]" in rendered_for_planner
