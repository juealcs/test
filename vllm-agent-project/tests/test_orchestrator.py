from agentic_solver.models import Plan, PlanStep, Review, StepResult, StepStatus
from agentic_solver.orchestrator import ProblemSolver


class FakeBackend:
    async def plan(self, goal):
        return Plan(
            goal=goal,
            steps=[
                PlanStep(id="research", objective="Find facts", success_criteria=["Facts found"]),
                PlanStep(id="answer", objective="Draft answer", success_criteria=["Draft exists"], depends_on=["research"]),
            ],
        )

    async def execute(self, goal, step, prior):
        return StepResult(
            step_id=step.id,
            status=StepStatus.COMPLETED,
            output=f"done {step.id}",
            evidence=["test evidence"],
        )

    async def review(self, goal, plan, results):
        return Review(passed=True, summary="All criteria met")

    async def synthesize(self, goal, results, review):
        return "verified answer"


async def test_executes_dependencies_in_order():
    result = await ProblemSolver(FakeBackend()).run("Solve something")
    assert [item.step_id for item in result.steps] == ["research", "answer"]
    assert result.answer == "verified answer"
