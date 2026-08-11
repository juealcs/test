from .backend import SolverBackend
from .models import Plan, RunResult, StepResult, StepStatus


class ProblemSolver:
    def __init__(self, backend: SolverBackend, *, max_steps: int = 8, max_replans: int = 2):
        if max_steps < 1 or max_replans < 0:
            raise ValueError("Invalid execution bounds")
        self.backend = backend
        self.max_steps = max_steps
        self.max_replans = max_replans

    @staticmethod
    def _validate_plan(plan: Plan, completed_ids: set[str] | None = None) -> None:
        completed_ids = completed_ids or set()
        ids = [step.id for step in plan.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan step IDs must be unique")
        known = set(ids) | completed_ids
        for step in plan.steps:
            unknown = set(step.depends_on) - known
            if unknown:
                raise ValueError(f"Step {step.id} has unknown dependencies: {sorted(unknown)}")

    async def run(self, goal: str) -> RunResult:
        if not goal.strip():
            raise ValueError("Goal must not be empty")

        plan = await self.backend.plan(goal)
        original_plan = plan
        results: list[StepResult] = []
        replans = 0

        while True:
            self._validate_plan(plan, {result.step_id for result in results})
            for step in plan.steps:
                if len(results) >= self.max_steps:
                    break
                completed = {
                    result.step_id
                    for result in results
                    if result.status == StepStatus.COMPLETED
                }
                attempted = {result.step_id for result in results}
                if step.id in attempted or not set(step.depends_on) <= completed:
                    continue
                result = await self.backend.execute(goal, step, results)
                if result.step_id != step.id:
                    raise ValueError(
                        f"Backend returned result for {result.step_id}; expected {step.id}"
                    )
                results.append(result)

            review = await self.backend.review(goal, plan, results)
            should_stop = (
                review.passed
                or review.revised_plan is None
                or replans >= self.max_replans
                or len(results) >= self.max_steps
            )
            if should_stop:
                break
            plan = review.revised_plan
            replans += 1

        answer = await self.backend.synthesize(goal, results, review)
        return RunResult(
            answer=answer,
            plan=original_plan,
            steps=results,
            review=review,
            replan_count=replans,
        )
