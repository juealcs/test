import argparse
import asyncio
import os

from dotenv import load_dotenv

from .backend import VllmAgentBackend
from .orchestrator import ProblemSolver


async def _run(goal: str) -> None:
    load_dotenv()
    required = ["VLLM_BASE_URL", "VLLM_API_KEY", "VLLM_MODEL"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")

    backend = VllmAgentBackend(
        base_url=os.environ["VLLM_BASE_URL"],
        api_key=os.environ["VLLM_API_KEY"],
        model=os.environ["VLLM_MODEL"],
    )
    solver = ProblemSolver(
        backend,
        max_steps=int(os.getenv("MAX_STEPS", "8")),
        max_replans=int(os.getenv("MAX_REPLANS", "2")),
    )
    result = await solver.run(goal)
    print(result.answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan, execute, verify, and answer")
    parser.add_argument("goal", help="Problem or outcome to solve")
    args = parser.parse_args()
    asyncio.run(_run(args.goal))


if __name__ == "__main__":
    main()
