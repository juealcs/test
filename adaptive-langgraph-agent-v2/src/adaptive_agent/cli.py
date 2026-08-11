import argparse

from dotenv import load_dotenv

from .context import AgentContext
from .graph import build_agent


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the adaptive LangGraph agent")
    parser.add_argument("task")
    parser.add_argument("--user", default="default-user")
    parser.add_argument("--thread", default="default-thread")
    parser.add_argument("--mode", choices=["fast", "deliberate"])
    args = parser.parse_args()
    with build_agent() as agent:
        result = agent.invoke(args.task, AgentContext(args.user, args.mode), args.thread)
        print(result["final_answer"])


if __name__ == "__main__":
    main()

