import argparse

from .runtime import AgentRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agent with tools and durable memory")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ask", "chat"):
        command = sub.add_parser(name)
        if name == "ask":
            command.add_argument("prompt")
        command.add_argument("--user", default="default")
        command.add_argument("--thread", default="default")
    args = parser.parse_args()

    with AgentRuntime() as runtime:
        if args.command == "ask":
            print(runtime.invoke(args.prompt, args.user, args.thread))
            return
        print("Agent ready. Type /quit to exit.")
        while True:
            prompt = input("you> ").strip()
            if prompt.lower() in {"/quit", "/exit"}:
                break
            if prompt:
                print("agent>", runtime.invoke(prompt, args.user, args.thread))


if __name__ == "__main__":
    main()
