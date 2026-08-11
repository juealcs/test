import argparse
import json
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .runtime import MultiAgentRuntime

console = Console()

AGENT_COLORS = {
    "system": "white",
    "memory_manager": "magenta",
    "router": "cyan",
    "planner": "bright_blue",
    "supervisor": "yellow",
    "researcher": "green",
    "analyst": "bright_cyan",
    "writer": "blue",
    "reviewer": "bright_magenta",
    "finalizer": "bright_green",
    "fast_agent": "green",
}


def show_event(event: dict, details: bool = True) -> None:
    agent = str(event.get("agent", "system"))
    color = AGENT_COLORS.get(agent, "white")
    timestamp = str(event.get("timestamp", ""))
    try:
        timestamp = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
    except ValueError:
        pass
    console.print(
        f"[{color}][{timestamp}] {agent.upper():<15}[/{color}] "
        f"[dim]{event.get('kind', '')}[/dim]  {event.get('message', '')}"
    )
    if not details:
        return
    payload = event.get("details") or {}
    kind = event.get("kind")
    if kind == "plan":
        for task in payload.get("tasks", []):
            console.print(
                f"    [bold]{task.get('id')}[/bold] → {task.get('agent')}: "
                f"{task.get('instruction')}"
            )
    elif kind == "tool":
        console.print(f"    args: {json.dumps(payload.get('arguments', {}), default=str)}")
        console.print(f"    result: {payload.get('output_preview', '')}", soft_wrap=True)
    elif kind in {"agent_output", "revision"}:
        preview = payload.get("output_preview") or payload.get("draft_preview")
        if preview:
            console.print(Panel(str(preview), border_style=color, padding=(0, 1)))
    elif kind in {"memory_read", "memory_write"}:
        values = payload.get("memories") or payload.get("saved") or []
        for value in values:
            console.print(f"    • {value}")


def show_answer(answer: str, run_id: str) -> None:
    console.print()
    console.print(Panel(Markdown(answer), title="Final answer", border_style="bright_green"))
    console.print(f"[dim]run_id: {run_id}[/dim]")


def add_run_arguments(parser: argparse.ArgumentParser, include_prompt: bool = False) -> None:
    if include_prompt:
        parser.add_argument("prompt")
    parser.add_argument("--user", default="default", help="Long-term memory namespace")
    parser.add_argument("--thread", default="default", help="Short-term chat-history thread")
    parser.add_argument("--force-plan", action="store_true", help="Use all agents for any query")
    parser.add_argument("--quiet", action="store_true", help="Hide per-agent trace output")
    parser.add_argument(
        "--compact", action="store_true", help="Show stages without detailed previews"
    )


def run_one(runtime: MultiAgentRuntime, args, prompt: str) -> None:
    handler = None
    if not args.quiet:
        handler = lambda event: show_event(event, details=not args.compact)
    result = runtime.run(
        prompt,
        user_id=args.user,
        thread_id=args.thread,
        force_plan=args.force_plan,
        on_event=handler,
    )
    show_answer(result.answer, result.run_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-agents", description="Observable LangGraph multi-agent research framework"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Run one query")
    add_run_arguments(ask, include_prompt=True)
    chat = commands.add_parser("chat", help="Start an interactive session")
    add_run_arguments(chat)

    memory = commands.add_parser("memory", help="Inspect or delete long-term memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    memory_list = memory_commands.add_parser("list")
    memory_list.add_argument("--user", default="default")
    memory_delete = memory_commands.add_parser("delete")
    memory_delete.add_argument("memory_id", type=int)
    memory_delete.add_argument("--user", default="default")

    history = commands.add_parser("history", help="Show checkpointed chat history")
    history.add_argument("--user", default="default")
    history.add_argument("--thread", default="default")

    trace = commands.add_parser("trace", help="Replay a run from the audit log")
    trace.add_argument("run_id")
    commands.add_parser("graph", help="Print the compiled graph as Mermaid")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with MultiAgentRuntime() as runtime:
        if args.command == "ask":
            run_one(runtime, args, args.prompt)
        elif args.command == "chat":
            console.print("[bold green]Multi-agent system ready.[/bold green] Type /quit to exit.")
            while True:
                prompt = console.input("[bold]you> [/bold]").strip()
                if prompt.lower() in {"/quit", "/exit"}:
                    break
                if prompt:
                    run_one(runtime, args, prompt)
        elif args.command == "memory" and args.memory_command == "list":
            rows = runtime.memory.list(args.user)
            table = Table("ID", "Category", "Memory", "Updated")
            for row in rows:
                table.add_row(str(row["id"]), row["category"], row["text"], row["updated_at"])
            console.print(table if rows else "No long-term memories found.")
        elif args.command == "memory" and args.memory_command == "delete":
            deleted = runtime.memory.delete(args.user, args.memory_id)
            console.print("Memory deleted." if deleted else "Memory was not found.")
        elif args.command == "history":
            rows = runtime.history(args.user, args.thread)
            for row in rows:
                console.print(Panel(str(row["content"]), title=row["type"]))
            if not rows:
                console.print("No checkpointed history found.")
        elif args.command == "trace":
            rows = runtime.audit.get_run(args.run_id)
            for row in rows:
                show_event(row)
            if not rows:
                console.print("Run ID was not found.")
        elif args.command == "graph":
            console.print(runtime.graph_mermaid())


if __name__ == "__main__":
    main()
