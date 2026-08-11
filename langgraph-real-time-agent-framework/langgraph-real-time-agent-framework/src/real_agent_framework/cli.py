import argparse
import json
from datetime import datetime
from uuid import uuid4

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .runtime import AgentRuntime

console = Console()
COLORS = {
    "system": "white",
    "problem_analyzer_service": "cyan",
    "planner_agent": "bright_blue",
    "solver_agent": "green",
    "tool_router_service": "yellow",
    "verifier_agent": "bright_magenta",
    "memory_service": "magenta",
    "output_service": "bright_green",
}


class LiveTrace:
    def __init__(self, compact: bool = False):
        self.compact = compact
        self.in_final_stream = False

    def __call__(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "token":
            console.print(str(event.get("message", "")), end="", markup=False)
            return
        if kind == "final_start":
            self.in_final_stream = True
        elif kind == "final_end" and self.in_final_stream:
            console.print()
            self.in_final_stream = False

        actor = str(event.get("actor", "system"))
        color = COLORS.get(actor, "white")
        timestamp = str(event.get("timestamp", ""))
        try:
            timestamp = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
        except ValueError:
            timestamp = "--:--:--"
        console.print(
            f"[{color}][{timestamp}] {actor.upper():<26}[/{color}] "
            f"[dim]{kind or ''}[/dim]  {event.get('message', '')}"
        )
        if self.compact:
            return
        details = event.get("details") or {}
        if kind == "plan_created":
            for step in details.get("steps", []):
                console.print(f"    [bold]{step['id']}[/bold] {step['description']}")
        elif kind == "step_result" and details.get("output"):
            console.print(Panel(details["output"], border_style=color, padding=(0, 1)))
        elif kind == "tool_start":
            console.print(f"    args: {json.dumps(details.get('arguments', {}), default=str)}")
        elif kind == "tool_result":
            console.print(f"    result: {details.get('output', '')}", soft_wrap=True)
        elif kind == "verification":
            console.print(f"    feedback: {details.get('feedback', '')}")
            for item in details.get("missing_items", []):
                console.print(f"    • {item}")
        elif kind in {"memory_read", "memory_write"}:
            values = details.get("recalled_memories") or details.get("saved") or []
            for item in values:
                console.print(f"    • {item if isinstance(item, str) else item.get('text', item)}")


def add_run_args(parser: argparse.ArgumentParser, prompt: bool = False) -> None:
    if prompt:
        parser.add_argument("prompt")
    parser.add_argument("--user", default="default")
    parser.add_argument("--thread", default="default")
    parser.add_argument("--force-plan", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compact", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-framework",
        description="Durable real-time Planner-Solver-Verifier LangGraph framework",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Run one request")
    add_run_args(ask, prompt=True)
    chat = commands.add_parser("chat", help="Continue a durable conversation")
    add_run_args(chat)

    sessions = commands.add_parser("sessions", help="Manage saved conversations")
    session_commands = sessions.add_subparsers(dest="session_command", required=True)
    session_list = session_commands.add_parser("list")
    session_list.add_argument("--user", default="default")
    session_new = session_commands.add_parser("new")
    session_new.add_argument("--user", default="default")
    session_new.add_argument("--title", default="New conversation")
    session_rename = session_commands.add_parser("rename")
    session_rename.add_argument("thread")
    session_rename.add_argument("title")
    session_rename.add_argument("--user", default="default")
    session_delete = session_commands.add_parser("delete")
    session_delete.add_argument("thread")
    session_delete.add_argument("--user", default="default")

    history = commands.add_parser("history", help="Show the complete stored transcript")
    history.add_argument("--user", default="default")
    history.add_argument("--thread", default="default")
    history.add_argument("--json", action="store_true")

    memory = commands.add_parser("memory", help="Inspect durable user memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    memory_list = memory_commands.add_parser("list")
    memory_list.add_argument("--user", default="default")
    memory_delete = memory_commands.add_parser("delete")
    memory_delete.add_argument("memory_id", type=int)
    memory_delete.add_argument("--user", default="default")

    runs = commands.add_parser("runs", help="List saved runs and result files")
    runs.add_argument("--user", default="default")
    runs.add_argument("--thread")
    trace = commands.add_parser("trace", help="Replay a run's stage events")
    trace.add_argument("run_id")
    commands.add_parser("graph", help="Print the compiled graph as Mermaid")
    return parser


def execute(runtime: AgentRuntime, args, prompt: str) -> None:
    handler = None if args.quiet else LiveTrace(args.compact)
    result = runtime.run(
        prompt,
        user_id=args.user,
        thread_id=args.thread,
        force_plan=args.force_plan,
        on_event=handler,
    )
    console.print(Panel(Markdown(result.answer), title="Final answer", border_style="bright_green"))
    console.print(f"[dim]run_id: {result.run_id}[/dim]")
    console.print(f"[dim]result: {result.result_path}[/dim]")


def main() -> None:
    args = build_parser().parse_args()
    with AgentRuntime() as runtime:
        if args.command == "ask":
            execute(runtime, args, args.prompt)
        elif args.command == "chat":
            console.print(
                f"[bold green]Conversation ready[/bold green] "
                f"user={args.user} thread={args.thread}. Type /quit to exit."
            )
            while True:
                prompt = console.input("[bold]you> [/bold]").strip()
                if prompt.lower() in {"/quit", "/exit"}:
                    break
                if prompt == "/history":
                    for row in runtime.chat.get_messages(args.user, args.thread):
                        console.print(Panel(row["content"], title=row["role"]))
                    continue
                if prompt == "/memory":
                    for row in runtime.long_term_memory.list(args.user):
                        console.print(f"{row['id']}: [{row['category']}] {row['text']}")
                    continue
                if prompt:
                    execute(runtime, args, prompt)
        elif args.command == "sessions" and args.session_command == "list":
            rows = runtime.chat.list_conversations(args.user)
            table = Table("Thread", "Title", "Messages", "Updated")
            for row in rows:
                table.add_row(
                    row["thread_id"], row["title"], str(row["message_count"]), row["updated_at"]
                )
            console.print(table if rows else "No saved conversations.")
        elif args.command == "sessions" and args.session_command == "new":
            thread = str(uuid4())[:8]
            runtime.chat.ensure_conversation(args.user, thread, args.title)
            console.print(f"Created thread: {thread}")
        elif args.command == "sessions" and args.session_command == "rename":
            console.print(
                "Conversation renamed."
                if runtime.chat.rename(args.user, args.thread, args.title)
                else "Conversation not found."
            )
        elif args.command == "sessions" and args.session_command == "delete":
            console.print(
                "Conversation and checkpoint deleted."
                if runtime.delete_conversation(args.user, args.thread)
                else "Conversation not found."
            )
        elif args.command == "history":
            rows = runtime.chat.get_messages(args.user, args.thread)
            if args.json:
                console.print_json(json.dumps(rows, ensure_ascii=False, default=str))
            else:
                for row in rows:
                    console.print(Panel(row["content"], title=f"{row['role']} #{row['id']}"))
            if not rows:
                console.print("No messages found.")
        elif args.command == "memory" and args.memory_command == "list":
            rows = runtime.long_term_memory.list(args.user)
            table = Table("ID", "Category", "Confidence", "Memory", "Source thread")
            for row in rows:
                table.add_row(
                    str(row["id"]),
                    row["category"],
                    f"{row['confidence']:.2f}",
                    row["text"],
                    row["source_thread"],
                )
            console.print(table if rows else "No long-term memories.")
        elif args.command == "memory" and args.memory_command == "delete":
            console.print(
                "Memory deleted."
                if runtime.long_term_memory.delete(args.user, args.memory_id)
                else "Memory not found."
            )
        elif args.command == "runs":
            rows = runtime.audit.list_runs(args.user, args.thread)
            table = Table("Run ID", "Thread", "Status", "Started", "Result")
            for row in rows:
                table.add_row(
                    row["run_id"],
                    row["thread_id"],
                    row["status"],
                    row["started_at"],
                    row["result_path"] or "",
                )
            console.print(table if rows else "No runs found.")
        elif args.command == "trace":
            rows = runtime.audit.get_events(args.run_id)
            tracer = LiveTrace()
            for row in rows:
                tracer(row)
            if not rows:
                console.print("Run ID not found.")
        elif args.command == "graph":
            console.print(runtime.graph_mermaid())


if __name__ == "__main__":
    main()
