from __future__ import annotations

import json
import os
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()

SYSTEM_PROMPT = """You are a concise research assistant.
Use Tavily search when you need current or factual web information.
Answer the user's question directly and include source URLs when available.
"""


def require_env(name: str, instructions: str) -> None:
    if os.getenv(name):
        return
    console.print(f"[bold red]Missing {name}[/bold red]")
    console.print(instructions)
    raise typer.Exit(code=1)


def message_text(message: Any) -> str:
    """Extract streamed text from a LangChain message or message chunk."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def flush() -> None:
    console.file.flush()


def truncate(value: Any, limit: int = 900) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def print_tool_call(name: str, args: Any) -> None:
    console.print()
    console.print(
            Panel(
                Text(truncate(args, limit=700)),
                title=f"Tool call: {name}",
                border_style="yellow",
                )
            )


def format_tool_result(content: Any) -> str:
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return truncate(content)

    if not isinstance(payload, dict) or "results" not in payload:
        return truncate(payload)

    lines = [f"Query: {payload.get('query', '')}", ""]
    for index, result in enumerate(payload.get("results", [])[:5], start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = " ".join(result.get("content", "").split())
        lines.append(f"{index}. {title}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {truncate(snippet, limit=220)}")
        lines.append("")
    return "\n".join(lines).strip()


def print_tool_result(message: Any) -> None:
    name = getattr(message, "name", None) or "tool"
    content = format_tool_result(getattr(message, "content", ""))
    console.print()
    console.print(
            Panel(
                Text(content),
                title=f"Tool result: {name}",
                border_style="yellow",
                )
            )


@app.command()
def main(name: Annotated[str, typer.Argument(help="Person's first and last name")], 
         description: Annotated[str, typer.Argument(help="Pasted job description,title, and company text, if available")]) -> None:

    print(f"Name: {name}")
    print(f"Title: {description}")


if __name__ == "__main__":
    app()
