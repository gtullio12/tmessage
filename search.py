from __future__ import annotations

import json
import os
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from rich.console import Console

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()
chat_model = ChatNebius(model="moonshotai/Kimi-K2.6")

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

def extract_info_from_description(description: Annotated[str, typer.Argument(help="Pasted job description,title, and company text, if available")]) -> dict:

    EXTRACTION_SYSTEM_PROMT = """
    I'm giving you a copy-pasted job description section in LinkedIn,
    please parse the entire string and extract the company name, job title, and job description
    if it's listed. With the job description split it into 2 lists, one is key_facts which should
    be a list of job responsibilities (close to paraphrasing) — each key_fact should describe a
    concrete responsibility or activity, not a characterization of how the person works — and
    persona_inference which is a list of strings of how this person communicates/operates
    (hands-on, technical, builds from scratch, decision maker), don't assume for this one,
    if you can't find something then return an empty list. Respond with ONLY valid JSON
    """

    response = chat_model.invoke([
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMT},
        {"role": "user", "content": description}
        ])

    try:
        response = json.loads(message_text(response))

        # USER has to enter company name, otherwise EXIT
        if (response['company_name'] is None):
            print("company: None. Please input a company")
            raise typer.Exit(code=1)

        return response 
    except json.JSONDecodeError:
        print("Failed to parse extraction response as JSON:")
        print(response.content)
        raise typer.Exit(code=1)


@app.command()
def main(name: Annotated[str, typer.Argument(help="Person's first and last name")], 
         description: Annotated[str, typer.Argument(help="Pasted job description,title, and company text, if available")]) -> None:
    # Now we need to extract the company, title, and optionally the job description

    parsed_description = extract_info_from_description(description)

    print(parsed_description)


if __name__ == "__main__":
    app()
