from __future__ import annotations

import json
import os
from typing import Annotated, Any
from datetime import datetime, timedelta

from langchain_tavily import TavilySearch
import typer
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

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

    # Now we need to extract the company, title, and optionally the job description
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


def search_relevant_resources(relevant_title: str, company_name: str) -> Any:
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)

    # Search based off of title 
    title_specific_search = TavilySearch(
            search_depth="basic",
            max_results=5,
            include_answer=False,
            # Only search for the last 365 days
            start_date= one_year_ago.strftime("%Y-%m-%d"),
            end_date= today.strftime("%Y-%m-%d"))

    result = title_specific_search.invoke(f"{company_name} {relevant_title}")
    return result


# Takes in a unstructured title -> And extracts out only the relevant bits. 
def title_extraction(title: str) -> str:
    TITLE_RELEVANCE_PROMPT = """
    Given this job title, identify the specific area of AI/technology focus, if any,
    that would be most relevant to a company that builds a web search API for AI agents.
    If the title has no clear AI/tech focus, say so rather than guessing.
    Respond with a short phrase (2-5 words) or "none" if nothing is clearly relevant.
    """

    response = chat_model.invoke([
        {"role": "system", "content": TITLE_RELEVANCE_PROMPT},
        {"role": "user", "content": title}
        ])
    return message_text(response)

@app.command()
def main(name: Annotated[str, typer.Argument(help="Person's first and last name")], 
         description: Annotated[str, typer.Argument(help="Pasted job description,title, and company text, if available")]) -> None:


    parsed_description = extract_info_from_description(description)

    console.print(
    Columns([
        Panel.fit(name, title="Person", border_style="cyan"),
        Panel.fit(parsed_description["job_title"], title="Title", border_style="cyan"),
    ])
    )
    console.rule("[bold blue]Generating Message")

    relevant_title = title_extraction(parsed_description['job_title'])

    relevant_resources = search_relevant_resources(relevant_title, parsed_description['company_name'])

    print(json.dumps(relevant_resources, indent=2))



if __name__ == "__main__":
    app()
