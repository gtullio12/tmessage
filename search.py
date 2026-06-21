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
from rich.text import Text
from rich.columns import Columns
from rich.pretty import pprint

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()

# Open example messages and save results locally
with open('example_messages.txt') as f:
    templates = [msg.strip() for msg in f.read().split("---") if msg.strip()]
example_templates = "\n\n---\n\n".join(templates)

def print_search_results(results: dict) -> None:
    for r in results['results']:
        content = f"{r['url']}\n\n{r['content']}"
        console.print(
            Panel(
                Text(content),
                title=f"[bold yellow]{r['title']}[/bold yellow]",
                border_style="yellow",
            )
        )

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
    console.print("\n[bold yellow]Extracting person info...[/bold yellow]")

    text_extraction_model = ChatNebius(model="Qwen/Qwen3-30B-A3B-Instruct-2507")

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

    response = text_extraction_model.invoke([
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMT},
        {"role": "user", "content": description}
        ])

    try:
        response = json.loads(message_text(response))

        pprint(response)

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
    console.print("\n[bold yellow]Extracting relevant title bits...[/bold yellow]")


    title_extraction_model = ChatNebius(model="Qwen/Qwen3-30B-A3B-Instruct-2507")

    TITLE_RELEVANCE_PROMPT = """
    Given this job title, identify the specific area of AI/technology focus, if any,
    that would be most relevant to a company that builds a web search API for AI agents.
    If the title has no clear AI/tech focus, say so rather than guessing.
    Respond with a short phrase (2-5 words) or "none" if nothing is clearly relevant.
    """

    response = title_extraction_model.invoke([
        {"role": "system", "content": TITLE_RELEVANCE_PROMPT},
        {"role": "user", "content": title}
        ])
    pprint(response.content)
    return message_text(response)

def create_message(name: str, company_name: str, job_title: str, key_facts_text: list[str], persona_text: list[str], results_text) -> str:

    console.print("\n[bold yellow]Generating Message...[/bold yellow]")

    message_model = ChatNebius(model="deepseek-ai/DeepSeek-V3.2-fast")

    CREATE_MESSAGE_PROMPT=f"""
    Please write an outbound LinkedIn message. This is the very first message to someone
    I connected with. The goal is to reach someone who would be interested in Tavily for
    AI-related projects.

    Match this tone and style exactly: 
    {example_templates}
    
    I will give you: the person's name, company name, job title, key_facts (concrete
    responsibilities/activities about this person), persona_inference (how this person
    communicates/operates — may be an empty list if nothing was confidently inferred),
    and role-specific search results about the company.
    
    Steps to follow:
    1. Review each search result. Only use a result if it is clearly relevant to this
       specific person's role and part of the organization. If a result is about a
       different org or department, skip it — do not use it.
    2. Based on the relevant results (if any), determine how Tavily could plausibly fit:
       (a) the company sells a product with an AI backend that could use a search/retrieval
       layer, (b) the company has an internal AI system or agentic workflow that could use
       search, or (c) no clear fit is evident from the results.
    3. If no search results are relevant, or case (c) applies, write a short, generic
       message that doesn't force a connection that isn't there — do not fabricate a
       reason Tavily is relevant.
    4. If key_facts or persona_inference are sparse or empty, rely more on whatever
       company research is available rather than inventing personal detail.
    5. Do not state anything as fact unless it's supported by the provided key_facts,
       persona_inference, or search results.
    
    Output format: respond with ONLY valid JSON, no markdown, no preamble, in this shape:
    {{
      "message": "the LinkedIn message text",
      "sources_used": ["url1", "url2"]
    }}
    
    The message should be under 100 words, written in a natural, conversational tone —
    not salesy, no exclamation points, no buzzwords like "synergy" or "leverage." End with
    a low-pressure ask (e.g., open to a quick chat) rather than a hard pitch.
    """

    user_content = json.dumps({
    "name": name,
    "company_name": company_name,
    "job_title": job_title,
    "key_facts": key_facts_text,
    "persona_inference": persona_text,
    "search_results": results_text,
    })
    
    response = message_model.invoke([
        {"role": "system", "content": CREATE_MESSAGE_PROMPT},
        {"role": "user", "content": user_content},
    ])

    try:
        response = json.loads(message_text(response))
        return response 
    except json.JSONDecodeError:
        print("Failed to parse extraction response as JSON:")
        print(response.content)
        raise typer.Exit(code=1)



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
    print_search_results(relevant_resources)

    message = create_message(name, parsed_description['company_name'], parsed_description['job_title'], 
                             parsed_description['job_description']['key_facts'], parsed_description['job_description']['persona_inference'], relevant_resources)
    console.print(message['message'], markup=False)


if __name__ == "__main__":
    app()
