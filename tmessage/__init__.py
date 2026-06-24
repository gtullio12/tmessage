from __future__ import annotations

import json
import os
import sys
from typing import Annotated, Any
from datetime import datetime, timedelta
from pathlib import Path
from importlib.resources import files

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
with files("tmessage").joinpath("example_messages.txt").open() as f:
    templates = [msg.strip() for msg in f.read().split("---") if msg.strip()]
example_templates = "\n\n---\n\n".join(templates)

# Define where to store API keys
CONFIG_PATH = Path.home() / ".config" / "tmessage" / ".env"

def setup_api_keys():
    if not CONFIG_PATH.exists():
        console.print("[bold]First time setup — enter your API keys:[/bold]")
        nebius_key = typer.prompt("Nebius API key")
        tavily_key = typer.prompt("Tavily API key")
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(f"NEBIUS_API_KEY={nebius_key}\nTAVILY_API_KEY={tavily_key}\n")
        console.print("[bold green]Keys saved![/bold green]")

    load_dotenv(CONFIG_PATH)

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
    # use faster/cheaper models for basic text parsing and more complex for making customized message
    text_extraction_model = ChatNebius(model="Qwen/Qwen3-30B-A3B-Instruct-2507")
    console.print("\n[bold yellow]Extracting person info...[/bold yellow]")


    # Now we need to extract the company, title, and optionally the job description
    EXTRACTION_SYSTEM_PROMT = """
    I'm giving you a copy-pasted job description section in LinkedIn,
    please parse the entire string and extract the company name, job title, and job description
    if it's listed. With the job description split it into 2 lists, one is key_facts which should
    be a list of job responsibilities (close to paraphrasing) — each key_fact should describe a
    concrete responsibility or activity, not a characterization of how the person works — and
    persona_inference which is a list of strings of how this person communicates/operates
    (hands-on, technical, builds from scratch, decision maker), don't assume for this one,
    if you can't find something then return an empty list. 

    Return ONLY valid JSON in this exact shape:
    {{
      "company_name": "...",
      "job_title": "...",
      "job_description": {
        "key_facts": [],
        "persona_inference": []
      },
      "search_query": "..."
    }}



    If no job description body is present and only a title and company are provided, return empty lists for both key_facts and persona_inference — do not infer or generate responsibilities from the title alone.

    Also return a "search_query" field: a short, disambiguated search query (under 10 words) that could be used to find recent news about this specific company. Include enough context to distinguish it from other companies with similar names (e.g. "Greenlight fintech kids finance app" not just "Greenlight").
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


def search_relevant_resources(search_query, relevant_title) -> Any:
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

    result = title_specific_search.invoke(f"{search_query} {relevant_title}")
    if (len(result['results']) == 0):
        console.print("\n[bold red]0 Search Results Found[/bold red]")
    return result


# Takes in a unstructured title -> And extracts out only the relevant bits. 
def title_extraction(title: str) -> str:
    title_extraction_model = ChatNebius(model="Qwen/Qwen3-30B-A3B-Instruct-2507")
    console.print("\n[bold yellow]Extracting relevant title bits...[/bold yellow]")



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
    message_model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct")

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


def message_evaluation_check(msg: str, key_facts: list[str], persona: list[str], search_results) -> bool:
    message_evaluator = ChatNebius(model="Qwen/Qwen3-30B-A3B-Instruct-2507")

    EVALUATE_PROMPT = f"""
    You are evaluating an outbound LinkedIn message written by a Tavily sales rep.

    IMPORTANT: Tavily is the company sending this message. Any mention of Tavily by name is ALWAYS valid and should NEVER be flagged as unsupported. Do not penalize the message for mentioning Tavily.
    
    Evaluate against these criteria:
    1. Under 100 words
    2. No buzzwords (synergy, leverage, etc.)
    3. Ends with a low-pressure ask
    4. Specific technical claims about the prospect's stack or architecture (e.g. "you use RAG", "you need a retrieval layer") must be supported by context. Generic statements like "Tavily could support your AI work" are always acceptable — that is the pitch, not a factual claim.
    5. If no clear use case is evident, the message is generic rather than fabricating a specific connection — this is acceptable and should pass
    6. Sounds casual and matches this tone: {example_templates}

    Note: the job title and company name are always provided directly by the user and are always considered supported facts. Only flag claims that go beyond what is in key_facts, persona_inference, and search_results.

    Context provided:
    key_facts: {key_facts}
    persona_inference: {persona}
    search_results: {search_results}

    Message to evaluate:
    {msg}

    Respond with ONLY valid JSON:
    {{{{
      "passed": true or false,
      "reason": "brief explanation if failed, null if passed"
    }}}}
    """
    user_content = json.dumps({
        "message": msg,
        })

    res = message_evaluator.invoke([
            {"role": "system", "content": EVALUATE_PROMPT},
            {"role": "user", "content": user_content}
            ])
    result = json.loads(message_text(res))

    if (result['passed'] == False):
        console.print(f"Message was rejected by evaluation loop. Reason: {result['reason']}\n Generated Message: {msg}")
        return False

    return True 


@app.command()
def main() -> None:

    setup_api_keys()

    EVAL_LOOP_RETRIES = 3

    while True:

        name = typer.prompt("Name")
        console.print("Paste LinkedIn description, then press Ctrl+D when done:")
        description = sys.stdin.read()

        parsed_description = extract_info_from_description(description)

        console.print(
                Columns([
                    Panel.fit(name, title="Person", border_style="cyan"),
                    Panel.fit(parsed_description["job_title"], title="Title", border_style="cyan"),
                    ])
                )
        relevant_title = title_extraction(parsed_description['job_title'])

        relevant_resources = search_relevant_resources(parsed_description['search_query'], relevant_title)
        print_search_results(relevant_resources)


        console.print('\n\nGenerating and Evaluating Message...')

        for _ in range(EVAL_LOOP_RETRIES):
            message = create_message(name, parsed_description['company_name'], parsed_description['job_title'], 
                                 parsed_description['job_description']['key_facts'], parsed_description['job_description']['persona_inference'], relevant_resources)
            evaluator = message_evaluation_check(message, parsed_description['job_description']['key_facts'], parsed_description['job_description']['persona_inference'], relevant_resources)

            if evaluator:
                break
            console.print('Regenerating message...')

        else:
            console.print('[bold red]Evaluator Loop Retries Exhausted.[/bold red]')
            continue

        console.print(message['message'], markup=False, style="green")
        another = typer.confirm("Generate another message?")
        if not another:
            break

if __name__ == "__main__":
    app()
