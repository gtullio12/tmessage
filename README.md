# tmessage

A CLI tool that generates personalized LinkedIn outreach messages using AI. Built for sales and account teams who need to send high-quality, contextually relevant messages at scale — without spending 10 minutes researching each person manually.

## Demo

> Loom walkthrough link here

---

## Installation

### Requirements
- [Homebrew](https://brew.sh)
- A [Nebius](https://nebius.com) API key
- A [Tavily](https://tavily.com) API key

### Install via Homebrew

```bash
brew tap gtullio12/tmessage
brew install tmessage
```

### First run

On first run, `tmessage` will prompt you to enter your API keys. These are saved to `~/.config/tmessage/.env` and reused on every subsequent run — you only need to do this once.

---

## Usage

```bash
tmessage
```

The tool will prompt you for:
1. **Name** — the person's first and last name
2. **LinkedIn description** — paste the job description section from their LinkedIn profile, then press `Ctrl+D` when done

`tmessage` then runs a three-stage pipeline and outputs a ready-to-send LinkedIn message. Press `y` when prompted to generate another message, or `n` to exit.

---

## How it works

### Pipeline overview

```
LinkedIn description paste
        │
        ▼
┌─────────────────────┐
│  Stage 1: Extraction │  Qwen3-30B
│  company, title,     │
│  key_facts,          │
│  persona_inference   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 2: Research  │  Tavily Search
│  title relevance    │
│  + company search   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 3: Message   │  DeepSeek-V3.2
│  generation         │
└─────────────────────┘
```

### Stage 1 — Person extraction

The user pastes the raw LinkedIn job description block. A fast, lightweight model (Qwen3-30B) extracts structured information:

- `company_name` — extracted from the pasted text
- `job_title` — extracted from the pasted text
- `key_facts` — a list of concrete responsibilities and activities, close to paraphrasing. Each fact describes what the person *does*, not how they do it.
- `persona_inference` — a list of inferred characteristics about how this person operates (e.g. hands-on, technically engaged, builds from scratch). Critically, this list is only populated when the inference is *directly supported* by specific language in the source text. If nothing is confidently inferable, this returns an empty list.

**Design decision:** the persona_inference constraint — preferring honest-but-generic over confident-but-wrong — is a core principle of the tool. A message that admits it doesn't know much about someone is better than one that fabricates detail and sounds off.

### Stage 2 — Company research

The job title is first distilled to its most AI/tech-relevant component (e.g. "VP of Generative AI Marketing" → "Generative AI"). This focused query is then used to search Tavily for recent, role-specific company news and initiatives — scoped to the last 365 days to keep results current.

**Design decision:** searching by distilled title rather than full title produces more relevant results. A full title like "Vice President & Global Head of Generative AI // Marketing Transformation Office" retrieves noisier results than "Generative AI."

### Stage 3 — Message generation

A larger, higher-quality model (DeepSeek-V3.2-fast) writes the final message using:

- Structured person context from Stage 1
- Tavily search results from Stage 2
- A library of real example messages to match tone and style

The model follows explicit rules: only reference facts supported by the provided context, don't fabricate a Tavily use case if one isn't evident, and write under 100 words with a low-pressure ask. If search results aren't relevant, it falls back to a generic message rather than forcing a connection that isn't there.

**Design decision:** two separate models for extraction vs. message generation. Extraction is a structured, deterministic task — a smaller, faster model handles it well. Message generation requires more nuance and writing quality, so a stronger model is used only where it matters.

---

## Design philosophy

- **Prefer honest-but-generic over confident-but-wrong.** The tool will not fabricate relevance or invent personal detail. A generic message is better than a weird one.
- **Minimize user input.** The user provides a name and a LinkedIn paste — nothing else. All structure is inferred.
- **Degrade gracefully.** If search returns nothing useful, the tool still produces a message. It doesn't stall or crash.
- **Speed over completeness where possible.** Smaller models are used for simpler tasks to keep the pipeline fast enough for batches of 50+ people.

---

## Tech stack

- [LangChain](https://langchain.com) — LLM orchestration
- [Nebius](https://nebius.com) — model hosting (Qwen3-30B, DeepSeek-V3.2-fast)
- [Tavily](https://tavily.com) — web search API
- [Typer](https://typer.tiangolo.com) — CLI framework
- [Rich](https://rich.readthedocs.io) — terminal formatting

---

## Project structure

```
tmessage/
├── tmessage.py          # main pipeline and CLI entrypoint
├── example_messages.txt # message style templates
├── pyproject.toml
├── requirements.txt
└── README.md
```
