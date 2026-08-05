---
title: "Financial AI Agent"
excerpt: "A Telegram-based financial AI agent built with FastAPI, LangChain, and LangGraph."
header:
  teaser: # optional: add a screenshot/diagram path here, e.g. /assets/images/financial-agent-teaser.png
order: 1
---

## 🤖 Overview

**Financial AI Agent** is a conversational AI agent that talks to customers over **Telegram** and recommends their **Next Best Action (NBA)** — a personalized financial recommendation. It's built with **FastAPI**, **LangChain**, **LangGraph**, and **Pydantic v2**.

Repository: [github.com/LucasMVasconcelos/financial-agent](https://github.com/LucasMVasconcelos/financial-agent)

## 🎯 What the agent does

- Answers customer questions about their profile, products, and financial options.
- Recommends a personalized Next Best Action, backed by a pluggable model gateway (mock today, swappable for a real SageMaker endpoint).
- Answers general questions by searching a knowledge base (Retrieval-Augmented Generation) over articles about CDs, treasury bonds, insurance, and loan policies.
- Handles loan requests end-to-end, including human approval for larger amounts.

## 🛠 What was built

- **Layered architecture** (domain, repository, gateway, service, agent, API) with strict one-directional dependencies, so any layer's implementation — an in-memory repository, a mocked ML model — can be swapped without touching the rest of the code.
- **The whole conversational agent as a LangGraph state machine**: an explicit `reason ⇄ validate tool calls ⇄ execute tool ⇄ self-correct` loop, checkpointed and resumable, replacing an opaque, single-shot agent executor.
- **Five tools with a strict contract**: each call is validated, exceptions never leak to the agent, and every result comes back as a structured success/error envelope the model can react to and self-correct from.
- **A second, dedicated graph for loan approval**, with a genuine human-in-the-loop pause: small loans auto-approve, larger ones pause the workflow until an admin approves or rejects it — potentially hours or days later — and the customer is notified once a decision is made.
- **A model router** that picks between a smaller/faster and a larger/more capable language model, both by task (reasoning vs. a quick holding reply vs. summarization) and by how complex the incoming message is — balancing latency and cost against answer quality.
- **Three layers of memory**: the raw recent conversation, a rolling summary once it grows too long, and a long-term semantic memory that resurfaces relevant context from past conversations.
- **Security by construction**: the customer's identity always comes from the authenticated Telegram payload, never from user-supplied text or a parameter the model could fill in; the Telegram webhook is validated via a secret token, and admin endpoints require separate service authentication.
- **Observability**: structured logs with correlation IDs, per-tool timing, and end-to-end LLM tracing.
- **Testing beyond unit tests**: a "golden transcripts" suite runs real conversation scenarios against the live agent, asserting both on behavior (which tools get called) and — via a second LLM acting as a judge — on the quality and compliance of the final answer (e.g., never implying a loan was approved while it's still pending human review).
- **Deployable as-is**: runs locally via Docker Compose, or as an AWS Lambda function behind API Gateway, with an optional SageMaker-backed recommendation model.

## 🧰 Tech stack

FastAPI · LangChain · LangGraph · Pydantic v2 · OpenAI · Redis · structlog · AWS Lambda / SageMaker (optional) · pytest
