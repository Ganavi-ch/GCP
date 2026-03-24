# Agentic AI E-Commerce Assistant (GCP)

This project provides a GCP-ready, agentic AI service for e-commerce operations using natural language queries.

## Features

- Coupon enquiry (list, detail, status)
- Order status enquiry
- Order details enquiry
- Update order (cancel, address update, item quantity update)
- Register user
- User profile and order history enquiry
- Platform usage questions answered through a lightweight RAG knowledge agent

## Tech Stack

- Python + FastAPI
- Agent orchestration layer in `app/agent.py`
- Backend store in `app/store.py`
- RAG module in `app/knowledge_base.py`
- Containerized deployment to Cloud Run via Cloud Build

## Project Structure

- `app/main.py`: API entrypoint
- `app/agent.py`: Intent detection and tool/action routing
- `app/store.py`: E-commerce backend operations
- `app/knowledge_base.py`: RAG knowledge Q&A
- `cloudbuild.yaml`: CI/CD pipeline to Cloud Run
- `Dockerfile`: Container build config

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

Open:

- API docs: `http://127.0.0.1:8080/docs`
- Health: `http://127.0.0.1:8080/health`

## Next.js Chat Frontend

The project includes a ChatGPT-style frontend in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

The frontend calls `http://127.0.0.1:8080/chat` by default.
To use a different API endpoint, set:

```bash
set NEXT_PUBLIC_API_BASE=http://your-api-host:8080
```

## Test with Example Queries

POST `http://127.0.0.1:8080/chat`

```json
{
  "user_id": "u1001",
  "message": "Show available coupons"
}
```

```json
{
  "message": "Check coupon SAVE20"
}
```

```json
{
  "user_id": "u1001",
  "message": "What is the status of my order 12345?"
}
```

```json
{
  "message": "Cancel order 10234"
}
```

```json
{
  "message": "Register me with name John and email john_new@example.com"
}
```

```json
{
  "message": "How can I apply a coupon?"
}
```

## Deploy on GCP

1. Enable APIs:
   - Cloud Build API
   - Cloud Run Admin API
   - Artifact Registry API

2. Build and deploy:

```bash
gcloud builds submit --config cloudbuild.yaml .
```

3. The Cloud Run service URL can be used for your frontend/chat client.

## Vertex AI / ADK Integration Path

To move from rule-based intent parsing to full LLM orchestration:

- Replace `detect_intent()` with a Vertex AI Gemini call for intent + entity extraction.
- Keep `run_agent()` action handlers as deterministic backend tools.
- Add function-calling/tool-calling schema for:
  - `get_coupon`, `list_coupons`
  - `get_order_status`, `get_order_details`, `cancel_order`, `update_order_address`, `update_order_item_quantity`
  - `register_user`, `get_user`, `list_user_orders`
  - `rag_answer`
- Use Vertex AI Search or embeddings + vector store for scalable RAG.

This pattern gives you an agent that is both flexible (LLM) and reliable (deterministic backend operations).

## Plan A: Vertex AI Function-Calling Agent

The backend will try a Vertex AI Gemini agent first (tool/function calling) and fall back to the local deterministic agent if Vertex isn’t available.

### 1) Enable required APIs

In your project `project-ca436020-0c75-4cc6-b84`, enable:

- `Vertex AI API` (`aiplatform.googleapis.com`)

If you just enabled it, it can take a few minutes to propagate.

### 2) Authentication

Set Application Default Credentials (ADC), for example:

- `gcloud auth application-default login`
- or set `GOOGLE_APPLICATION_CREDENTIALS` to a service account key file

### 3) Region mapping (Mumbai)

Your “`asia-mumbai`” choice maps to Vertex AI region id `asia-south1` in code.

### 4) Optional environment overrides

You can override defaults by setting:

- `VERTEX_PROJECT_ID` (default: `project-ca436020-0c75-4cc6-b84`)
- `VERTEX_LOCATION` (default: `asia-south1`)
- `VERTEX_MODEL_NAME` (default: `gemini-1.5-flash`)
- `VERTEX_EMBEDDING_MODEL_NAME` (default: `text-embedding-004`)

Note: You can also create a `.env` file in the root directory to manage these variables.
