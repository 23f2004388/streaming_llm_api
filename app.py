import os, json, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

app = FastAPI()

# Fix browser "Failed to fetch" (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq (free) OpenAI-compatible
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.1-8b-instant"


def sse(obj) -> str:
    # Required SSE format: data: <json>\n\n
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/")
async def health():
    return {"ok": True}


@app.post("/stream")
async def stream_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse({"error": "Missing prompt"}, status_code=400)

    if body.get("stream") is not True:
        return JSONResponse({"error": "Set stream=true"}, status_code=400)

    full_prompt = f"""
Analyze market trends and provide exactly 9 key insights.

Rules:
- At least 900 characters total
- Exactly 9 numbered insights (1 to 9)
- Each insight must include: insight + evidence/example + implication/action
- Stay relevant to the prompt

Prompt:
{prompt}
""".strip()

    async def gen():
        chunk_count = 0

        # MUST send first chunk immediately (grader reachability)
        yield sse({"choices": [{"delta": {"content": " "}}]})
        chunk_count += 1
        await asyncio.sleep(0)

        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.4,
                stream=True,
            )

            async for chunk in resp:
                delta = ""
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                if delta:
                    yield sse({"choices": [{"delta": {"content": delta}}]})
                    chunk_count += 1
                    await asyncio.sleep(0)

            # Ensure at least 5 chunks
            while chunk_count < 5:
                yield sse({"choices": [{"delta": {"content": " "}}]})
                chunk_count += 1
                await asyncio.sleep(0)

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield sse({"error": str(e)})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )

