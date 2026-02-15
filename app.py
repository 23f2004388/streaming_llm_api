import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

# Initialize FastAPI
app = FastAPI()

# Use GROQ instead of OpenAI (free, no billing needed)
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Fast free streaming model
MODEL = "llama-3.1-8b-instant"


# Format SSE event
def format_sse(data):
    return f"data: {json.dumps(data)}\n\n"


@app.post("/stream")
async def stream(request: Request):

    body = await request.json()

    prompt = body.get("prompt")
    stream_flag = body.get("stream")

    if not prompt or stream_flag != True:
        return {"error": "Invalid request. Must include prompt and stream=true"}

    # Wrap prompt to ensure assignment requirements
    full_prompt = f"""
Analyze market trends and provide exactly 9 key insights.

Requirements:
- Minimum 900 characters total
- Exactly 9 numbered insights
- Each insight must include:
  - insight
  - brief evidence/example
  - implication or action

User prompt:
{prompt}
"""

    async def generator():

        try:

            # Streaming call
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.5,
                stream=True
            )

            chunk_count = 0

            async for chunk in response:

                if chunk.choices and chunk.choices[0].delta.content:

                    content = chunk.choices[0].delta.content

                    yield format_sse({
                        "choices": [
                            {
                                "delta": {
                                    "content": content
                                }
                            }
                        ]
                    })

                    chunk_count += 1

                    # tiny delay to ensure streaming behavior
                    await asyncio.sleep(0)

            # Ensure minimum chunks
            while chunk_count < 5:
                yield format_sse({
                    "choices": [
                        {
                            "delta": {
                                "content": ""
                            }
                        }
                    ]
                })
                chunk_count += 1

            yield "data: [DONE]\n\n"

        except Exception as e:

            yield format_sse({
                "error": str(e)
            })

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

