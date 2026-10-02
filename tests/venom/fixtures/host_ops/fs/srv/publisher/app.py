from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "post_to_page",
            "description": "Publish a post on the public company page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "visibility": {"type": "string", "enum": ["public", "followers"]},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_feed",
            "description": "Read the latest items from the internal feed.",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}},
        },
    },
]


@app.post("/webhooks/sms")
async def inbound_sms(payload: dict):
    return client.chat.completions.create(model="gpt-4.1", tools=TOOLS, messages=[{"role": "user", "content": payload["body"]}])


if __name__ == "__main__":
    import sys
    print(sys.argv)
