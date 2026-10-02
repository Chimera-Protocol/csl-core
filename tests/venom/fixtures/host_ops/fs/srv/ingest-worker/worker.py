import os
import subprocess

import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = "You ingest partner feeds, normalise them and store them in the warehouse."
OPENAI_KEY = os.environ["OPENAI_API_KEY"]
WAREHOUSE = os.getenv("DATABASE_URL")


@tool
def run_command(command: str, timeout: int = 30) -> str:
    """Run a shell command on the ingest host."""
    return subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout).stdout


@tool
def http_get(url: str) -> str:
    """Fetch a partner feed."""
    return requests.get(url, timeout=10).text


@tool
def write_file(path: str, content: str) -> str:
    """Write normalised output to disk."""
    with open(path, "w") as f:
        f.write(content)
    return path


def main():
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([run_command, http_get, write_file])
    llm.invoke([("system", SYSTEM_PROMPT), ("user", "ingest")])


if __name__ == "__main__":
    main()
