# Proves Venom never imports discovered code: importing this module leaves a marker.
from pathlib import Path

Path(__file__).with_name("IMPORTED.marker").write_text("imported")

from langchain_core.tools import tool


@tool
def sideeffect_probe() -> str:
    """Harmless."""
    return "ok"
