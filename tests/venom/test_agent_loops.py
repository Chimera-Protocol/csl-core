"""A stopped call does not end the agent: in the real loops of LangChain (AgentExecutor), LangGraph
(a prebuilt agent and its ToolNode), the OpenAI Agents SDK and CrewAI, a wired tool whose limits stop a call returns Blocked
and the model reads why, then the loop goes on. Plain Python functions still raise PermissionError
(as in 0.6.8). Each framework's test runs where it is installed and is skipped elsewhere; the
models are scripted, nothing goes over the network."""

from __future__ import annotations

import importlib.util

import pytest

from .conftest import run_cli

HEADERS = {
    "langchain": "from langchain_core.tools import tool\n",
    "agents": "from agents import function_tool as tool\n",
    "crewai": "from crewai.tools import tool\n",
    "plain": "def tool(fn):\n    return fn\n",
}
BODY = '''

@tool{arg}
def transfer_funds(amount: int, to_wallet: str) -> str:
    """Send money to a wallet."""
    return f"sent {{amount}}"
'''


def _wired(tmp_path, capsys, kind: str):
    """A one-tool agent of this kind, set up and wired for real, in block mode, never above 1,000."""
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / ".git").mkdir()
    arg = '("transfer_funds")' if kind == "crewai" else ""
    (root / "app/agent.py").write_text(HEADERS[kind] + BODY.format(arg=arg))
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire", "--limit", "repo.transfer_funds=1000", "--no-anim"], capsys)
    assert rc == 0, out
    text = (root / "app/agent.py").read_text()
    assert "_csl_guard.tool(" in text
    spec = importlib.util.spec_from_file_location(f"loop_{kind}_{id(tmp_path)}", root / "app/agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, text


def test_plain_functions_still_raise(tmp_path, capsys):
    mod, text = _wired(tmp_path, capsys, "plain")
    assert 'on_block' not in text  # the file's own decorator: a plain function, as in 0.6.8
    assert mod.transfer_funds(amount=5, to_wallet="w") == "sent 5"
    with pytest.raises(PermissionError):
        mod.transfer_funds(amount=5_000, to_wallet="w")


def test_langchain_agent_executor_goes_on(tmp_path, capsys):
    pytest.importorskip("langchain_classic")
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.prompts import ChatPromptTemplate

    mod, _ = _wired(tmp_path, capsys, "langchain")
    seen = []

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            seen.append(messages)
            return super()._generate(messages, stop, run_manager, **kw)

    model = Model(responses=[
        AIMessage("", tool_calls=[{"name": "transfer_funds", "args": {"amount": 5, "to_wallet": "w"}, "id": "1"}]),
        AIMessage("", tool_calls=[{"name": "transfer_funds", "args": {"amount": 5_000, "to_wallet": "w"}, "id": "2"}]),
        AIMessage("I could not send 5,000: the limit stops it."),
    ])
    prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("placeholder", "{agent_scratchpad}")])
    executor = AgentExecutor(agent=create_tool_calling_agent(model, [mod.transfer_funds], prompt),
                             tools=[mod.transfer_funds])
    result = executor.invoke({"input": "pay"})
    assert result["output"] == "I could not send 5,000: the limit stops it."
    last_input = " ".join(str(m.content) for m in seen[-1])
    assert "sent 5" in last_input and "transfer_funds was not run" in last_input and "never above 1,000" in last_input


def test_langgraph_agent_goes_on(tmp_path, capsys):
    pytest.importorskip("langgraph")
    import warnings

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.prebuilt import create_react_agent

    mod, _ = _wired(tmp_path, capsys, "langchain")

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kw):
            return self

    model = Model(responses=[
        AIMessage("", tool_calls=[{"name": "transfer_funds", "args": {"amount": 5, "to_wallet": "w"}, "id": "1"},
                                  {"name": "transfer_funds", "args": {"amount": 5_000, "to_wallet": "w"}, "id": "2"}]),
        AIMessage("sent 5; 5,000 was stopped by its limit"),
    ])
    graph = create_react_agent(model, [mod.transfer_funds])  # its ToolNode runs the tools
    out = graph.invoke({"messages": [("human", "pay")]})
    tools = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tools[0].content == "sent 5" and "transfer_funds was not run" in tools[1].content
    assert all(m.status == "success" for m in tools)  # a result, not an error
    assert out["messages"][-1].content == "sent 5; 5,000 was stopped by its limit"


def test_openai_agents_runner_goes_on(tmp_path, capsys):
    pytest.importorskip("agents")
    import asyncio

    from agents import Agent, ModelResponse, Runner, Usage
    from agents.models.interface import Model
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

    mod, _ = _wired(tmp_path, capsys, "agents")
    seen = []

    def call(i, amount):
        return ResponseFunctionToolCall(id=f"f{i}", call_id=f"c{i}", name="transfer_funds", type="function_call",
                                        arguments=f'{{"amount": {amount}, "to_wallet": "w"}}')

    class Scripted(Model):
        turn = 0

        async def get_response(self, system_instructions, input, *args, **kw):
            seen.append(input)
            Scripted.turn += 1
            if Scripted.turn == 1:
                return ModelResponse(output=[call(1, 5), call(2, 5_000)], usage=Usage(), response_id=None)
            text = ResponseOutputText(text="done", type="output_text", annotations=[])
            msg = ResponseOutputMessage(id="m", content=[text], role="assistant", status="completed", type="message")
            return ModelResponse(output=[msg], usage=Usage(), response_id=None)

        def stream_response(self, *a, **k):
            raise NotImplementedError

    agent = Agent(name="cashier", model=Scripted(), tools=[mod.transfer_funds])
    result = asyncio.run(Runner.run(agent, "pay"))
    assert result.final_output == "done"
    outputs = [str(i.get("output")) for i in seen[-1] if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert outputs[0] == "sent 5" and "transfer_funds was not run" in outputs[1]


def test_crewai_agent_goes_on(tmp_path, capsys):
    pytest.importorskip("crewai")
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM

    mod, _ = _wired(tmp_path, capsys, "crewai")
    seen = []

    class Scripted(BaseLLM):
        def call(self, messages, *a, **k):
            seen.append(messages)
            text = str(messages)
            if "was not run" in text:
                return "Thought: the limit stopped the large one.\nFinal Answer: sent 5; 5,000 was stopped"
            if "sent 5" in text:
                return ('Thought: now the large one.\nAction: transfer_funds\n'
                        'Action Input: {"amount": 5000, "to_wallet": "w"}')
            return 'Thought: pay.\nAction: transfer_funds\nAction Input: {"amount": 5, "to_wallet": "w"}'

        def supports_function_calling(self):
            return False

        def supports_stop_words(self):
            return False

        def get_context_window_size(self):
            return 8192

    agent = Agent(role="cashier", goal="pay", backstory="pays", tools=[mod.transfer_funds], llm=Scripted(model="scripted"),
                  allow_delegation=False, max_iter=5, verbose=False)
    task = Task(description="pay", expected_output="what happened", agent=agent)
    out = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    assert "5,000 was stopped" in str(out)
    assert any("transfer_funds was not run" in str(m) for m in seen)


@pytest.fixture(autouse=True)
def _no_crewai_telemetry(monkeypatch):
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
