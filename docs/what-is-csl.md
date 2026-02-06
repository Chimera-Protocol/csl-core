# What is CSL-Core?

> **"Solidity for AI Policies"**

CSL-Core (Chimera Specification Language) is an open-core policy language and runtime designed to bring **deterministic safety** to probabilistic AI systems.

It allows developers to define rigid, formally verified "laws" for AI Agents and enforce them at runtime with near-zero latency, completely independent of the model's training or prompt.

---

## The Core Problem: The Probabilistic Trap

Modern Generative AI is inherently probabilistic. While this makes it creative, it also makes it unreliable for critical constraints.
- **Prompts are suggestions**, not rules.
- **Fine-tuning** biases behavior but guarantees nothing.
- **Post-hoc classifiers** are just another probabilistic layer (more AI watching AI).

CSL-Core flips this model. Instead of asking the AI to behave, **you force it to comply** using an external, deterministic logic layer.

---

## How It Works: The 3-Stage Pipeline

CSL-Core is not just a syntax; it is a complete lifecycle for governance. It separates **Policy Definition** from **Runtime Enforcement**.

### 1. The Compiler (`compiler.py`)
Unlike standard config files (YAML/JSON), CSL is a compiled language.
- It parses your `.csl` policy file into an Abstract Syntax Tree (AST).
- It converts high-level logic into an optimized **Intermediate Representation (IR)** consisting of lightweight Python functors (`OpBinary`, `OpVariable`).
- This compilation step ensures that no heavy parsing happens during the critical runtime path.

### 2. The Verifier (`verifier.py`)
Before a policy is ever deployed, CSL-Core performs **Static Formal Verification** using the **Z3 Theorem Prover**.
It doesn't just check syntax; it checks *mathematical consistency*.

* **Reachability Analysis:** Are there rules that can never be triggered?
* **Contradiction Detection:** Do two rules conflict? (e.g., Rule A says "Must be < 100" and Rule B says "Must be > 200" for the same condition).
* **Shadowing:** Does one rule render another obsolete?

If your policy contains a logical contradiction, **it will not compile.**

### 3. The Runtime Guard (`runtime.py`)
The runtime is the enforcement layer that sits between your Agent and the World (API, Tool, or User).
- **Fail-Closed:** If something goes wrong, the action is blocked by default.
- **Zero-Dependency:** The runtime does not require Z3 or heavy libraries; it runs pure Python.
- **Deterministic:** Given the same context and policy, the result is always identical.

---

## Technical Highlights

* **Logic Engine:** Microsoft Z3 (at compile time).
* **Enforcement:** Functor-based evaluation (at runtime).
* **Integration:** Drop-in support for LangChain, Python functions, and REST APIs.
* **Auditability:** Every decision produces a `GuardResult` containing triggered rules, violations, and latency metrics.

## Example Workflow

1.  **Write Law:** Define a `.csl` file constraining financial transaction limits.
2.  **Verify:** Run `cslcore verify policy.csl`. The Z3 engine proves your rules don't contradict.
3.  **Deploy:** Wrap your LangChain tool with `guard_tools()`.
4.  **Enforce:** When the LLM tries to hallucinate a transaction over the limit, CSL-Core intercepts and blocks it before execution.