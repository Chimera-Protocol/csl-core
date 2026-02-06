# Philosophy

> "AI must obey explicit law."

The development of CSL-Core is driven by a belief that the future of Artificial Intelligence is **Neuro-Symbolic-Causal**.

We are moving away from "pure black box" systems toward architectures where a neural brain (LLM) is paired with a symbolic spine (CSL). This document outlines the guiding principles behind Chimera and CSL.

---

## 1. Governance is a Separate Layer
In traditional software, logic and data are often mixed. In AI, "logic" (the prompt) and "intelligence" (the model) are deeply entangled.

We believe **Governance** must be decoupled from **Intelligence**.
* **Intelligence** is fluid, creative, and probabilistic.
* **Governance** must be rigid, explicit, and deterministic.

You shouldn't need to retrain a model or rewrite a 10-page system prompt just to change a business rule. Governance belongs in a dedicated layer, governed by a dedicated language.

## 2. Code is Law (Literally)
In high-stakes domains like Finance, Healthcare, and Law, "99% accuracy" is not a success metric; it is a liability.

CSL-Core treats policy as code that is:
* **Immutable:** Once compiled, the rules cannot change dynamically based on context.
* **Verifiable:** We use formal methods (Z3) to prove correctness.
* **Auditable:** Every decision leaves a trace.

If an AI Agent creates a contract, that contract must adhere to the laws defined in CSL. No hallucinations allowed.

## 3. Fail-Closed by Design
The default state of any secure system must be "Block."
If a CSL policy encounters data it cannot understand, or a state that is ambiguous, it blocks the action. We prioritize **safety over convenience**.

## 4. The Neuro-Symbolic Future
Current AI safety relies heavily on RLHF (Reinforcement Learning from Human Feedback)—essentially training the model to "want" to be good.

We believe this is insufficient for enterprise autonomy. A truly safe autonomous agent requires:
1.  **Neural Component:** To understand intent and generate plans.
2.  **Symbolic Component:** To validate those plans against strict constraints.

CSL-Core is that symbolic component. It provides the "guardrails" that are mathematically proven to hold, allowing the neural component to be as creative as possible within those bounds.