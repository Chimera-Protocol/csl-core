# CSL-Core Syntax Specification

This document defines the official syntax, grammar, and semantics for **Chimera Specification Language (CSL-Core)**.

CSL is a domain-specific language (DSL) designed for defining deterministic policies for AI agents. It compiles into an executable Intermediate Representation (IR) verified by Z3.

---

## 1. File Structure

A standard `.csl` file consists of two main blocks:

1. **`CONFIG`** (Optional): Runtime and compiler settings.
2. **`DOMAIN`** (Required): The problem space definitions and constraints.

```csl
CONFIG {
  ENFORCEMENT_MODE: BLOCK
}

DOMAIN MyDomain {
  VARIABLES { ... }
  STATE_CONSTRAINT MyRule { ... }
}
```

---

## 2. Configuration (CONFIG)

The configuration block controls the behavior of the CSL Compiler and Runtime Guard.

| Key | Type | Allowed Values | Default | Description |
|-----|------|----------------|---------|-------------|
| `ENFORCEMENT_MODE` | Enum | `BLOCK`, `WARN`, `LOG` | `BLOCK` | Action to take on violation. |
| `CHECK_LOGICAL_CONSISTENCY` | Bool | `TRUE`, `FALSE` | `TRUE` | Enable Z3 formal verification at compile time. |
| `ENABLE_FORMAL_VERIFICATION` | Bool | `TRUE`, `FALSE` | `FALSE` | (Enterprise) Enable TLA+ engine. |
| `ENABLE_CAUSAL_INFERENCE` | Bool | `TRUE`, `FALSE` | `FALSE` | (Enterprise) Enable Causal Inference engine. |
| `INTEGRATION` | String | `"native"`, `"langchain"` | `"native"` | Integration context metadata. |

---

## 3. Domain Definition (DOMAIN)

The `DOMAIN` block encapsulates the context (variables) and the laws (constraints) of your system.

### 3.1 Variable Declarations (VARIABLES)

All variables referenced in constraints must be declared here to pass semantic validation. CSL supports three types of domains:

- **Range (Intervals)**: Defined by `start..end` (inclusive).
- **Set (Enums)**: Defined by explicit values `{ "A", "B" }`.
- **Primitive Types**: `Int`, `Nat`, `BOOLEAN`.

```csl
VARIABLES {
  amount: 0..100000            // Interval
  currency: {"USD", "EUR"}     // Set (String)
  risk_score: 0.0..1.0         // Interval (Float)
  is_verified: BOOLEAN         // Primitive
  kyc_level: Int               // Primitive
}
```

---

## 4. Constraints

Constraints are the core logic units. CSL-Core supports `STATE_CONSTRAINT` (invariants).

**Structure:**

```csl
STATE_CONSTRAINT rule_name {
  WHEN <condition_expression>
  THEN <action_expression>
}
```

### 4.1 Temporal Operators (WHEN Clause)

Defines when a rule applies.

| Operator | Description |
|----------|-------------|
| `WHEN` | Standard trigger. Applies if condition is true in current state. |
| `ALWAYS` | Invariant. Condition is effectively TRUE (applies to all states). |
| `BEFORE` | (Context-dependent) Pre-condition check. |
| `AFTER` | (Context-dependent) Post-condition check. |
| `EVENTUALLY` | (Liveness) Requires condition to be true in a future state. |

### 4.2 Modal Operators (THEN Clause)

Defines the obligation or prohibition.

| Operator | Usage | Logic Equivalent |
|----------|-------|------------------|
| `MUST BE` | `THEN status MUST BE "ACTIVE"` | `status == "ACTIVE"` |
| `MUST NOT BE` | `THEN risk MUST NOT BE "HIGH"` | `risk != "HIGH"` |
| `MAY BE` | `THEN status MAY BE "PENDING"` | (Permissive / No-op) |
| Comparisons | `THEN amount <= 1000` | `amount <= 1000` |

---

## 5. Expressions & Operators

CSL supports standard logic, arithmetic, and comparison operations within expressions.

### 5.1 Binary Operators

- **Logic**: `AND`, `OR`
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`

### 5.2 Unary Operators

- **Logic**: `NOT` (e.g., `NOT is_admin`)
- **Arithmetic**: `-` (e.g., `-amount`)

### 5.3 Built-in Functions

Only the following safe, side-effect-free functions are supported in CSL-Core:

- `len(x)`: Length of a string or list.
- `max(a, b)`: Maximum of two values.
- `min(a, b)`: Minimum of two values.
- `abs(x)`: Absolute value.

**Note**: Attempting to call other functions will result in a Compilation Error.

### 5.4 Member Access

Dot notation is supported for accessing nested JSON properties.

```csl
WHEN user.profile.age > 18
```

---

## 6. Lexical Structure

- **Comments**: Support C-style single line `//` and multi-line `/* ... */`.
- **Strings**: Double-quoted `"value"`.
- **Booleans**: Case-insensitive `TRUE`, `FALSE`.
- **Identifiers**: Alphanumeric, must start with a letter (`[a-zA-Z_][a-zA-Z0-9_]*`).

---

## 7. Example Policy

```csl
CONFIG {
  ENFORCEMENT_MODE: BLOCK
}

DOMAIN PaymentGuard {
  VARIABLES {
    action: {"TRANSFER", "REFUND"}
    amount: 0..50000
    user_tier: {"BASIC", "PREMIUM"}
  }

  STATE_CONSTRAINT limit_basic_tier {
    WHEN action == "TRANSFER" AND user_tier == "BASIC"
    THEN amount <= 1000
  }

  STATE_CONSTRAINT no_refund_abuse {
    WHEN action == "REFUND"
    THEN amount MUST NOT BE > 5000
  }
}
```
