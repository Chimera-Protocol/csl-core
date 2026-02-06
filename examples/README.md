# CSL-Core Examples

Learn CSL through progressive examples, from basic policies to production-ready governance systems.

## 🚀 Quick Start (60 seconds)

### Run All Examples
```bash
# From repository root
python3 examples/run_examples.py

# Or run specific policy
python3 examples/run_examples.py dao_treasury_guard
python3 examples/run_examples.py agent_tool_guard
python3 examples/run_examples.py banking
python3 examples/run_examples.py langchain_agent_demo
```

### Test Single Policy
```bash
# Verify policy compiles
cslcore verify examples/dao_treasury_guard.csl

# Test with specific input
cslcore simulate examples/dao_treasury_guard.csl \
  --input '{"transfer_amount": 50000, "total_balance": 1000000, "proposer_reputation": 500, "approval_count": 5, "proposal_age_hours": 48, "destination_type": "EXTERNAL", "action": "TRANSFER", "proposal_id": 1}'
```

---

## 📚 Examples by Difficulty

### 🟢 Beginner (5 minutes)

- `00_hello_world.csl` - Simplest possible policy
- `01_age_verification.csl` - Basic numeric comparisons

**Want to contribute?** See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

### 🟡 Intermediate (15 minutes)

#### [agent_tool_guard.csl](agent_tool_guard.csl)

**Domain:** AI Agent Tool Governance  
**Use Case:** Prevent AI agents from performing dangerous operations

**Key Features:**
- Role-based access control (ADMIN/USER/ANALYST)
- Tool permission enforcement
- PII protection for email operations
- Approval token requirements for transfers
- Hard-banned destructive operations

**Constraints:**
```csl
// Non-admin users cannot transfer funds
WHEN user_role == "USER" OR user_role == "ANALYST"
THEN tool MUST NOT BE "TRANSFER_FUNDS"

// PII cannot be sent externally
WHEN tool == "SEND_EMAIL" AND pii_present == "YES"
THEN recipient_domain MUST NOT BE "EXTERNAL"

// Secrets table is forbidden
WHEN tool == "QUERY_DB"
THEN db_table MUST NOT BE "SECRETS"
```

**Test It:**
```bash
python3 examples/run_examples.py agent_tool_guard

# Or manually
cslcore simulate agent_tool_guard.csl \
  --input-file json_files/agent_tool_guard_tests.json
```

**Learn More:** See [LangChain Integration](integration/) for AI agent examples

---

#### [chimera_banking_case_study.csl](chimera_banking_case_study.csl)

**Domain:** Banking Transaction Safety  
**Use Case:** Multi-factor risk management for financial transactions

**Key Features:**
- Sanctions enforcement (hard-ban specific countries)
- VIP tier system with progressive limits
- Risk-based transaction blocking
- Device trust requirements
- KYC level verification

**Constraints:**
```csl
// Hard sanctions
WHEN country == country
THEN country MUST NOT BE "NK"

// Progressive limits by VIP status
WHEN action == "TRANSFER" AND is_vip == "FALSE"
THEN amount <= 1000

WHEN action == "TRANSFER" AND is_vip == "TRUE"
THEN amount <= 10000

// Risk ceiling
WHEN action == "TRANSFER"
THEN risk_score <= 0.8

// Device trust for medium+ transfers
WHEN action == "TRANSFER" AND amount > 300
THEN device_trust >= 0.7
```

**Test It:**
```bash
python3 examples/run_examples.py banking
```

---

### 🔴 Advanced (30 minutes)

#### [dao_treasury_guard.csl](dao_treasury_guard.csl)

**Domain:** Web3 DAO Treasury Governance  
**Use Case:** Multi-sig protection for DAO fund transfers

**Key Features:**
- Catastrophic transfer protection (>10% of balance)
- Timelock requirements for large transfers
- Progressive approval thresholds by destination type
- Reputation-based access control
- Emergency action bypass with unanimous approval
- Bridge transfer extra security

**Architecture:**
```
TIER 1: Low Reputation (< 100)
  → INTERNAL only
  → Max 10K transfers

TIER 2: High Reputation (>= 100)
  → EXTERNAL allowed (5+ approvals if > 50K)
  → BRIDGE allowed (7+ approvals if > 10K)
  → All BRIDGE transfers need 24h timelock

TIER 3: Universal Rules
  → Catastrophic (>10% balance): 3+ approvals
  → Large transfers (>5% balance): 24h timelock
  → Emergency actions: 10+ approvals (bypass timelock)
```

**Constraints:**
```csl
// Low reputation restrictions
WHEN proposer_reputation < 100
THEN destination_type == "INTERNAL"

WHEN proposer_reputation < 100
THEN transfer_amount <= 10000

// High reputation external transfers
WHEN proposer_reputation >= 100 
     AND destination_type == "EXTERNAL" 
     AND transfer_amount > 50000
THEN approval_count >= 5

// Bridge always requires timelock
WHEN proposer_reputation >= 100 
     AND destination_type == "BRIDGE" 
     AND action != "EMERGENCY"
THEN proposal_age_hours >= 24

// Emergency bypass
WHEN action == "EMERGENCY"
THEN approval_count >= 10
```

**Test It:**
```bash
python3 examples/run_examples.py dao_treasury_guard
```

**Why This Example?**
- Real-world Web3 governance problem
- Demonstrates tiered permission system
- Shows emergency bypass patterns
- Z3-verified consistency (no contradictions)

---
## Integrations & Showcases
See how CSL-Core integrates with real-world frameworks.

###LangChain Agent Guard

File: [lancgchain_agent_demo.py](integrations/lancgchain_agent_demo.py)

A production-grade simulation of a AI Agent using langchain-core and rich for visualization. It demonstrates how to wrap tools and inject secure context.

** Scenarios Covered: **
- Prompt Injection Defense: User tries to trick the agent into sending PII externally.
- RBAC Enforcement: Standard user attempts restricted admin actions.
- Business Logic: Admin attempts to exceed transfer limits without approval.

---

## 🧪 Testing Examples

### Run All Tests
```bash
python3 examples/run_examples.py
```

**Output:**
```
╭──────────────────────────────────────╮
│ CSL-Core Examples Test Runner        │
╰──────────────────────────────────────╯

DAO_TREASURY_GUARD
✅ Policy compiled successfully

Test Results
┌────┬──────────────────────┬──────────┬────────┬────────┐
│ #  │ Test Case            │ Expected │ Result │ Status │
├────┼──────────────────────┼──────────┼────────┼────────┤
│ 1  │ Low rep: small...    │ ALLOW    │ ALLOW  │ ✅ PASS│
│ 2  │ High rep: large...   │ ALLOW    │ ALLOW  │ ✅ PASS│
...
└────┴──────────────────────┴──────────┴────────┴────────┘

🎉 All tests passed for dao_treasury_guard!
```

### Run Specific Policy
```bash
python3 examples/run_examples.py agent_tool_guard
```

### Show Detailed Failures
```bash
python3 examples/run_examples.py --details
```

### List Available Policies
```bash
python3 examples/run_examples.py --list
```

---

## 📖 Policy Pattern Library

Common patterns extracted from examples for reuse.

### Pattern 1: Role-Based Access Control (RBAC)

```csl
VARIABLES { user_role: String, operation: String }

CONSTRAINT admin_only {
    WHEN operation == "SENSITIVE_ACTION"
    THEN user_role MUST BE "ADMIN"
}
```

**Example:** `agent_tool_guard.csl` (lines 30-33)

---

### Pattern 2: PII Protection

```csl
VARIABLES { pii_present: String, destination: String }

CONSTRAINT no_external_pii {
    WHEN pii_present == "YES"
    THEN destination MUST NOT BE "EXTERNAL"
}
```

**Example:** `agent_tool_guard.csl` (lines 55-58)

---

### Pattern 3: Progressive Limits by Tier

```csl
VARIABLES { amount: 0..1000000, tier: String }

CONSTRAINT basic_tier_limit {
    WHEN tier == "BASIC"
    THEN amount <= 1000
}

CONSTRAINT premium_tier_limit {
    WHEN tier == "PREMIUM"
    THEN amount <= 50000
}
```

**Example:** `chimera_banking_case_study.csl` (lines 28-38)

---

### Pattern 4: Hard Sanctions (Fail-Closed)

```csl
VARIABLES { country: String }

CONSTRAINT sanctions {
    ALWAYS True
    THEN country MUST NOT BE "SANCTIONED_COUNTRY"
}
```

**Example:** `chimera_banking_case_study.csl` (lines 22-25)

---

### Pattern 5: Tiered Permissions (Guard Pattern)

```csl
VARIABLES { permission_level: 0..100, feature: String }

// LOW TIER: Block advanced features
CONSTRAINT low_tier_restriction {
    WHEN permission_level < 50
    THEN feature MUST NOT BE "ADVANCED"
}

// HIGH TIER: Advanced feature with requirements
CONSTRAINT high_tier_feature {
    WHEN permission_level >= 50 AND feature == "ADVANCED"
    THEN additional_requirements
}
```

**Example:** `dao_treasury_guard.csl` (lines 25-48)

---

### Pattern 6: Emergency Bypass

```csl
VARIABLES { action: String, approval_count: 0..100 }

// Normal rule with bypass
CONSTRAINT normal_with_bypass {
    WHEN condition AND action != "EMERGENCY"
    THEN requirement
}

// Emergency gate
CONSTRAINT emergency_gate {
    WHEN action == "EMERGENCY"
    THEN approval_count >= 10  // Higher threshold
}
```

**Example:** `dao_treasury_guard.csl` (lines 60-67)

---

## 🔍 Understanding Test Files

Test files in `json_files/` follow this structure:

```json
{
  "allow_cases": [
    {
      "name": "Human-readable description",
      "input": {
        "variable1": "value1",
        "variable2": 123
      }
    }
  ],
  "block_cases": [
    {
      "name": "Human-readable description",
      "input": {
        "variable1": "value1"
      },
      "expected_violation": "constraint_name"
    }
  ]
}
```

**Creating Your Own Tests:**
1. Match variable names to policy VARIABLES
2. Provide all required fields
3. For block cases, specify which constraint should trigger

---

## 🛠️ CLI Commands Cheatsheet

```bash
# Verify policy compiles and passes Z3
cslcore verify policy.csl

# Simulate single input
cslcore simulate policy.csl --input '{"key": "value"}'

# Batch test
cslcore simulate policy.csl --input-file tests.json

# Interactive REPL
cslcore repl policy.csl

# Dashboard mode (rich UI)
cslcore simulate policy.csl --input-file tests.json --dashboard

# Dry-run (don't block, just report)
cslcore simulate policy.csl --input '...' --dry-run

# JSON output for CI/CD
cslcore simulate policy.csl --input-file tests.json --json --quiet
```

---

## 🎯 Next Steps

### For Beginners
1. ✅ Run `python3 examples/run_examples.py` to see all tests
2. ✅ Read `agent_tool_guard.csl` - simplest intermediate example
3. ✅ Modify a constraint and see tests fail
4. ✅ Try `cslcore repl agent_tool_guard.csl` for interactive testing in examples/ folder

### For Advanced Users
1. ✅ Study `dao_treasury_guard.csl` tiered permission pattern
2. ✅ Create your own domain (healthcare, supply chain, etc.)
3. ✅ Integrate with LangChain
4. ✅ Submit your example as a PR!

### For Contributors
We need examples in these domains:
- 🏥 Healthcare (HIPAA compliance)
- 📦 Supply Chain (provenance tracking)
- 🎮 Gaming (anti-cheat, economy balance)
- 📱 Social Media (content moderation)
- 🏗️ Construction (safety protocols)

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## ❓ FAQ

**Q: Why did my policy fail Z3 verification?**  
A: Z3 found a logical contradiction. Read the error carefully - it shows which constraints conflict. Common issues:
- Two rules require opposite values for same variable
- Impossible conditions (e.g., `amount > 100 AND amount < 50`)

**Q: Can I skip Z3 verification?**  
A: Set `check_logical_consistency: false` in CONFIG, but NOT recommended for production.

**Q: How do I debug which constraint blocked my input?**  
A: Use `--dashboard` flag or check `result.violations` in Python API.

**Q: What's the difference between ALLOW and BLOCK tests?**  
A: ALLOW tests should pass all constraints. BLOCK tests should violate at least one.

**Q: Can I use CSL with other frameworks besides LangChain?**  
A: Yes! See `chimera_core/plugins/base.py` for creating custom integrations.

---

## 🤝 Contributing Examples

Good examples are:
- ✅ Self-contained (single `.csl` file)
- ✅ Real-world use case
- ✅ Clear comments explaining intent
- ✅ Comprehensive test coverage (both allow and block cases)
- ✅ Z3-verified (no logical contradictions)

Submit via Pull Request with:
1. Policy file (`examples/your_example.csl`)
2. Test file (`examples/json_files/your_example_tests.json`)
3. Update to this README

---

**Built with ❤️ by the Chimera Team**
