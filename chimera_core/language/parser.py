"""
CSL Parser - Text to AST Conversion

Parses Chimera Specification Language (CSL) text into Abstract Syntax Tree.

V0.1: Regex-based parser (simple but functional)
V1.0: ANTLR/Lark parser (robust, better error messages)
"""

import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .ast import (
    # Top-level
    Constitution,
    Domain,
    Constraint,
    VariableDeclaration,
    
    # Causal
    CausalGraph,
    CausalEdge,
    StructuralEquation,
    
    # Formal
    Invariant,
    LivenessProperty,
    
    # Constraint components
    ConditionClause,
    ActionClause,
    CausalProof,
    CounterfactualStatement,
    FormalProof,
    TLASpec,
    ModelCheckingResult,
    EnforcementClause,
    
    # Expressions
    Expression,
    Variable,
    Literal,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    MemberAccess,
    
    # Operators
    TemporalOperator,
    ModalOperator,
    ComparisonOperator,
    LogicalOperator,
    ArithmeticOperator,
)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ParseError(Exception):
    """Base exception for parsing errors"""
    
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Line {line}, Col {column}: {message}")


# ============================================================================
# TOKENIZER
# ============================================================================

@dataclass
class Token:
    """Lexical token"""
    type: str
    value: str
    line: int
    column: int


class Tokenizer:
    """
    Simple regex-based tokenizer for CSL.
    
    Token types:
    - KEYWORD: DOMAIN, CONSTRAINT, WHEN, THEN, etc.
    - IDENTIFIER: variable names
    - NUMBER: integers and floats
    - STRING: quoted strings
    - OPERATOR: →, ==, !=, <, >, etc.
    - DELIMITER: {, }, (, ), ,, ;
    - COMMENT: // or /* */
    """
    
    # Keywords
    KEYWORDS = {
        
    # Config Keys
    'CONFIG', 'ENFORCEMENT_MODE', 'BLOCK', 'WARN', 'LOG', 
    'CHECK_LOGICAL_CONSISTENCY',  # Z3 (Core)
    'ENABLE_FORMAL_VERIFICATION', # TLA+ (Enterprise)
    'ENABLE_CAUSAL_INFERENCE', 'OPTIMIZE_VERIFICATION_SCOPE',
    'INTEGRATION',
    
    # Domain Elements
    'DOMAIN', 'VARIABLES', 'CAUSAL_GRAPH', 'STRUCTURAL_EQUATIONS', 'INVARIANTS', 'LIVENESS', 
    'STATE_CONSTRAINT', 'NEXT_CONSTRAINT',
    
    # Temporal Logic
    'WHEN', 'BEFORE', 'AFTER', 'ALWAYS', 'EVENTUALLY',
    'THEN', 'MUST', 'NOT', 'MAY', 'BE',
    
    # Proof & Causal
    'CAUSAL_PROOF', 'MECHANISM', 'COUNTERFACTUAL', 'IDENTIFICATION',
    'METHOD', 'ADJUSTMENT', 
    'FORMAL_PROOF', 'TLA_SPEC', 'MODEL_CHECKING',

    # Enforcement & Logic
    'ENFORCEMENT', 'DEFAULT_ACTION', 'NOTIFY', 'OVERRIDE_REQUIRES',
    'IF', 'AND', 'OR', 'DATA', 'TRUE', 'FALSE'
    }

    
    # Token patterns
    PATTERNS = [
        ('COMMENT_MULTI', r'/\*.*?\*/'),
        ('COMMENT_SINGLE', r'//.*?$'),
        ('STRING', r'"([^"\\]|\\.)*"'),
        ('BI_ARROW', r'<->|<=>'),
        ('ARROW', r'→|->'),
        ('NUMBER', r'\d+(\.\d+)?'),
        ('OPERATOR', r'==|!=|<=|>=|<|>|\+|-|\*|/|%|='),
        ('DELIMITER', r'[{}()\[\]:,;|\.]'),
        ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('WHITESPACE', r'\s+'),
    ]
    
    def __init__(self, text: str):
        self.text = text
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
    def tokenize(self) -> List[Token]:
        """Tokenize entire text"""
        while self.position < len(self.text):
            self._next_token()
        return self.tokens
    
    def _next_token(self):
        """Extract next token"""
        if self.position >= len(self.text):
            return
        
        # Try each pattern
        for token_type, pattern in self.PATTERNS:
            regex = re.compile(pattern, re.MULTILINE)
            match = regex.match(self.text, self.position)
            
            if match:
                value = match.group(0)
                
                # Skip comments and whitespace
                if token_type in ('COMMENT_MULTI', 'COMMENT_SINGLE', 'WHITESPACE'):
                    self._advance(len(value))
                    return
                
                # Check if identifier is actually a keyword
                if token_type == 'IDENTIFIER' and value.upper() in self.KEYWORDS:
                    token_type = 'KEYWORD'
                    value = value.upper()  # Normalize keywords
                
                # Create token
                token = Token(
                    type=token_type,
                    value=value,
                    line=self.line,
                    column=self.column
                )
                self.tokens.append(token)
                
                self._advance(len(value))
                return
        
        # No pattern matched - unexpected character
        raise ParseError(
            f"Unexpected character: '{self.text[self.position]}'",
            self.line,
            self.column
        )
    
    def _advance(self, count: int):
        """Advance position and update line/column"""
        for _ in range(count):
            if self.position < len(self.text):
                if self.text[self.position] == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.position += 1


# ============================================================================
# PARSER
# ============================================================================

class CSLParser:
    """
    Recursive descent parser for CSL.
    
    Grammar (simplified):
        constitution ::= domain constraint+
        domain ::= "DOMAIN" id "{" domain_body "}"
        constraint ::= "CONSTRAINT" id "{" constraint_body "}"
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.position = 0
        self.current_token: Optional[Token] = None
        
    def parse(self, text: str) -> Constitution:
        """
        Parse CSL text into Constitution AST.
        
        Args:
            text: CSL source code
            
        Returns:
            Constitution AST
            
        Raises:
            ParseError: On syntax error
        """
        # Tokenize
        tokenizer = Tokenizer(text)
        self.tokens = tokenizer.tokenize()
        self.position = 0
        self.current_token = self.tokens[0] if self.tokens else None
        
        try:
            # Parse constitution
            return self._parse_constitution()
        except RecursionError:
            # ROBUSTNESS FIX: Catch stack overflow and convert to safe ParseError
            # Fix: Line/Column bilgisini current_token üzerinden alıyoruz.
            line = self.current_token.line if self.current_token else 0
            column = self.current_token.column if self.current_token else 0
            
            raise ParseError(
                "Input too complex: Maximum nesting depth exceeded (Stack Overflow Protection)", 
                line, 
                column
            )
    
    def parse_file(self, filepath: str) -> Constitution:
        """Parse CSL file with strict UTF-8 encoding"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.parse(text)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _advance(self):
        """Move to next token"""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
    
    def _peek(self, offset: int = 1) -> Optional[Token]:
        """Look ahead at token"""
        pos = self.position + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def _expect(self, token_type: str, value: Optional[str] = None) -> Token:
        """
        Expect specific token type/value.
        
        Raises ParseError if not matched.
        """
        if not self.current_token:
            raise ParseError("Unexpected end of file")
        
        if self.current_token.type != token_type:
            raise ParseError(
                f"Expected {token_type}, got {self.current_token.type}",
                self.current_token.line,
                self.current_token.column
            )
        
        if value and self.current_token.value != value:
            raise ParseError(
                f"Expected '{value}', got '{self.current_token.value}'",
                self.current_token.line,
                self.current_token.column
            )
        
        token = self.current_token
        self._advance()
        return token
    
    def _match(self, token_type: str, value: Optional[str] = None) -> bool:
        """Check if current token matches"""
        if not self.current_token:
            return False
        
        if self.current_token.type != token_type:
            return False
        
        if value and self.current_token.value != value:
            return False
        
        return True
    
    def _consume_if(self, token_type: str, value: Optional[str] = None) -> bool:
        """Consume token if it matches"""
        if self._match(token_type, value):
            self._advance()
            return True
        return False
    
    # ========================================================================
    # PARSING METHODS
    # ========================================================================
    
    def _parse_constitution(self) -> Constitution:
        """Parse top-level constitution"""
        
        # 1. Parse Optional CONFIG block
        config = None
        if self._match('KEYWORD', 'DOMAIN'):
            pass # No config, proceed to domain
        elif self._match('KEYWORD', 'CONFIG'):
            config = self._parse_config()
        
        # DOMAIN header
        self._expect('KEYWORD', 'DOMAIN')
        name_token = self._expect('IDENTIFIER')
        name = name_token.value
        self._expect('DELIMITER', '{')
    
        # Domain contents
        variable_declarations = []
        causal_graph = None
        structural_equations = []
        invariants = []
        liveness_properties = []
        constraints = []
    
        while self.current_token and not self._match('DELIMITER', '}'):
            
            if self._match('KEYWORD', 'VARIABLES'):
                variable_declarations = self._parse_variable_declarations()
                continue
    
            if self._match('KEYWORD', 'CAUSAL_GRAPH'):
                causal_graph = self._parse_causal_graph()
                continue
    
            if self._match('KEYWORD', 'STRUCTURAL_EQUATIONS'):
                structural_equations = self._parse_structural_equations()
                continue
    
            if self._match('KEYWORD', 'INVARIANTS'):
                invariants = self._parse_invariants()
                continue
    
            if self._match('KEYWORD', 'LIVENESS'):
                liveness_properties = self._parse_liveness()
                continue
    
           
            if self._match('KEYWORD', 'STATE_CONSTRAINT') or self._match('KEYWORD', 'NEXT_CONSTRAINT'):
                constraints.append(self._parse_constraint())
                continue
    
            tok = self.current_token
            raise ParseError(
                f"Unexpected token in DOMAIN: {tok.type}({tok.value})",
                tok.line,
                tok.column
            )
    
        # close DOMAIN
        self._expect('DELIMITER', '}')
    
        domain = Domain(
            name=name,
            causal_graph=causal_graph,
            structural_equations=structural_equations,
            invariants=invariants,
            liveness_properties=liveness_properties,
            location=(name_token.line, name_token.column),
            variable_declarations=variable_declarations
        )
    
        while self.current_token:
            if self._match('KEYWORD', 'STATE_CONSTRAINT') or \
               self._match('KEYWORD', 'NEXT_CONSTRAINT') or \
               self._match('KEYWORD', 'CONSTRAINT'):
                constraints.append(self._parse_constraint())
                continue
            break  
    
        return Constitution(
            domain=domain, 
            constraints=constraints, 
            causal_graph=causal_graph,
            config=config,
        )
    
    def _parse_config(self):
        """
        Parse configuration block.
        
        CONFIG {
            enforcement_mode: WARN
            check_logical_consistency: true   // Z3 (Core - Default: True)
            enable_formal_verification: false // TLA+ (Enterprise - Default: False)
            enable_causal_inference: false    // Causal (Default: False)
        }
        """
        self._expect('KEYWORD', 'CONFIG')
        self._expect('DELIMITER', '{')
        
        from .ast import Configuration, EnforcementMode
        # Default:
        # check_logical_consistency=True
        # enable_formal_verification=False
        # enable_causal_inference=False 
        config = Configuration()
        
        while not self._match('DELIMITER', '}'):
            # 1. Enforcement Mode (BLOCK/WARN/LOG)
            if self._match('KEYWORD', 'ENFORCEMENT_MODE'):
                self._advance(); self._expect('DELIMITER', ':')
                
                if self._match('KEYWORD', 'BLOCK'):
                    config.enforcement_mode = EnforcementMode.BLOCK; self._advance()
                elif self._match('KEYWORD', 'WARN'):
                    config.enforcement_mode = EnforcementMode.WARN; self._advance()
                elif self._match('KEYWORD', 'LOG'):
                    config.enforcement_mode = EnforcementMode.LOG; self._advance()
                else:
                    raise ParseError(f"Expected BLOCK, WARN, LOG", self.current_token.line, self.current_token.column)

            # 2. Logic Consistency (Z3 - Core)
            elif self._match('KEYWORD', 'CHECK_LOGICAL_CONSISTENCY'):
                self._advance(); self._expect('DELIMITER', ':')
                config.check_logical_consistency = self._parse_boolean()

            # 3. Formal Verification (TLA+ - Enterprise) - İSİM KORUNDU
            elif self._match('KEYWORD', 'ENABLE_FORMAL_VERIFICATION'):
                self._advance(); self._expect('DELIMITER', ':')
                config.enable_formal_verification = self._parse_boolean()

            # 4. Integration
            elif self._match('KEYWORD', 'INTEGRATION'):
                self._advance(); self._expect('DELIMITER', ':')
                val = self._expect('STRING').value.strip('"')
                config.integration = val

            # 5. Causal Inference
            elif self._match('KEYWORD', 'ENABLE_CAUSAL_INFERENCE'):
                self._advance(); self._expect('DELIMITER', ':')
                config.enable_causal_inference = self._parse_boolean()
                
            elif self._match('KEYWORD', 'OPTIMIZE_VERIFICATION_SCOPE'):
                self._advance(); self._expect('DELIMITER', ':')
                config.optimize_verification_scope = self._parse_boolean()
            
            else:
                # Forward Compatibility
                self._advance()
                if self._match('DELIMITER', ':'): 
                    self._advance(); self._parse_expression()
        
        self._expect('DELIMITER', '}')
        return config

    def _parse_boolean(self) -> bool:
        """Helper to parse explicit boolean values"""
        if self._match('KEYWORD', 'TRUE') or (self._match('IDENTIFIER') and self.current_token.value.lower() == 'true'):
            self._advance()
            return True
        if self._match('KEYWORD', 'FALSE') or (self._match('IDENTIFIER') and self.current_token.value.lower() == 'false'):
            self._advance()
            return False
        
        # Fallback for literals parsed as numbers (0/1) or unexpected
        if self._match('NUMBER'):
            val = self.current_token.value
            self._advance()
            return float(val) > 0
            
        raise ParseError(f"Expected boolean (true/false), got {self.current_token.value}", 
                        self.current_token.line, self.current_token.column)
    
    def _parse_domain(self) -> Domain:
        """
        Parse domain declaration.
        
        DOMAIN name {
            CAUSAL_GRAPH { ... }
            STRUCTURAL_EQUATIONS { ... }
            INVARIANTS { ... }
            LIVENESS { ... }
        }
        """
        self._expect('KEYWORD', 'DOMAIN')
        name_token = self._expect('IDENTIFIER')
        name = name_token.value
        
        self._expect('DELIMITER', '{')
        
        # Parse domain body
        causal_graph = None
        structural_equations = []
        invariants = []
        liveness_properties = []
        variable_declarations = []
        
        while not self._match('DELIMITER', '}'):
            if self._match('KEYWORD', 'VARIABLES'):
                variable_declarations = self._parse_variable_declarations()
            elif self._match('KEYWORD', 'CAUSAL_GRAPH'):
                causal_graph = self._parse_causal_graph()
            elif self._match('KEYWORD', 'STRUCTURAL_EQUATIONS'):
                structural_equations = self._parse_structural_equations()
            elif self._match('KEYWORD', 'INVARIANTS'):
                invariants = self._parse_invariants()
            elif self._match('KEYWORD', 'LIVENESS'):
                liveness_properties = self._parse_liveness()
            else:
                # Skip unknown block
                self._advance()
        
        self._expect('DELIMITER', '}')
        
        return Domain(
            name=name,
            causal_graph=causal_graph,
            structural_equations=structural_equations,
            invariants=invariants,
            liveness_properties=liveness_properties,
            location=(name_token.line, name_token.column),
            variable_declarations=variable_declarations
        )
    
    def _parse_causal_graph(self) -> CausalGraph:
        """
        Parse causal graph.
        
        CAUSAL_GRAPH {
            A → B ("mechanism")
            B → C
        }
        """
        self._expect('KEYWORD', 'CAUSAL_GRAPH')
        self._expect('DELIMITER', '{')
        
        edges = []
        while not self._match('DELIMITER', '}'):
            # Parse edge
            source_token = self._expect('IDENTIFIER')
            source = source_token.value
            
            # Check edge type
            edge_type = "directed"
            if self._match('BI_ARROW'):
                edge_type = "bidirected"
                self._advance()
            else:
                self._expect('ARROW')
            
            target_token = self._expect('IDENTIFIER')
            target = target_token.value
            
            # Optional mechanism
            mechanism = None
            if self._match('DELIMITER', '('):
                self._advance()
                mech_token = self._expect('STRING')
                mechanism = mech_token.value.strip('"')
                self._expect('DELIMITER', ')')
            
            edges.append(CausalEdge(
                source=source,
                target=target,
                mechanism=mechanism,
                edge_type=edge_type,
                location=(source_token.line, source_token.column)
            ))
        
        self._expect('DELIMITER', '}')
        
        return CausalGraph(edges=edges)
    
    def _parse_variable_declarations(self) -> List[VariableDeclaration]:
        """
        Parse VARIABLES block.
        
        Syntax:
            VARIABLES {
                price: 0..100000
                action: {"BUY", "SELL", "HOLD"}
                balance: Int
            }
        """
        self._expect('KEYWORD', 'VARIABLES')
        self._expect('DELIMITER', '{')
        
        declarations = []
        
        while not self._match('DELIMITER', '}'):
            # Variable name
            var_name = self._expect('IDENTIFIER').value
            
            # Colon
            self._expect('DELIMITER', ':')
            
            # Domain specification
            domain = self._parse_domain_spec()
            
            declarations.append(VariableDeclaration(
                name=var_name,
                domain=domain
            ))
            
            # Optional comma
            self._consume_if('DELIMITER', ',')
        
        self._expect('DELIMITER', '}')
        
        return declarations
    
    def _parse_domain_spec(self) -> str:
        """
        Parse domain specification.
        
        Examples:
            0..100000
            {"BUY", "SELL", "HOLD"}
            Nat
            Int
            BOOLEAN
        """
        # Type name (Nat, Int, BOOLEAN, etc.)
        if self._match('IDENTIFIER'):
            val = self.current_token.value
            self._advance() 
            return val
        
        # Range: 0..100
        if self._match('NUMBER'):
            start = self.current_token.value
            self._advance()
            
            if self._match('DELIMITER', '.'):
                self._advance()
                if self._match('DELIMITER', '.'):
                    self._advance()
                    end = self._expect('NUMBER').value
                    return f"{start}..{end}"
            
            # Single number (treat as 0..n)
            return f"0..{start}"
        
        # Set: {"A", "B", "C"}
        if self._match('DELIMITER', '{'):
            self._advance()
            
            elements = []
            while not self._match('DELIMITER', '}'):
                if self._match('STRING'):
                    elements.append(self.current_token.value)
                    self._advance()
                elif self._match('IDENTIFIER'):
                    elements.append(f'"{self.current_token.value}"')
                    self._advance()
                elif self._match('NUMBER'):
                    elements.append(self.current_token.value)
                    self._advance()
                
                self._consume_if('DELIMITER', ',')
            
            self._expect('DELIMITER', '}')
            
            # Format as TLA+ set
            return "{" + ", ".join(elements) + "}"
        
        # Fallback
        return "DOMAIN"
    
    def _parse_structural_equations(self) -> List[StructuralEquation]:
        """
        Parse structural equations.
        
        STRUCTURAL_EQUATIONS {
            volatility = std(price_change, window=24h)
            risk = position * volatility
        }
        """
        self._expect('KEYWORD', 'STRUCTURAL_EQUATIONS')
        self._expect('DELIMITER', '{')
        
        equations = []
        while not self._match('DELIMITER', '}'):
            # Parse: variable = expression
            var_token = self._expect('IDENTIFIER')
            var_name = var_token.value
            
            self._expect('OPERATOR', '=')
            
            expr = self._parse_expression()
            
            equations.append(StructuralEquation(
                variable=var_name,
                expression=expr,
                location=(var_token.line, var_token.column)
            ))
        
        self._expect('DELIMITER', '}')
        
        return equations
    
    def _parse_invariants(self) -> List[Invariant]:
        """
        Parse invariants.
        
        INVARIANTS {
            position_bounds: position >= 0
            solvency: portfolio_value > 0
        }
        """
        self._expect('KEYWORD', 'INVARIANTS')
        self._expect('DELIMITER', '{')
        
        invariants = []
        while not self._match('DELIMITER', '}'):
            # Parse: name: formula
            name_token = self._expect('IDENTIFIER')
            name = name_token.value
            
            self._expect('DELIMITER', ':')
            
            formula = self._parse_expression()
            
            invariants.append(Invariant(
                name=name,
                formula=formula,
                location=(name_token.line, name_token.column)
            ))
        
        self._expect('DELIMITER', '}')
        
        return invariants
    
    def _parse_liveness(self) -> List[LivenessProperty]:
        """Parse liveness properties (similar to invariants)"""
        self._expect('KEYWORD', 'LIVENESS')
        self._expect('DELIMITER', '{')
        
        properties = []
        while not self._match('DELIMITER', '}'):
            name_token = self._expect('IDENTIFIER')
            name = name_token.value
            
            self._expect('DELIMITER', ':')
            
            formula = self._parse_expression()
            
            properties.append(LivenessProperty(
                name=name,
                formula=formula,
                location=(name_token.line, name_token.column)
            ))
        
        self._expect('DELIMITER', '}')
        
        return properties
    
    def _parse_constraint(self) -> Constraint:
        """
        Parse constraint.
        
        CONSTRAINT name {
            WHEN condition
            THEN action
            CAUSAL_PROOF { ... }
            FORMAL_PROOF { ... }
            ENFORCEMENT { ... }
        }
        """
        from .ast import ConstraintType # Import the new enum
        
        # 1. Tipi Belirle (STATE mi NEXT mi?)
        if self._match('KEYWORD', 'STATE_CONSTRAINT'):
            constraint_type = ConstraintType.STATE
            self._advance()
        else:
            constraint_type = ConstraintType.NEXT
            self._advance()
            
        name_token = self._expect('IDENTIFIER')
        name = name_token.value
        
        self._expect('DELIMITER', '{')
        
        # Parse constraint body
        condition = None
        action = None
        causal_proof = None
        formal_proof = None
        enforcement = None
        
        while not self._match('DELIMITER', '}'):
            if self._match('KEYWORD', 'WHEN') or self._match('KEYWORD', 'BEFORE') or \
               self._match('KEYWORD', 'AFTER') or self._match('KEYWORD', 'ALWAYS') or \
               self._match('KEYWORD', 'EVENTUALLY'):
                condition = self._parse_condition_clause()
            elif self._match('KEYWORD', 'THEN'):
                action = self._parse_action_clause()
            elif self._match('KEYWORD', 'CAUSAL_PROOF'):
                causal_proof = self._parse_causal_proof()
            elif self._match('KEYWORD', 'FORMAL_PROOF'):
                formal_proof = self._parse_formal_proof()
            elif self._match('KEYWORD', 'ENFORCEMENT'):
                enforcement = self._parse_enforcement()
            else:
                # Skip unknown
                self._advance()
        
        self._expect('DELIMITER', '}')
        
        if not condition or not action:
            raise ParseError(
                f"Constraint {name} must have WHEN and THEN clauses",
                name_token.line,
                name_token.column
            )
        
        return Constraint(
            name=name,
            condition=condition,
            constraint_type=constraint_type,
            action=action,
            causal_proof=causal_proof,
            formal_proof=formal_proof,
            enforcement=enforcement,
            location=(name_token.line, name_token.column)
        )
    
    def _parse_condition_clause(self) -> ConditionClause:
        """
        Parse condition clause.
        
        WHEN price_change < -0.05
        BEFORE action.type == "spend"
        """
        # Get temporal operator
        temporal_op_map = {
            'WHEN': TemporalOperator.WHEN,
            'BEFORE': TemporalOperator.BEFORE,
            'AFTER': TemporalOperator.AFTER,
            'ALWAYS': TemporalOperator.ALWAYS,
            'EVENTUALLY': TemporalOperator.EVENTUALLY,
        }
        
        op_token = self.current_token
        op_value = op_token.value
        self._advance()
        
        temporal_op = temporal_op_map.get(op_value)
        if not temporal_op:
            raise ParseError(
                f"Unknown temporal operator: {op_value}",
                op_token.line,
                op_token.column
            )
        
        # Parse condition expression
        condition_expr = self._parse_expression()
        
        return ConditionClause(
            temporal_operator=temporal_op,
            condition=condition_expr,
            location=(op_token.line, op_token.column)
        )
    
    def _parse_action_clause(self) -> ActionClause:
        """
        Parse action clause.
        THEN action MUST NOT BE "SELL"
        THEN position <= 1000
        THEN position MUST BE <= 1000
        """
        self._expect('KEYWORD', 'THEN')
        
        # Variable name
        var_token = self._expect('IDENTIFIER')
        var_name = var_token.value
        
        modal_op = None
        negation = False
        
        # 1. Keyword (MUST / MAY)
        if self._match('KEYWORD', 'MUST'):
            self._advance()
            if self._match('KEYWORD', 'NOT'):
                self._advance()
                negation = True
            self._expect('KEYWORD', 'BE')
            
            modal_op = ModalOperator.MUST_NOT_BE if negation else ModalOperator.MUST_BE
            
        elif self._match('KEYWORD', 'MAY'):
            self._advance()
            self._expect('KEYWORD', 'BE')
            modal_op = ModalOperator.MAY_BE
            
        # 2. Operator (THEN x <= 10 gibi)
        elif self._match('OPERATOR'):
            op_token = self.current_token
            # op_map control
            op_map = {
                '==': ModalOperator.EQ,
                '!=': ModalOperator.NEQ,
                '<': ModalOperator.LT,
                '>': ModalOperator.GT,
                '<=': ModalOperator.LTE,
                '>=': ModalOperator.GTE,
            }
            
            if op_token.value in op_map:
                modal_op = op_map[op_token.value]
                self._advance() 
            else:

                 modal_op = op_token.value
                 self._advance()

        else:
            raise ParseError(
                f"Expected modal operator (MUST, MAY) or comparison, got {self.current_token.value}",
                self.current_token.line,
                self.current_token.column
            )
        
        if modal_op in (ModalOperator.MUST_BE, ModalOperator.MAY_BE):
            if self._match('OPERATOR'):
                op_token = self.current_token
                op_map = {
                    '==': ModalOperator.EQ, '!=': ModalOperator.NEQ,
                    '<': ModalOperator.LT, '>': ModalOperator.GT,
                    '<=': ModalOperator.LTE, '>=': ModalOperator.GTE,
                }
                
                if op_token.value in op_map:
                    modal_op = op_map[op_token.value] 
                    self._advance() 

        # Value (can be comparison or literal)
        value_expr = self._parse_expression()
        
        return ActionClause(
            variable=var_name,
            modal_operator=modal_op,
            value=value_expr,
            location=(var_token.line, var_token.column)
        )
    
    def _parse_causal_proof(self) -> CausalProof:
        """
        Parse causal proof.
        
        CAUSAL_PROOF {
            MECHANISM: A → B → C
            COUNTERFACTUAL { ... }
        }
        """
        self._expect('KEYWORD', 'CAUSAL_PROOF')
        self._expect('DELIMITER', '{')
        
        mechanism = []
        counterfactuals = []
        identification = None
        confidence = 1.0
        
        while not self._match('DELIMITER', '}'):
            if self._match('KEYWORD', 'MECHANISM'):
                self._advance()
                self._expect('DELIMITER', ':')
                
                # Parse chain: A → B → C
                mechanism = []
                mechanism.append(self._expect('IDENTIFIER').value)
                
                while self._match('ARROW'):
                    self._advance()
                    mechanism.append(self._expect('IDENTIFIER').value)
            
            elif self._match('KEYWORD', 'COUNTERFACTUAL'):
                self._advance()
                self._expect('DELIMITER', '{')
                
                # Parse counterfactual statements
                while not self._match('DELIMITER', '}'):
                    if self._match('KEYWORD', 'IF'):
                        cf = self._parse_counterfactual_statement()
                        counterfactuals.append(cf)
                    else:
                        self._advance()
                
                self._expect('DELIMITER', '}')
            
            elif self._match('KEYWORD', 'IDENTIFICATION'):
                # ADMG/Identification Parsing
                self._advance()
                self._expect('DELIMITER', '{')
                
                method = "BACKDOOR"
                variables = []
                
                while not self._match('DELIMITER', '}'):
                    if self._match('KEYWORD', 'METHOD'):
                        self._advance()
                        self._expect('DELIMITER', ':')
                        method = self._expect('IDENTIFIER').value
                    
                    elif self._match('KEYWORD', 'ADJUSTMENT') or self._match('KEYWORD', 'VARIABLES'):
                        self._advance()
                        self._expect('DELIMITER', ':')
                        self._expect('DELIMITER', '{')
                        while not self._match('DELIMITER', '}'):
                            variables.append(self._expect('IDENTIFIER').value)
                            self._consume_if('DELIMITER', ',')
                        self._expect('DELIMITER', '}')
                    else:
                        self._advance()
                
                self._expect('DELIMITER', '}')
                from .ast import IdentificationSpec # Ensure import available
                identification = IdentificationSpec(method=method, variables=variables)
            
            else:
                self._advance()
        
        self._expect('DELIMITER', '}')
        
        return CausalProof(
            mechanism=mechanism,
            counterfactuals=counterfactuals,
            identification=identification,
            confidence=confidence
        )
    
    def _parse_counterfactual_statement(self) -> CounterfactualStatement:
        """
        Parse counterfactual: IF intervention | condition THEN outcome
        Example: IF action == "SELL" | market_crash == true THEN regret > 0.5
        """
        self._expect('KEYWORD', 'IF')
        

        intervention = self._parse_causal_dict_expression()
        
        condition = {}
        if self._match('DELIMITER', '|'):
            self._advance() 
            condition = self._parse_causal_dict_expression()
        
        self._expect('KEYWORD', 'THEN')

        outcome = self._parse_causal_dict_expression()
        

        return CounterfactualStatement(
            intervention=intervention,
            condition=condition,
            outcome=outcome
        )

    def _parse_causal_dict_expression(self) -> Dict[str, Any]:
        """
        Helper: CSL ifadelerini (x == 1 AND y == 2) Causal Engine sözlüğüne {x:1, y:2} çevirir.
        Recursively traverses the Expression AST to extract assignments.
        """
        result = {}
        
        expr = self._parse_expression()
        
        from .ast import BinaryOp, Variable, Literal, LogicalOperator, ComparisonOperator
        
        def _extract_assignments(node):
            """Recursive extractor for logic trees"""
            if isinstance(node, BinaryOp):
                # 1
                if node.operator == LogicalOperator.AND:
                    _extract_assignments(node.left)
                    _extract_assignments(node.right)
                    return

                # 2
                is_assignment = node.operator in (ComparisonOperator.EQ, '==', '=')
                if is_assignment:
                    if isinstance(node.left, Variable) and isinstance(node.right, Literal):
                        result[node.left.name] = node.right.value
                    elif isinstance(node.left, Variable) and isinstance(node.right, Variable):
                         result[node.left.name] = node.right.name
            
            # 3
            pass

        _extract_assignments(expr)
        
        if not result:
            # Fallback for simple "var" boolean checks (if needed)
            pass
            
        return result
    
    def _parse_formal_proof(self) -> FormalProof:
        """
        Parse formal proof.
        
        FORMAL_PROOF {
            TLA_SPEC {
                NoSell == [](...) 
            }
            MODEL_CHECKING { ... }
        }
        """
        self._expect('KEYWORD', 'FORMAL_PROOF')
        self._expect('DELIMITER', '{')
        
        tla_specs = []
        model_checking = None
        
        while not self._match('DELIMITER', '}'):
            if self._match('KEYWORD', 'TLA_SPEC'):
                self._advance()
                self._expect('DELIMITER', '{')
                
                while not self._match('DELIMITER', '}'):
                    if self._match('IDENTIFIER'):
                        prop_name = self._expect('IDENTIFIER').value
                        
                        if self._match('OPERATOR', '=='):
                            self._advance()
                        else:
                            self._expect('OPERATOR', '=')

                        formula_parts = []
                        while not self._match('DELIMITER', '}'):
                            if self.current_token.type == 'IDENTIFIER':
                                next_token = self._peek()
                                if next_token and next_token.value in ('==', '='):
                                    break
                            
                            formula_parts.append(self.current_token.value)
                            self._advance()
                        
                        tla_specs.append(TLASpec(
                            property_name=prop_name,
                            formula=' '.join(formula_parts)
                        ))
                    else:
                        self._advance()
                
                self._expect('DELIMITER', '}')
            
            elif self._match('KEYWORD', 'MODEL_CHECKING'):
                # Skip for now
                self._skip_block()
            
            else:
                self._advance()
        
        self._expect('DELIMITER', '}')
        
        return FormalProof(
            tla_spec=tla_specs,
            model_checking=model_checking
        )
    
    def _parse_enforcement(self) -> EnforcementClause:
        """
        Parse enforcement clause.
        
        ENFORCEMENT {
            DEFAULT_ACTION: HOLD
            NOTIFY: risk_manager
        }
        """
        self._expect('KEYWORD', 'ENFORCEMENT')
        self._expect('DELIMITER', '{')
        
        default_action = None
        notify = None
        override_requires = None
        
        while not self._match('DELIMITER', '}'):
            if self._match('KEYWORD', 'DEFAULT_ACTION'):
                self._advance()
                self._expect('DELIMITER', ':')
                default_action = self._expect('IDENTIFIER').value
            
            elif self._match('KEYWORD', 'NOTIFY'):
                self._advance()
                self._expect('DELIMITER', ':')
                notify = self._expect('IDENTIFIER').value
            
            elif self._match('KEYWORD', 'OVERRIDE_REQUIRES'):
                self._advance()
                self._expect('DELIMITER', ':')
                override_requires = self._parse_expression()
            
            else:
                self._advance()
        
        self._expect('DELIMITER', '}')
        
        return EnforcementClause(
            default_action=default_action or "HOLD",
            notify=notify,
            override_requires=override_requires
        )
    
    def _skip_block(self):
        """Skip entire block (for unimplemented features)"""
        depth = 0
        
        while self.current_token:
            if self._match('DELIMITER', '{'):
                depth += 1
            elif self._match('DELIMITER', '}'):
                if depth == 0:
                    return
                depth -= 1
            
            self._advance()
    
    # ========================================================================
    # EXPRESSION PARSING
    # ========================================================================
    
    def _parse_expression(self) -> Expression:
        """Parse expression (recursive descent)"""
        return self._parse_logical_or()
    
    def _parse_logical_or(self) -> Expression:
        """Parse OR expressions"""
        left = self._parse_logical_and()
        
        while self._match('KEYWORD', 'OR'):
            self._advance()
            right = self._parse_logical_and()
            left = BinaryOp(left=left, operator=LogicalOperator.OR, right=right)
        
        return left
    
    def _parse_logical_and(self) -> Expression:
        """Parse AND expressions"""
        left = self._parse_comparison()
        
        while self._match('KEYWORD', 'AND'):
            self._advance()
            right = self._parse_comparison()
            left = BinaryOp(left=left, operator=LogicalOperator.AND, right=right)
        
        return left
    
    def _parse_comparison(self) -> Expression:
        """Parse comparison expressions"""
        left = self._parse_additive()
        
        # Check for comparison operators
        comp_ops = {
            '==': ComparisonOperator.EQ,
            '!=': ComparisonOperator.NEQ,
            '<': ComparisonOperator.LT,
            '>': ComparisonOperator.GT,
            '<=': ComparisonOperator.LTE,
            '>=': ComparisonOperator.GTE,
        }
        
        if self._match('OPERATOR'):
            op_value = self.current_token.value
            if op_value in comp_ops:
                self._advance()
                right = self._parse_additive()
                return BinaryOp(
                    left=left,
                    operator=comp_ops[op_value],
                    right=right
                )
        
        return left
    
    def _parse_additive(self) -> Expression:
        """Parse addition/subtraction"""
        left = self._parse_multiplicative()
        
        while self._match('OPERATOR') and self.current_token.value in ('+', '-'):
            op_value = self.current_token.value
            self._advance()
            right = self._parse_multiplicative()
            
            op = ArithmeticOperator.ADD if op_value == '+' else ArithmeticOperator.SUB
            left = BinaryOp(left=left, operator=op, right=right)
        
        return left
    
    def _parse_multiplicative(self) -> Expression:
        """Parse multiplication/division"""
        left = self._parse_unary()
        
        while self._match('OPERATOR') and self.current_token.value in ('*', '/', '%'):
            op_value = self.current_token.value
            self._advance()
            right = self._parse_unary()
            
            op_map = {
                '*': ArithmeticOperator.MUL,
                '/': ArithmeticOperator.DIV,
                '%': ArithmeticOperator.MOD,
            }
            left = BinaryOp(left=left, operator=op_map[op_value], right=right)
        
        return left
    
    def _parse_unary(self) -> Expression:
        """Parse unary expressions"""
        if self._match('OPERATOR', '-'):
            self._advance()
            operand = self._parse_unary()
            return UnaryOp(operator=ArithmeticOperator.SUB, operand=operand)
        
        if self._match('KEYWORD', 'NOT'):
            self._advance()
            operand = self._parse_unary()
            return UnaryOp(operator=LogicalOperator.NOT, operand=operand)
        
        return self._parse_postfix()
    
    def _parse_postfix(self) -> Expression:
        """Parse postfix expressions (member access, function calls)"""
        expr = self._parse_primary()
        
        while True:
            # Member access: obj.member
            if self._match('DELIMITER', '.'):
                self._advance()
                member = self._expect('IDENTIFIER').value
                expr = MemberAccess(object=expr, member=member)
            
            # Function call: func(args)
            elif self._match('DELIMITER', '('):
                expr = self._parse_function_call(expr)
            
            else:
                break
        
        return expr
    
    def _parse_function_call(self, func_expr: Expression) -> FunctionCall:
        """Parse function call"""
        if not isinstance(func_expr, Variable):
            raise ParseError("Function name must be an identifier")
        
        func_name = func_expr.name
        
        self._expect('DELIMITER', '(')
        
        args = []
        kwargs = {}
        
        while not self._match('DELIMITER', ')'):
            # Check if this is keyword argument
            if self._match('IDENTIFIER') and self._peek() and \
               self._peek().type == 'OPERATOR' and self._peek().value == '=':
                # Keyword argument
                key = self._expect('IDENTIFIER').value
                self._expect('OPERATOR', '=')
                value = self._parse_expression()
                kwargs[key] = value
            else:
                # Positional argument
                arg = self._parse_expression()
                args.append(arg)
            
            # Consume comma if present
            self._consume_if('DELIMITER', ',')
        
        self._expect('DELIMITER', ')')
        
        return FunctionCall(name=func_name, args=args, kwargs=kwargs)
    
    def _parse_primary(self) -> Expression:
        """Parse primary expressions (literals, variables, parenthesized)"""
        
        # 1. NUMBER
        if self._match('NUMBER'):
            token = self.current_token
            self._advance()
            
            value_str = token.value
            if '.' in value_str:
                value = float(value_str)
                value_type = "float"
            else:
                value = int(value_str)
                value_type = "int"
            
            return Literal(value=value, type=value_type, location=(token.line, token.column))
        
        # 2. STRING
        if self._match('STRING'):
            token = self.current_token
            self._advance()
            value = token.value.strip('"')
            return Literal(value=value, type="string", location=(token.line, token.column))
        
        # 3. BOOLEAN 
        if self._match('KEYWORD', 'TRUE'):
            token = self.current_token
            self._advance()
            return Literal(value=True, type="bool", location=(token.line, token.column))

        if self._match('KEYWORD', 'FALSE'):
            token = self.current_token
            self._advance()
            return Literal(value=False, type="bool", location=(token.line, token.column))

        # 4. BOOLEAN - Old
        if self._match('IDENTIFIER') and self.current_token.value.lower() in ("true", "false"):
            token = self.current_token
            self._advance()
            return Literal(
                value=(token.value.lower() == "true"),
                type="bool",
                location=(token.line, token.column)
            )

        # 5. VARIABLE / IDENTIFIER
        if self._match('IDENTIFIER'):
            token = self.current_token
            self._advance()
            return Variable(name=token.value, location=(token.line, token.column))
        
        # 6. PARENTHESIZED EXPRESSION
        if self._match('DELIMITER', '('):
            self._advance()
            expr = self._parse_expression()
            self._expect('DELIMITER', ')')
            return expr
        
        # Error Handler
        if self.current_token:
            raise ParseError(
                f"Unexpected token: {self.current_token.type}({self.current_token.value})",
                self.current_token.line,
                self.current_token.column
            )
        else:
            raise ParseError("Unexpected end of file")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def parse_csl(text: str) -> Constitution:
    """Parse CSL text into Constitution AST"""
    parser = CSLParser()
    return parser.parse(text)


def parse_csl_file(filepath: str) -> Constitution:
    """Parse CSL file into Constitution AST"""
    parser = CSLParser()
    return parser.parse_file(filepath)