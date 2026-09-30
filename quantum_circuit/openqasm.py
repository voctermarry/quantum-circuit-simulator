"""Lexer, parser and semantic validation for the OpenQASM 2.0 subset.

Supported subset::

    OPENQASM 2.0;
    include "qelib1.inc";
    qreg name[positive-size];
    creg name[positive-size];
    x qubit;
    h qubit;
    cx qubit, qubit;
    measure qubit -> cbit;

Only one ``qreg`` and one ``creg`` may be declared, and gates/measures only
accept single bits with constant integer indices. Line comments (``//``) are
supported.
"""

from __future__ import annotations

from dataclasses import dataclass


class ParseError(Exception):
    """A lexical or syntactic problem (reported as ``parse_error``)."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column


class ValidationError(Exception):
    """A syntactically valid but semantically invalid program."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column


@dataclass(frozen=True)
class Token:
    kind: str  # 'ident', 'int', 'real', 'string' or a literal punctuation word
    value: str
    line: int  # 1-based
    column: int  # 1-based, points at the first character of the token


_KEYWORDS = {"OPENQASM", "include", "qreg", "creg", "measure"}
_BUILTIN_GATES = {"x", "h", "cx"}

# A state vector has 2**n amplitudes; larger registers cannot be simulated
# and are rejected as illegal sizes rather than crashing.
_MAX_REGISTER_SIZE = 20


def tokenize(source: str) -> list[Token]:
    """Split *source* into tokens.

    Raises :class:`ParseError` (with 1-based position) on invalid input.
    """
    tokens: list[Token] = []
    line = 1
    col = 1
    n = len(source)
    i = 0

    while i < n:
        ch = source[i]

        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if ch in " \t\r":
            i += 1
            col += 1
            continue

        # Line comment: runs to end of line (end of input also terminates it).
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            col += 2
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue

        start_line, start_col = line, col

        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            kind = word if word in _KEYWORDS else "ident"
            tokens.append(Token(kind, word, start_line, start_col))
            col += j - i
            i = j
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            j = i
            seen_dot = False
            seen_exp = False
            while j < n:
                c = source[j]
                if c.isdigit():
                    j += 1
                elif c == "." and not seen_dot and not seen_exp:
                    seen_dot = True
                    j += 1
                elif c in "eE" and not seen_exp:
                    seen_exp = True
                    j += 1
                    if j < n and source[j] in "+-":
                        j += 1
                else:
                    break
            word = source[i:j]
            kind = "real" if seen_dot or seen_exp else "int"
            tokens.append(Token(kind, word, start_line, start_col))
            col += j - i
            i = j
            continue

        if ch == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == "\n":
                    raise ParseError("unterminated string literal", start_line, start_col)
                j += 1
            if j >= n:
                raise ParseError("unterminated string literal", start_line, start_col)
            tokens.append(Token("string", source[i + 1 : j], start_line, start_col))
            col += j - i + 1
            i = j + 1
            continue

        if ch == "-" and i + 1 < n and source[i + 1] == ">":
            tokens.append(Token("->", "->", start_line, start_col))
            i += 2
            col += 2
            continue

        if ch in "[];,":
            tokens.append(Token(ch, ch, start_line, start_col))
            i += 1
            col += 1
            continue

        raise ParseError(f"unexpected character {ch!r}", start_line, start_col)

    tokens.append(Token("eof", "", line, col))
    return tokens


@dataclass(frozen=True)
class Operation:
    """A validated gate application or measurement."""

    kind: str  # 'x', 'h', 'cx' or 'measure'
    targets: tuple[int, ...]  # qubit indices; measure appends the cbit index


@dataclass(frozen=True)
class Program:
    num_qubits: int
    num_clbits: int
    operations: tuple[Operation, ...]


def _parse_decimal(token: Token) -> int:
    """Parse a non-negative decimal literal, rejecting oversized ones."""
    if len(token.value) > 18:
        raise ValidationError("integer literal is too large", token.line, token.column)
    return int(token.value)


def parse(source: str) -> Program:
    """Parse and validate the OpenQASM subset, returning a :class:`Program`."""
    tokens = tokenize(source)
    pos = 0

    def peek() -> Token:
        return tokens[pos]

    def advance() -> Token:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def expect(kind: str) -> Token:
        tok = peek()
        if tok.kind != kind:
            raise ParseError(f"expected {kind!r} but found {tok.value!r}", tok.line, tok.column)
        return advance()

    # Header: OPENQASM 2.0;  (a real token "2.0", exactly)
    first = peek()
    if first.kind != "OPENQASM":
        raise ParseError("expected 'OPENQASM' declaration", first.line, first.column)
    advance()
    version = expect("real") if peek().kind == "real" else expect("int")
    if version.value != "2.0":
        raise ParseError(f"unsupported OpenQASM version {version.value!r}", version.line, version.column)
    expect(";")

    # Exactly: include "qelib1.inc";
    expect("include")
    path = expect("string")
    if path.value != "qelib1.inc":
        raise ValidationError(f"unknown include file {path.value!r}", path.line, path.column)
    expect(";")

    qreg_name: str | None = None
    qreg_size: int | None = None
    creg_name: str | None = None
    creg_size: int | None = None
    operations: list[Operation] = []
    measured_qubits: set[int] = set()
    measured_clbits: set[int] = set()
    measurement_started = False

    def parse_index(reg_kind: str) -> tuple[int, Token, Token]:
        """Parse ``name[const]`` returning (index, name token, index token)."""
        name_tok = expect("ident")
        expect("[")
        idx_tok = expect("int")
        expect("]")
        if reg_kind == "q":
            if name_tok.value != qreg_name:
                if creg_name is not None and name_tok.value == creg_name:
                    raise ValidationError(
                        "classical register used where a quantum register is required",
                        name_tok.line,
                        name_tok.column,
                    )
                raise ValidationError(
                    f"undeclared quantum register {name_tok.value!r}",
                    name_tok.line,
                    name_tok.column,
                )
            size = qreg_size
        else:
            if name_tok.value != creg_name:
                if qreg_name is not None and name_tok.value == qreg_name:
                    raise ValidationError(
                        "quantum register used where a classical register is required",
                        name_tok.line,
                        name_tok.column,
                    )
                raise ValidationError(
                    f"undeclared classical register {name_tok.value!r}",
                    name_tok.line,
                    name_tok.column,
                )
            size = creg_size
        idx = _parse_decimal(idx_tok)
        assert size is not None
        if not 0 <= idx < size:
            raise ValidationError(
                f"register index {idx} out of range for register of size {size}",
                idx_tok.line,
                idx_tok.column,
            )
        return idx, name_tok, idx_tok

    while peek().kind != "eof":
        tok = peek()

        if tok.kind == "qreg" or tok.kind == "creg":
            advance()
            is_qreg = tok.kind == "qreg"
            name_tok = expect("ident")
            expect("[")
            size_tok = expect("int")
            expect("]")
            expect(";")
            size = _parse_decimal(size_tok)
            if not 1 <= size <= _MAX_REGISTER_SIZE:
                raise ValidationError(
                    f"register size must be between 1 and {_MAX_REGISTER_SIZE}, got {size}",
                    size_tok.line,
                    size_tok.column,
                )
            if is_qreg:
                if qreg_name is not None:
                    raise ValidationError(
                        "duplicate declaration of quantum register",
                        name_tok.line,
                        name_tok.column,
                    )
                if creg_name is not None and name_tok.value == creg_name:
                    raise ValidationError(
                        f"register name {name_tok.value!r} already declared",
                        name_tok.line,
                        name_tok.column,
                    )
                qreg_name = name_tok.value
                qreg_size = size
            else:
                if creg_name is not None:
                    raise ValidationError(
                        "duplicate declaration of classical register",
                        name_tok.line,
                        name_tok.column,
                    )
                if qreg_name is not None and name_tok.value == qreg_name:
                    raise ValidationError(
                        f"register name {name_tok.value!r} already declared",
                        name_tok.line,
                        name_tok.column,
                    )
                creg_name = name_tok.value
                creg_size = size
            continue

        if tok.kind == "measure":
            advance()
            q_idx, q_name_tok, _q_idx_tok = parse_index("q")
            expect("->")
            c_idx, c_name_tok, _c_idx_tok = parse_index("c")
            expect(";")
            if q_idx in measured_qubits:
                raise ValidationError(
                    f"qubit {qreg_name}[{q_idx}] is measured more than once",
                    q_name_tok.line,
                    q_name_tok.column,
                )
            if c_idx in measured_clbits:
                raise ValidationError(
                    f"clbit {creg_name}[{c_idx}] is the target of more than one measurement",
                    c_name_tok.line,
                    c_name_tok.column,
                )
            measured_qubits.add(q_idx)
            measured_clbits.add(c_idx)
            measurement_started = True
            operations.append(Operation("measure", (q_idx, c_idx)))
            continue

        if tok.kind == "ident":
            gate_name = tok.value
            if gate_name not in _BUILTIN_GATES:
                raise ValidationError(
                    f"undeclared identifier or unsupported gate {gate_name!r}",
                    tok.line,
                    tok.column,
                )
            if measurement_started:
                raise ValidationError(
                    f"quantum gate {gate_name!r} appears after a measurement",
                    tok.line,
                    tok.column,
                )
            advance()
            if gate_name == "cx":
                ctrl, _, _ = parse_index("q")
                expect(",")
                tgt, tgt_name_tok, _ = parse_index("q")
                expect(";")
                if ctrl == tgt:
                    raise ValidationError(
                        "cx control and target must be different qubits",
                        tgt_name_tok.line,
                        tgt_name_tok.column,
                    )
                operations.append(Operation("cx", (ctrl, tgt)))
            else:
                target, _, _ = parse_index("q")
                expect(";")
                operations.append(Operation(gate_name, (target,)))
            continue

        raise ParseError(
            f"unexpected token {tok.value!r}",
            tok.line,
            tok.column,
        )

    eof = peek()
    if qreg_name is None:
        raise ValidationError("missing required quantum register declaration", eof.line, eof.column)
    if creg_name is None:
        raise ValidationError("missing required classical register declaration", eof.line, eof.column)

    return Program(
        num_qubits=qreg_size if qreg_size is not None else 0,
        num_clbits=creg_size if creg_size is not None else 0,
        operations=tuple(operations),
    )
