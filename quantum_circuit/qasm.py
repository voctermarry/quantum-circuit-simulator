"""Lexer, parser and semantic validation for the OpenQASM 2.0 subset.

Supported grammar (one quantum and one classical register)::

    OPENQASM 2.0;
    include "qelib1.inc";
    qreg name[size];
    creg name[size];
    x q[i];
    h q[i];
    cx q[i], q[j];
    measure q[i] -> c[j];

Line comments (``//``) and whitespace are ignored.  Lexical/syntactic
violations raise :class:`ParseError`; resolvable-but-invalid programs raise
:class:`ValidationError`.  Both carry a 1-based line/column position.
"""

from __future__ import annotations

from dataclasses import dataclass


class QASMError(Exception):
    """Base class for errors located in the source text."""

    kind = "qasm_error"

    def __init__(self, line: int, column: int):
        super().__init__(f"{self.kind} at line {line}, column {column}")
        self.line = line
        self.column = column


class ParseError(QASMError):
    """The source is not lexically/grammatically valid for the subset."""

    kind = "parse_error"


class ValidationError(QASMError):
    """The source parses but cannot be executed as specified."""

    kind = "validation_error"


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    column: int


@dataclass(frozen=True)
class Gate:
    name: str  # "x", "h" or "cx"
    qubits: tuple[int, ...]
    line: int
    column: int


@dataclass(frozen=True)
class Measurement:
    qubit: int
    clbit: int
    line: int
    column: int


@dataclass(frozen=True)
class Program:
    num_qubits: int
    num_clbits: int
    gates: tuple[Gate, ...]
    measurements: tuple[Measurement, ...]


@dataclass(frozen=True)
class _Register:
    name: str
    kind: str  # "q" or "c"
    size: int


def _is_ascii_alpha(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _is_name_start(ch: str) -> bool:
    return _is_ascii_alpha(ch) or ch == "_"


def _is_name_part(ch: str) -> bool:
    return _is_name_start(ch) or _is_digit(ch)


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1
    column = 1
    length = len(text)

    while i < length:
        ch = text[i]
        if ch in " \t\r":
            i += 1
            column += 1
        elif ch == "\n":
            i += 1
            line += 1
            column = 1
        elif ch == "/" and i + 1 < length and text[i + 1] == "/":
            # Line comment runs to (but excludes) the newline.
            while i < length and text[i] != "\n":
                i += 1
                column += 1
        elif ch == "-" and i + 1 < length and text[i + 1] == ">":
            tokens.append(Token("ARROW", "->", line, column))
            i += 2
            column += 2
        elif ch == '"':
            start_line, start_column = line, column
            i += 1
            column += 1
            value: list[str] = []
            while i < length and text[i] != '"':
                if text[i] == "\n":
                    raise ParseError(line, column)
                value.append(text[i])
                i += 1
                column += 1
            if i >= length:
                raise ParseError(start_line, start_column)
            i += 1  # closing quote
            column += 1
            tokens.append(Token("STRING", "".join(value), start_line, start_column))
        elif _is_name_start(ch):
            start_column = column
            start = i
            while i < length and _is_name_part(text[i]):
                i += 1
                column += 1
            tokens.append(Token("NAME", text[start:i], line, start_column))
        elif _is_digit(ch):
            start_column = column
            start = i
            while i < length and _is_digit(text[i]):
                i += 1
                column += 1
            kind = "INT"
            if i < length and text[i] == ".":
                kind = "REAL"
                i += 1
                column += 1
                while i < length and _is_digit(text[i]):
                    i += 1
                    column += 1
            tokens.append(Token(kind, text[start:i], line, start_column))
        elif ch in "[];,":
            tokens.append(Token(ch, ch, line, column))
            i += 1
            column += 1
        else:
            raise ParseError(line, column)

    tokens.append(Token("EOF", "", line, column))
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens
        self._pos = 0
        self._registers: dict[str, _Register] = {}
        self._qreg: _Register | None = None
        self._creg: _Register | None = None
        self._gates: list[Gate] = []
        self._measurements: list[Measurement] = []
        self._measured_qubits: set[int] = set()
        self._measured_clbits: set[int] = set()
        self._measurement_seen = False

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _expect(self, kind: str, text: str | None = None) -> Token:
        token = self._peek()
        if token.kind != kind or (text is not None and token.text != text):
            raise ParseError(token.line, token.column)
        return self._advance()

    def parse(self) -> Program:
        # OPENQASM 2.0;
        header = self._peek()
        if header.kind != "NAME" or header.text != "OPENQASM":
            raise ParseError(header.line, header.column)
        self._advance()
        version = self._peek()
        if version.kind != "REAL" or version.text != "2.0":
            raise ParseError(version.line, version.column)
        self._advance()
        self._expect(";")

        # include "qelib1.inc";
        include = self._peek()
        if include.kind != "NAME" or include.text != "include":
            raise ParseError(include.line, include.column)
        self._advance()
        library = self._expect("STRING")
        if library.text != "qelib1.inc":
            raise ParseError(library.line, library.column)
        self._expect(";")

        while self._peek().kind != "EOF":
            self._parse_statement()

        eof = self._peek()
        if self._qreg is None or self._creg is None:
            raise ValidationError(eof.line, eof.column)

        return Program(
            num_qubits=self._qreg.size,
            num_clbits=self._creg.size,
            gates=tuple(self._gates),
            measurements=tuple(self._measurements),
        )

    def _parse_statement(self) -> None:
        token = self._peek()
        if token.kind != "NAME":
            raise ParseError(token.line, token.column)
        if token.text in ("qreg", "creg"):
            self._parse_declaration()
        elif token.text == "measure":
            self._parse_measurement()
        elif token.text in ("x", "h"):
            self._parse_single_qubit_gate()
        elif token.text == "cx":
            self._parse_cx()
        else:
            raise ParseError(token.line, token.column)

    def _parse_declaration(self) -> None:
        keyword = self._advance()  # qreg / creg
        kind = "q" if keyword.text == "qreg" else "c"
        name = self._expect("NAME")
        self._expect("[")
        size_token = self._peek()
        if size_token.kind != "INT":
            raise ParseError(size_token.line, size_token.column)
        self._advance()
        self._expect("]")
        self._expect(";")

        existing = self._registers.get(name.text)
        if existing is not None or (kind == "q" and self._qreg is not None) or (
            kind == "c" and self._creg is not None
        ):
            raise ValidationError(keyword.line, keyword.column)
        size = int(size_token.text)
        if size < 1:
            raise ValidationError(size_token.line, size_token.column)

        register = _Register(name.text, kind, size)
        self._registers[name.text] = register
        if kind == "q":
            self._qreg = register
        else:
            self._creg = register

    def _parse_operand(self) -> tuple[Token, Token]:
        name = self._expect("NAME")
        bracket = self._peek()
        if bracket.kind != "[":
            # Bare registers are outside the subset: single bits only.
            raise ParseError(bracket.line, bracket.column)
        self._advance()
        index = self._peek()
        if index.kind != "INT":
            raise ParseError(index.line, index.column)
        self._advance()
        self._expect("]")
        return name, index

    def _resolve(self, name: Token, index: Token, kind: str) -> int:
        register = self._registers.get(name.text)
        if register is None or register.kind != kind:
            raise ValidationError(name.line, name.column)
        value = int(index.text)
        if not 0 <= value < register.size:
            raise ValidationError(index.line, index.column)
        return value

    def _parse_single_qubit_gate(self) -> None:
        keyword = self._advance()  # x / h
        if self._measurement_seen:
            raise ValidationError(keyword.line, keyword.column)
        name, index = self._parse_operand()
        self._expect(";")
        qubit = self._resolve(name, index, "q")
        self._gates.append(Gate(keyword.text, (qubit,), keyword.line, keyword.column))

    def _parse_cx(self) -> None:
        keyword = self._advance()  # cx
        if self._measurement_seen:
            raise ValidationError(keyword.line, keyword.column)
        first_name, first_index = self._parse_operand()
        self._expect(",")
        second_name, second_index = self._parse_operand()
        self._expect(";")
        control = self._resolve(first_name, first_index, "q")
        target = self._resolve(second_name, second_index, "q")
        self._gates.append(
            Gate("cx", (control, target), keyword.line, keyword.column)
        )

    def _parse_measurement(self) -> None:
        keyword = self._advance()  # measure
        qubit_name, qubit_index = self._parse_operand()
        self._expect("ARROW", "->")
        clbit_name, clbit_index = self._parse_operand()
        self._expect(";")
        qubit = self._resolve(qubit_name, qubit_index, "q")
        clbit = self._resolve(clbit_name, clbit_index, "c")
        if qubit in self._measured_qubits or clbit in self._measured_clbits:
            raise ValidationError(keyword.line, keyword.column)
        self._measured_qubits.add(qubit)
        self._measured_clbits.add(clbit)
        self._measurement_seen = True
        self._measurements.append(
            Measurement(qubit, clbit, keyword.line, keyword.column)
        )


def parse(text: str) -> Program:
    """Parse an OpenQASM 2.0 subset program."""
    return _Parser(tokenize(text)).parse()
