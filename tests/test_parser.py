"""Tests for the OpenQASM subset lexer/parser and semantic validation."""

from __future__ import annotations

import math

import pytest

from quantum_circuit.openqasm import ParseError, ValidationError, parse, tokenize


def _parse(body: str, header: bool = True):
    return parse((('OPENQASM 2.0;\ninclude "qelib1.inc";\n') if header else "") + body)


# ---------------------------------------------------------------- tokenizer


def test_tokenizer_skips_line_comments():
    tokens = tokenize("// only a comment\nqreg // inline\n q[1];")
    kinds = [t.kind for t in tokens]
    assert kinds == ["qreg", "ident", "[", "int", "]", ";", "eof"]


def test_tokenizer_tracks_line_and_column():
    tokens = tokenize("\n  h q[0];")
    h_tok = tokens[0]
    assert (h_tok.line, h_tok.column) == (2, 3)
    q_tok = tokens[1]
    assert (q_tok.line, q_tok.column) == (2, 5)


def test_tokenizer_rejects_illegal_character_with_position():
    with pytest.raises(ParseError) as info:
        tokenize("qreg q[1];\n @")
    assert info.value.line == 2
    assert info.value.column == 2


def test_tokenizer_accepts_real_literal_for_version():
    tokens = tokenize("OPENQASM 2.0;")
    assert tokens[1].kind == "real"
    assert tokens[1].value == "2.0"


# --------------------------------------------------------------- valid programs


def test_parse_minimal_program_structure():
    program = _parse("qreg q[2];\ncreg c[2];\nx q[0];\nh q[1];\ncx q[0],q[1];\nmeasure q[0]->c[0];\nmeasure q[1]->c[1];\n")
    assert program.num_qubits == 2
    assert program.num_clbits == 2
    kinds = [op.kind for op in program.operations]
    assert kinds == ["x", "h", "cx", "measure", "measure"]
    assert program.operations[2].targets == (0, 1)
    assert program.operations[3].targets == (0, 0)


def test_program_with_no_gates_or_measures_is_valid():
    program = _parse("qreg q[1];\ncreg c[1];\n")
    assert program.operations == ()


def test_register_declarations_may_appear_in_either_order():
    program = _parse("creg c[1];\nqreg q[1];\n")
    assert (program.num_qubits, program.num_clbits) == (1, 1)


# ----------------------------------------------------------------- parse errors


@pytest.mark.parametrize(
    "source",
    [
        "",
        "qreg q[1];",
        "OPENQASM 2.0;\n",
        "OPENQASM 2.0 include \"qelib1.inc\";\n",
        "OPENQASM 3.0;\ninclude \"qelib1.inc\";\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\"",
        "OPENQASM 2.0;\ninclude qelib1.inc;\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[];\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1]\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1]; creg c[1]; h q0;\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1]; creg c[1]; cx q[0];\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1]; creg c[1]; measure q[0] c[0];\n",
    ],
)
def test_syntax_errors_raise_parse_error(source):
    with pytest.raises(ParseError):
        parse(source)


def test_gate_without_declared_registers_is_validation_error():
    with pytest.raises(ValidationError):
        parse('OPENQASM 2.0;\ninclude "qelib1.inc";\nx q[0];\n')


def test_parse_error_position_is_one_based():
    with pytest.raises(ParseError) as info:
        parse("OPENQASM 2.0;\ninclude \"qelib1.inc\"\nqreg q[1];\n")
    # Missing semicolon is reported at the following token ('qreg', line 3).
    assert (info.value.line, info.value.column) == (3, 1)


# ------------------------------------------------------------- validation errors


def _validation(body: str):
    with pytest.raises(ValidationError) as info:
        _parse(body)
    return info.value


def test_unknown_include_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse('OPENQASM 2.0;\ninclude "other.inc";\n')
    assert (info.value.line, info.value.column) == (2, 9)


def test_duplicate_qreg_rejected():
    exc = _validation("qreg a[1];\nqreg b[1];\ncreg c[1];\n")
    assert exc.line == 4


def test_duplicate_creg_rejected():
    exc = _validation("qreg q[1];\ncreg a[1];\ncreg b[1];\n")
    assert exc.line == 5


def test_reusing_register_name_rejected():
    exc = _validation("qreg q[1];\ncreg q[1];\n")
    assert "already declared" in exc.message


def test_register_size_must_be_positive():
    exc = _validation("qreg q[0];\ncreg c[1];\n")
    assert exc.line == 3


def test_register_index_out_of_range():
    exc = _validation("qreg q[1];\ncreg c[1];\nx q[1];\n")
    assert exc.line == 5
    assert "out of range" in exc.message


def test_undeclared_register_name():
    exc = _validation("qreg q[1];\ncreg c[1];\nh r[0];\n")
    assert "undeclared" in exc.message
    assert exc.line == 5


def test_quantum_register_required_in_gate_arguments():
    exc = _validation("qreg q[1];\ncreg c[1];\nx c[0];\n")
    assert "classical register" in exc.message


def test_classical_register_required_as_measure_target():
    exc = _validation("qreg q[1];\ncreg c[1];\nmeasure q[0] -> q[0];\n")
    assert "quantum register" in exc.message


def test_unsupported_gate_is_validation_error():
    exc = _validation("qreg q[1];\ncreg c[1];\nw q[0];\n")
    assert "w" in exc.message


def test_qubit_measured_twice_rejected():
    exc = _validation(
        "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nmeasure q[0] -> c[1];\n"
    )
    assert exc.line == 6


def test_clbit_written_twice_rejected():
    exc = _validation(
        "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[0];\n"
    )
    assert exc.line == 6


def test_gate_after_measurement_rejected():
    exc = _validation("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nx q[0];\n")
    assert exc.line == 6
    assert "after a measurement" in exc.message


def test_missing_qreg_rejected():
    with pytest.raises(ValidationError):
        parse('OPENQASM 2.0;\ninclude "qelib1.inc";\ncreg c[1];\n')


def test_missing_creg_rejected():
    with pytest.raises(ValidationError):
        parse('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n')


def test_cx_control_and_target_must_differ():
    exc = _validation("qreg q[2];\ncreg c[1];\ncx q[0], q[0];\nmeasure q[0]->c[0];\n")
    assert "different" in exc.message


# ------------------------------------------------------- parameterized gates


def test_tokenizer_accepts_angle_expression_tokens():
    tokens = tokenize("rx(-pi/2) q[0];")
    kinds = [t.kind for t in tokens]
    assert kinds == ["ident", "(", "-", "ident", "/", "int", ")", "ident", "[", "int", "]", ";", "eof"]


def test_parse_parameterized_gates():
    program = _parse("qreg q[2];\ncreg c[2];\nrx(pi/2) q[0];\nry(0.25) q[1];\nrz(1e-3) q[0];\n")
    kinds = [op.kind for op in program.operations]
    assert kinds == ["rx", "ry", "rz"]
    assert program.operations[0].targets == (0,)
    assert program.operations[1].targets == (1,)
    assert program.operations[0].params == (math.pi / 2,)
    assert program.operations[1].params == (0.25,)
    assert program.operations[2].params == (1e-3,)


def test_angle_expression_precedence_and_associativity():
    program = _parse(
        "qreg q[1];\ncreg c[1];\n"
        "rx(1+2*3) q[0];\n"
        "ry((1+2)*3) q[0];\n"
        "rz(10-4-3) q[0];\n"
        "rx(8/4/2) q[0];\n"
    )
    params = [op.params[0] for op in program.operations]
    assert params == [7.0, 9.0, 3.0, 1.0]


def test_angle_expression_unary_signs_and_pi():
    program = _parse("qreg q[1];\ncreg c[1];\nrx(-pi/2) q[0];\nry(+2) q[0];\nrz(--3) q[0];\n")
    params = [op.params[0] for op in program.operations]
    assert params == [-math.pi / 2, 2.0, 3.0]


def test_angle_expression_scientific_and_nested_parens():
    program = _parse("qreg q[1];\ncreg c[1];\nrx(2.5e-1*(1+(2))) q[0];\n")
    assert program.operations[0].params == (0.75,)


@pytest.mark.parametrize(
    "source",
    [
        "rx q[0];",  # missing parentheses and parameter
        "rx() q[0];",  # missing parameter
        "rx(1,2) q[0];",  # more than one parameter
        "rx(1+) q[0];",  # missing operand
        "rx(*2) q[0];",  # operator without left operand
        "rx(1 2) q[0];",  # two operands without operator
        "rx((1) q[0];",  # unbalanced parenthesis
        "rx(1)) q[0];",  # extra closing parenthesis
        "rx(1) q[0]",  # missing semicolon after the gate
    ],
)
def test_malformed_angle_expressions_raise_parse_error(source):
    with pytest.raises(ParseError):
        _parse("qreg q[1];\ncreg c[1];\n" + source)


def test_angle_expression_unknown_name_is_validation_error():
    exc = _validation("qreg q[1];\ncreg c[1];\nrx(theta) q[0];\n")
    assert "theta" in exc.message
    assert exc.line == 5


def test_angle_expression_division_by_zero_is_validation_error():
    exc = _validation("qreg q[1];\ncreg c[1];\nrx(1/0) q[0];\n")
    assert "zero" in exc.message
    assert exc.line == 5


def test_angle_expression_literal_overflow_is_validation_error():
    exc = _validation("qreg q[1];\ncreg c[1];\nrx(1e999) q[0];\n")
    assert exc.line == 5


def test_angle_expression_non_finite_result_is_validation_error():
    exc = _validation("qreg q[1];\ncreg c[1];\nrx(1e308*10) q[0];\n")
    assert "finite" in exc.message
    assert exc.line == 5


def test_parameterized_gate_after_measurement_rejected():
    exc = _validation("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nrz(pi) q[0];\n")
    assert "after a measurement" in exc.message


def test_parameterized_gate_requires_quantum_register():
    exc = _validation("qreg q[1];\ncreg c[1];\nrx(pi) c[0];\n")
    assert "classical register" in exc.message


def test_parameterized_gate_index_out_of_range():
    exc = _validation("qreg q[1];\ncreg c[1];\nry(pi) q[1];\n")
    assert "out of range" in exc.message
