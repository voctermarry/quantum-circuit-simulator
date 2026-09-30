"""Tests for the OpenQASM 2.0 subset parser and validator."""

from __future__ import annotations

import pytest

from quantum_circuit.qasm import ParseError, ValidationError, parse


def parse_one(text: str):
    return parse(text)


HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def test_minimal_program_with_registers_and_no_statements():
    program = parse_one(HEADER + "qreg q[2];\ncreg c[2];\n")
    assert program.num_qubits == 2
    assert program.num_clbits == 2
    assert program.gates == ()
    assert program.measurements == ()


def test_comments_and_whitespace_are_ignored():
    text = (
        "// leading comment\n"
        "OPENQASM 2.0; // version\n"
        "include \"qelib1.inc\"; // lib\n"
        "qreg q[1];   // register\n"
        "creg c[1];\n"
        "x q[0]; // flip\n"
        "measure q[0] -> c[0]; // done\n"
    )
    program = parse_one(text)
    assert [gate.name for gate in program.gates] == ["x"]
    assert program.measurements[0].qubit == 0
    assert program.measurements[0].clbit == 0


def test_gate_operands_are_resolved_to_indices():
    program = parse_one(
        HEADER + "qreg q[3];\ncreg c[3];\nx q[2];\nh q[0];\ncx q[0], q[2];\n"
    )
    assert program.gates[0].name == "x"
    assert program.gates[0].qubits == (2,)
    assert program.gates[1].qubits == (0,)
    assert program.gates[2].name == "cx"
    assert program.gates[2].qubits == (0, 2)


def test_measurement_records_both_indices():
    program = parse_one(
        HEADER
        + "qreg q[2];\ncreg c[2];\nmeasure q[1] -> c[0];\nmeasure q[0] -> c[1];\n"
    )
    assert [(m.qubit, m.clbit) for m in program.measurements] == [(1, 0), (0, 1)]


def test_header_must_be_openqasm_2():
    with pytest.raises(ParseError) as info:
        parse_one("OPENQASM 2;")
    assert (info.value.line, info.value.column) == (1, 10)


def test_include_must_be_qelib1():
    with pytest.raises(ParseError) as info:
        parse_one('OPENQASM 2.0;\ninclude "other.inc";\n')
    assert info.value.line == 2


def test_missing_semicolon_is_parse_error():
    with pytest.raises(ParseError) as info:
        parse_one(HEADER + "qreg q[1]\ncreg c[1];\n")
    assert (info.value.line, info.value.column) == (4, 1)


def test_unknown_gate_name_is_parse_error():
    with pytest.raises(ParseError) as info:
        parse_one(HEADER + "qreg q[1];\ncreg c[1];\ny q[0];\n")
    assert (info.value.line, info.value.column) == (5, 1)


def test_bare_register_operand_is_parse_error():
    with pytest.raises(ParseError) as info:
        parse_one(HEADER + "qreg q[1];\ncreg c[1];\nx q;\n")
    assert (info.value.line, info.value.column) == (5, 4)


def test_illegal_character_reports_position():
    with pytest.raises(ParseError) as info:
        parse_one(HEADER + "qreg q[1];\ncreg c[1];\nx q[0])\n")
    # ')' on line 5 right after x q[0]
    assert (info.value.line, info.value.column) == (5, 7)


def test_empty_input_is_parse_error_at_start():
    with pytest.raises(ParseError) as info:
        parse_one("")
    assert (info.value.line, info.value.column) == (1, 1)


def test_duplicate_qreg_declaration_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse_one(HEADER + "qreg q[1];\nqreg r[1];\n")
    assert (info.value.line, info.value.column) == (4, 1)


def test_duplicate_creg_name_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse_one(HEADER + "qreg q[1];\ncreg q[1];\n")
    assert info.value.line == 4


def test_zero_sized_register_is_validation_error_at_size():
    with pytest.raises(ValidationError) as info:
        parse_one(HEADER + "qreg q[0];\n")
    assert (info.value.line, info.value.column) == (3, 8)


def test_index_out_of_range_is_validation_error_at_index():
    with pytest.raises(ValidationError) as info:
        parse_one(HEADER + "qreg q[1];\ncreg c[1];\nx q[2];\n")
    assert (info.value.line, info.value.column) == (5, 5)


def test_undeclared_name_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse_one(HEADER + "qreg q[1];\ncreg c[1];\nx r[0];\n")
    assert (info.value.line, info.value.column) == (5, 3)


def test_classical_register_used_as_qubit_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse_one(HEADER + "qreg q[1];\ncreg c[1];\nx c[0];\n")
    assert (info.value.line, info.value.column) == (5, 3)


def test_gate_after_measurement_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse_one(
            HEADER
            + "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nx q[1];\n"
        )
    assert (info.value.line, info.value.column) == (6, 1)


def test_duplicate_qubit_measurement_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse_one(
            HEADER
            + "qreg q[2];\ncreg c[2];\n"
            + "measure q[0] -> c[0];\nmeasure q[0] -> c[1];\n"
        )
    assert info.value.line == 6


def test_duplicate_clbit_measurement_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse_one(
            HEADER
            + "qreg q[2];\ncreg c[2];\n"
            + "measure q[0] -> c[0];\nmeasure q[1] -> c[0];\n"
        )
    assert info.value.line == 6


def test_missing_registers_at_eof_is_validation_error():
    with pytest.raises(ValidationError) as info:
        parse_one(HEADER)
    assert info.value.line == 3
