import re
import subprocess
import sys


def run(exp):
    cmd = [sys.executable, "strategy_tqqq_reserve.py", "--no-show", "--experiment", exp]
    out = subprocess.check_output(cmd, text=True)
    m = re.search(r"Strategy CAGR: ([0-9.]+)%", out)
    assert m, "CAGR output not found"
    return float(m.group(1))


def test_a1_cagr():
    cagr = run("A1")
    assert abs(cagr - 30.43) < 0.01


def test_a2_cagr():
    cagr = run("A2")
    assert abs(cagr - 31.10) < 0.01


def test_a3_cagr():
    cagr = run("A3")
    assert abs(cagr - 31.14) < 0.01


def test_a4_cagr():
    cagr = run("A4")
    assert abs(cagr - 31.74) < 0.01


def test_a5_cagr():
    cagr = run("A5")
    assert abs(cagr - 32.41) < 0.01


def test_a7_cagr():
    cagr = run("A7")
    assert abs(cagr - 33.63) < 0.01


def test_a11_cagr():
    cagr = run("A11")
    assert abs(cagr - 33.96) < 0.01
