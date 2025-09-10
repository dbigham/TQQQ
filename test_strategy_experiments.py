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
    assert abs(cagr - 29.70) < 0.01


def test_a2_cagr():
    cagr = run("A2")
    assert abs(cagr - 30.58) < 0.01


def test_a3_cagr():
    cagr = run("A3")
    assert abs(cagr - 31.17) < 0.01
