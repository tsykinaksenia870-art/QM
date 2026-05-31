import unittest
import json
import subprocess
import os
from pathlib import Path
from parse_brsccp import (
    preprocess, parse_header, parse_input_params,
    parse_results, parse_diagnostics, parse_status,
    parse_log, EnergyState, AtomShift
)

class TestBrsccpParser(unittest.TestCase):

    def test_preprocess_timestamp_inheritance(self):
        raw = "[01/01/25 10:00:00] Line 1\nLine 2\n[01/01/25 10:05:00] Line 3"
        processed = preprocess(raw)
        self.assertEqual(processed[0], "[01/01/25 10:00:00] Line 1")
        self.assertEqual(processed[1], "[01/01/25 10:00:00] Line 2")
        self.assertEqual(processed[2], "[01/01/25 10:05:00] Line 3")

    def test_parse_header(self):
        lines = ["brsccp version 2.1.16 from 27.01.2024", "[01/01/25 10:00:00] Start"]
        v, d, t = parse_header(lines, [])
        self.assertEqual(v, "2.1.16")
        self.assertEqual(d, "27.01.2024")
        self.assertEqual(t, "01/01/25 10:00:00")

    def test_parse_input_params_multiline(self):
        lines = [
            "Input parameters",
            "                    n_runs = 10",
            "                    remove_by_resname = ['CL', 'PO4',",
            "                    'SO4']",
            "--------------------------------------------------"
        ]
        params = parse_input_params(lines, [])
        self.assertEqual(params['n_runs'], 10)
        self.assertEqual(params['remove_by_resname'], ['CL', 'PO4', 'SO4'])

    def test_parse_results_abnormal(self):
        lines = ["Some log lines", "ABNORMAL TERMINATION"]
        results = parse_results(lines, [])
        self.assertIsNone(results)

    def test_parse_results_normal(self):
        lines = [
            "                    2V8X to 2V8Y",
            "                    Energy of complex 1 after optimization: -697.84919 h",
            "                    Energy of ligand 1 after optimization: -96.38567 h",
            "                    Energy of ligand 1 in dot: -96.36690 h",
            "                    Energy of protein 1 in dot: -601.37746 h",
            "                    Relative binding energy, ddG (dG(ligand 2) - dG(ligand 1)): 2.00 kcal/mol",
            "                    Activity ratio, exp(-ddG/RT): 0.03"
        ]
        res = parse_results(lines, [])
        self.assertIsNotNone(res)
        self.assertEqual(res.transformation_from, "2V8X")
        self.assertEqual(res.ddG_kcal_mol, 2.00)
        self.assertEqual(res.state1.complex_opt, -697.84919)

    def test_parse_diagnostics(self):
        lines = [
            "INFO      Best optimization trajectory: x00009; E=-697.849194118701 Eh",
            "WARNING    C2' @ MGQ(1218) has shift 1.1264949090518903 > 1.0 A!"
        ]
        diag = parse_diagnostics(lines, [])
        self.assertEqual(diag.best_trajectory, "x00009")
        self.assertEqual(diag.best_energy_Eh, -697.849194118701)
        self.assertEqual(len(diag.atom_shifts), 1)
        self.assertEqual(diag.atom_shifts[0].atom, "C2'")

    def test_parse_status(self):
        lines = ["TOTAL RUN TIME: 0:51:12.085976", "----------------- NORMAL TERMINATION -----------------"]
        term, rt = parse_status(lines)
        self.assertEqual(term, "NORMAL")
        self.assertEqual(rt, "0:51:12.085976")

    def test_integration_full_log(self):
        log_path = Path("brsccp2V8X.log")
        if not log_path.exists():
            self.skipTest("brsccp2V8X.log not found")

        log = parse_log(log_path)
        self.assertEqual(log.version, "2.1.16")
        self.assertEqual(log.termination, "NORMAL")
        self.assertIsNotNone(log.results)
        self.assertEqual(log.results.ddG_kcal_mol, 2.0)
        self.assertEqual(len(log.diagnostics.atom_shifts), 24)

    def test_cli_json_output(self):
        if not Path("brsccp2V8X.log").exists():
            self.skipTest("brsccp2V8X.log not found")

        result = subprocess.run(
            ["python3", "parse_brsccp.py", "brsccp2V8X.log", "--format", "json"],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertEqual(data['version'], "2.1.16")

    def test_cli_csv_output(self):
        if not Path("brsccp2V8X.log").exists():
            self.skipTest("brsccp2V8X.log not found")

        result = subprocess.run(
            ["python3", "parse_brsccp.py", "brsccp2V8X.log", "--format", "csv"],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        # 2V8X,2V8X to 2V8Y,2.0,0.03,NORMAL,24
        parts = result.stdout.strip().split(',')
        self.assertEqual(parts[0], "2V8X")
        self.assertEqual(parts[4], "NORMAL")

if __name__ == "__main__":
    unittest.main()
