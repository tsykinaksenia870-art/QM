import re
import json
import csv
import sys
import argparse
import ast
import io
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

@dataclass
class EnergyState:
    complex_opt: float
    ligand_opt: float
    ligand_dot: float
    protein_dot: float

@dataclass
class AtomShift:
    atom: str
    resname: str
    resnum: int
    shift_A: float
    threshold_A: float

@dataclass
class Results:
    transformation_from: str
    transformation_to: str
    state1: EnergyState
    state2: EnergyState
    ddG_kcal_mol: float
    activity_ratio: float

@dataclass
class Diagnostics:
    best_trajectory: Optional[str] = None
    best_energy_Eh: Optional[float] = None
    alt_structures: List[Tuple[str, float]] = field(default_factory=list)
    atom_shifts: List[AtomShift] = field(default_factory=list)
    pdb_warnings: List[str] = field(default_factory=list)

@dataclass
class BrsccpLog:
    version: str
    build_date: str
    start_time: str
    input_params: Dict[str, Any]
    results: Optional[Results]
    diagnostics: Diagnostics
    termination: str
    total_runtime: Optional[str]
    parse_warnings: List[str] = field(default_factory=list)

def preprocess(raw: str) -> List[str]:
    lines = raw.splitlines()
    processed_lines = []
    last_timestamp_str = None
    timestamp_re = re.compile(r"\[\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\]")
    
    for line in lines:
        line_rstrip = line.rstrip()
        match = timestamp_re.search(line_rstrip)
        if match:
            last_timestamp_str = match.group(0)
            processed_lines.append(line_rstrip)
        elif last_timestamp_str:
            processed_lines.append(f"{last_timestamp_str} {line_rstrip}")
        else:
            processed_lines.append(line_rstrip)
    return processed_lines

def parse_header(lines: List[str], warnings: List[str]) -> Tuple[str, str, str]:
    version = "UNKNOWN"
    build_date = "UNKNOWN"
    start_time = "UNKNOWN"
    
    version_re = re.compile(r"brsccp\s+version\s+(\S+)\s+from\s+(\S+)")
    time_re = re.compile(r"\[(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]")
    
    for line in lines:
        v_match = version_re.search(line)
        if v_match:
            version = v_match.group(1)
            build_date = v_match.group(2)
        
        t_match = time_re.search(line)
        if t_match and start_time == "UNKNOWN":
            start_time = t_match.group(1)
            
    return version, build_date, start_time

def parse_input_params(lines: List[str], warnings: List[str]) -> Dict[str, Any]:
    params_raw = {}
    in_block = False
    current_key = None
    
    block_lines = []
    for line in lines:
        content = re.sub(r"^\[.*?\]\s*", "", line)
        if "Input parameters" in content:
            in_block = True
            continue
        if in_block and "---" in content:
            if content.strip().count('-') > 20:
                break
        if in_block:
            block_lines.append(line)
            
    param_re = re.compile(r"^\s*(?:\[.*?\]\s*)?([A-Za-z_]\w*)\s*=\s*(.+)$")
    
    for line in block_lines:
        content = re.sub(r"^\[.*?\]\s*", "", line)
        if not content.strip():
            continue
            
        match = param_re.match(content)
        if match:
            current_key = match.group(1)
            params_raw[current_key] = match.group(2).strip()
        elif current_key:
            params_raw[current_key] += " " + content.strip()
            
    params = {}
    for k, v in params_raw.items():
        try:
            params[k] = ast.literal_eval(v)
        except Exception as e:
            params[k] = v
            warnings.append(f"Failed to parse param {k}: {v}. Error: {e}")
            
    return params

def parse_results(lines: List[str], warnings: List[str]) -> Optional[Results]:
    energies = {}
    ddG = None
    activity_ratio = None
    trans_from = "UNKNOWN"
    trans_to = "UNKNOWN"
    
    energy_opt_re = re.compile(r"Energy of (\w+) (\d+) after optimization:\s*(-?[\d.]+)\s*h")
    energy_dot_re = re.compile(r"Energy of (\w+) (\d+) in dot:\s*(-?[\d.]+)\s*h")
    ddg_re = re.compile(r"Relative binding energy, ddG.*?:\s*(-?[\d.]+)\s*kcal/mol")
    ratio_re = re.compile(r"Activity ratio, exp\(-ddG/RT\):\s*(-?[\d.]+)")
    trans_re = re.compile(r"^\s*(?:\[.*?\]\s*)?(\w+)\s+to\s+(\w+)\s*$")
    
    for line in lines:
        content = re.sub(r"^\[.*?\]\s*", "", line)
        
        m = energy_opt_re.search(content)
        if m:
            energies[(m.group(1), int(m.group(2)), 'opt')] = float(m.group(3))
            continue
            
        m = energy_dot_re.search(content)
        if m:
            energies[(m.group(1), int(m.group(2)), 'dot')] = float(m.group(3))
            continue
            
        m = ddg_re.search(content)
        if m:
            ddG = float(m.group(1))
            continue
            
        m = ratio_re.search(content)
        if m:
            activity_ratio = float(m.group(1))
            continue
            
        m = trans_re.match(content)
        if m:
            trans_from = m.group(1)
            trans_to = m.group(2)
            continue

    if ddG is None and activity_ratio is None:
        return None
        
    try:
        def get_state(num):
            return EnergyState(
                complex_opt=energies.get(('complex', num, 'opt'), 0.0),
                ligand_opt=energies.get(('ligand', num, 'opt'), 0.0),
                ligand_dot=energies.get(('ligand', num, 'dot'), 0.0),
                protein_dot=energies.get(('protein', num, 'dot'), 0.0)
            )
        return Results(
            transformation_from=trans_from,
            transformation_to=trans_to,
            state1=get_state(1),
            state2=get_state(2),
            ddG_kcal_mol=ddG if ddG is not None else 0.0,
            activity_ratio=activity_ratio if activity_ratio is not None else 0.0
        )
    except Exception as e:
        warnings.append(f"Error assembling results: {e}")
        return None

def parse_diagnostics(lines: List[str], warnings: List[str]) -> Diagnostics:
    best_traj = None
    best_energy = None
    alt_structures = []
    atom_shifts = []
    pdb_warnings = []
    
    best_re = re.compile(r"Best optimization trajectory:\s*(\S+);\s*E=(-?[\d.]+)\s*Eh")
    struct_re = re.compile(r"Structure (\S+):\s*(-?[\d.]+)\s*Eh")
    shift_re = re.compile(r"([A-Z0-9']+)\s+@\s+([A-Z]+)\((\d+)\)\s+has shift\s+([\d.]+)\s+>\s+([\d.]+)\s+A")
    pdb_warn_re = re.compile(r"PDBConstructionWarning:.*")
    
    for line in lines:
        content = re.sub(r"^\[.*?\]\s*", "", line)
        
        m = best_re.search(content)
        if m:
            best_traj = m.group(1)
            best_energy = float(m.group(2))
            continue
            
        m = struct_re.search(content)
        if m:
            alt_structures.append((m.group(1), float(m.group(2))))
            continue
            
        m = shift_re.search(content)
        if m:
            atom_shifts.append(AtomShift(
                atom=m.group(1), resname=m.group(2), resnum=int(m.group(3)),
                shift_A=float(m.group(4)), threshold_A=float(m.group(5))
            ))
            continue
            
        m = pdb_warn_re.search(content)
        if m:
            pdb_warnings.append(m.group(0))
            continue
            
    return Diagnostics(
        best_trajectory=best_traj,
        best_energy_Eh=best_energy,
        alt_structures=alt_structures,
        atom_shifts=atom_shifts,
        pdb_warnings=pdb_warnings
    )

def parse_status(lines: List[str]) -> Tuple[str, Optional[str]]:
    termination = "UNKNOWN"
    total_runtime = None
    
    term_re = re.compile(r"(NORMAL|ABNORMAL) TERMINATION")
    runtime_re = re.compile(r"TOTAL RUN TIME:\s*(\S+)")
    
    for line in lines:
        m = term_re.search(line)
        if m:
            termination = m.group(1)
        
        m = runtime_re.search(line)
        if m:
            total_runtime = m.group(1)
            
    return termination, total_runtime

def parse_log(path: Path) -> BrsccpLog:
    try:
        raw = path.read_text(encoding='utf-8')
    except Exception as e:
        return BrsccpLog(
            version="UNKNOWN", build_date="UNKNOWN", start_time="UNKNOWN",
            input_params={}, results=None, diagnostics=Diagnostics(),
            termination="ERROR", total_runtime=None,
            parse_warnings=[f"Could not read file: {e}"]
        )
        
    lines = preprocess(raw)
    warnings = []
    
    version, build_date, start_time = "UNKNOWN", "UNKNOWN", "UNKNOWN"
    try:
        version, build_date, start_time = parse_header(lines, warnings)
    except Exception as e:
        warnings.append(f"Error in parse_header: {e}")

    input_params = {}
    try:
        input_params = parse_input_params(lines, warnings)
    except Exception as e:
        warnings.append(f"Error in parse_input_params: {e}")

    results = None
    try:
        results = parse_results(lines, warnings)
    except Exception as e:
        warnings.append(f"Error in parse_results: {e}")

    diagnostics = Diagnostics()
    try:
        diagnostics = parse_diagnostics(lines, warnings)
    except Exception as e:
        warnings.append(f"Error in parse_diagnostics: {e}")

    termination, total_runtime = "UNKNOWN", None
    try:
        termination, total_runtime = parse_status(lines)
    except Exception as e:
        warnings.append(f"Error in parse_status: {e}")
    
    return BrsccpLog(
        version=version, build_date=build_date, start_time=start_time,
        input_params=input_params, results=results, diagnostics=diagnostics,
        termination=termination, total_runtime=total_runtime,
        parse_warnings=warnings
    )

def to_json(log: BrsccpLog) -> str:
    return json.dumps(asdict(log), indent=2, ensure_ascii=False)

def to_csv(log: BrsccpLog) -> str:
    model = log.input_params.get('model_name', 'UNKNOWN')
    res = log.results
    trans = f"{res.transformation_from} to {res.transformation_to}" if res else "N/A"
    ddg = res.ddG_kcal_mol if res else "N/A"
    ratio = res.activity_ratio if res else "N/A"
    term = log.termination
    shifts = len(log.diagnostics.atom_shifts)
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow([model, trans, ddg, ratio, term, shifts])
    return si.getvalue().strip()

def main():
    parser = argparse.ArgumentParser(description="Parse BRSCCP log file")
    parser.add_argument("log", type=Path, nargs='?', help="Path to the log file")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    parser.add_argument("--out", type=Path, help="Output file path")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero if abnormal or warnings")
    parser.add_argument("--self-check", action="store_true", help="Run self-check on brsccp2V8X.log")
    
    args = parser.parse_args()
    
    if args.self_check:
        log_path = Path("brsccp2V8X.log")
        if not log_path.exists():
            print("brsccp2V8X.log not found for self-check")
            sys.exit(1)
        log = parse_log(log_path)
        try:
            assert log.version == "2.1.16"
            assert log.results is not None
            assert log.results.transformation_from == "2V8X"
            assert log.results.transformation_to == "2V8Y"
            assert log.results.ddG_kcal_mol == 2.00
            assert log.results.activity_ratio == 0.03
            assert log.diagnostics.best_trajectory == "x00009"
            assert len(log.diagnostics.atom_shifts) == 24
            assert log.termination == "NORMAL"
            print("Self-check passed!")
        except AssertionError as e:
            print(f"Self-check failed: {e}")
            # print some details to help debug
            print(f"Version: {log.version}")
            if log.results:
                print(f"Transformation: {log.results.transformation_from} to {log.results.transformation_to}")
                print(f"ddG: {log.results.ddG_kcal_mol}")
                print(f"Activity ratio: {log.results.activity_ratio}")
            print(f"Best trajectory: {log.diagnostics.best_trajectory}")
            print(f"Atom shifts count: {len(log.diagnostics.atom_shifts)}")
            print(f"Termination: {log.termination}")
            sys.exit(1)
        return

    if not args.log:
        parser.print_help()
        sys.exit(0)

    if not args.log.exists():
        print(f"Error: File {args.log} does not exist", file=sys.stderr)
        sys.exit(1)
        
    log_data = parse_log(args.log)
    
    if args.format == "json":
        output = to_json(log_data)
    else:
        output = to_csv(log_data)
        
    if args.out:
        args.out.write_text(output, encoding='utf-8')
    else:
        print(output)
        
    if args.strict:
        if log_data.termination != "NORMAL" or log_data.parse_warnings:
            sys.exit(1)

if __name__ == "__main__":
    main()
