import os
import pandas as pd
from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.problems.generators import NQueensProblem
from lattice_weaver.core.csp_engine.tracing import ExecutionTracer

def generate_nqueens_trace(n: int = 4, output_path: str = "nqueens_4_trace.csv"):
    print(f"Generando trace para N-Queens {n}x{n}...")
    nqueens_problem_generator = NQueensProblem()
    csp = nqueens_problem_generator.generate(n=n)

    tracer = ExecutionTracer(enabled=True)
    solver = CSPSolver(csp, tracer=tracer)
    solver.solve()

    events_data = []
    for event in tracer.events:
        events_data.append({
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "variable": event.variable,
            "value": event.value,
            "depth": event.metadata.get("depth"),
            "domain_size": event.metadata.get("domain_size"),
            "pruned_values": event.metadata.get("pruned_values"),
            "constraint": event.metadata.get("constraint"),
            "result": event.metadata.get("result"),
        })

    df = pd.DataFrame(events_data)
    df.to_csv(output_path, index=False)
    print(f"Trace guardado en: {output_path}")

if __name__ == "__main__":
    output_dir = "/home/ubuntu/lattice-weaver/examples"
    os.makedirs(output_dir, exist_ok=True)
    generate_nqueens_trace(n=4, output_path=os.path.join(output_dir, "nqueens_4_trace.csv"))

