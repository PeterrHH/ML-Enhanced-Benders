"""Run flowfirst training jobs on Modal GPUs.

One-time setup on your machine:
    .venv/bin/pip install modal          # already installed in this venv
    .venv/bin/modal setup                # browser login, stores the token

Create the two volumes (once), then upload a dataset (once per dataset; the 20-node pickle is 1.6 GB):
    .venv/bin/modal volume create flowfirst-datasets
    .venv/bin/modal volume create flowfirst-runs
    .venv/bin/modal volume put flowfirst-datasets flowfirst/datasets/flowfirst-20node-allgen_smp18.pkl /flowfirst-20node-allgen_smp18.pkl

Run a jobs file, one GPU container per job, all concurrently (same format as run_jobs.sh):
    FLOWFIRST_GPU=A100 .venv/bin/modal run --detach flowfirst/modal/train.py --jobs flowfirst/jobs/jobs-20node-modal.txt
or a single job:
    FLOWFIRST_GPU=A100 .venv/bin/modal run --detach flowfirst/modal/train.py --args "--variant flowfirst-gnn --config flowfirst/configs/config-20node-x8.json --epochs 2 --train-size 2048 --valid-size 1024 --tag smoke"

Fetch the run directories (also possible during a run: the volume is committed every 5 minutes):
    .venv/bin/modal volume get flowfirst-runs / flowfirst/runs/

--device cuda is appended to every job unless it names a device. The GPU type comes from the environment
variable FLOWFIRST_GPU (default A10G). Training is float64: measured on the 20-node GNN at batch 128, the A10G
(no fp64 tensor cores, ~$1.1/h) took 90 ms per step and the A100 (~$2.1/h) 21.7 ms, so use FLOWFIRST_GPU=A100.
Always pass --detach to modal run, otherwise closing the terminal or Ctrl-C kills every job.
"""
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent
# Peter's modules that flowfirst.dataset / flowfirst.train import (needed to unpickle the dataset and to build the nets)
ROOT_MODULES = ["networks.py", "gep_problem_operational.py", "gep_problem.py", "gep_config_parser.py",
                "create_gep_dataset.py", "data_wrangling.py"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "pandas==2.2.3", "scipy==1.13.1", "toml==0.10.2", "Pyomo==6.8.2",
                 "gurobipy==12.0.0", "tensorboard==2.18.0", "networkx==3.2.1", "matplotlib==3.9.2", "scikit-learn==1.6.1", "setuptools<81")
    .add_local_dir(REPO / "flowfirst", remote_path="/repo/flowfirst", ignore=["datasets", "runs", "__pycache__", "*.pkl"])
)
for module in ROOT_MODULES:
    image = image.add_local_file(REPO / module, remote_path=f"/repo/{module}")

datasets = modal.Volume.from_name("flowfirst-datasets", create_if_missing=True)
runs = modal.Volume.from_name("flowfirst-runs", create_if_missing=True)
app = modal.App("flowfirst")


@app.function(image=image, gpu=os.environ.get("FLOWFIRST_GPU", "A10G"), timeout=24 * 3600,
              volumes={"/repo/flowfirst/datasets": datasets, "/repo/flowfirst/runs": runs})
def train(args: list[str]) -> int:
    """Run one flowfirst.train job inside the container; the run directory lands on the runs volume."""
    os.chdir("/repo")
    if "--device" not in args:
        args = [*args, "--device", "cuda"]
    print("launching: python -m flowfirst.train " + " ".join(args), flush=True)
    proc = subprocess.Popen([sys.executable, "-u", "-m", "flowfirst.train", *args])
    last_commit = time.time()
    while proc.poll() is None:
        time.sleep(15)
        if time.time() - last_commit > 300:
            runs.commit()      # makes train.log / events readable from your machine while the job runs
            last_commit = time.time()
    runs.commit()
    return proc.returncode


def read_jobs(path):
    lines = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            lines.append(shlex.split(line))
    return lines


@app.local_entrypoint()
def main(jobs: str = "", args: str = ""):
    job_list = read_jobs(jobs) if jobs else []
    if args:
        job_list.append(shlex.split(args))
    assert job_list, "pass --jobs <file> and/or --args '<train.py arguments>'"
    print(f"submitting {len(job_list)} job(s) to {os.environ.get('FLOWFIRST_GPU', 'A10G')}")
    calls = [train.spawn(job) for job in job_list]
    codes = [call.get() for call in calls]
    for job, code in zip(job_list, codes, strict=True):
        tag = next((job[i + 1] for i, a in enumerate(job) if a == "--tag"), "?")
        print(f"job {tag}: exit code {code}")
    print("fetch results with:  .venv/bin/modal volume get flowfirst-runs / flowfirst/runs/")
    sys.exit(max(codes))
