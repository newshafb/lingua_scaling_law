# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import random
import shutil
import time
import subprocess
import requests
from huggingface_hub import snapshot_download

SBATCH_COMMAND = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --job-name={name}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-gpu={ncpu}
#SBATCH --time={time}
#SBATCH --mem={mem}

#SBATCH --output={dump_dir}/%A_%a/stdout
#SBATCH --error={dump_dir}/%A_%a/stderr

#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block

# Mimic the effect of "conda init", which doesn't work for scripts
eval "$({conda_exe} shell.bash hook)"
source activate {conda_env_path}

{command1}

{command2}

{command3}

echo "Done!"
"""

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    print(f"------")


def run_sbatch(args, snapshot_id, filter_name, cmd1, cmd2, cmd3):

    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = os.path.dirname(os.path.dirname("/data/home/new/anaconda3/envs/lingua_241022"))
    job_name = f"terashuf_{snapshot_id}_{filter_name}"
    dump_dir = f"{args.dump_dir}/{snapshot_id}/{filter_name}"
    os.makedirs(dump_dir, exist_ok=True)
    sbatch = SBATCH_COMMAND.format(
        account=args.account,
        qos=args.qos,
        name=job_name,
        dump_dir=dump_dir,
        nodes=args.nodes,
        ngpus=args.ngpus,
        ncpu=args.ncpus,
        mem=args.memory,
        time=args.time,
        partition=args.partition,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        command1=cmd1,
        command2=cmd2,
        command3=cmd3,
    )

    print("Writing sbatch command ...")
    #print(sbatch)
    with open(f"{dump_dir}/submit.slurm", "w") as f:
        f.write(sbatch)

    print("Submitting job ...")
    print(f"sbatch --array=0-7 {dump_dir}/submit.slurm")
    os.system(f"sbatch --array=0-7 {dump_dir}/submit.slurm")



def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16, # Don't hesitate to increase this number to lower the download time
                )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename="chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def main(args):
    dataset = args.dataset
    memory = args.memory
    data_dir = args.data_dir
    seed = args.seed
    snapshot_id = args.snapshot_id
    filter_name = args.filter
    
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
        "quality_pajama": None,
    }[dataset]
    src_dir = f"{data_dir}/{dataset}"
    if "quality_pajama" in dataset:
        src_dir = f"{data_dir}/{snapshot_id}/head/{filter_name}"
    out_dir = f"{src_dir}_shuffled"
    #os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    work_dir = os.getcwd()
    prefix = f"chunk."
    orig_extension = {
        "fineweb_edu": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
        "dclm_baseline_1.0_10prct": ".jsonl.zst",
        "quality_pajama": ".jsonl",
    }[dataset]
    cat_command = {
        "fineweb_edu": "cat",
        "fineweb_edu_10bt": "cat",
        "dclm_baseline_1.0": "zstdcat",
        "dclm_baseline_1.0_10prct": "zstdcat",
        "quality_pajama": "cat",
    }[dataset]
    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "dclm_baseline_1.0": "*.jsonl.zst",
        "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
        "quality_pajama": None,
    }[dataset]
    suffix = ".jsonl"
    nchunks = 8
    k_validation = 10000  # Number of lines to take from each chunk for validation

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Download dataset
    #download_dataset(repo_id, src_dir, allow_patterns)

    if "fineweb" in dataset:
        parquet_to_jsonl(dataset, work_dir, src_dir, src_dir)

    # Set up environment variables
    os.environ["MEMORY"] = f"{memory}"
    os.environ["SEED"] = f"{seed}"

    # Run the original shuffling and splitting command
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    

    
    nfiles = sum(1 for file in os.listdir(f"{src_dir}") if file.endswith('.jsonl'))
    numbers = list(range(nfiles))
    
    # Shuffle the list
    random.seed(seed)
    random.shuffle(numbers)
    
    # Break the list into nchunks chunks
    chunk_size = len(numbers) // nchunks
    chunks = [numbers[i * chunk_size:(i + 1) * chunk_size] for i in range(nchunks)]
    
    # Write each list to a separate file
    for i, chunk in enumerate(chunks):
        with open(f"{out_dir}/chunk_{i}.txt", "w") as file:
            file.write("\n".join(map(lambda x: f"{src_dir}/{x:04d}{suffix}", chunk)))
    
    chunk_id="${SLURM_ARRAY_TASK_ID}"
    chunk_file = f"{out_dir}/{prefix}{chunk_id}{suffix}"
    validation_file = f"{out_dir}/{prefix}{chunk_id}.val{suffix}"
    cmd1 = (f"time (ulimit -n 100000 && "
            f"cat {out_dir}/chunk_{chunk_id}.txt | "
            f"xargs {cat_command} | {terashuf_executable} >| "
            f"{chunk_file}); "
            f"trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;")
    cmd2 = f"time head -n {k_validation} {chunk_file} >> {validation_file}"
    cmd3 = f"time sed -i '1,{k_validation}d' {chunk_file}"

    run_sbatch(args, snapshot_id, filter_name,cmd1, cmd2, cmd3)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="/fsx-data-quality-law/new/data/RedPajama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot_id", type=str, default="2014-15")

    parser.add_argument("--account", type=str, default="atom")
    parser.add_argument("--qos", type=str, default="atom_high")
    parser.add_argument("--time", type=str, default="5:00:00")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--ncpus", type=int, default=12)
    parser.add_argument("--dump_dir", type=str, default="/fsx-checkpoints/new/terashuf")
    parser.add_argument("--partition", type=str, default="learn")
    parser.add_argument("--filter", type=str, default="orig")

    args = parser.parse_args()

    main(args)

