import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Command to run when GPU becomes available
COMMAND_TO_RUN = "python run_finetuning.py"
# COMMAND_TO_RUN = 'echo ">>> Running command because GPU memory is sufficient. <<<"'

# Memory threshold in GB
MEM_THRESHOLD = 20
CUDA_INDEX = 0

# Check interval in seconds (10 minutes)
CHECK_INTERVAL = 10 * 60


def get_gpu_free_memory():
    """Get free memory on cuda:0 in GB."""
    try:
        # Run nvidia-smi and capture output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_list = result.stdout.splitlines()

        # Parse the output to get the free memory in MB
        free_memory_mb = float(gpu_list[CUDA_INDEX].strip())

        # Convert to GB
        free_memory_gb = free_memory_mb / 1024.0

        return free_memory_gb
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return 0


def main():
    print(
        f"Starting GPU monitor. Waiting for cuda:0 to have more than {MEM_THRESHOLD}GB free memory."
    )
    print(f"Will check every {CHECK_INTERVAL/60} minutes.")

    while True:
        free_memory = get_gpu_free_memory()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{timestamp}] Free GPU memory: {free_memory:.2f}GB")

        if free_memory > MEM_THRESHOLD:
            print(
                f"GPU has {free_memory:.2f}GB of free memory, which exceeds threshold of {MEM_THRESHOLD}GB."
            )
            print(f"Running command: {COMMAND_TO_RUN}")

            try:
                # Run the command when memory threshold is met
                subprocess.run(COMMAND_TO_RUN, shell=True, check=True)
                print("Command completed successfully. Exiting.")
                break
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                sys.exit(1)
        else:
            print(
                f"Not enough GPU memory available. Waiting {CHECK_INTERVAL/60} minutes before checking again..."
            )
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    print(">>> Running GPU monitor <<<")
    main()
