"""
Master file to run all evaluations at once
Run the commands using a config file (benchmark_config.yaml)
"""

import yaml
import subprocess
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'benchmark_config.yaml')
BENCHMARKS_DIR = os.path.dirname(__file__)

# Helper to run a command in a specific folder
def run_command(cmd, cwd=None):
    print(f"Running: {cmd} in {cwd if cwd else os.getcwd()}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config(CONFIG_PATH)

    # CHAMMI
    if config.get('CHAMMI', False):
        morphem_dir = os.path.join(BENCHMARKS_DIR, 'morphem')
        chammi_cmd = (
            f"python feature_extraction.py "
            f"--root_dir {config['CHAMMI_IMAGES_PATH']} "
            f"--feat_dir {config['CHAMMI_FEATURES_PATH']} "
            f"--model {config['MODEL_TYPE']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--batch_size {config['CHAMMI_BATCH_SIZE']} "
        )
        run_command(chammi_cmd, cwd=morphem_dir)

        # Run scoring from benchmarks folder (not morphem)
        benchmark_cmd = (
            f"python -c \"from morphem.benchmark import run_benchmark; "
            f"run_benchmark('{config['CHAMMI_IMAGES_PATH']}', '{config['CHAMMI_SCORE_PATH']}', "
            f"'{config['CHAMMI_FEATURES_PATH']}', 'pretrained_vit_features.npy')\""
        )
        run_command(benchmark_cmd, cwd=BENCHMARKS_DIR)

    # HPA
    if config.get('HPA', False):
        hpa_dir = os.path.join(BENCHMARKS_DIR, 'hpa')
        hpa_cmd = (
            f"accelerate launch --multi_gpu --num_processes={config['NUM_GPUS']} accelerate_hpa_features.py "
            f"--model {config['MODEL_TYPE']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--image_folder {config['HPA_IMAGES_PATH']} "
            f"--batch_size {config['HPA_BATCH_SIZE']} "
            f"--num_workers {config['HPA_NUM_WORKERS']} "
            f"--output_folder {config['HPA_FEATURES_PATH']}"
        )
        run_command(hpa_cmd, cwd=hpa_dir)

        train_cmd = (
            f"python train_classification.py -f {config['HPA_SCORE_PATH']} -cc locations -uc challenge_cats"
        )
        run_command(train_cmd, cwd=hpa_dir)

    # Neuron Features
    if config.get('NEURON_FEATURES', False):
        neuron_dir = os.path.join(BENCHMARKS_DIR, 'neuron_features')
        neuron_cmd = (
            f"accelerate launch --multi_gpu --num_processes={config['NUM_GPUS']} extraction.py "
            f"--model {config['MODEL_TYPE']} "
            f"--image_folder {config['NEURON_IMAGES_PATH']} "
            f"--model_path {config['MODEL_PATH']} "
            f"--output_folder {config['NEURON_FEATURES_PATH']} "
            f"--num_workers {config['NEURON_NUM_WORKERS']}"
        )
        run_command(neuron_cmd, cwd=neuron_dir)

        classifier_cmd = (
            f"python classifier.py --embedding_path {config['NEURON_FEATURES_PATH']}"
        )
        run_command(classifier_cmd, cwd=neuron_dir)

if __name__ == "__main__":
    main()
