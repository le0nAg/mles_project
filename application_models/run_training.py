#!/usr/bin/env python3
"""
run_training.py - Train and export both model candidates

Usage:
    python run_training.py --samples-dir ../samples
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", default="../samples")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only export")
    args = parser.parse_args()
    
    models = [
        ("mcunet_tiny", True),   # (model_name, early_exit)
        ("tflm_tiny", False),
    ]
    
    for model_name, early_exit in models:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print("="*60)
        
        if not args.skip_train:
            # Train
            cmd = [
                sys.executable, "train.py",
                "--model", model_name,
                "--epochs", str(args.epochs),
                "--samples-dir", args.samples_dir,
            ]
            if not early_exit:
                cmd.append("--no-early-exit")
            
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        
        # Export
        suffix = "_ee" if early_exit else "_no_ee"
        model_path = f"checkpoints/{model_name}{suffix}_final.keras"
        
        if Path(model_path).exists():
            cmd = [
                sys.executable, "export.py",
                model_path,
                "--output-dir", "exports",
                "--samples-dir", args.samples_dir,
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            print(f"Model not found: {model_path}")
    
    # Compare results
    print("\n" + "="*60)
    print("Comparing exported models")
    print("="*60)
    
    tflite_files = list(Path("exports").glob("*_int8.tflite"))
    if tflite_files:
        cmd = [sys.executable, "evaluate.py"] + [str(f) for f in tflite_files]
        cmd += ["--samples-dir", args.samples_dir, "--output", "exports/comparison.json"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
