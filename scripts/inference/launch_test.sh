#!/bin/bash

# Directory for results
save_dir="/data/MPDresultsSTOMPSamplesExplore/"

# Define parameters
model_ids=("EnvSpheres3D-RobotPanda" "EnvSimple2D-RobotPointMass" "EnvNarrowPassageDense2D-RobotPointMass" "EnvDense2D-RobotPointMass")
sample_vers=("noise_ddpm")
seeds=($(seq 0 299))
n_samples=100
step_size=0.5
guidance_samples=(5 10 15 20 25 30 40 50)

# Iterate over all combinations of parameters
for model_id in "${model_ids[@]}"; do
  for seed in "${seeds[@]}"; do
    for sample_ver in "${sample_vers[@]}"; do
      for sample_num in "${guidance_samples[@]}"; do
        # Run inference.py with specified arguments
        python ./scripts/inference/inference.py \
          --model_id "$model_id" \
          --seed "$seed" \
          --sample_ver "$sample_ver" \
          --n_samples "$n_samples" \
          --step_size "$step_size" \
          --guidance_sample "$sample_num" \
          --save_dir "$save_dir"
        
        # Optional: Add a small delay to prevent overwhelming system resources
        sleep 0.1
      done
    done
  done
done