# STOMP-Guided Diffusion

---
## Installation

Pre-requisites:
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

Clone this repository with
```bash
cd ~
git clone https://github.com/rml-unist/STOMP-Guided_Diffusion.git
cd STOMP-Guided_Diffusion
```

Download [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) and extract it under `deps/isaacgym`
```bash
mv ~/Downloads/IsaacGym_Preview_4_Package.tar.gz ~/mpd-public/deps/
cd ~/mpd-public/deps
tar -xvf IsaacGym_Preview_4_Package.tar.gz
```

Run the bash setup script to install everything.
```
cd ~/mpd-public
bash setup.sh
```

---
## Running the MPD inference

To try out the MPD inference, first download the data and the trained models. 

```bash
conda activate mpd
```

```bash
gdown --id 1mmJAFg6M2I1OozZcyueKp_AP0HHkCq2k
tar -xvf data_trajectories.tar.gz
gdown --id 1I66PJ5QudCqIZ2Xy4P8e-iRBA8-e2zO1
tar -xvf data_trained_models.tar.gz
```

After downloading, please change the below variables
```
TRAINED_MODELS_DIR (scripts/inference/inference.py)
data_dir (mpd/datasets/trajectories.py)
```

Run the inference script
```bash
cd scripts/inference
python inference.py
```

Comment out the `model-id` variable in `scripts/inference/inference.py` to try out different models
```python
model_id: str = 'EnvDense2D-RobotPointMass'
model_id: str = 'EnvNarrowPassageDense2D-RobotPointMass'
model_id: str = 'EnvSimple2D-RobotPointMass'
model_id: str = 'EnvSpheres3D-RobotPanda'
```
The results will be saved under `data_trained_models/[model_id]/results_inference/`.

To run multiple experiment, use the below command
```bash
python ./scripts/inference/launch_test.py
```

---
## Credits

The most of this repository is from

Carvalho, J.; Le, A.T.; Baierl, M.; Koert, D.; Peters, J. (2023). **_Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models_**, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

[<img src="https://img.shields.io/badge/arxiv-%23B31B1B.svg?&style=for-the-badge&logo=arxiv&logoColor=white" />](https://arxiv.org/abs/2308.01557)


