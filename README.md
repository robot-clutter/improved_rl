# Improved Reinforcement Learning Pushing Policies via Heuristic Rules


**CAUTION**: The code is under development so some bugs and missing documentation is to be expected.

# Install

The code has been tested on a PC with Ubuntu 20.04 equipped with an Nvidia GPU.

```bash
mkdir robot-clutter
cd robot-clutter

sudo apt-get install python3-tk python3-pip
sudo pip3 install virtualenv
virtualenv env --python=python3 --prompt='[clutter-env] '
echo "export ROBOT_CLUTTER_WS=$PWD" >> env/bin/activate
source env/bin/activate

git clone https://github.com/robot-clutter/clt_core.git
cd clt_core
pip install -e .
cd ..

git clone https://github.com/robot-clutter/clt_assets.git
cd clt_assets
pip install -e .
cd ..

git clone https://github.com/robot-clutter/clt_models.git

git clone https://github.com/robot-clutter/bridging_the_gap.git
cd bridging_the_gap
pip install -e .
cd ..

```

Then install PyTorch with CUDA support:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

# Run

Everything is run by the `run.py` script. Try `python run.py --help` for a list of the arguments. The available experiments (`--exp` argument) are the following:

| --exp                    | Description                                                                    | Related args                                                |
|--------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------|
| `collect_dataset`        | Collects a dataset of transitions using random actions                         |`--seed`, `--dataset_dir`, `n_episodes`                      | 
| `train_rl`               | Preprocesses the dataset and trains the `RL` policy using offline RL           |`--seed`, `--resume_model`                                   | 
| `eval_rl`                | Run evaluation episodes for the trained `RL` policy                            |`--seed`, `--model_to_eval`, `--n_episodes`, `--compare_with`| 
| `train_rl_es`            | Preprocesses the dataset and trains the `RL-ES` hybrid policy using offline RL |`--seed`, `--resume_model`                                   | 
| `eval_rl_es`             | Run evaluation episodes for the trained `RL-ES` policy                         |`--seed`, `--model_to_eval`, `n_episodes`                    | 
| `train_rl_les`           | Preprocesses the dataset and trains the `RL-LES` policy using offline RL       |`--seed`, `--resume_model`                                   | 
| `eval_rl_les`            | Run evaluation episodes for the trained `RL-LES` policy                        |`--seed`, `--model_to_eval`, `n_episodes`                    | 
| `eval_es`                | Run evaluation episodes for the Empty-Space (ES) heuristic policy              |`--seed`, `--model_to_eval`, `n_episodes`                    |
| `eval_les`               | Run evaluation episodes for the Local Empty Space (LES) heuristic policy       |`--seed`, `--model_to_eval`, `n_episodes`                    |

By default, the logs of the experiments are saved in `$ROBOT_CLUTTER_WS/clt_logs`.

E.g.

```bash
python run.py --exp=collect_dataset --n_episodes=10000
python run.py --exp=train_rl
```
