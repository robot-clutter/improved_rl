# bridging_the_gap

Code of "Bridging the gap between learning and heuristic based pushing policies"

https://robot-clutter.github.io/bridging-the-gap

# Install

```bash
mkdir clutter
cd clutter

sudo apt-get install python3-tk python3-pip
sudo pip3 install virtualenv
virtualenv env --python=python3 --prompt='[clutter-env] '
. env/bin/activate

git clone https://github.com/robot-clutter/clt_core.git
cd clt_core
pip install -e .
cd ..
git clone https://github.com/robot-clutter/bridging_the_gap.git
cd bridging_the_gap
pip install -e .
```

For nvidia users:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```