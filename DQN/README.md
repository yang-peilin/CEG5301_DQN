# DQN

This code for DQN are adapted from [Kchu](https://github.com/Kchu/DeepRL_PyTorch)

## Speed Up:

Speed up sampling by having parallel threads for multiple environments: 
In train.py, for example, depending on your CUP cores and memories, you can set the number environments as below
```bash
# number of environments
    N_ENVS = 4
```
In our pendulum scenario, the default number of parallel threads is set to 1, because the percentage of the sampling period in all traning time is very small. In other words, the sampling costs very little time, and hence the optimization space is very narrow even by employing parallel coding technique.

GPU will help accelerate the processing. Make sure that you installed Pytorch correctly with the CUDA option. More details please visit [PyTorch official websites for installation](https://pytorch.org/get-started/locally/). 
For example, the following command install the PyTorch with CUDA 11.8 on Windows or Linux: 
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Installing Dependency:
You can use anaconda to create a python3 environment:

```bash
conda env create -f environment.yml
```

If some error messages from Anaconda are raised, you could choose to install the required python3 package manually. Run the following command with CMD in Windows or Shell in Linux or MacOS:

```bash
pip3 install torch pygame gym opencv_python
```

How to use
Enter the DQN directory, and run the python3 command 'python3 train.py':

```bash
cd DQN-pytorch # 
python3 train.py
```

When testing the bulit environment, you could let the code idle with the following command:

```bash
python3 train.py --idling
```

When you run these codes, it can automatically create two subdirectories under the current directory: ./data/model/ & ./data/plots/. These two directories are used to store the models and the results.

After training, you can plot the results by running result_show.py with appropriate parameters.

## References:
Human-level control through deep reinforcement learning (DQN) | [Paper](https://www.nature.com/articles/nature14236) |