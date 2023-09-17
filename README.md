# F23-Reinforcement-Learning
Project Lead: Elijah Grubbs (grubbse@umich.edu)  
Project Co-Lead: Casper Guo (casperg@umich.edu)

## Introduction  

Reinforcement learning gives ChatGPT its human-like feel, lets AlphaZero beat chess grandmasters, and empowers robots to teach themselves how to walk.

RL stands as a fundamental pillar of artificial intelligence, empowering AI agents to discern optimal decisions by learning from their interactions within a dynamic environment. 
Much like honing a skill through trial and error, RL enables AI to evolve continuously, mastering tasks and achieving heightened proficiency over time.

The most magical part is that you can use human intelligence to refine behavior, and human intelligence alone.  

## Description
This project consists of two major parts. Initially, we will place heavy emphasis on learning about deep reinforcement learning. Then, once we understand the basics, we will create synthetic reward signals from human preferences using supervised machine learning, so that we can train agents to perform tasks free from the constraints of pre-defined reward signals.

We will use the gymnasium python package to interact with environments and will create our own algorithms to train the agent from both pre-defined reward signals and human feedback.

## Goals  

- Introduce participants to the fundamental concepts, terminology, and principles of reinforcement learning, cultivating a strong grasp of the field's underlying mechanics.
- Provide participants with practical exposure to industry-standard tools such as Python, PyTorch, and NumPy.
- Challenge participants to implement essential reinforcement learning algorithms.
- Guide members through the intricate process of incorporating human feedback a synthetic reward mechanism that aligns AI agents' behavior with human preferences.
- Share results and visualize trained agents.
- Have fun and learn something :)

---
## Timeline  

#### Week 1 Sep 17, 2023:
- Introduction to RL
- Lab 0
- Dynamic Programming and Monte Carlo Solutions
- Simple Dynamic Programming gridworld lab in Python

#### Week 2 Sep 24, 2023:
- Monte Carlo Lab
- Explain REINFORCE
- Pytorch tutorial
- Implement REINFORCE

#### Week 3 Oct 1, 2023:
- Implement REINFORCE
- Explain A2C

#### Week 4 Oct 8, 2023:
- Recap A2C
- Implement A2C

#### Week 5 Oct 22, 2023:
- Synthesize Reward Signal from Human Preferences

#### Week 6 Oct 29, 2023:
- Tune environments to human preferences

#### Week 7 Nov 5, 2023:
- Tune environments to human preferences

#### Week 8 Nov 12, 2023:
- Wrap Up

#### Project Expo Nov 19, 2023:
- Show off what we have done

---
## Setup

First, clone this repo (via ssh)

```bash
git clone git@github.com:MichiganDataScienceTeam/F23-Reinforcement-Learning.git
```

### Virtual Environment

You can choose whether or not to use a virtual environment for this project (though it is recommended). The setup guide shows how to create a venv through pip, but it can also be done via Conda if you want. The important thing is that you can run test-setup.py 

We are going to initialize a Python virtual environment with all the required packages. We use a virtual environment here to isolate our development environment from the rest of your computer. This is helpful in not leaving messes and keeping project setups contained.

First create a Python 3.8+ virtual environment. The virtual environment creation code for Linux/MacOS is below:

```bash
python3 -m venv venv
```


Now that you have a virtual environment installed, you need to activate it. This may depend on your system, but on Linux/MacOS, this can be done using

```bash
source ./venv/bin/activate
```

Now your computer will know to use the Python installation in the virtual environment rather than your default installation.

After the virtual environment has been activated, we can install the required dependencies into this environment using

```bash
pip install -r requirements.txt
```

Once the packages are installed, you should be able to run `test-setup.py` without problems

```bash
python3 test-setup.py
```

### Note for Windows display problems using WSL  

If the `test-setup.py` runs fine but you never see a window pop up with the environment. Full instructions are at [this link.](https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2)  
Here I have typed out the top answer from the linked StackOverflow page.

First you'll have to install VcXsrv:  
1. `sudo apt-get update`
2. `sudo apt-get install python*-tk` where the `*` you replace with your python version. E.g. for Python 3.8 do `sudo apt-get install python3.8-tk`

Next, change where outputs are send on the back-end using the following command.  
1. `export DISPLAY=localhost:0.0`. Add this command to the end of `~/bashrc` to make it permanent, else you'll have to run this command every time you re-open your terminal.

Finally, quit VcXsrv if you have it open. And re-run the application with "Diable Access Control" selected.

