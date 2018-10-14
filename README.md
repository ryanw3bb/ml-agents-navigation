## Unity ML-Agents
This repository contains a solution to the ML-Agents Navigation project, part of Udacity's Deep Reinforcement Learning Nanodegree.

The objective is to train a machine learning agent to navigate around a large square world, collecting yellow bananas and avoiding blue ones.

This project has been developed using Python 3.6 but should work using newer releases.
## Environment
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available:

0 - move forward  
1 - move backward  
2 - turn left  
3 - turn right  

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Dependencies
This project requires the PyTorch (torch), NumPy and UnityAgents packages to run, please make sure they are present in your Python environment. You can install them using pip:
```
python3.6 -m pip install torch 
```

## Running the Project
Browse to the project directory and run `train.py` to generate training data based on the implemented learning model.
```
python3.6 Train.py
```
Run `replay.py` to watch an episode where the agent navigates around the world using the generated training data.