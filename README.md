## ML-Agents Navigation
![gif](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

The objective of this project is to train a machine learning agent to navigate around a large square world, collecting yellow bananas and avoiding blue ones.

## Environment
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Dependencies
Make sure you have python 3.6 installed and a virtual environment activated, then install the PyTorch (torch), NumPy and UnityAgents packages which are required to run. You can install them using pip:
```
python3.6 -m pip install torch 
```

## Running the Project
Browse to the project directory and run `train.py` to generate training data based on the implemented learning model.
```
python train.py
```
Run `play.py` to watch an episode where the agent navigates around the world using the training data.