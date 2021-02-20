## ML-Agents Navigation
![gif](images/trained.gif)

The objective of this project is to train a machine learning agent to navigate around a large square world, collecting yellow bananas and avoiding blue ones.

## Environment
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Setup
1. Clone this repository to your local drive.

2. Download the environment from one of the links below and extract in the repository directory:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. Navigate to the repository directory then create and activate a new environment with Python 3.6:
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```  

4. Install the required packages torch, numpy and unityagents. They can be installed using pip:
    ```
    pip install torch numpy unityagents
    ```

## Running the Project
With the environment activated run `train.py` to begin training.  

The trained weights are saved to the file `checkpoints.pth` once the required score is reached.

Run `play.py` to watch an episode where a smart agent navigates around the world using the trained data.