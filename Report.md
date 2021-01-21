# Project report

## Learning algorithm

The learning algorithm used is Deep Q Learning as described in this research paper:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

The model architecture for the neural network is configured with the following layers:

- input: 37 (state size), output: 64
- input: 64, output: 64
- input: 64, output: 4 (action size)

All layers are fully connected (not convolutional) and linear.

The following hyperparameters were used in the final training solution:

- Maximum steps per episode: 2000
- Start epsilon: 1.0
- End epsilon: 0.01
- Decay rate: 0.995

## Results

![results](images/results.png)

Episode 100	Average Score: 1.15  
Episode 200	Average Score: 4.70  
Episode 300	Average Score: 8.28  
Episode 400	Average Score: 10.80  
Episode 454	Average Score: 13.03  
Environment solved in 354 episodes!	Average Score: 13.03

## Ideas for future work

Some areas that could be investigated to improve the amount of time taken to train and the accuracy of final data:
- Optimising hyperparameters
- Double DQN
- Prioritized Experience Replay
- Dueling DQN
- Rainbow DQN
- Learning from pixels (using CNN)