# Homewrok5 Submission

Problem2.py contains training and testing code. Problem2_render.py contains visulization code.
I choose DQN pipeline to achieve the homework

## Training Result
<p align="center">
    <img src="https://github.com/Nicholas-Yang/Figures/blob/335f8c154b84e04ebb20c55abfb2c6aeb01ca1b5/tensorboard.png"><br/>
    <em>Training Result.</em>
</p>

## Testing Result
<p align="center">
    <img src="https://github.com/Nicholas-Yang/Figures/blob/219626723e1854507c5c4c2fe6f29c21aa0d2d4b/Testing%20Vis.gif"><br/>
    <em>Training Result.</em>
</p>

## Comments

DQN (Deep Q Learning) uses a neural network to replace Q table. The Q network will predict the optimal actions based on Q value using temporal difference method. Also the parameters of the network can be updated using gradient descent method.
The code can be divided into three parts including training,testing and visulization.
Since DQN is too simple and the hyperparameter of the network is not very optimal. The mean reward is around 18 after 20k iteration. And the visulization shows the control result is not very optimal.

