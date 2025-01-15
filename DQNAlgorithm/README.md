# Reinforcement Learning
This repository is a creation of a simple Reinforcment learning model inteaded to get the basics of creating them using pytorch and gymnasmim

# Requirements
Pytorch
gymnasium
matplotlib

# Task
This uses the cart pull problem where the system can choice to move the cart right or left and will try to keep the sick in the air 

# Replay Memory
For this model we are using replay memory meaning it sotres the transitions that the agnet observes, allowing us to reuse the data for later. By sampling it randomly we are able to much better stabilize are DQN training

# DQN Algorithm
The goal is to train a decision making strategy that maximizes the discounted cumulative reward meaning we are trying to set up are algorithm so we have control over if we care more about it's results now or in the future