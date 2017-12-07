# Reinforcement Learning Analysis in the OpenAI Gym Environments
Final project for CS182 by Sam Oh, Brandon Hills, and Collin Price.

## Overview
In this project, we calculated optimal solutions for three games (Taxi, Frozen Lakes, N-Chain) and used these solutions as a baseline for testing the effectiveness of Q-learning on these games. We modelled Taxi as a search problem, and used A* search to find the optimal policy. Frozen Lakes and N-Chain were stochastic games, so we modelled them as MDPs and used value iteration to create optimal policies for them.

## Requirements
You need to have python 2.7, as well as have the gym module installed. If you arlready are running python 2.7, you can simply run **pip install gym** or run **pip install -r requirements.txt**.

## Specification
All of our learning agents (Q-learning, Incentivized Q-learning, Random agent) can be found in learningAgents.py. Our MDP models for Frozen Lakes and N-Chain can be found in MDP.py, and the corresponding Value Iteration agents for these games in valueIterationAgents.py. Our optimal solution for the taxi game can be found in searchTaxi.py, where we implement A* search. Utils.py contains some helper classes (mainly the Counter and PriorityQueue) that were provided in our problem sets, which can be found [here](http://ai.berkeley.edu/project_overview.html).

## Running
All of our results are compiled in our test files for each game: testTaxi.py, testFrozenLakes.py, and testNChain.py. Running any of these files will compare the results of the Q-learning and incentivized Q-learning agents with the optimal solution for that game. With N-Chain, we just use regular Q-learning.
