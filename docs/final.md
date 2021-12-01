---
layout: default
title: Final Report
---
## Video Here

## Project Summary
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Gathering animal resources is a crucial part of *Minecraft* that can often be tedious. When a player wants milk or wool for baking a cake, healing status effects, making beds, or building a new colorful project, they must find the correct tools to use and then track down the respective animal for harvesting. On average, it takes a minute for a sheep's wool to grow back in *Minecraft* (see Sheep Wool Statistic in References). This means that most of the time, the player must wait for the cooldown to end if they need a specific color of wool from a specific sheep. This makes animal resource harvesting time consuming, and forces the player to sit around and wait instead of exploring the world, mining, or working on a new build.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BarnyardBot solves this problem by automating the animal resource harvesting process. With the power of reinforcement learning, BarnyardBot can navigate throughout the animal pen and harvest resources for the player. The player can specify whether they need milk, a certain color of wool, or a ratio of resources. BaryardBot collects the requested items and gives the player more free time, removing the needs to sit around and wait for cooldowns or chase around animals. This gives the player more time to explore the world, go mining for precious stones, cook food, or build their next work of art.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BarnyardBot takes the agent's current target and the player's requests as input and determines the current best solution for meeting the player's requests. For example, if the player only wants milk and does not want wool, BarnyardBot will find cows to milk and ignore the sheep it comes across. Taking the player's requests is important for solving the animal resource harvesting problem because the player won't always need every resource. BarnyardBot aims to use this information to harvest resources as efficiently and accurately as possible. If the player only needs a certain color of wool for a build, BarnyardBot won't waste time collecting other colors. The goal of this project is to maximize the requested resource output.

## Approaches
- observations from grid vs observationsf rom target
- three approaches to ratios: text, chat, blocks

## Evaluation

## References
[RLlib](https://docs.ray.io/en/latest/rllib.html)\
[RLlib PPO](https://docs.ray.io/en/latest/rllib-algorithms.html#ppo)\
[PPO Algorithm Source](https://blogs.oracle.com/ai-and-datascience/post/reinforcement-learning-proximal-policy-optimization-ppo)\
[Gym](https://gym.openai.com/)\
[Malmo](https://www.microsoft.com/en-us/research/project/project-malmo/)\
[Malmo Gitter Chat](https://gitter.im/Microsoft/malmo?at=578aa4fd3cb52e8b24cee1af)\
[Malmo XML](https://microsoft.github.io/malmo/0.21.0/Schemas/MissionHandlers.html)\
[Minecraft](https://www.minecraft.net/en-us/login)\
[Sheep Wool Statistic](https://minecraft.fandom.com/wiki/Tutorials/Wool_farming)\
[Writing 3x3 Letters in Minecraft (For Home Page/Video Image)](https://www.youtube.com/watch?v=vHExVqV-FD8)\
[Reinforcement Learning Algorithm Flowchart](https://static.us.edusercontent.com/files/eS20DbiGQfi4P2skbCN9WYeD)\
[CS175 Assignment 2 for understanding of RLlib/Gym](https://canvas.eee.uci.edu/courses/40175/files/folder/assignment2?preview=16066666)\
[Displaying Images Side by Side in GitHub](https://stackoverflow.com/questions/24319505/how-can-one-display-images-side-by-side-in-a-github-readme-md)
