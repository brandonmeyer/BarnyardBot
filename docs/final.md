---
layout: default
title: Final Report
---
## BarnyardBot Video

## Project Summary
<img src="https://user-images.githubusercontent.com/51243475/144334516-68004b17-8994-4385-a4c4-7a60bf262afb.png" alt="BarnyardBot Logo" width="425" height="300">  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Gathering animal resources is a crucial part of *Minecraft*<sup>1</sup> that can often be tedious. When a player wants milk or wool for baking a cake, healing status effects, making beds, or building a new colorful project, they must find the correct tools to use and then track down the respective animal for harvesting. On average, it takes a minute for a sheep's wool to grow back<sup>2</sup> in *Minecraft*. This means that most of the time, the player must wait for the cooldown to end if they need a specific color of wool from a specific sheep. This makes animal resource harvesting time consuming, and forces the player to sit around and wait instead of exploring the world, mining, or working on a new build.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BarnyardBot solves this problem by automating the animal resource harvesting process. With the power of reinforcement learning, BarnyardBot can navigate throughout the animal pen and harvest resources for the player. The player can specify whether they need milk, a certain color of wool, or a ratio of resources. BaryardBot collects the requested items and gives the player more free time, removing the needs to sit around and wait for cooldowns or chase around animals. This gives the player more time to explore the world, go mining for precious stones, cook food, or build their next work of art.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BarnyardBot takes the agent's current target and the player's requests as input and determines the current best solution for meeting the player's requests. For example, if the player only wants milk and does not want wool, BarnyardBot will find cows to milk and ignore the sheep it comes across. Taking the player's requests is important for solving the animal resource harvesting problem because the player won't always need every resource. BarnyardBot aims to use this information to harvest resources as efficiently and accurately as possible. If the player only needs a certain color of wool for a build, BarnyardBot won't waste time collecting other colors. Whenever the player wants a different resource, they can tell BarnyardBot and it will adjust accordingly. This prevents the player from needing multiple bots to perform a similar task, since BarnyardBot can harvest whatever they need from sheep and cows. The goal of this project is to maximize resource output in *Minecraft* using *Malmo*<sup>3</sup> and *RLlib*<sup>4</sup>, with the output based on the requests of the player.

## Approaches
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BarnyardBot uses reinforcement learning through *RLlib*, *Gym*<sup>5</sup>, and *Malmo* to adjust agent behavior in *Minecraft* for harvesting resources. When creating the first baseline for the project, the most important details were how the agent would observe its environment (the observation space) and how the agent would perform its actions (the action space).

- discrete vs continuous
- observations from grid vs observationsf rom target
- three approaches to ratios: text, chat, blocks

## Evaluation

## References
[Minecraft<sup>1</sup>](https://www.minecraft.net/en-us/login)\
[Sheep Wool Statistic<sup>2</sup>](https://minecraft.fandom.com/wiki/Tutorials/Wool_farming)\
[Malmo<sup>3</sup>](https://www.microsoft.com/en-us/research/project/project-malmo/)\
[RLlib<sup>4</sup>](https://docs.ray.io/en/latest/rllib.html)\
[Gym<sup>4</sup>](https://gym.openai.com/)\
[RLlib PPO](https://docs.ray.io/en/latest/rllib-algorithms.html#ppo)\
[PPO Algorithm Source](https://blogs.oracle.com/ai-and-datascience/post/reinforcement-learning-proximal-policy-optimization-ppo)\
[Malmo Gitter Chat](https://gitter.im/Microsoft/malmo?at=578aa4fd3cb52e8b24cee1af)\
[Malmo XML](https://microsoft.github.io/malmo/0.21.0/Schemas/MissionHandlers.html)\
[Writing 3x3 Letters in Minecraft (For Home Page/Video Image)](https://www.youtube.com/watch?v=vHExVqV-FD8)\
[Reinforcement Learning Algorithm Flowchart](https://static.us.edusercontent.com/files/eS20DbiGQfi4P2skbCN9WYeD)\
[CS175 Assignment 2 for understanding of RLlib/Gym](https://canvas.eee.uci.edu/courses/40175/files/folder/assignment2?preview=16066666)\
[Displaying Images Side by Side in GitHub](https://stackoverflow.com/questions/24319505/how-can-one-display-images-side-by-side-in-a-github-readme-md)
