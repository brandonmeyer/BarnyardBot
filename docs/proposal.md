---
layout: default
title: Proposal
---

## Summary
Gathering resources from farm animals in *Minecraft* can be tedious. When a player needs bulk buckets of milk or stacks of wool, they have to walk to the animal pen and manually gather all of the resources. With cooldowns, the player could find themself waiting for more milk or for wool to grow back. This is time consuming, and prevents the player from quickly accessing these items for projects like baking, healing status effects, creating colorful builds, or making beds to sleep through the night. Animal-AI will fix this problem by standing in an animal pen and automatically gathering these resources for the player. It will take the agent's vision as input and output the correct action to harvest the current animal's resources. For example, seeing a cow promts the agent to use a bucket and seeing a sheep prompts the agent to use shears. The goal of this project is to use *Malmo* for animal resource gathering automation and give the player more time to explore their creativity in *Minecraft*, rather than harvest resources.

## Algorithms
Reinforcement learning with actor critic policy gradient will help the agent determine which tools to use (for example, shears should be used if a sheep is seen).

## Evaluation Plan
Quantitative evaluation for the project will focus on items harvested and points scored by the agent. When the agent correctly responds to an animal, it will receive a set number of points. More items harvested by the agent will result in more points scored. After each mission, the agent will be evaluated based on how many points it earned in the timeframe.  The agent will be successful if it can harvest useful numbers of resources and score highly after reinforcement learning is complete. Ideally, the agent will harvest enough items for the player to use without having to wait. This could be half of a chest full of milk or a couple of stacks of wool.

## Appointment
10/21/21 at 7:30pm
