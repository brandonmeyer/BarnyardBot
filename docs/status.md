---
layout: default
title: Status
---

video here  

## Project Summary
Gathering resources from cows and sheep in *Minecraft* can be tedious. When a player needs bulk buckets of milk or stacks of wool, they have to walk to the animal pen and manually gather all of the resources. If a player only wants one color of wool, they have to chase after the sheep in their pen and find the right colors. With cooldowns, the player could also find themself waiting for more milk or for wool to grow back. This is time consuming, and prevents the player from quickly accessing these items for projects like baking, healing status effects, creating colorful builds, or making beds to sleep through the night. Animal-AI fixes this problem by navigating throughout an animal pen and automatically gathering these resources for the player. It takes the agent's current target as input and output the correct action to harvest the current animal's resources. For example, seeing a cow promts the agent to use a bucket and seeing a sheep prompts the agent to use shears. BarynardBot will only harvest resources that the player specifies. If the agent is looking at the wrong animal, the wrong color sheep, or no animal at all, it will walk around the pen until it finds what it its looking for. The goal of this project is to use *Malmo* and *RLlib* for animal resource gathering automation, and give the player more free time to explore their creativity in *Minecraft*.  

## Approach
approach here  

## Evaluation
eval here  

## Remaining Goals and Challenges
Over the next few weeks, our goal is to implement user input through *Minecraft* to set the target item ratios. Currently, if a player wants milk only, wool only, or a specific color of wool only, the user input is done outside of *Minecraft*. While this is working for the intended purpose, the goal is to get this function working through a new *Malmo* observation such as chat commands or block placement. If the player types "0:1:1" in chat, for example, BarnyardBot will harvest no milk, and equal parts blue and red wool. For block placement, the agent would observe a designated grid that represents the ratio. For example, a 10x3 grid where each column represets a resource. One block in each column would mean a 1:1:1 ratio, five blocks in the middle column would mean a 1:5:1 ratio, or no blocks in the last two columns would mean a 1:0:0 ratio. This would be more challenging to implement than ratios through chat, since the agent has to learn how the block ratios work on top of learning to collect resources. In the current state of the project, the player has to update a line in a file whenever they want to update the ratio.  
It would also be interesting to try a hand-coded policy since the observation space is not overwhelmingly large. Comparing the results from a hand-coded policy to that of the current *RLlib* algorithm would be a good comparison for the final evaluation. The current *RLlib* algorithm performs well, but writing a policy designed specifically for BarnyardBot could potentially increase the performance.
Future challenges are mainly centered around player input for ratios. If the approach of using blocks to represent ratios is taken, the agent will need to spend more time learning for each run. This might also increase the time needed for each mission, because the agent needs enough time to learn from the increased observation space. One of the early challenges was having an observation space that was far too large, and it is possible for the grid to cause the same issue. A possible solution to this would be decreasing the size of the grid, but that limits the number of ratios that can be selected. If the chat option is taken for ratios, the challenge will be parsing out chat messages that are not related to adjusting BarnyardBot's policy. A possible solution would be adding a phrase to the start of every message, or whispering directly to BarnyardBot instead of typing in chat.

## Resources used
resources here  
