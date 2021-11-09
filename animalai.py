from __future__ import print_function
from __future__ import division
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

from builtins import range
from malmo import MalmoPython
import os
import sys
import time
import numpy as np
import json

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# Variables
targetWool="BLUE"
totalReward = 0
action_dict = {
    0: 'move 1',  # Move one block forward
    1: 'turn 1',  # Turn 90 degrees to the right
    2: 'turn -1',  # Turn 90 degrees to the left
    3: 'use 1', # Use item
    4: 'hotbar.1 1', # Swap to first hotbar slot
    5: 'hotbar.2 1' # Swap to second hotbar slot
}

# Malmo setup XML
missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
                <About>
                    <Summary>AnimalAI: Project for collecting resouces from animals in Minecraft</Summary>
                </About>
              
            <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>6000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
                </ServerInitialConditions>
                <ServerHandlers>
                    <FlatWorldGenerator generatorString="2;7,2x3,2;1;"/>
                    <DrawingDecorator>
                        <DrawCuboid x1="-6" y1="4" z1="-6" x2="6" y2="4" z2="6" type="fence"/>
                        <DrawCuboid x1="-5" y1="4" z1="-5" x2="5" y2="4" z2="5" type="air"/>
                    </DrawingDecorator>
                    <ServerQuitFromTimeUp timeLimitMs="30000"/>
                    <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
            </ServerSection>
                <AgentSection mode="Survival">
                    <Name>MalmoTutorialBot</Name>
                    <AgentStart>
                        <Placement x="0" y="4.0" z="0" pitch="30" yaw="90"/>
                        <Inventory>
                            <InventoryItem slot="0" type="shears"/>
                            <InventoryItem slot="1" type="bucket"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <ObservationFromFullStats/>
                        <ContinuousMovementCommands turnSpeedDegs="180"/>
                        <ChatCommands />
                        <InventoryCommands/>
                        <RewardForCollectingItem>
                            <Item type="wool" colour="''' + str(targetWool) + '''" reward="1"/>
                            <Item type="leather" reward="-1"/>
                            <Item type="beef" reward="-1"/>
                            <Item type="mutton" reward="-1"/>
                        </RewardForCollectingItem>
                        <ObservationFromHotBar/>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''

# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()

while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
print()
print("Mission running ", end=' ')

# Adjust the tick speed after the mission starts so grass grows back quicker and sheep eat quicker
agent_host.sendCommand("chat /gamerule randomTickSpeed 20")

# Spawn 8 sheep with a given color at random locations
def spawnSheep(colorString):
    for _ in range(8):
        x = np.random.randint(-5,5)
        z = np.random.randint(-5,5)
        agent_host.sendCommand("chat {}".format('/summon minecraft:sheep ' + str(x) + ' 4 ' + str(z) + ' {Color:' + str(colorString) + '}'))

# Spawn 8 cows at random locations
def spawnCows():
    for _ in range(8):
        x = np.random.randint(-5,5)
        z = np.random.randint(-5,5)
        agent_host.sendCommand("chat {}".format('/summon minecraft:cow ' + str(x) + ' 4 ' + str(z)))

spawnSheep(11) # blue
spawnSheep(14) # red
spawnCows()

# Choose a random action from action_dict
# Adjust this once the learning algorithm has been written
def step(action_dict):
    action = np.random.randint(0,6)
    agent_host.sendCommand(action_dict[action])

# Loop until mission ends:
while world_state.is_mission_running:
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

    # Choose an action and take a step
    step(action_dict)

    # Check hotbar for milk if observations have been made
    if world_state.number_of_observations_since_last_state > 0:
        obsText = world_state.observations[-1].text
        obsJson = json.loads(obsText)
        if (obsJson['Hotbar_1_item'] == 'milk_bucket'): # If the agent has milk, add a point and replace the bucket
            totalReward += 1
            print(totalReward)
            agent_host.sendCommand("chat /replaceitem entity @p slot.hotbar.1 minecraft:bucket")
            time.sleep(0.1) # Allow time for the item to be replaced (prevents scoring multiple points)

    # Check the wool collection rewards and print if reward has changed
    reward = 0
    for r in world_state.rewards:
        reward += r.getValue()
    if (reward != 0):
        totalReward += reward
        print(totalReward)

print()
print("Mission ended")
# Mission has ended.
