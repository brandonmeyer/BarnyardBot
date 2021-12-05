from builtins import range

from malmo import MalmoPython
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo, sac, ddpg, dqn

class AnimalAI(gym.Env):

    ###########################################################################
    # Initialize all variables/objects needed
    ###########################################################################
    def __init__(self, env_config):
        # Variables
        self.targetWool="BLUE" # Current wool to reward
        self.totalReward = 0 # Current Reward Total
        self.totalSteps = 0 # Current steps for mission
        self.rewardList = []
        self.stepList = []
        self.milkList = []
        self.blueList = []
        self.redList = []
        self.obs_size = 16
        self.obs = None
        self.view_size = 5
        self.currentItem = 1
        self.agent_x = 5.5
        self.agent_z = 5.5
        self.blueReward = 1
        self.redReward = 1
        self.milkReward = 1
        self.blueRatio = 1
        self.redRatio = 1
        self.milkRatio = 1
        self.milkObs = 1
        self.redObs = 1
        self.blueObs = 1
        self.missionMilkTotal = 0
        self.missionBlueTotal = 0
        self.missionRedTotal = 0
        self.discrete_action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'move -1',  # Move back
            2: 'strafe 1',  # Move right
            3: 'strafe -1', # Move left
            4: 'use 1', # Use item
            5: 'hotbar.1 1', # Swap to first hotbar slot
            6: 'hotbar.2 1' # Swap to second hotbar slot
        }

        # Malmo Objects
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # RLLib objects
        # Discrete action space
        self.action_space= Discrete(len(self.discrete_action_dict))
        # Observation space: 0=air,1=cow,2=red_sheep,3=blue_sheep
        self.observation_space = Box(low=0, high=np.array([3, 1, 2, 2, 2]), dtype=np.float32)

    ###########################################################################
    # Return the mission XML with the current rewards
    ###########################################################################
    def getMissionXML(self):
        missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                                <DrawCuboid x1="-1" y1="4" z1="-1" x2="11" y2="4" z2="11" type="fence"/>
                                <DrawCuboid x1="0" y1="4" z1="0" x2="10" y2="4" z2="10" type="air"/>
                            </DrawingDecorator>
                            <ServerQuitFromTimeUp timeLimitMs="30000"/>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>
                        <AgentSection mode="Survival">
                            <Name>AnimalAIBot</Name>
                            <AgentStart>
                                <Placement x="5.5" y="4" z="5.5" pitch="22" yaw="0"/>
                                <Inventory>
                                    <InventoryItem slot="0" type="shears"/>
                                    <InventoryItem slot="1" type="bucket"/>
                                </Inventory>
                            </AgentStart>
                            <AgentHandlers>
                                <DiscreteMovementCommands>
                                    <ModifierList type="deny-list">
                                        <command>use</command>
                                        <command>turn</command>
                                    </ModifierList>
                                </DiscreteMovementCommands>
                                <ContinuousMovementCommands turnSpeedDegs="360">
                                    <ModifierList type="allow-list">
                                        <command>use</command>
                                        <command>turn</command>
                                    </ModifierList>
                                </ContinuousMovementCommands>
                                <AbsoluteMovementCommands />
                                <ChatCommands />
                                <InventoryCommands/>
                                <RewardForCollectingItem>
                                    <Item type="wool" colour="BLUE" reward="''' + str(self.blueReward) + '''"/>
                                    <Item type="wool" colour="RED" reward="''' + str(self.redReward) + '''"/>
                                </RewardForCollectingItem>
                                <ObservationFromChat/>
                                <ObservationFromFullStats/>
                                <ObservationFromHotBar/>
                                <ObservationFromRay/>
                                <ObservationFromNearbyEntities>
                                    <Range name="entities" xrange="50" yrange="10" zrange="50"/>
                                </ObservationFromNearbyEntities>
                            </AgentHandlers>
                        </AgentSection>
                    </Mission>'''
        return missionXML

    ###########################################################################
    # Set up malmo mission
    ###########################################################################
    def initMalmo(self):
        self.setRatios()

        # Start the mission
        my_mission = MalmoPython.MissionSpec(self.getMissionXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'AnimalAI' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2)

        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.KEEP_ALL_OBSERVATIONS)

        # Make sure that mission initializes
        world_state = self.agent_host.getWorldState()

        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
        print('Mission started!\n')

        # Adjust the tick speed after the mission starts so grass grows back quicker and sheep eat quicker
        self.agent_host.sendCommand("chat /gamerule randomTickSpeed 20")

        # Spawn animals for pen
        self.spawnSheep(11) # blue
        self.spawnSheep(14) # red
        self.spawnCows()

        # Set the ratio observations for the current mission
        # These do not change live because the rewards can only update at mission start
        self.milkObs = self.milkRatio
        self.redObs = self.redRatio
        self.blueObs = self.blueRatio
        print('RATIO: ' + str(self.milkObs) + ':' + str(self.redObs) + ':' + str(self.blueObs))
        print('REWARD: ' + str(self.milkReward) + ':' + str(self.redReward) + ':' + str(self.blueReward))

        # Make sure there are no items on the ground
        self.agent_host.sendCommand("chat /kill @e[type=item]")

        world_state = self.agent_host.getWorldState()
        return world_state

    ###########################################################################
    # Run Malmo
    ###########################################################################
    def reset(self):
        # Initialize Malmo
        world_state = self.initMalmo()

        # Reset Variables
        self.rewardList.append(self.totalReward)
        current_step = self.stepList[-1] if len(self.stepList) > 0 else 0
        self.stepList.append(current_step + self.totalSteps)
        self.milkList.append(self.missionMilkTotal)
        self.blueList.append(self.missionBlueTotal)
        self.redList.append(self.missionRedTotal)
       

        print('MISSION REWARD: ' + str(self.totalReward))
        print('MISSION MILK TOTAL: ' + str(self.missionMilkTotal))
        print("MISSION RED TOTAL: " + str(self.missionRedTotal))
        print("MISSION BLUE TOTAL: " + str(self.missionBlueTotal))

        self.totalReward = 0
        self.totalSteps = 0
        self.missionMilkTotal = 0
        self.missionBlueTotal = 0
        self.missionRedTotal = 0

        # Log graph
        # if len(self.rewardList) > 11 and len(self.rewardList) % 10 == 0:
        #     box = np.ones(10) / 10
        #     returns_smooth = np.convolve(self.rewardList[1:], box, mode='same')
        #     plt.clf()
        #     plt.plot(self.stepList[1:], returns_smooth)
        #     plt.title('ANIMAL AI')
        #     plt.ylabel('Return')
        #     plt.xlabel('Steps')
        #     plt.savefig('animal_returns.png')

        if len(self.milkList) > 11 and len(self.milkList) % 10 == 0:
            box = np.ones(10) / 10
            milk_smooth = np.convolve(self.milkList[1:], box, mode='same')
            red_smooth = np.convolve(self.redList[1:], box, mode='same')
            blue_smooth = np.convolve(self.blueList[1:], box, mode='same')
            plt.clf()
            plt.plot(self.stepList[1:], milk_smooth, 'k')
            plt.plot(self.stepList[1:], red_smooth, 'r')
            plt.plot(self.stepList[1:], blue_smooth, 'b')
            plt.title('ANIMAL AI')
            plt.ylabel('Number Harvested')
            plt.xlabel('Steps')
            plt.savefig('ratio_graph.png')

        # Log the mission rewards in txt form
        with open('animalai_returns.txt', 'w') as f:
            for step, value in zip(self.stepList[1:], self.rewardList[1:]):
                f.write("{}\t{}\n".format(step, value)) 

        with open('milk_returns.txt', 'w') as f:
            for step, value in zip(self.stepList[1:], self.milkList[1:]):
                f.write("{}\t{}\n".format(step, value)) 
        
        with open('red_returns.txt', 'w') as f:
            for step, value in zip(self.stepList[1:], self.redList[1:]):
                f.write("{}\t{}\n".format(step, value)) 

        with open('blue_returns.txt', 'w') as f:
            for step, value in zip(self.stepList[1:], self.blueList[1:]):
                f.write("{}\t{}\n".format(step, value)) 

        # Get Observation
        self.obs = self.getObservation(world_state)

        return self.obs

    ###########################################################################
    # Get Observation
    # Change this to return the current view of the agent
    ###########################################################################
    def getObservation(self, world_state):
        obs = np.zeros((5, ))

        # Loop until mission ends:
        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                self.parseChat(world_state.observations)
                obs = self.parseObservation(obs, observations['LineOfSight'])
                break
        return obs

    ###########################################################################
    # Parse Observation
    ###########################################################################
    def parseObservation(self, obs, los):
        # Take line of sight and return what object is visible
        if los['type'] == 'Cow':
            obs[0] = 1
        elif los['type'] == 'Red':
            obs[0] = 2
        elif los['type'] == 'Blue':
            obs[0] = 3
        else:
            obs[0] = 0

        # Add the current held item to the observation
        if self.currentItem == 1:
            obs[1] = 0
        else:
            obs[1] = 1

        obs[2] = self.milkObs
        obs[3] = self.redObs
        obs[4] = self.blueObs

        return obs

    ###########################################################################
    # Parse Chat
    ###########################################################################
    def parseChat(self, observations):
        for iter in observations:
            if 'Chat' in iter.text:
                for message in json.loads(iter.text)['Chat']:
                    if '!RATIO' in message:
                        try:
                            ratio = message.split('!RATIO ')[1].split(' ')[0].split(':')
                            if (int(ratio[0]) < 3 and int(ratio[0]) > -1 and int(ratio[1]) < 3 and int(ratio[1]) > -1 and int(ratio[2]) < 3 and int(ratio[2]) > -1):
                                self.milkRatio = int(ratio[0])
                                self.redRatio = int(ratio[1])
                                self.blueRatio = int(ratio[2])
                            else:
                                self.agent_host.sendCommand("chat Msg error. Ratio values must be >-1 and <= 2")
                        except:
                            self.agent_host.sendCommand("chat Msg error. Example: !RATIO 1:0:1")
                        else:
                            self.agent_host.sendCommand("chat Ratio " + str(self.milkRatio) + ':' + str(self.redRatio) + ':' + str(self.blueRatio) + " loaded, and will be live next mission")

    ###########################################################################
    # Spawn 8 sheep with a given color at random locations
    ###########################################################################
    def spawnSheep(self, colorString):
        for _ in range(8):
            x = np.random.randint(0,10)
            z = np.random.randint(0,10)
            name = 'Red'
            if colorString == 11:
                name = 'Blue'
            self.agent_host.sendCommand("chat {}".format('/summon minecraft:sheep ' + str(x) + ' 4 ' + str(z) + ' {CustomName:' + name + ',Color:' + str(colorString) + '}'))

    ###########################################################################
    # Spawn 8 cows at random locations
    ###########################################################################
    def spawnCows(self):
        for _ in range(8):
            x = np.random.randint(0,10)
            z = np.random.randint(0,10)
            self.agent_host.sendCommand("chat {}".format('/summon minecraft:cow ' + str(x) + ' 4 ' + str(z)))

    ###########################################################################
    # Execute the continuous action
    ###########################################################################
    def step(self, action):
        ###################
        ## DISCRETE
        ###################
        act = self.discrete_action_dict[action]
        if act[0:4] == 'move' or act[0:6] == 'strafe':
            # Center the agent on the block before moving
            # This prevents the agent from getting pushed off by an animal and being unable to move
            self.agent_host.sendCommand('tp ' + str(self.agent_x) + ' 4 ' + str(self.agent_z))
            time.sleep(0.1)
            self.agent_host.sendCommand(act)
            if act == 'move 1' and self.agent_z <= 10:
                self.agent_z += 1
            elif act == 'move -1' and self.agent_z >= 1:
                self.agent_z -= 1
            elif act == 'strafe -1' and self.agent_x <= 10:
                self.agent_x += 1
            elif act == 'strafe 1' and self.agent_x >= 1:
                self.agent_x -= 1
        elif act == 'use 1':
            self.agent_host.sendCommand(act)
            time.sleep(0.1)
            self.agent_host.sendCommand('use 0')
            self.collect()
        elif act == 'turn 1' or act == 'turn -1':
            self.agent_host.sendCommand(act)
            time.sleep(0.25)
            self.agent_host.sendCommand('turn 0')
        else:
            self.agent_host.sendCommand(act)
            if act == 'hotbar.1 1':
                self.currentItem = 1
                self.agent_host.sendCommand('hotbar.1 0')
            elif act == 'hotbar.2 1':
                self.currentItem = 2
                self.agent_host.sendCommand('hotbar.2 0')
        
        # Calculate reward
        reward = 0

        # Get the current world state
        world_state = self.agent_host.getWorldState()

        # Check the wool collection rewards and print if reward has changed
        for r in world_state.rewards:
            reward += r.getValue()

        # Check hotbar for milk if observations have been made
        if world_state.number_of_observations_since_last_state > 0:
            obsText = world_state.observations[-1].text
            obsJson = json.loads(obsText)
            if (obsJson['Hotbar_1_item'] == 'milk_bucket'):
                 # If the agent has milk, add a point and replace the bucket
                reward += self.milkReward
                self.missionMilkTotal += 1
                self.agent_host.sendCommand("chat /replaceitem entity @p slot.hotbar.1 minecraft:bucket")
                time.sleep(0.1) # Allow time for the item to be replaced (prevents scoring multiple points)

                # Kill all cows and spawn them in a new location so the agent can't keep milking the same cow
                self.agent_host.sendCommand("chat /kill @e[type=cow]")
                self.spawnCows()
            for i in range(2,9):
                if (obsJson['Hotbar_' + str(i) + '_item'] == 'wool'):
                    if (obsJson['Hotbar_' + str(i) + '_colour'] == 'RED'):
                        self.missionRedTotal = obsJson['Hotbar_' + str(i) + '_size']
                    elif (obsJson['Hotbar_' + str(i) + '_colour'] == 'BLUE'):
                        self.missionBlueTotal = obsJson['Hotbar_' + str(i) + '_size']
        self.obs = self.getObservation(world_state)

        self.totalReward += reward
        self.totalSteps += 1

        # Check if the mission is still running
        done = not world_state.is_mission_running

        print(reward)

        return self.obs, reward, done, dict()

    ###########################################################################
    # Collect - Move 2 forwards and 2 back to collect use reward
    ###########################################################################
    def collect(self):
        # 2 forward, 2 back
        if (self.agent_z <= 8.5):
            self.agent_host.sendCommand('tp ' + str(self.agent_x) + ' 4 ' + str(self.agent_z + 1))
            time.sleep(0.1)
            self.agent_host.sendCommand('tp ' + str(self.agent_x) + ' 4 ' + str(self.agent_z + 2))
            time.sleep(0.3)
            self.agent_host.sendCommand('tp ' + str(self.agent_x) + ' 4 ' + str(self.agent_z))

    ###########################################################################
    # Set Ratios
    ###########################################################################
    def setRatios(self):
        if self.milkRatio == 0:
            self.milkReward = -1
        else:
            self.milkReward = self.milkRatio
        if self.redRatio == 0:
            self.redReward = -1
        else:
            self.redReward = self.redRatio
        if self.blueRatio == 0:
            self.blueReward = -1
        else:
            self.blueReward = self.blueRatio
        print(self.milkReward, self.redReward, self.blueReward)

if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=AnimalAI, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
