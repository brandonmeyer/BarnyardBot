from builtins import range

from malmo import MalmoPython
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

class AnimalAI(gym.Env):

    ###########################################################################
    # Initialize all variables/objects needed
    ###########################################################################
    def __init__(self, env_config):
        # Variables
        self.targetWool="BLUE" # Current wool to reward
        self.totalReward = 0 # Current Reward Total
        self.rewardList = []
        self.obs_size = 16
        self.obs = None
        self.discrete_action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            3: 'use 1', # Use item
            4: 'hotbar.1 1', # Swap to first hotbar slot
            5: 'hotbar.2 1' # Swap to second hotbar slot
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
        # Continuous action space: [move, turn, hotbar, use]: hotbar < 0 = hotbar1, hotbar > 0 = hotbar2
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Discrete action space
        # self.action_space= Discrete(len(self.discrete_action_dict))
        # Observation space: 0=air,1=agent,2=cow,3=red_sheep,4=blue_sheep
        self.observation_space = Box(low=0, high=4, shape=(self.obs_size * self.obs_size, ), dtype=np.float32)

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
                                <DrawCuboid x1="-1" y1="4" z1="-1" x2="15" y2="4" z2="15" type="fence"/>
                                <DrawCuboid x1="0" y1="4" z1="0" x2="14" y2="4" z2="14" type="air"/>
                            </DrawingDecorator>
                            <ServerQuitFromTimeUp timeLimitMs="30000"/>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>
                        <AgentSection mode="Survival">
                            <Name>AnimalAIBot</Name>
                            <AgentStart>
                                <Placement x="7.5" y="4" z="7.5" pitch="30" yaw="0"/>
                                <Inventory>
                                    <InventoryItem slot="0" type="shears"/>
                                    <InventoryItem slot="1" type="bucket"/>
                                </Inventory>
                            </AgentStart>
                            <AgentHandlers>
                                <ContinuousMovementCommands turnSpeedDegs="180"/>
                                <ChatCommands />
                                <InventoryCommands/>
                                <RewardForCollectingItem>
                                    <Item type="wool" colour="''' + str(self.targetWool) + '''" reward="1"/>
                                </RewardForCollectingItem>
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
        self.totalReward = 0

        # Log graph
        if (len(self.rewardList) > 1):
            plt.clf()
            plt.plot(self.rewardList)
            plt.title('ANIMAL AI')
            plt.ylabel('Return')
            plt.xlabel('Missions')
            plt.savefig('animal_returns.png')

        # Log the mission rewards in txt form
        with open('animalai_returns.txt', 'w') as f:
            for i,x in enumerate(self.rewardList):
                f.write("{}\t{}\n".format(i, x))

        # Get Observation
        self.obs = self.getObservation(world_state)

        return self.obs

    ###########################################################################
    # Get Observation
    # Change this to return the current view of the agent
    ###########################################################################
    def getObservation(self, world_state):
        obs = np.zeros((self.obs_size * self.obs_size, ))

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
                obs = self.parseObservation(obs, observations['entities'])

                obs = obs.reshape((1, self.obs_size, self.obs_size))
                yaw = observations['Yaw']
                if (yaw > 45 and yaw <= 135) or (yaw < -225 and yaw >= -315):
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif (yaw > 135 and yaw <= 225) or (yaw < -135 and yaw >= -225):
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif (yaw > 225 and yaw <= 315) or (yaw < -45 and yaw >= -135):
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                obs = obs.flatten()
                
                break
        
        # self.printGrid(obs) # optional: print the grid to view the current observation state
        return obs

    ###########################################################################
    # Parse Observation
    ###########################################################################
    def parseObservation(self, obs, entities):
        # Take entities list and return a grid for the agent
        for entry in entities:
            name = entry['name']
            # convert the x,z coords to an index in the observation grid, 15,15 top left (index 0), 0,0 bottom right
            index = (self.obs_size * self.obs_size)-1 - round(entry['x']) - (self.obs_size*round(entry['z']))
            if name == 'Cow':
                obs[index] = 2
            elif name == 'Red':
                obs[index] = 3
            elif name == 'Blue':
                obs[index] = 4
            elif name == 'AnimalAIBot':
                obs[index] = 1
        return obs

    ###########################################################################
    # Print Grid
    ###########################################################################
    def printGrid(self, obs):
        # Print a readable grid from observations
        count = 0
        printStr = ""
        for entry in obs:
            if (count == self.obs_size):
                printStr += '\n'
                count = 0
            printStr += str(entry) + ' '
            count += 1
        print(printStr + '\n')

    ###########################################################################
    # Spawn 8 sheep with a given color at random locations
    ###########################################################################
    def spawnSheep(self, colorString):
        for _ in range(8):
            x = np.random.randint(0,14)
            z = np.random.randint(0,14)
            name = 'Red'
            if colorString == 11:
                name = 'Blue'
            self.agent_host.sendCommand("chat {}".format('/summon minecraft:sheep ' + str(x) + ' 4 ' + str(z) + ' {CustomName:' + name + ',Color:' + str(colorString) + '}'))

    ###########################################################################
    # Spawn 8 cows at random locations
    ###########################################################################
    def spawnCows(self):
        for _ in range(8):
            x = np.random.randint(0,14)
            z = np.random.randint(0,14)
            self.agent_host.sendCommand("chat {}".format('/summon minecraft:cow ' + str(x) + ' 4 ' + str(z)))

    ###########################################################################
    # Execute the continuous action
    ###########################################################################
    def step(self, action):
        # action = [move, turn, hotbar, use]
        # move
        self.agent_host.sendCommand('move ' + str(action[0]))

        # turn
        self.agent_host.sendCommand('turn ' + str(action[1]))

        # hotbar, if <= 0 press hotbar 1, if > 0 press hotbar 2
        if (action[2] <= 0):
            self.agent_host.sendCommand('hotbar.1 1')
        else:
            self.agent_host.sendCommand('hotbar.2 1')

        # use
        if (action[3] <= 0):
            self.agent_host.sendCommand('use 0')
        else:
            self.agent_host.sendCommand('use 1')

        # Give the agent time to execute the action
        time.sleep(0.3)
        
        # Calculate reward
        reward = 0

        # Check hotbar for milk if observations have been made
        world_state = self.agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            obsText = world_state.observations[-1].text
            obsJson = json.loads(obsText)
            if (obsJson['Hotbar_1_item'] == 'milk_bucket'):
                 # If the agent has milk, add a point and replace the bucket
                reward += 1
                self.agent_host.sendCommand("chat /replaceitem entity @p slot.hotbar.1 minecraft:bucket")
                time.sleep(0.1) # Allow time for the item to be replaced (prevents scoring multiple points)

                # Kill all cows and spawn them in a new location so the agent can't keep milking the same cow
                self.agent_host.sendCommand("chat /kill @e[type=cow]")
                self.spawnCows()
        self.obs = self.getObservation(world_state)

        # Check the wool collection rewards and print if reward has changed
        for r in world_state.rewards:
            reward += r.getValue()
        self.totalReward += reward

        # Check if the mission is still running
        done = not world_state.is_mission_running

        return self.obs, reward, done, dict()

if __name__ == '__main__':
    ray.init()
    # Change from ppo??
    trainer = ppo.PPOTrainer(env=AnimalAI, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
