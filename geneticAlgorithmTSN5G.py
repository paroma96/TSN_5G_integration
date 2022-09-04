#!/usr/bin/env python3
#pylint: skip-file
# -*- coding: utf-8 -*-
# pip install networkx
# pip install pygad
# pip install scipy
"""
This code defines the genetic algorithm that allows the use of a topology matrix (1 if 
node i is linked to node j, 0 if not) in order to find the optimal solution in a TSN+5G 
network for several restrictive flows from a source to a destination participating with its 
own constraints (max. delay, period, frame length, transceptors' capacity). It will optimize 
every flow's route and its scheduling based on its end-to-end maximum permissible delay 
and bandwidth usage on the links. Scheduler is then made up by the revision of every flow's 
normalized delay and the count of time within the gaps as explained in the, However, the 
system of Size-Based Queuing Algorithm and the compression of scheduling of flows presented 
in "No-wait Packet Scheduling for IEEE TSN" do not contemplate the existence of certain number
of phases in a so-called "hyperperiod" calculated with all flows' different periods, so might
lead to collisions. Additionally, the second publication is computationally far more expensive
with compression as it will have to re-schedule all routes in order to reduce de end-to-end 
delay but also gaps. This work presents then a solution with Artificial Intelligence Genetic 
Algorithms in a simpler scheduler to minimize the set of flows' maximum delays, overall the 
most critical ones (higher periodicity and lower e2e delay); and, at the same time, reduce the 
gaps so best-effort traffic and guard bands can fit in them in a large number of scheduled
flows with very restrictive parameters (Industry 4.0). Optionally, there is the possibility of 
choosing a full-duplex or a half-duplex configuration taking into account the link's usage in 
previous flows and the usage of left side scheduling, that means a flow arriving earlier can 
be placed before all the already scheduled sequence. With this last configuration may happen
that a flow's slow down must be performed in order to reduce that delay so it spends no time 
in queue, but there is no gap enough to satisfy the scrolling of time Finally, this model 
offers the possibility to use 5G logic bridges with guard band as solution of the NFVs
fluctuations, so additional delay is added in radioaccess.

@author    Pablo Rodríguez Martín
           (pablorodrimar@correo.ugr.es)

MSc. Telecommunication Engineering Final Project (TFM) - University of Granada

Title: Synchronous TSN Topologies Configuration for transportation of 5G Network Slices. 
       Machine Learning Optimization of Scheduler.

Date: September, 2022
"""

from scipy.io import savemat
from goto import with_goto

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import pygad
import copy
import math
import sys
import os

"""Class that implements the whole TSN network topology object from the given topology matrix, 
links' capacities and flows' constraints files. It contains the nodes and ports that the flows
go through along the different phases within a calculated hyperperiod from periods (LCM)."""
class TSN_NETWORK:
    # =========================================================================================
    # INITIALIZATION
    def __init__(self: object, path: list = [], reduceSpace: int = 4, pathLen: int = 4, \
                bidirec: bool = 1, maxLenFrame: int = 1500) -> object:
        #--------------------------------------------------------------------------------------
        # DEFAULT:  NULL
        #           4 shortest paths
        #           4 nodes per path (max.)
        #           Half-duplex
        #           1500 Bytes
        #--------------------------------------------------------------------------------------
        # Matrix with the TSN network's topology contained
        self.topology: list = self.readTopology(path)
        # Number of nodes involved in the described topology
        self.N: int = len(self.topology)
        if(self.N != len(self.topology[0])):
            print("ERROR: TOPOLOGY MATRIX MISMATCHES DIMENSIONS. INPUT MUST BE A SQUARE \
                MATRIX OF NxN NODES")
            return
        #--------------------------------------------------------------------------------------
        # Vector with the nodes' processing time delay. Different values may cause bottlenecks!
        self.procDelay: list = self.readProcDelay(path)
        if(self.N != len(self.procDelay)):
            print("ERROR: PROCESSING DELAY VECTOR MISMATCHES DIMENSIONS.")
            return    
        #--------------------------------------------------------------------------------------
        # Matrix with guardband percentages between TSN and 5G nodes
        self.fluctuations: list = []
        if(mode5G):
            self.fluctuations = self.readFluct(path)
            if(self.N != len(self.fluctuations)  or self.N != len(self.fluctuations[0])):
                print("ERROR: FLUCTUATIONS VECTOR MISMATCHES DIMENSIONS.")
                return
        else:
            self.fluctuations = np.zeros((self.N,self.N))
        #--------------------------------------------------------------------------------------
        # Matrix with radioaccess delay
        self.radioDelay: list = []
        if(mode5G):
            files: str = os.listdir(path + "/Inputs/")
            for file in files:
                if(file.startswith('radioDelMatrix_node_')):
                    self.radioDelay.append(self.readRadioDelay(path, file))
        else:
            for zero_cnt in range(self.N):
                self.radioDelay.append(np.zeros((self.N,self.N)))
        #--------------------------------------------------------------------------------------
        # Matrix with the flows' constraints e2e (src-dst, max. delay, tx period, frame size,
        # talker's throughput, listener's throughput)
        if(path):
            flows: list = self.readFlows(path)
        else:
            print("ERROR: NO FLOWS FILE SPECIFIED")
            return
        if(flows):
            flows = self.sortFlows(flows) # Sorted list of flows
            self.flows: list = []
            for flowLine in flows:
                flow: object = FLOW(flowLine) # Call to flow object
                self.flows.append(flow)
        else:
            print("ERROR: NO FLOWS' CONSTRAINTS SPECIFIED")
            return
        # Number of flows involved in the described topology
        self.M: int = len(self.flows)
        #--------------------------------------------------------------------------------------
        # Vector with all flows' periods' Least Common Multiple (LCM) -- Hyperperiod
        self.LCM: float = 1.0
        self.minPeriod: float = 100000.0
        self.order: float = 1e-3 # ms
        for i in range(self.M):
            period: int = int(self.flows[i].T_tx * 100 / self.order)
            self.LCM = int((self.LCM * period) / (GCD(self.LCM, period))) # All flows' LCM
            if(period <= self.minPeriod):
                self.minPeriod = period
        self.LCM = self.LCM * self.order / 100
        self.minPeriod = self.minPeriod * self.order / 100 
        #--------------------------------------------------------------------------------------
        # Precision of time: ns
        self.precision: int = round(abs(math.log(10e-9, 10))+1) 
        #--------------------------------------------------------------------------------------
        # Matrix with all possible flows' paths/routes e2e
        self.reduceSpace: int = reduceSpace # Number of max paths per flow
        self.pathLen: int = pathLen # Number of max nodes per path
        self.pathFlows: list = []
        for i in range(len(self.flows)):
            calc: bool = 1
            for j in range(len(self.pathFlows)):
                if(self.pathFlows[j][0][0] == self.flows[i].src and \
                    self.pathFlows[j][0][-1] == self.flows[i].dst): # Already calculated
                    self.pathFlows.append(self.pathFlows[j])
                    calc = 0
                    break
            if(calc): # Not calculated yet
                self.pathFlows.append(self.listPaths(self.topology, self.flows[i], \
                    self.reduceSpace, self.pathLen))
        #--------------------------------------------------------------------------------------
        # Possibility of bidirectional links between nodes, so scheduling is shared for both
        self.bidirec: bool = bidirec # 1 half-duplex links (symm. matrix), 0 full-duplex links
        #--------------------------------------------------------------------------------------
        # Network's biggest Ethernet frame size
        self.maxLenFrame: int = maxLenFrame # Bytes
        #--------------------------------------------------------------------------------------
        # Resetting
        self.nodes: list = []
        self.reset()
    # =========================================================================================
    # Resets TSN simulation in order to perform a new one with the creation of nodes with their 
    # own ports according to the topology. Each node has its own ID, ports and process delay 
    # due to conmutation.
    def reset(self: object) -> None:
        #First deletes previous objects
        aux1: int = len(self.nodes)
        for i in range(aux1): 
            aux2: int = len(self.nodes[0].ports)
            for j in range(aux2):
                del self.nodes[0].ports[0]
            del self.nodes[0]
        #New objects
        for x in range(self.N):
            self.nodes.append(NODE(x, self.topology[x], self.procDelay[x], \
                self.fluctuations[x], self.minPeriod, self.LCM))
    # =========================================================================================
    # Reads TSN topology through a matrix of linked nodes. 1 if node i is linked to node j,
    # 0 if not. Matrix may not be symmetric but NxN, being N the total number of TSN nodes.
    # A header with node's ID numbers is skipped.
    def readTopology(self: object, path: list) -> list:
        with open(path + '/Inputs/topologyMatrix.txt', 'r') as readM:
            next(readM)
            topologyMatrix: list = [[float(num) for num in line.split(',')] for line in readM]
        return topologyMatrix
    # =========================================================================================
    # Reads the delay value in seconds that introduces each node for the frame processing.
    # A header with node's ID numbers is skipped.
    def readProcDelay(self: object, path: list) -> list:
        with open(path + '/Inputs/procDelay.txt', 'r') as readM:
            next(readM)
            procDelay: list = [[float(num) for num in line.split(',')] for line in readM]
        return procDelay[0]
    # =========================================================================================
    # Reads the list of 5G logical bridge's port's fluctuations as a percentage of the total
    # length of the frames. This value acts as a factor for the guardbands in time assignement
    # in the planification of every flow. It is set over the link from a 5G node to others.
    def readFluct(self: object, path: list) -> list:
        with open(path + '/Inputs/fluctMatrix.txt', 'r') as readM:
            next(readM)
            fluctuationsVector: list = [[float(num) for num in line.split(',')] for line in readM]
        return fluctuationsVector
    # =========================================================================================
    # Reads the list of 5G logical bridge's wireless delay introduced by radioaccess from one
    # virtual port to other. If flow does not go through wireless channel then this delay is 
    # set to 0.
    def readRadioDelay(self: object, path: list, filename: str) -> list:
        with open(path + '/Inputs/' + filename, 'r') as readM:
            next(readM)
            radioDelayVector: list = [[float(num) for num in line.split(',')] for line in readM]
        return radioDelayVector
    # =========================================================================================
    # Reads the list of 5G-TSN flows within a constraints vector: source node, destination node, 
    # max. delay (s), transmission period (s), frame length (Bytes). Matrix may not be symmetric 
    # but NxN, being N the total number of TSN nodes. Period must be even and 2^x. A header with 
    # field names is omitted.
    def readFlows(self: object, path: list) -> list:
        with open(path + '/Inputs/FlowStates/' + flowState +'.txt', 'r') as readM:
            next(readM)
            flowsVector: list = [[float(num) for num in line.split(',')] for line in readM]
        return flowsVector
    # =========================================================================================
    # Searches for every possible path/route in a flow from a source node to a destination 
    # node following the links in the topology matrix. Used method: Depth-First Search 
    # algorithm (DFS). Runs over every node neighbors, if the destination cannot be 
    # reached, the path will automatically be discarded with the visited list and pop's
    # in the list.
    def listPaths(self: object, topologyMatrix: list, flow: list, reduce: int, pathLen: int) -> list:
        N: int = len(topologyMatrix[0]) # Number of nodes
        src: int = int(flow.src) # Flow's source
        dst: int = int(flow.dst) # Flow's destination
        checkedList: list = [] # Visited list
        paths2check: list = [] # Possible open paths
        paths2check.append([src]) # Starts from the source
        routes: list = [] # All-paths solution
        #--------------------------------------------------------------------------------------
        # Possible flow-paths discovery
        while(len(paths2check) > 0):
            path: list = paths2check.pop()
            if(len(path) <= pathLen):
                if(dst in path):
                    routes.append(path) # Solution path found
                else:
                    if(path not in checkedList):
                        checkedList.append(path) 
                        for node in range(N):
                            if(node not in path and topologyMatrix[path[-1]][node] > 0):
                                path.append(node)
                                paths2check.append(path[:]) # Adds a new possible path to explore
                                path.pop()
        #--------------------------------------------------------------------------------------
        # In case there is a reduction to a limited number of paths for every flow, shortest 
        # paths in a sorted list are selected
        if(reduce and reduce <= len(routes)):
            aux = sorted(routes, key=len)
            aux = aux[:reduce]
            routes = aux
        return routes
    # =========================================================================================
    # Sorts the list of flows by its parameters, defining the order of scheduling in all the
    # nodes in the topology involved in the different paths. Those parameters are: 
    # 1st) Transmission period, 2nd) Max. Delay, 3rd) source, 4th) destination, 5th) Frame size
    def sortFlows(self: object, flows: list) -> list:
        flows=sorted(flows, key = lambda x: (x[3], x[2], x[0], x[1], x[4]))
        for i in range(len(flows)):
            flows[i].insert(0, i) # Inserts FlowID    
        return flows
    # =========================================================================================
    # Plots ports' phase scheduling on every node depending on the flows' paths. 
    def plotScheduling(self: object, nodeID: int, neighborID: int) -> None:
        x: list = np.linspace(0, self.minPeriod, num = int(1e6)) # X-axis
        y: list = [] # Y-axis
        port: object
        for port_ in self.nodes[nodeID].ports:
            if(port_.neighborID == neighborID):
                port = port_
                print("  Gaps found in Node #" + str(nodeID) + ", Port #" + str(neighborID) + ": " \
                + str(port.gap))
                print("  Guard band: " + str(port.perc_gb))
                break
        if(not port):
            print("ERROR: NO PORT FOUND. PLEASE CHECK TOPOLOGY")
            return 0
        colors: list = ["b", "r", "g", "m", "c", "k", "y"] # List of colors
        color_idx: int = 0
        #--------------------------------------------------------------------------------------
        # All flows scheduled in this port
        fig, axs = plt.subplots(len(self.flows[0].schStart[0]))
        fig.suptitle("Phase scheduling in Node #" + str(nodeID) + ", Port #" + str(neighborID))
        count: int = 0
        for ph in range(len(self.flows[0].schStart[0])):
            for flow in port.schFlowID[ph]: 
                if(self.nodes[nodeID].id in self.flows[flow].pathNodes): 
                    y = [] # New values for new flow
                    posNode = self.flows[flow].pathNodes.index(self.nodes[nodeID].id)
                    posPort = self.flows[flow].pathNodes.index(self.nodes[neighborID].id)
                    if(posNode > posPort and self.bidirec): # Node order in flow's route
                        pos = posPort
                    else:
                        pos = posNode 
                    for i in x:
                        if((i >= self.flows[flow].schStart[pos][ph]) and \
                            (i <= self.flows[flow].schEnd[pos][ph])):
                            y.append(1) # Tx time
                        else:
                            y.append(0) # No Tx time
                    axs[count].plot(x, y, colors[color_idx], label = "Flow ID: " + \
                        str(self.flows[flow].id))
                    axs[count].legend(port.schFlowID[ph])
                    if(color_idx == (len(colors)-1)): # Color change
                        color_idx = 0
                    else:
                        color_idx += 1  
            plt.show(block = False)
            x = [round((i + self.minPeriod), network.precision) for i in x] # Adds minimum period
            color_idx = 0
            count += 1
    # =========================================================================================

"""Class that implements the TSN node/switch object from the given topology. Every node is 
identified by an ID. Also, it has its own process time delay to conmute TSN frames and the
information about every port that connects itself with other nodes through that identifier."""
class NODE:
    # =========================================================================================
    # INITIALIZATION
    def __init__(self, *args) -> object:
        self.id: int = args[0] # Node's ID
        self.neighbors: list = args[1] # Linked neighbors and their capacity
        self.procDelay: float = args[2] # Node's processing time
        self.fluctuations: list = args[3] # Values of fluctuations for ports
        self.minPeriod: float = args[4] # Node's minimum period to schedule
        self.LCM: float = args[5] # Node's LCM of periods
        #--------------------------------------------------------------------------------------
        # Node opens a Tx port with its neighbors by an identifier
        self.ports: list = [] 
        for id in range(len(self.neighbors)): 
            if(id != self.id and self.neighbors[id] > 0):
                port: object = PORT([id, self.neighbors[id], self.fluctuations[id], \
                    self.minPeriod, self.LCM])
                self.ports.append(port) # Adding port to node's port list
    # =========================================================================================

"""Class that implements a specific port that connects the proper node with its neighbor, 
identified with an ID. Every port is also chracterized by a vector for Time-Aware Shaper 
scheduling with a scheduled start (lower bound) and end (upper bound) times."""
class PORT:
    # =========================================================================================
    # INITIALIZATION
    def __init__(self, *args) -> object:
        self.neighborID: int = args[0][0] # Port to neighbor node with ID i (next-hop)
        self.speed: float = args[0][1] # Link speed to neighbor i (Gbps)
        self.perc_gb: float = args[0][2] # Percentage of fluctuation GB
        self.minPeriod: float = args[0][3] # Minimum period value
        self.LCM: float = args[0][4] # LCM (hyperperiod)
        self.schLW: list = [] # Scheduling time START/LOWERBOUND
        self.schUP: list = [] # Scheduling time END/UPPERBOUND
        self.latArrTime: list = [] # Latest arrival to port queue
        self.lastGB: list = []
        self.schFlowID: list = [] # Scheduled Flow IDs
        self.gap: list = [] # Not scheduled gaps
        #--------------------------------------------------------------------------------------
        # Scheduling is divided in phases with a duration of minimum period of all flows
        for phase in range(int(self.LCM / self.minPeriod)): 
            self.schLW.append(0) 
            self.schUP.append(0)
            self.latArrTime.append([])
            self.lastGB.append(0)
            self.schFlowID.append([])
            self.gap.append([])
    # =========================================================================================
    # Merges gaps in case both two gaps have consecutive intervals.
    def mergeGaps(self: object, ph: int) -> None:
        listGaps: list = self.gap[ph]
        if(len(listGaps) > 0):
            listGaps = list(filter(lambda x: x, listGaps)) # Removes empty values
            listGaps = sorted(listGaps, key = lambda x: x[0]) # Sort gaps by time
            toRemoveList: list = []
            for gap in listGaps:
                if(gap != listGaps[-1]):
                    for otherGap in listGaps[listGaps.index(gap)+1: ]:
                        if(gap[1] == otherGap[0]):
                            listGaps[listGaps.index(gap)] = [gap[0], otherGap[1]] # Unique gap
                            toRemoveList.append(otherGap)
                            break
            #-----------------------------------------------------------------------------------
            # Deletes all gaps that have been merged into another 
            for remGap in range(len(toRemoveList)): # Removes those which have been used to merge
                try:
                    listGaps.remove(toRemoveList[remGap])
                except ValueError:
                    pass
            self.gap[ph] = listGaps[ : ]
    # =========================================================================================
    
"""Class that implements a flow characterized by its constraints (ID, source & destination
nodes, max. delay, transmission period and frame length). It also contains the scheduling
times on every node."""
class FLOW:
    # =========================================================================================
    # INITIALIZATION
    def __init__(self, *args) -> object:
        self.id: int = int(args[0][0]) # Flow's ID
        self.src: int = int(args[0][1]) # Flow's source
        self.dst: int = int(args[0][2]) # Flow's destination
        self.maxDelay: float = args[0][3] # Flow's upper bound time delay (s)
        self.T_tx: float = args[0][4] # Flow's transmission period (s)
        self.length: int = int(args[0][5] * 8) # Flow's frame's length (bits)
        self.talkerSpeed: float = args[0][6] # Flow's talker speed
        self.listenerSpeed: float = args[0][7] # Flow's listener speed
        self.pathNodes: list = [] # Flow's scheduling nodes in a path [pos]
        self.schStart: list = [] # Flow's start time in a node [pos]
        self.schEnd: list = [] # Flow's start time in a node [pos]
    # =========================================================================================
    # Updates de route of a flow after finding all possible paths within a topology and 
    # selected one of them in a chromosome. It also resets scheduling times.
    def updateRoute(self: object, route: list) -> object:
        self.pathNodes = route
        self.schStart = []
        self.schEnd = []
    # =========================================================================================

"""-----------------------------------------------------------------------------------------"""
# Calculates the Greatest Common Divisor and returns the result. It is used to subsequently 
# compute the topology's LCM value.
def GCD(a: int, b: int) -> int:
    temp: int = 0
    while(b != 0):
        temp = b
        b = a % b
        a = temp
    return a
"""-----------------------------------------------------------------------------------------"""
# Initializes the very first chromosome with a subset of random paths, one per each flow (gene).
def initializeChromosome(possiblePathFlows: list) -> list:
    chromosome: list = []
    for i in range(len(possiblePathFlows)):
        chromosome.append(random.randrange(len(possiblePathFlows[i]))) # Random index for gene
    return chromosome
"""-----------------------------------------------------------------------------------------"""
# Performs the scheduling for an unique flow over all nodes in the path selected. Two main
# cases can be distinguished:
#        - Time of post-processing > Time of upper bound (new gap)
#        - Time of post-processing < Time of upper bound
#                · Time of Tx end < Time of lower bound (opt, new gap if no compression)
#                · Time of Tx end > Time of lower bound
@ with_goto
def scheduling(path: list, flowID: int) -> bool:
    time: float = 0.0 # Will accquire the value of the hyperperiod in case flow times goes out
    initialized: bool = 0 # 1 if flow's path's first node scheduled, 0 if not
    t0: list = [] # Arrival time to port's queue (dim = number of phases)
    for i in range(len(path)-1): # Nodes (last node is not scheduled, only sent to listener)
        for port in network.nodes[path[i]].ports:
            if(port.neighborID == path[i+1]): # Ports
                #------------------------------------------------------------------------------
                # Arrival time to node + SW processing delay
                if(not initialized): 
                    # CASE: FLOW'S FIRST NODE SCHEDULING
                    for ph in range(int(network.LCM / network.minPeriod)):
                        t0.append(0) # Any time restriction, as many 0's as phases
                else:
                    # CASE: NOT FLOW'S FIRST NODE SCHEDULING
                    for node_port in network.nodes[path[i]].ports:
                        if(node_port.neighborID == network.nodes[path[i-1]].id):
                            src_port: int = network.nodes[path[i]].ports.index(node_port)
                        elif(node_port.neighborID == network.nodes[path[i+1]].id):
                            dst_port: int = network.nodes[path[i]].ports.index(node_port)
                    for ph in range(len(t0)):
                        if(network.flows[flowID].schEnd[i-1][ph] != 0):
                            t0[ph] = (network.flows[flowID].schEnd[i-1][ph] + \
                            network.nodes[path[i]].procDelay + \
                            network.radioDelay[path[i]][src_port][dst_port]) # Prev. node end time + Bridge delay
                            t0[ph] = np.round(t0[ph], network.precision)
                #------------------------------------------------------------------------------
                # Adaptation of global hyperperiod to phase time (first schedulable)
                idx: int = next((i for i, x in enumerate(t0) if x), None) # First sched phase
                if(idx is None):
                    idx = 0
                phase: int = int(t0[idx] / network.minPeriod)
                for ph in range(len(t0)):
                    t0[ph] = round((t0[ph] % network.minPeriod), network.precision)
                #------------------------------------------------------------------------------
                # Current port's Guard Band
                guardBand: float = port.perc_gb * network.flows[flowID].length / port.speed
                #------------------------------------------------------------------------------
                # Previous node's port's Guard Band (equal to current's in case same speed)
                guardBand_p: float
                if(initialized):
                    guardBand_p = prev_port.perc_gb * network.flows[flowID].length / prev_port.speed 
                #------------------------------------------------------------------------------
                # No flow's phases scheduled yet in this port
                j: int = 0 # Number of phase after first scheduling
                scheduled: int = 0 # +1 if scheduled in phase, 0 if not
                phChange: bool = 1 # Possibility to continue next hyper
                phVectorStart: list = [] # Flow's phases' start time
                phVectorEnd: list = [] # Flow's phases' end time
                for ph in range(int(network.LCM / network.minPeriod)):
                    phVectorStart.append(0) # As many 0's as phases
                    phVectorEnd.append(0)
                #------------------------------------------------------------------------------
                t1: float # Tx start time
                t2: float # Tx end time
                #------------------------------------------------------------------------------
                # ASSIGNS TIME SCHEDULE FOR EVERY PHASE IN PORT
                while((phase+j) < (int(network.LCM / network.minPeriod))):
                    if(j * (network.minPeriod / network.order) % \
                       (network.flows[flowID].T_tx / network.order) == 0 and \
                        scheduled < (network.LCM / network.flows[flowID].T_tx)): # T
                        t0_gb: list
                        label .begin
                        if(initialized):
                            # Previous hop's port's GB is added
                            t0_gb = [round((ph + guardBand_p), network.precision) for ph in t0]
                        else:
                            # No GB is added (only in fitness delay eval.) 
                            t0_gb = t0[ : ]
                        if(port.schUP[phase+j] < t0_gb[phase+j]): 
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # CASE 1: TIME START > UPPERBOUND 
                            t1 = t0_gb[phase+j] # No queue time, just GB waiting
                            t2 = t1 + (network.flows[flowID].length / port.speed) # Tx time
                            t1 = round(t1, network.precision)
                            t2 = round(t2, network.precision)
                            if((initialized and (network.minPeriod - (t2 + guardBand)) >= 0) or \
                            (not initialized and (network.minPeriod - (t2 + guardBand)) >= \
                            margin * network.minPeriod)):
                                aux: bool = 0 # No gaps yet
                                if(not port.schFlowID[phase+j] and leftSide):  
                                    # Updates LW only in case it's the first flow for this phase 
                                    port.schLW[phase+j] = t1 # Port LW updated
                                # --- BIDIRECTIONAL ---
                                    # Inside
                                    if(network.bidirec):
                                        for next_port in network.nodes[port.neighborID].ports:
                                            if(next_port.neighborID == path[i]):
                                                next_port.schLW[phase+j] = t1 # Neighbor LW update
                                                break
                                # Outside
                                else:
                                    # Gap (saved counting last GB)
                                    port.gap[phase+j].append([port.schUP[phase+j], t1]) 
                                    if(guardBand > 0):
                                        port.gap[phase+j].append([t2, round((t2 + guardBand), \
                                            network.precision)])
                                    port.mergeGaps(phase+j)
                                    aux = 1 # New gap from last UP
                                if(network.bidirec):
                                    for next_port in network.nodes[port.neighborID].ports:
                                        if(next_port.neighborID == path[i]):
                                            if(aux):
                                                # Gap saved (saved counting last GB)
                                                next_port.gap[phase+j].append([port.schUP[phase+j], t1])
                                                if(guardBand > 0):
                                                    next_port.gap[phase+j].append([t2, round((t2 + guardBand), \
                                                        network.precision)])
                                                next_port.mergeGaps(phase+j)
                                            next_port.schUP[phase+j] = round((t2 + guardBand), \
                                                network.precision) # Neighbor's port's UP updated
                                            next_port.lastGB[phase+j] = guardBand
                                            next_port.schFlowID[phase+j].append(flowID) # FLOW ID
                                            next_port.latArrTime[phase+j] = t0_gb[phase+j] # New last arrival time 
                                            break
                                #----------------------
                                port.schUP[phase+j] = round((t2 + guardBand), network.precision)
                                port.lastGB[phase+j] = guardBand
                                port.schFlowID[phase+j].append(flowID) # FLOW ID
                                scheduled += 1 # Port's phase is scheduled           
                                port.latArrTime[phase+j] = t0[phase+j] # New last arrival time                     
                                phVectorStart[phase+j] = round(((phase+j) * network.minPeriod + t1 + \
                                    time), network.precision)
                                phVectorEnd[phase+j] = round(((phase+j) * network.minPeriod + t2 + \
                                    time), network.precision)
                                j += 1 # Phase is scheduled
                            else:
                                # IF NOT POSSIBLE, DO IT NEXT PHASE (phase+=1, keep j)
                                if(not scheduled and not initialized and 
                                network.flows[flowID].T_tx != network.minPeriod):
                                    # This case will not happen, as it would already be initialized
                                    phase += 1 # New phase but not schedule count
                                    if((phase+j) != (int(network.LCM / network.minPeriod))):
                                        t0[phase+j] = 0 # Starts with 0 in new phase
                                else:
                                    # Cannot accomplish periodicity after first phase scheduled
                                    if(debug):
                                        print("FLOW #" + str(flowID) + " COULD NOT BE SCHEDULED IN NODE #" + \
                                        str(network.nodes[path[i]].id))
                                    return 0 
                        else:
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # CASE 2: TIME START < UPPERBOUND -- Only lowerbound is updated
                            tpp: float = t0_gb[phase+j] + (network.flows[flowID].length / \
                                port.speed) + guardBand # Tx time
                            if(tpp <= port.schLW[phase+j] and leftSide and initialized):
                                # SUBCASE: CAN BE PLACED BEFORE ALL SEQUENCE
                                if(compression): # With left compression
                                    t1 = port.schLW[phase+j] - guardBand - \
                                        (network.flows[flowID].length / port.speed) 
                                    t2 = port.schLW[phase+j] - guardBand
                                else:
                                    t1 = t0_gb[phase+j] # Without left compression
                                    t2 = tpp - guardBand
                                t1 = round(t1, network.precision)
                                t2 = round(t2, network.precision)
                                if(not compression):
                                    port.gap[phase+j].append([t2, port.schLW[phase+j]]) # Gap saved
                                # --- BIDIRECTIONAL ---
                                if(network.bidirec):
                                    for next_port in network.nodes[port.neighborID].ports:
                                        if(next_port.neighborID == path[i]):
                                            if(not compression):
                                                next_port.gap[phase+j].append([t2, \
                                                    next_port.schLW[phase+j]]) # Gap saved
                                            next_port.schLW[phase+j] = t1 # Neighbor LW update
                                            next_port.schFlowID[phase+j].append(flowID)
                                            break
                                #----------------------
                                port.schLW[phase+j] = t1 # Port's LW updated
                                port.schFlowID[phase+j].append(flowID) # FLOW ID
                                scheduled += 1 # Port's phase is scheduled
                                phVectorStart[phase+j] = round(((phase+j) * network.minPeriod + t1 + \
                                    time), network.precision)
                                phVectorEnd[phase+j] = round(((phase+j) * network.minPeriod + t2 + \
                                    time), network.precision)
                                j += 1 # Phase is scheduled
                            else:
                                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                # SUBCASE 1: CANNOT BE PLACED BEFORE ALL SEQUENCE -- Only upperbound is updated
                                # Check first if arrival time to node is possible
                                if(t0_gb[phase+j] < port.schUP[phase+j] and initialized and not scheduled): # Check path
                                    dif: float = round((port.schUP[phase+j] - t0_gb[phase+j]), network.precision)
                                    dif_alt: float = round((port.latArrTime[phase+j] - t0[phase+j]), network.precision)
                                    if(port.latArrTime[phase+j] and dif_alt > dif): # To avoid FIFO non-compliance
                                        dif = dif_alt                               
                                    isPossible: bool = slowDown(flowID, path, dif, i, guardBand)
                                    if(not isPossible):
                                        if(debug):
                                            print("FLOW #" + str(flowID) + " COULD NOT BE SCHEDULED IN NODE #" + \
                                            str(network.nodes[path[i]].id) + " DUE TO SHORT GAP INTERVAL")
                                        return 0
                                    # Update t0 from previous node
                                    for ph in range(len(t0)):
                                        if(network.flows[flowID].schEnd[i-1][ph] != 0):
                                            # Prev. node end time + Bridge delay
                                            t0[ph] = (network.flows[flowID].schEnd[i-1][ph] + \
                                            network.nodes[path[i]].procDelay + \
                                            network.radioDelay[path[i]][src_port][dst_port]) 
                                            t0[ph] = round((t0[ph] % network.minPeriod), network.precision)
                                    goto .begin # In case it is slowed down so now t1=t0_gb > schUP
                                #---------------------------------------------------------------
                                t1 = port.schUP[phase+j]
                                t2 = t1 + (network.flows[flowID].length / port.speed)
                                t1 = round(t1, network.precision)
                                t2 = round(t2, network.precision)
                                if((initialized and (network.minPeriod - (t2 + guardBand)) >= 0) or \
                                (not initialized and (network.minPeriod - (t2 + guardBand)) >= \
                                margin * network.minPeriod)):
                                    # --- BIDIRECTIONAL ---
                                    if(network.bidirec):
                                        for next_port in network.nodes[port.neighborID].ports:
                                            if(next_port.neighborID == path[i]):
                                                if(guardBand > 0):
                                                    next_port.gap[phase+j].append([t2, round((t2 + guardBand), \
                                                        network.precision)])
                                                next_port.schUP[phase+j] = round((t2 + guardBand), \
                                                    network.precision) # Neighbor's port's UP updated
                                                next_port.lastGB[phase+j] = guardBand
                                                next_port.schFlowID[phase+j].append(flowID) # FLOW ID
                                                next_port.latArrTime[phase+j] = t0[phase+j] # New last arrival time
                                                break
                                    #----------------------
                                    if(guardBand > 0):
                                        port.gap[phase+j].append([t2, round((t2 + guardBand), network.precision)])
                                    port.schUP[phase+j] = round((t2 + guardBand), network.precision) # Port's UP updated
                                    port.lastGB[phase+j] = guardBand
                                    port.schFlowID[phase+j].append(flowID) # FLOW ID
                                    scheduled += 1 # Port's phase is scheduled
                                    port.latArrTime[phase+j] = t0_gb[phase+j] # New last arrival time
                                    phVectorStart[phase+j] = round(((phase+j) * network.minPeriod + t1 + \
                                        time), network.precision)
                                    phVectorEnd[phase+j] = round(((phase+j) * network.minPeriod + t2 + \
                                        time), network.precision)
                                    j += 1 # Phase is scheduled
                                else:
                                    # IF NOT POSSIBLE, DO IT NEXT PHASE (phase+=1, keep j)
                                    if(not scheduled and not initialized and 
                                    network.flows[flowID].T_tx != network.minPeriod):
                                        phase += 1 # New phase but not schedule count
                                        if((phase+j) != (int(network.LCM / network.minPeriod))):
                                            t0[phase+j] = 0 # Starts with 0 in new phase
                                    else:
                                        # Cannot accomplish periodicity after first phase scheduled 
                                        if(debug):
                                            print("FLOW #" + str(flowID) + " COULD NOT BE SCHEDULED IN NODE #" + \
                                            str(network.nodes[path[i]].id))
                                        return 0  
                    else:
                        j += 1 # Phase is not scheduled
                    #--------------------------------------------------------------------------
                    # Goes back with the start (with t0=0, but more delay)
                    if((phase+j) == (int(network.LCM / network.minPeriod)) \
                        and j < int(network.LCM / network.minPeriod) and \
                        scheduled < int(network.LCM / network.flows[flowID].T_tx)):
                        if(phChange):
                            # Goes back to 1st phase [0], only once
                            phase = -j # To make phase+j=0 
                            time = round((time + network.LCM), network.precision) # Sums hyperperiod
                            phChange = 0 # No come back again
                        else:
                            if(debug):
                                print("FLOW #" + str(flowID) + " COULD NOT BE SCHEDULED IN NODE #" + \
                                    str(network.nodes[path[i]].id))
                            return 0
                #------------------------------------------------------------------------------
                # FLOW SCHEDULING TIMES
                if(scheduled == int(network.LCM / network.flows[flowID].T_tx)): 
                    # Only if all periods/phases were scheduled 
                    network.flows[flowID].schStart.append(phVectorStart) # START
                    network.flows[flowID].schEnd.append(phVectorEnd) # END
                    if(debug and not conf_sol):
                        print("FLOW #" + str(flowID) + " WAS SUCCESFULLY SCHEDULED IN NODE #" + \
                            str(network.nodes[path[i]].id))
                else:
                    if(debug and not conf_sol):
                        print("FLOW #" + str(flowID) + " COULD NOT BE SCHEDULED IN NODE #" + \
                            str(network.nodes[path[i]].id) + "(t1=" + str(t1) + ", t2=" + str(t2) + ")")
                    return 0
                break
        initialized = 1 # First scheduled
        prev_port = copy.copy(port) # Port saved
    return 1                                                             
"""-----------------------------------------------------------------------------------------"""
# Applies a delay to the same flow in the previous node if it is possible to do so. This is 
# performed due to the TSN queing system's limits, as it uses FIFO queues, so frames must
# arrive to port's queue after the one awaiting to be sent. Delayed dif value, which is the
# difference between new flow's arrival time and previous flow's arrival time.
def slowDown(flowID: int, path: list, dif: float, nodeStart: int, guardBand: float) -> bool:                             
    rePath: list = path[ :nodeStart+1] # All previous nodes
    modifNodes: list = [] # List of modified nodes
    modifNodes.append(network.nodes[rePath[nodeStart]].id) # Current node
    done: int = 0 # Changes to 1 if at least one node's port's scheduling delay was performed
    possibleSD: bool = 0 # 1 if slow down was possible, 0 if not
    #------------------------------------------------------------------------------------------
    for k in range(len(rePath)-2, -1, -1): # All previous nodes, current one as last port
        for rePort in network.nodes[rePath[k]].ports:
            if(rePort.neighborID == rePath[k+1]):
                for ph in range(len(rePort.schLW)):
                    rePort.mergeGaps(ph)
                    if(flowID in rePort.schFlowID[ph]): # Periodicity
                        possibleSD = 0
                        lwFlow: float = round((network.flows[flowID].schStart[k][ph] % \
                            network.minPeriod), network.precision)
                        upFlow: float = round((network.flows[flowID].schEnd[k][ph] % \
                            network.minPeriod), network.precision)
                        lwPort: float = round(rePort.schLW[ph], network.precision)
                        upPort: float = round(rePort.schUP[ph], network.precision) 
                        if(round((upFlow + rePort.lastGB[ph]), network.precision) == upPort): # At the end
                            if(network.minPeriod - round((upPort + dif), network.precision) >= 0):
                                # Port
                                if(rePort.perc_gb > 0):
                                    rePort.gap[ph][-1] = [round((rePort.gap[ph][-1][0] + dif), network.precision), \
                                        round((rePort.gap[ph][-1][1] + dif), network.precision)]
                                rePort.gap[ph].append([lwFlow, round((lwFlow + dif), network.precision)]) # New gap
                                rePort.schUP[ph] = round((rePort.schUP[ph] + dif), network.precision)
                                rePort.mergeGaps(ph)
                                # --- BIDIRECTIONAL ---
                                if(network.bidirec):
                                    for next_rePort in network.nodes[rePort.neighborID].ports:
                                        if(next_rePort.neighborID == rePath[k]):
                                            if(next_rePort.perc_gb > 0):
                                                next_rePort.gap[ph][-1] = [round((next_rePort.gap[ph][-1][0] + dif), network.precision), \
                                                    round((next_rePort.gap[ph][-1][1] + dif), network.precision)]
                                            next_rePort.gap[ph].append([lwFlow, round((lwFlow + dif), network.precision)]) # New gap
                                            next_rePort.schUP[ph] = round((next_rePort.schUP[ph] + \
                                                dif), network.precision)
                                            next_rePort.mergeGaps(ph)
                                            break
                                #----------------------
                                # Flow
                                network.flows[flowID].schStart[k][ph] = round((network.flows[flowID].schStart[k][ph] + \
                                    dif), network.precision)
                                network.flows[flowID].schEnd[k][ph] = round((network.flows[flowID].schEnd[k][ph] + \
                                    dif), network.precision)
                                possibleSD = 1 # Delayed
                            else:
                                if(debug):
                                    print("LIMIT NOT ENOUGH IN NODE #" + str(network.nodes[rePath[k]].id))
                                return 0
                        else:
                            if(lwFlow == lwPort): # At the beginning (only used if leftSide=1)
                                for n in range(len(rePort.gap[ph])):
                                    if(upFlow == rePort.gap[ph][n][0]):
                                        if(round((rePort.gap[ph][n][1] - rePort.gap[ph][n][0]), \
                                            network.precision) >= (dif + guardBand)):
                                            # Port
                                            rePort.gap[ph][n] = [(upFlow + dif), rePort.gap[ph][n][1]] # Gap reduced
                                            rePort.schLW[ph] += dif
                                            # --- BIDIRECTIONAL ---
                                            if(network.bidirec):
                                                for next_rePort in network.nodes[rePort.neighborID].ports:
                                                    if(next_rePort.neighborID == rePath[k]):
                                                        next_rePort.gap[ph][n] = [(upFlow + dif), \
                                                            next_rePort.gap[ph][n][1]]
                                                        next_rePort.schLW[ph] += dif
                                                        break
                                            #----------------------
                                            # Flow
                                            network.flows[flowID].schStart[k][ph] += dif
                                            network.flows[flowID].schEnd[k][ph] += dif
                                            possibleSD = 1 # Delayed
                                            break
                                        else:
                                            if(debug):
                                                print("GAP NOT ENOUGH")
                                    else:
                                        if(debug):
                                            print("GAP NOT POSSIBLE")
                            else:
                                # This case is not possible
                                return 0
                break
    #------------------------------------------------------------------------------------------
        # Inserts node as modified
        if(possibleSD):
            modifNodes.insert(0, network.nodes[rePath[k]].id) # Modified nodes
            done += 1
        else:
            if(done > 0): # At least one previous node has been modified
                break
            else:
                if(debug):
                    print("NO GAP TO DELAY") 
                return 0 # No node has been modified, flow cannot be scheduled
    #------------------------------------------------------------------------------------------
    # Setting all node's last arrival time
    init: bool = 0 # Last (first) node was reached, very first t0
    if(done): 
        if(done == len(rePath)-1): # Very first scheduled node
            init = 1
        for v in range(len(modifNodes)-1):
            for chgPort in network.nodes[modifNodes[v]].ports:
                if(chgPort.neighborID == modifNodes[v+1]):
                    for phas in range(len(chgPort.latArrTime)):
                        if(init):
                            # Very first node start time
                            chgPort.latArrTime[phas] = round((network.flows[flowID].schStart[v][phas] % \
                                network.minPeriod), network.precision) 
                        else:
                            # Last node's end time
                            chgPort.latArrTime[phas] = round((network.flows[flowID].schEnd[v-1][phas] % \
                                network.minPeriod), network.precision) 
                    init = 0
        return 1
"""-----------------------------------------------------------------------------------------"""
# Fitness function evaluates the result of applying a chromosome solution to the whole TSN
# network, as it schedules flows with their own path selected by the genetic algorithm. It 
# uses the chromosome solution and its index in the population. The chromosome solution is
# a list of values that contains, for each flow, the index of the path in a list of possible
# paths found for it. It returns the fitness value of a given chromosome.
def fitness_func(sol_chr: list, chr_pop_idx: int) -> float:
    #------------------------------------------------------------------------------------------
    # Extraction of path chromosomes given index chromosomes
    chr: list = []
    for gen in range(len(sol_chr)):
        chr.append(network.pathFlows[gen][sol_chr[gen]]) # Extracts the path for flows
    #------------------------------------------------------------------------------------------
    # Network initialization (only nodes and ports are reset)
    network.reset()
    #------------------------------------------------------------------------------------------
    # Flows' scheduling
    delay: list = []
    delay_mean: float = 0
    delay_eval: float = 0
    delay_norm: float
    delay_flow: float
    if(debug):
        print("\n-- FLOWS' TIMES --")
    for numFlow in range(len(chr)): 
        path: list = chr[numFlow]
        network.flows[numFlow].updateRoute(path) # Flow reset
        if(debug):
            print("\nFlow: " + str(network.flows[numFlow].id) + ", Path: " + str(path) + \
                "  (Max. Delay = " + str(round((network.flows[numFlow].maxDelay / network.order), 2)) + \
                " ms, Period = " + str(round((network.flows[numFlow].T_tx / network.order), 2)) + \
                " ms, Size = " + str(int(network.flows[numFlow].length / 8)) + " Bytes)")
        possible: bool = scheduling(path, numFlow) # Flow scheduling on its corresponding path
        if(possible):
            if(debug):
                print("  Start time for flow #" + str(numFlow) + " per node: " + \
                str(network.flows[numFlow].schStart))
                print("  End time for flow #" + str(numFlow) + " per node:   " + \
                str(network.flows[numFlow].schEnd))
    #------------------------------------------------------------------------------------------
            # --- DELAY EVALUATION ---
            # Highest value of time in last node
            delay1: float = network.flows[numFlow].schEnd[-1][np.max(np.nonzero(network.flows[numFlow].schEnd[-1]))] 
            # Highest value of time in first node
            delay2: float
            try:
                delay2 = network.flows[numFlow].schStart[0][np.max(np.nonzero((network.flows[numFlow].schStart[0])))] 
            except ValueError:
                delay2 = 0
                pass
            # Total delay is the difference of the first and the last scheduled times in the TSN network 
            # plus the SW's processing delay in the first and last nodes plus the talker and listener 
            # throughput. Also, first node's guard band must be considered. Flow's max. delay normalization
            node_init: int = network.flows[numFlow].pathNodes[0]
            for p in network.nodes[node_init].ports:
                if(p.neighborID == network.flows[numFlow].pathNodes[1]):
                    port_init: object = p
            delay_flow = (abs(delay1 - delay2) + \
                (network.nodes[network.flows[numFlow].pathNodes[0]].procDelay) + \
                (network.nodes[network.flows[numFlow].pathNodes[-1]].procDelay) + \
                (network.flows[numFlow].length / network.flows[numFlow].talkerSpeed) * \
                (1 + port_init.perc_gb) + \
                (network.flows[numFlow].length / network.flows[numFlow].listenerSpeed))
            if(debug):
                print("  End-to-end delay:                 " + str(round(delay_flow, network.precision)))
            if(delay_flow <= network.flows[numFlow].maxDelay):
                delay_mean += delay_flow
                delay_norm = (delay_flow / network.flows[numFlow].maxDelay)
                delay.append(delay_norm)
                delay_eval += (delay_flow / network.flows[numFlow].length / len(network.flows[numFlow].pathNodes))
            else:
                if(debug):
                    print("FLOW #" + str(numFlow) + " EXCEEDED ITS MAXIMUM DELAY: " + str(chr))
                return 0
        else:
            if(debug):
                print("NO POSSIBLE SCHEDULING IN THIS CHROMOSOME")
            return 0 # fitness=0 too low
    delay_norm: float = (sum(delay) / network.M)
    delay_mean = (delay_mean / network.M)
    delay_eval = (delay_eval / network.M)
    #------------------------------------------------------------------------------------------
    # --- GAP EVALUATION ---
    count: float = 0.0 # Gap counter 
    gap_norm: float = 0.0 # General gap counter
    cnt_ph: int = 0 # Phase counter 
    for node_ in network.nodes:
        for port_ in node_.ports:
            for ph in range(len(port_.gap)):
                if(len(port_.gap[ph]) > 0): # Gaps observed
                    cnt_ph += 1
                    port_.mergeGaps(ph) # Sort
                    gapVal: float
                    for gap in port_.gap[ph]:
                        # Delete latest GB
                        if(gap[1] == port_.schUP[ph]):
                            del port_.gap[ph][-1]
                            if(len(port_.gap[ph]) == 0):
                                cnt_ph -= 1
                                break
                        else:
                            # Gap fitting max. size Ethernet frame 
                            gapVal = (gap[1] - gap[0])
                            if(gap[0] != 0.0):
                                if((gapVal / (network.maxLenFrame / port_.speed)) > 1):
                                    count += (network.maxLenFrame / port_.speed) # GB
                                else:
                                    count += gapVal # The whole gap
                    if(not leftSide):
                        # Space left + first gap next ph**
                        gapVal = (network.minPeriod - (port_.schUP[ph] - port_.lastGB[ph]))
                        if(len(port_.gap[ph]) > 0):
                            if(port_.gap[ph][0][0] == 0.0):
                                gapVal += (port_.gap[ph][0][1] - port_.gap[ph][0][0]) # First gap
                        if((gapVal / (network.maxLenFrame / port_.speed)) > 1):
                            count += (network.maxLenFrame / port_.speed) # GB
                        else:
                            count += gapVal # The whole gap
                    else:
                        # Space left + first space next ph (not a gap)
                        gapVal = port_.schLW[ph] + (network.minPeriod - \
                        (port_.schUP[ph] - port_.lastGB[ph]))
                        if((gapVal / (network.maxLenFrame / port_.speed)) > 1):
                            count += (network.maxLenFrame / port_.speed) # GB
                        else:
                            count += gapVal # The whole gap
    if(network.bidirec):
        count = 0.5 * count # Counted twice
    gap_norm = (count / ((cnt_ph) * network.minPeriod))
    gap_norm = (gap_norm) # Correction 
    usage: float = ((1 - gap_norm) * 100)
    #------------------------------------------------------------------------------------------
    if(debug):
        print("\n || Average Delay: " + str(delay_mean) + \
            " s || Network Link Usage: " + str(usage) + \
            " % || Norma-sized delay: " + str(delay_eval * 512 * 3)) # 512 aprox. average size
    #------------------------------------------------------------------------------------------
    # --- FITNESS FUNCTION EVALUATION ---
    fitness: float
    #print("DELAY: " + str(WEIGHT_DELAY * delay_norm))
    #print("GAPS:  " + str(WEIGHT_GAP * gap_norm * correction))
    fitness = ((WEIGHT_DELAY * delay_norm + WEIGHT_GAP * gap_norm * correction))
    fitness = (1.0 / (fitness + 1e-10)) # Fitness maximized by PyGAD
    return fitness
    #-----------------------------------------------------------------------------------------
"""-----------------------------------------------------------------------------------------"""
# Prints every evolutionary algorithm's generation's best solution.
def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)
"""-----------------------------------------------------------------------------------------"""


# ******************************************* MAIN ********************************************
#----------------------------------------------------------------------------------------------
# Flow's constraints can make the solution be quite different to shortest path (all 0's):
#     - Shorter max. delay (e.g. 0.5ms), overall in higher genes
#     - Lower Tx period (e.g. 1ms), not fitting many more flows in scheduling
#     - Larger frame length (e.g. 512 Bytes)
#     - More complex topology with less links
#     - Higher processing delay in switches (e.g. 100us)
#     - Lower link speed (e.g. 100Mbps)
#     - Half-duplex instead of full-duplex

# Also, execution time may increase due to:
#     - Number of flows
#     - Number of paths/flow
#     - Number of generations
#     - Number of initial population
#     - Not using boost (all 0's in initial population)
#----------------------------------------------------------------------------------------------
# Debugging case 
#flowState: str = "flowsVector_80"
#correction: int = 1
#WEIGHT_DELAY: float = 0.5
#WEIGHT_GAP: float = 0.5
#parent_selection_type: str = "rank"
#crossover_type: str = "two_points"
#mutation_probability: float = 0.05
#indexe: int = 1
#----------------------------------------------------------------------------------------------
# Execution path
path: str = os.path.dirname(__file__) # Files' directory
flowState: str = str(sys.argv[1]) # Name of input flows file to be executed
#----------------------------------------------------------------------------------------------
if(len(sys.argv) >= 3):
    correction : float = float(sys.argv[2]) # Correction factor between delay and gaps
else:
    correction = 1
#----------------------------------------------------------------------------------------------
# Network parameters
mode5G: bool = 1 # 1 if 5G+TSN (5G files are read), 0 if TSN only <ATTENTION: TOPOLOGY SELECTED>
reduceSpace: int = 6 # [Search space]: number of possible paths for every flow (default: 4)
pathLen: int = 6 # [Search space]: number of nodes per path for every flow (default: 4)
bidirec: bool = 0 # 1 if links are half-duplex, 0 if full-duplex (default: 0)
maxLenFrame: int = 1500 + 42 # Ethernet max. frame size in Bytes (default: 1500)
network = TSN_NETWORK(path = path, reduceSpace = reduceSpace, \
                      pathLen = pathLen, bidirec = bidirec, \
                      maxLenFrame = maxLenFrame) # Creation of topology
#----------------------------------------------------------------------------------------------
# Scheduling parameters
leftSide: bool = 0 # Use left side of scheduling (first schedule compression)
compression: bool = 0 # Compression in scheduling (always compression) [only if leftSide=1]
margin: float = 0.1 # Margin to complete phase scheduling (10 % of minPeriod)
#----------------------------------------------------------------------------------------------
# Checking
plot_top: bool = 1 # Plots the TSN topology (1 ON, 0 OFF)
plot_port: bool = 1 # Plots ports timing (1 ON, 0 OFF)
debug: bool = 0 # debug mode (1 ON, 0 OFF)
conf_sol: bool = 1 # Shows the configuration result (1 ON, 0 OFF)
print_paths: bool = 1 # Shows possible paths for each flow (1 ON, 0 OFF)
#----------------------------------------------------------------------------------------------
# Weights
WEIGHT_DELAY: float = float(sys.argv[6]) # Penalising weight for delay
WEIGHT_GAP: float = float(sys.argv[7]) # Penalising weight for gaps
#----------------------------------------------------------------------------------------------
# Genetic algorithm parameters: initial population and chromosomes
boost: bool = 0 # Shortest path solution in initial population (also used for test)
n_solutions: int = 100 # 100 [Search space]: Number o parents, at least 2
num_generations: int = 15000 # 15000 [Search space]: Number of iterations
stop_criteria: str = "saturate_5000" # Stops if fitness doesn't change for n iterations
initial_population: list = []
if(boost):
    n_solutions -= 1
    initial_population.append(np.zeros(len(network.flows, ), dtype = int)) # Boost
for x in range(n_solutions): # Initial population of 100
        initial_population.append(initializeChromosome(network.pathFlows)) 
#----------------------------------------------------------------------------------------------
# Genetic algorithm parameters: genes
num_genes: int = len(network.flows)
gene_type = int # Indices
gene_space: list = []
for flow in range(len(network.pathFlows)):
    gene_space.append(list(range(0, len(network.pathFlows[flow])))) # Only possible indices
#----------------------------------------------------------------------------------------------
# Genetic algorithm parameters: selection
keep_parents: int = 1 # Number of parents kept in next generation (<=mating parents)
parent_selection_type: str = str(sys.argv[3])
    # sss - Steady-State Selection
    # rws - Roulette Wheel Selection
    # sus - Stochastic Universal Selection
    # rank - Rank Selection
    # random - Random Selection
    # tournament - Tournament Selection
K_tournament: int = 5 # Only in case "tournament" is set (default 3)
# NOTE: Higher value (and higher pressure) leads to a fast convergence to local maximum, but
#       a shorter number of generations.
#----------------------------------------------------------------------------------------------
# Genetic algorithm parameters: crossover
num_parents_mating: int = 2 # Mating parents on each combination
crossover_type: str = str(sys.argv[4])
    # single_point - Single-Point Crossover
    # two_points - Two-Points Crossover
    # uniform - Uniform Crossover
    # scattered - Scattered Crossover
crossover_probability = None # No probability to be selected as parent, all are selected
#----------------------------------------------------------------------------------------------
# Genetic algorithm parameters: mutation
mutation_type: str = "random" # Random Mutation (available with different gene spaces)
mutation_probability: float = float(sys.argv[5]) # Probability of gene mutation (10%)
mutation_by_replacement: bool = True # Replacement in mutation
#----------------------------------------------------------------------------------------------
# Genetic algorithm parameters: solution
save_best_solutions: bool = True # Save best solutions
suppress_warnings: bool = True
#----------------------------------------------------------------------------------------------
#Execution counter
indexe: int = int(sys.argv[8]) 
#----------------------------------------------------------------------------------------------
# Genetic algorithm execution
try:
    ga_instance: object = pygad.GA(num_generations = num_generations,
                                   num_parents_mating = num_parents_mating,
                                   stop_criteria = stop_criteria,
                                   initial_population = initial_population,
                                   num_genes = num_genes,
                                   fitness_func = fitness_func,
                                   #on_generation = on_generation,
                                   keep_parents = keep_parents,
                                   parent_selection_type = parent_selection_type,
                                   crossover_type = crossover_type,
                                   crossover_probability = crossover_probability,
                                   mutation_type = mutation_type,
                                   mutation_probability = mutation_probability,
                                   save_best_solutions = save_best_solutions,
                                   suppress_warnings = suppress_warnings,
                                   gene_type = gene_type,
                                   gene_space = gene_space,
                                   parallel_processing = None)
    ga_instance.run()
    solution: list
    solution_fitness: float
    solution_idx: int
    solution, solution_fitness, solution_idx = ga_instance.best_solution() # Best solution and fitness
#----------------------------------------------------------------------------------------------
# Plotting results
    # Graph
    plt.figure()
    plt.plot(ga_instance.best_solutions_fitness)
    plt.title("Fitness Vs. Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(path + "/Results/test_" + flowState + "_" + parent_selection_type \
    + "_" + crossover_type + "_" + str(int(mutation_probability * 100)) + "_" + str(WEIGHT_DELAY) + \
        "_" + str(WEIGHT_GAP) + "_" + str(indexe) + ".png", dpi = 500) # Save results
    plt.show(block = False)
    # File .mat
    fitVector: list = {"generations": ga_instance.generations_completed, "fitness": ga_instance.best_solutions_fitness}
    savemat(path + "/Results/fitness_vs_generations_" + flowState + "_" + parent_selection_type + \
    "_" + crossover_type + "_" + str(int(mutation_probability * 100)) + "_" + str(WEIGHT_DELAY) + \
        "_" + str(WEIGHT_GAP) + "_" + str(indexe) + ".mat", fitVector)
    # Logs
    print("Best scheduling solution found: " + str(solution) + ", fitness: " + str(solution_fitness))
    fileRes: object = open(path + "/Results/test_log_" + flowState + "_" + parent_selection_type \
    + "_" + crossover_type + "_" + str(int(mutation_probability * 100)) + "_" + str(WEIGHT_DELAY) + \
        "_" + str(WEIGHT_GAP) + "_" + str(indexe) + ".txt", 'a+')
    fileRes.write("\nSolution to flows " + flowState + " --> " + str(solution) + \
                  ", fitness=" + str(solution_fitness))
    fileRes.close()
#----------------------------------------------------------------------------------------------   
except Exception as e:
    print("AN EXCEPTION ERROR OCCURRED WITH THE OBJECT'S CALL: ")
    print(e)
    sys.exit()
#----------------------------------------------------------------------------------------------
# Checking best solution: flow times and nodes configuration
if(conf_sol):
    debug = 1
    fit_: list = fitness_func(solution, 0)
    # Representing final nodes' ports' configuration
    print("\n-- NODES' CONFIGURATION --")
    for node in network.nodes:
        print("\nConfiguration for ports in Node #" + str(node.id) + ": ")
        aux: list = []
        for port in node.ports:
            aux.append([port.neighborID, port.speed, port.schLW, port.schUP, port.schFlowID])
            print("  " + str([port.neighborID, port.speed, port.schLW, port.schUP, port.schFlowID]))
    print("\n")
#----------------------------------------------------------------------------------------------
if(print_paths):
    print("-- POSSIBLE PATHS PER FLOW --\n")
    for i in range(len(network.pathFlows)):
        print("Flow #" +str(i) + ": " + str(network.pathFlows[i]))     
    print("\n")
#---------------------------------------------------------------------------------------------- 
# Plotting TSN network's topology
if(plot_top):
    topology: object = nx.from_numpy_matrix(np.matrix(network.topology))
    plt.figure()
    graph: object = nx.Graph()
    for i in range(len(network.topology)):
        graph.add_node(i)
        for j in range(len(network.topology[i])):
            if(i != j and network.topology[i][j] > 0):
                graph.add_edge(i, j, label = str(network.topology[i][j] * 1e-9) + " Gbps")
    pos: object = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos, with_labels = 1)
    labels: object = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
    plt.show(block = False)
    x: str = input("\n\nPress [enter] to continue.")
#----------------------------------------------------------------------------------------------
# Plotting solution's schedule per phases in nodes and ports
if(plot_port and conf_sol):
    iter = 1
    print("Checking port's timing...")
    while(iter):
        try:
            nodeID = int(input("  INSERT NODE ID: "))
            portNeighborID = int(input("  INSERT PORT (NEIGHBOR) ID: "))
            network.plotScheduling(nodeID, portNeighborID)
            iter = int(input("  INSERT NEW VALUES? (1 YES, 0 NO): "))
        except UnboundLocalError:
            print("  UNEXISTENT PORT CONNECTION. PLEASE CHECK TOPOLOGY")
#----------------------------------------------------------------------------------------------



