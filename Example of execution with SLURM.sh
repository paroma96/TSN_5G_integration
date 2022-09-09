#!/bin/bash
#SBATCH --job-name=ga_TSN5G_PRM_test_80_80_20                                           # Job name
#SBATCH --mail-type=END,FAIL                                                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=xxxx@correo.ugr.es                                                  # Where to send mail	
#SBATCH --nodes=1                                                                       # Number of nodes
#SBATCH --partition xxxx                                                                # Partition
#SBATCH --mem=2gb                                                                       # Job memory request
#SBATCH --time=24:00:00                                                                 # Time limit hrs:min:sec
#SBATCH --output=/path_to/Results/BashLogs/logs_ag.out                                  # Standard output and error log

pwd; hostname; date

source /path_to/pyenvi/bin/activate

echo "Starting process..."

flowState="flowsVector_80"
selection="rank"
crossover="two_points"
mutation="0.05"
correction=5
W_delay=0.8
W_gap=0.2
iterator=1


until [ $iterator -gt 10 ] 
do
    python3 /path_to/geneticAlgorithmTSN5G.py $flowState $correction $selection $crossover $mutation $W_delay $W_gap $iterator > /path_to/Results/"logs_${flowState}_${selection}_${crossover}_${mutation}_${W_delay}_${W_gap}_${iterator}.txt"
    ((iterator=iterator+1))
done

date