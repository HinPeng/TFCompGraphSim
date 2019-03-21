import os
import sys

def getSwapOutNodeID(result):
    swapconflict = "Swap in when swapping out not finish"
    nodeIds = []
    with open(result, "r") as fout:
        for line in fout:
            if swapconflict in line:
                nodeIds.append(line.split(" ")[-1].strip('\n'))
    nodeIds = list(set(nodeIds))
    return nodeIds

def deleteConflictNodes(filepath, result):
    nodeIds = getSwapOutNodeID(result)
    lines = []
    with open(filepath, "r") as fin:
        lines = fin.readlines()
        lines = [line for line in lines if line.split('\t')[0] not in nodeIds]
    with open(filepath, "w") as fin:
        fin.writelines(lines)

if __name__ == '__main__':
    filepath = '/home/frog/maweiliang/tmp/swap_policy.txt'
    result = './result.log'
    deleteConflictNodes(filepath, result)

            




