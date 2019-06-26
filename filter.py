import os
import sys

def getSwapOutNodeID(result):
  swapconflict = "not finish swap in before comp."
  nodes = dict()
  flag = False
  iteration = 1
  nodes[iteration] = list()
  with open(result, "r") as fout:
    for line in fout:
      if swapconflict in line:
        flag=True
        nodes[iteration].append(line.split(" ")[-7]+'\n')
      else:
        if flag:
          flag=False
          iteration += 1
          nodes[iteration] = list()
  return nodes

def compareBatch_size(nodes1, nodes2):
  diff = []
  common = []
  for node in nodes1:
    if node not in nodes2:
      diff.append(node)
    else:
      common.append(node)
  return diff, common

def compareIteration(nodes):
  iteration = 1
  base = nodes[iteration]
  diff = dict()
  # print nodes.keys()
  while iteration < len(nodes.keys()):
    iteration += 1
    diff[iteration] = list()
    for node in base:
      # print node
      if node not in nodes[iteration]:
          diff[iteration].append(node)

  return diff


def deleteConflictNodes(filepath, result):
  nodes = getSwapOutNodeID(result)[1]
  lines = []
  with open(filepath, "r") as fin:
    lines = fin.readlines()
    lines = [line for line in lines if line.split('\t')[0] not in nodes]
  with open(filepath, "w") as fin:
    fin.writelines(lines)

if __name__ == '__main__':
  # filepath = '/home/frog/maweiliang/tmp/swap_policy.txt'
  # result = './result.log'
  # deleteConflictNodes(filepath, result)
  result = './resnet50_result.txt'
  # filepath = './inception3_160_p100/swapping_decision.log'
  filepath1 = './resnet50_190_p100/resnet50_result_600.txt'
  filepath2 = './resnet50_190_p100/resnet50_result_650.txt'
  # diff, common = compareBatch_size(filepath1, filepath2)
  # diff = compareIteration(nodes)
  # deleteConflictNodes(filepath, result)
  print diff
  print common
            




