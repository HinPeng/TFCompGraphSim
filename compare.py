

def getInnodes(filepath):
    nodes = dict()
    with open(filepath, 'r') as fout:
        lines = fout.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            line = line.split('\t')
            assert len(line) == 3 and line[0]=='SrcNode'
            outslot = i+int(line[2])
            nodename = line[1]
            nodes[nodename] = list()
            while True:
                i += 1
                if i <= outslot:
                    line = lines[i].split('\t')
                    assert len(line) == 3 and line[0]=='InputNode'
                    tensor_name = line[1]+":"+line[2]
                    nodes[nodename].append(tensor_name)
                else:
                    break
    return nodes

def compareNodesInput(nodes1, nodes2):
    diff_nodes = []
    for node in nodes1.keys():
        if node in nodes2.keys():
            for input_ in nodes1[node]:
                if input_ not in nodes2[node]:
                    diff_nodes.append([node, nodes1[node], nodes2[node]])
                    print node
                    print nodes1[node]
                    print nodes2[node]
                    break

    return diff_nodes
if __name__ == "__main__":
  filepath1 = './resnet50_64_eager/1_innodes_64_new.txt'
  filepath2 = './resnet50_64_eager/backup_64/1_innodes.txt'
  filepath3 = './resnet50_64_eager/backup_120/1_innodes.txt'

  nodes1 = getInnodes(filepath1)
  nodes2 = getInnodes(filepath2)
  nodes3 = getInnodes(filepath3)
  print "64_new VS 64"
  diff_nodes = compareNodesInput(nodes1, nodes2)
  print "64 VS 64_new"
  diff_nodes = compareNodesInput(nodes2, nodes1)
  print "64_new VS 120"
  diff_nodes = compareNodesInput(nodes1, nodes3)
  print "120 VS 64_new"
  diff_nodes = compareNodesInput(nodes3, nodes1)
#   print "64 VS 32"
#   diff_nodes = compareNodesInput(nodes2, nodes3)
#   print "32 VS 64"
#   diff_nodes = compareNodesInput(nodes3, nodes2)
    # filepath1 = "./spinn_1024_eager/1_innodes.txt"
    # filepath2 = "./spinn_1024_eager/1_innodes_ite_535.txt"
    # nodes1 = getInnodes(filepath1)
    # nodes2 = getInnodes(filepath2)
    # print "? VS 535"
    # diff_nodes = compareNodesInput(nodes1, nodes2)
    # print "535 VS ?"
    # diff_nodes = compareNodesInput(nodes2, nodes1)