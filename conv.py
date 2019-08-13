import re
import pandas as pd
from node import Node

# metadata_dir = './resnet50_190_p100/'
# metadata_dir = './resnet152_86_p100/'
# metadata_dir = './inception3_160_p100/'
# metadata_dir = './inception4_88_p100/'
metadata_dir = './vgg16_226_p100/'
GpuNodeInfo_filename = 'gpu_0_stream_all_nodetime.txt'

filter_keys = ['gradient']

def InitNodesExecutionTime():
    nodes = dict()
    with open(metadata_dir+GpuNodeInfo_filename, 'r') as fin:
      for line in fin:
        tmp = line.split()
        assert (len(tmp) == 3)
        node_name = tmp[0]
        node = Node(tmp[0], int(tmp[1]), int(tmp[2]), gpu_time=True)
        assert (not nodes.__contains__(node_name))
        nodes[node_name] = node
    return nodes


def IsGradient(node_name):
    node_name = node_name.lower()
    for keys in filter_keys:
        if keys in node_name:
            return True
    return False


def getConvNodes(nodes):
    convs_forward = dict()
    convs_backword = dict()
    i = 0
    while True:
        flag = False
        # convs_str = 'conv%d/conv2d/Conv2D' % i
        # relu_str = '.*/Relu$'
        pool_str = '.*Pool$'
        for node_name in nodes.keys():
            if IsGradient(node_name):
                convs = convs_backword
            else:
                convs = convs_forward
            # if conv_relu_str in node_name:
            if re.match(pool_str, node_name) != None:
                # if "conv" not in node_name:
                #     continue
                if node_name not in convs.keys():
                    flag = True
                    convs[node_name] = list()
                    convs[node_name].append((node_name, (nodes[node_name].start_time, nodes[node_name].end_time)))
                else:
                    convs[node_name].append((node_name, (nodes[node_name].start_time, nodes[node_name].end_time)))
        i = i + 1
        if not flag:
            break
    return convs_forward, convs_backword


# def getConvTime(convs):
#     layer_times = dict()
#     for conv in convs.keys():
#         start_time_1 = -1
#         start_time_2 = -1
#         end_time_1 = -1
#         end_time_2 = -1
#         for _, ct in convs[conv]:
#             if start_time_1 == -1:
#                 start_time_1 = ct[0]
#             if start_time_1 > ct[0]:
#                 start_time_1 = ct[0]
#             if start_time_2 < ct[0]:
#                 start_time_2 = ct[0]
#             if end_time_1 == -1:
#                 end_time_1 = ct[1]
#             if end_time_1 > ct[1]:
#                 end_time_1 = ct[1]
#             if end_time_2 < ct[1]:
#                 end_time_2 = ct[1]
#         layer_times[conv] = [start_time_1, start_time_2, start_time_2-start_time_1, end_time_1, end_time_2, end_time_2-end_time_1, end_time_2-start_time_1]
#     return layer_times

def getConvTime(convs):
    layer_times = dict()
    for conv in convs.keys():
        # print convs[conv],type(convs[conv][1])
        layer_times[conv] = convs[conv][0][1][1]-convs[conv][0][1][0]
    return layer_times

if __name__ == "__main__":
    nodes = InitNodesExecutionTime() 
    convs_forward, convs_backword = getConvNodes(nodes)
    layer_times_forward = getConvTime(convs_forward)
    # layer_times_backward = getConvTime(convs_backword)
    # pd.DataFrame(list(layer_times.items())).to_csv("%s/convExecTime.csv" % metadata_dir, index=False, header=False)
    # pd.DataFrame([[key]+layer_times_forward[key] for key in layer_times_forward.keys()]).to_csv("%s/conv2d_forward.csv" % metadata_dir, index=False, header=False)
    # pd.DataFrame([[key]+layer_times_backward[key] for key in layer_times_backward.keys()]).to_csv("%s/conv_backward.csv" % metadata_dir, index=False, header=False)
    # pd.DataFrame(layer_times_forward.items()).to_csv("%s/conv2d_forward.csv" % metadata_dir, index=False, header=False)
    # pd.DataFrame(layer_times_forward.items()).to_csv("%s/conv_relu_forward.csv" % metadata_dir, index=False, header=False)
    pd.DataFrame(layer_times_forward.items()).sort_values(by=1).to_csv("%s/pool_forward.csv" % (metadata_dir), index=False, header=False)
    