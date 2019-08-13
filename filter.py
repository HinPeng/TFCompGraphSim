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

def filterBert(filepath):
  # filterStrs = ['dropout/Floor', 'mul_1', 'begin', 'intermediate/dense/Tanh'
  # 'bert/embeddings/word_embeddings/read', 'intermediate/dense/BiasAdd',
  # 'bert/encoder/layer_2/attention/self/transpose_2']
  # 70
  # filterStrs = ['dropout/Floor', 'mul_1', 'begin',
  # 'bert/embeddings/word_embeddings/read:0']
  # 100 d=5
  # filterStrs = ['bert/embeddings/Slice/begin', 'PolynomialDecay/Cast/x', 'Mul_4/x',
  # 'dropout/Floor', 'bert/encoder/layer_11/attention/self/mul_1', 'clip_by_global_norm/mul',
  # 'bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1', 'bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1',
  # 'bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1', 'bert/embeddings/LayerNorm/batchnorm/Rsqrt',
  # 'bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1', 'bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/Rsqrt',
  # 'bert/encoder/layer_0/output/LayerNorm/batchnorm/Rsqrt']
  
  # 110 d=10
  # filterStrs = ['bert/embeddings/Slice/begin', 'PolynomialDecay/Cast/x', 'bert/embeddings/LayerNorm/batchnorm/Rsqrt',
  # 'Mul_4/x','bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/Rsqrt', 'bert/encoder/layer_0/output/LayerNorm/batchnorm/Rsqrt',
  # 'dropout/Floor', 'bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1', 'bert/encoder/layer_11/attention/self/mul_1',
  # 'bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1', 'bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1',
  # 'clip_by_global_norm/mul']

  # 100 dep=10 new
  # filterStrs = ['attention/output/add:0', 'attention/self/Reshape_3:0',
  # 'attention/self/dropout/mul:0', 'attention/self/Softmax:0', 'Mul_5/y:0',
  # 'bert/embeddings/one_hot', 'bert/embeddings/LayerNorm/batchnorm/Rsqrt',
  # 'bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1', 'bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1',
  # 'bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1' , 'bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/Rsqrt',
  # 'bert/encoder/layer_0/output/LayerNorm/batchnorm/Rsqrt', 'bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/Rsqrt',
  # 'bert/encoder/layer_1/output/LayerNorm/batchnorm/Rsqrt']
  # ,'attention/output/LayerNorm/batchnorm/Rsqrt',
  # 'output/LayerNorm/batchnorm/Rsqrt', 'attention/output/LayerNorm/batchnorm/add_1']

  # 150 d=15 

  # filterStrs = ['bert/embeddings/LayerNorm/batchnorm/Rsqrt', 'PolynomialDecay/Cast/x', 'Mul_4/x',
  # 'dropout/Floor', 'bert/embeddings/Slice/begin', 'bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/Rsqrt',
  # 'output/LayerNorm/batchnorm/Rsqrt', 'bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/Rsqrt',
  # 'bert/encoder/layer_1/output/LayerNorm/batchnorm/Rsqrt', 'bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/Rsqrt',
  # 'bert/encoder/layer_2/output/LayerNorm/batchnorm/Rsqrt', 'bert/encoder/layer_11/attention/self/mul_1', 
  # 'bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1', 'bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1', 
  # 'bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1', ]
  ## 70
  # filterStrs = ['dropout/Floor', 'mul_1', 'begin',
  # 'bert/embeddings/word_embeddings/read:0','PolynomialDecay/Cast/x', 
  # 'bert/encoder/layer_1/attention/self/dropout/mul', 'bert/encoder/layer_1/intermediate/dense/BiasAdd',
  # 'bert/encoder/layer_0/intermediate/dense/BiasAdd', 'bert/encoder/layer_1/intermediate/dense/Tanh',
  # 'bert/encoder/layer_0/intermediate/dense/Tanh']
  # 80
  # filterStrs = ['dropout/Floor', 'mul_1', 'begin',
  # 'bert/embeddings/word_embeddings/read:0', 'bert/encoder/layer_1/intermediate/dense/BiasAdd',
  # 'bert/encoder/layer_1/attention/self/transpose_2', 'bert/encoder/layer_0/intermediate/dense/BiasAdd',
  # 'bert/encoder/layer_1/intermediate/dense/Tanh']
  
  # 90
  # filterStrs = ['/dropout/Floor','begin','bert/encoder/layer_11/attention/self/mul_1', 
  # 'bert/embeddings/word_embeddings/read:0', 'bert/embeddings/LayerNorm/batchnorm/Rsqrt',
  # 'bert/encoder/layer_2/attention/self/transpose_2', 'bert/encoder/layer_4/attention/self/dropout/mul',
  # 'bert/encoder/layer_1/intermediate/dense/Tanh', 'bert/encoder/layer_2/intermediate/dense/BiasAdd',
  # 'bert/encoder/layer_0/intermediate/dense/mul_2', 'bert/encoder/layer_2/intermediate/dense/Tanh']
  ## 90 depth=6 no cpu time input
  # filterStrs = ['bert/encoder/layer_1/attention/self/dropout/Floor', 'bert/embeddings/Slice/begin',
  # 'bert/encoder/layer_2/attention/output/dropout/Floor', 'bert/encoder/layer_0/attention/self/dropout/Floor',
  # 'bert/encoder/layer_1/intermediate/dense/Tanh', 'bert/encoder/layer_3/attention/self/dropout/Floor',
  # 'bert/encoder/layer_2/attention/self/dropout/Floor', 'bert/encoder/layer_11/attention/output/dropout/Floor',
  # 'bert/encoder/layer_2/intermediate/dense/Tanh', 'bert/encoder/layer_11/attention/self/dropout/Floor',
  # 'bert/encoder/layer_11/attention/self/mul_1', 'bert/encoder/layer_0/intermediate/dense/BiasAdd', 
  # 'bert/embeddings/word_embeddings/read:0']

  ## 90 depth=1 cpu time input
  # filterStrs = ['intermediate/dense/BiasAdd', 'bert/encoder/layer_2/attention/output/dropout/Floor', 'bert/encoder/layer_1/attention/self/dropout/Floor',
  # 'bert/embeddings/Slice/begin', 'bert/encoder/layer_0/attention/self/transpose_2', 'bert/encoder/layer_0/attention/self/dropout/Floor',
  # 'bert/encoder/layer_3/attention/self/dropout/Floor', 'bert/encoder/layer_2/intermediate/dense/BiasAdd', 'PolynomialDecay/Cast/x',
  # 'bert/encoder/layer_2/attention/self/transpose_2', 'cls/predictions/transform/dense/BiasAdd', 'bert/encoder/layer_11/intermediate/dense/BiasAdd',
  # 'bert/encoder/layer_11/attention/self/dropout/Floor', 'bert/encoder/layer_11/attention/self/mul_1', 'bert/encoder/layer_10/intermediate/dense/BiasAdd',
  # 'bert/encoder/layer_9/intermediate/dense/BiasAdd', 'bert/encoder/layer_8/intermediate/dense/BiasAdd',
  # 'bert/encoder/layer_7/intermediate/dense/BiasAdd', 'bert/encoder/layer_0/attention/output/add', 
  # 'bert/encoder/layer_6/intermediate/dense/BiasAdd', 'bert/encoder/layer_0/output/add']


  # 100
  # filterStrs = ['bert/embeddings/LayerNorm/batchnorm/Rsqrt','bert/embeddings/Slice/begin',
  # 'bert/encoder/layer_3/intermediate/dense/Tanh', 'bert/encoder/layer_2/attention/self/transpose_2',
  # 'bert/encoder/layer_4/attention/self/dropout/mul', 'bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1',
  # 'bert/encoder/layer_0/attention/self/transpose_2', 'attention/self/dropout/Floor',
  # 'bert/encoder/layer_1/attention/output/dropout/Floor', 'bert/encoder/layer_1/intermediate/dense/Tanh']
  ## 100 depth=6
  # filterStrs = ['bert/embeddings/Slice/begin', 'bert/embeddings/LayerNorm/batchnorm/Rsqrt', 
  # 'bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1', 'bert/encoder/layer_1/attention/output/dropout/Floor',
  # 'bert/encoder/layer_1/attention/self/dropout/Floor', 'bert/encoder/layer_0/attention/self/dropout/Floor', 
  # 'bert/encoder/layer_2/attention/self/dropout/Floor', 'bert/encoder/layer_3/attention/output/dropout/Floor',
  # 'bert/encoder/layer_4/attention/self/dropout/Floor', 'bert/encoder/layer_11/attention/output/dropout/Floor', 
  # 'bert/encoder/layer_3/attention/self/dropout/Floor', 'bert/encoder/layer_3/attention/self/dropout/Floor', 
  # 'bert/encoder/layer_3/attention/self/transpose:0', 'bert/encoder/layer_11/attention/self/dropout/Floor',
  # 'bert/encoder/layer_11/attention/self/mul_1', 'bert/encoder/layer_1/intermediate/dense/Tanh']
  ## 70
  # filterStrs = ['cls/predictions/transform/dense/BiasAdd', 'bert/embeddings/Slice/begin',
  # 'intermediate/dense/BiasAdd', 'attention/output/dropout/Floor',
  # 'attention/self/mul_1', 'bert/embeddings/word_embeddings/read:0']

  filterStrs = ['attention/output/add:0', 'attention/self/Reshape_3:0',
'attention/self/dropout/mul:0', 'attention/self/Softmax:0']
  
  policy = []
  comment = '#'
  with open(filepath, "r") as fout:
    lines = fout.readlines()
    for line in lines:
      flag = False
      if comment not in line:
        for filter_str in filterStrs:
          if filter_str in line:
            policy.append("# "+line)
            flag = True
            break
            # print line
      if not flag:
        policy.append(line)
  with open(filepath, "w") as fin:
    fin.writelines(policy)


def filterDensenet(filepath):
    # depth 0
    filterNode = ['SplitV_40', 'SplitV_52', 'Floor','Floor_4:0','Floor_6:0', 'Floor_3:0', 'Floor_10:0', 'Floor_8:0']
    # filterNode = ['SplitV_40', 'SplitV_52', 'Floor_4', 'Floor_6:0', 'Floor_3:0', 'Floor_7:0', 'Floor_8:0', 'Floor_10:0', 'Floor_2:0', 'Floor_12:0', 'Floor_0:0']
    # filterNode = ['SplitV_40','SplitV_52', 'Floor_4:0', 'Floor_6:0', 'Floor_3:0', '']
    # depth 1
    # filterNode = ['SplitV_40', 'SplitV_52', 'Floor_3:0', 'Floor_4:0', 'Floor_5:0', 'ConcatV2_5:0', 'Floor_6:0','Floor_2:0','Floor_7:0','Floor']    
    # filterNode = ['SplitV_40', 'SplitV_52','Floor_3:0','Floor_4:0','Floor_5:0','ConcatV2_5:0','Floor_6:0','Floor_2:0','Floor_7:0','Floor_8:0'\
    #   , 'ConcatV2_1:0','Floor_10:0','Floor_11:0','ConcatV2_11:0', 'Floor_12:0','Floor_1:0','RealDiv_2:0','ConcatV2_13:0','Floor_13:0'\
    #   , 'Floor_0:0','Floor_14:0', 'Floor']
    # # filterNode = ['SplitV_40','SplitV_52','Conv2D_1:0','Relu_0:0','MaxPool_0:0', 'RealDiv_0:0','Floor_0:0'] # save multi tensor
    # depth 2
    # filterNode = ['SplitV_40', 'SplitV_52', 'Floor','Floor_3:0', 'Floor_4:0']
    # filterNode = ['SplitV_40', 'SplitV_52','Relu_2:0', 'Conv2D_1:0', 'Relu_0:0','Conv2D_0:0', 'RealDiv_0:0', 'Floor_0:0']
    # depth 3
    # filterNode = ['SplitV_40', 'SplitV_52', 'RealDiv_5:0', 'Floor_5:0', 'Floor_4:0','ConcatV2_5:0', 'Relu_13:0', 'Conv2D_13:0',
    # 'Floor_6:0', 'Floor_3:0', 'Floor_7:0','Floor_8:0', 'Floor']
    # filterNode = ['SplitV_40', 'SplitV_52']
    # depth 4
    # filterNode = ['SplitV_40', 'SplitV_52', 'Floor']
    # filterNode = ['SplitV_40', 'SplitV_52']
    # depth 5
    # filterNode = ['SplitV_40', 'SplitV_52', 'Floor']
    # filterNode = ['SplitV_40', 'SplitV_52']
    # depth 6
    # filterNode = ['SplitV_40', 'SplitV_52', 'Floor']
    # filterNode = ['SplitV_40', 'SplitV_52', 'Floor', 'ConcatV2_16:0','Conv2D_27:0', 'Relu_29:0','Floor_13:0','RealDiv_13:0','ConcatV2_5:0','Conv2D_4:0','Relu_5:0']    
    # depth 7
    # filterNode = ['SplitV_40', 'SplitV_52', 'Floor']
    # filterNode = ['SplitV_40', 'SplitV_52','Floor']
    


    comment = '# '
    lines = []
    with open(filepath, 'r') as fout:
      lines = fout.readlines()
      print len(lines)
      tensor_num = int(lines[0].split('\t')[0])
      for i in range(1, tensor_num+1):
        if comment not in lines[i]:
          for node in filterNode:
            if node in lines[i+tensor_num]:
              lines[i+tensor_num] = comment+lines[i+tensor_num]
              lines[i] = comment + lines[i]
              break
      # for i in range(1, tensor_num+1):
      #     if comment not in lines[i]:
      #       for tensor in filterTensor:
      #         if tensor in lines[i]:
      #           lines[i] = comment + lines[i]
      #           lines[i+tensor_num] = comment+lines[i+tensor_num]
      #           break
    with open(filepath, 'w') as fin:
      for line in lines:
        fin.write(line)
        # print line
      # fin.writelines(lines) 


if __name__ == '__main__':
  # filepath = '/home/frog/maweiliang/tmp/swap_policy.txt'
  # result = './result.log'
  # deleteConflictNodes(filepath, result)
  # result = './resnet50_result.txt'
  # filepath = './inception3_160_p100/swapping_decision.log'
  # filepath1 = './resnet50_190_p100/resnet50_result_600.txt'
  # filepath2 = './resnet50_190_p100/resnet50_result_650.txt'
  # diff, common = compareBatch_size(filepath1, filepath2)
  # diff = compareIteration(nodes)
  # deleteConflictNodes(filepath, result)
  # print diff
  # print common
  # recompute_policy = "./bert_66_p100/recompute.log"
  recompute_policy = "./densenet_64_eager/recompute.log"
  filterDensenet(recompute_policy)
            




