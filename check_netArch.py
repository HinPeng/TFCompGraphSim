import os
import logger
import logging

def FilterByKeys(name, filters=None):
  if filters == None:
    return True

  l_name = name.lower()
  for f in filters:
    if f in l_name:
      return True
  
  return False


def Check_netArch(data_dir, net, log2file):
  # For vDNN(conv) swapping
  # layerkeys = ["conv"]
  layerkeys = ["conv", "avgpool", "mpool", "relu"]
  # filter the conv layer with the kernel keyword as it's the weights
  filterconvkeys = ["kernel", "bias"]
  # layerkeys = ["conv", "mpool", "apool"]
  filterkeys = ["batchnorm", "Relu", "gradient"]

  forwardkey = dict()
  backwardkey = dict()

  assert (data_dir != None)
  if log2file:
    log_dir = data_dir+"layerArch/"
    if not os.path.exists(log_dir):
      os.mkdir(log_dir)

  # innodes_file=data_dir+"1_innodes.txt"

  for node in net.values():
    # Check if the backpropagation
    backPropa = False
    if "back" in node.node_name.lower():
      backPropa = True

    # if not backpropagation, filter the keywords we didn't need
    if not backPropa:
      if FilterByKeys(node.node_name, filters=filterkeys):
        continue

    # Extract the layerkey, like "conv0, mpool0"
    kn = ""
    for layerkey in layerkeys:
      if layerkey in node.node_name.lower():
        kn = layerkey
        break

    if kn == "":
      continue

    tmp = node.node_name.split('/')
    for ttmp in tmp:
      if kn in ttmp:
        kn = ttmp
        break

    # kn should be like 'conv2'
    if "conv" in kn:
      if FilterByKeys(node.node_name, filterconvkeys):
        continue
      # convfilterflag = False
      # for filterconvkey in filterconvkeys:
      #   if filterconvkey in node.node_name:
      #     convfilterflag = True
      #     break
      # if convfilterflag:
      #   continue

    if backPropa:
      if not backwardkey.__contains__(kn):
        backwardkey[kn] = []
      backwardkey[kn].append(node.node_name)
    else:
      if not forwardkey.__contains__(kn):
        forwardkey[kn] = []
      forwardkey[kn].append(node.node_name)


  if log2file:
    for k,v in forwardkey.items():
      filename = "forward_"+k+".log"
      with open (log_dir+filename, 'w') as fout:
        for vv in v:
          fout.write(vv+'\n')

    for k,v in backwardkey.items():
      filename = "backward_"+k+".log"
      with open (log_dir+filename, 'w') as fout:
        for vv in v:
          fout.write(vv+'\n')

  # Get the tensors which are needed swapped out
  swapped_tensors = dict()
  for k,v in forwardkey.items():
    for vv in v:
      # vv: node name
      node = net[vv]
      for fanin_tensor in node.fanin_tensors:
        # ERROR here
        if FilterByKeys(fanin_tensor.name(), filters=filterconvkeys):
          continue
        if not swapped_tensors.__contains__(vv):
          swapped_tensors[vv] = []
        swapped_tensors[vv].append(fanin_tensor.name())

  swap_info = dict()
  conv_fanout_nodes = dict()
  for k,v in swapped_tensors.items():
    logging.info("%s fanin tensor: %s" % (k, v))
    # Only one input of conv layer is needed swapped out, maybe not a good assumption
    assert (len(v) == 1)
    # for each tensor
    for vv in v:
      index_ = vv.rfind('_')
      node_name = vv[:index_]
      output_id = vv[index_+1:]
      # node_name = vv.split('_')[0]  # swap out tensor's node name
      # output_id = vv.split('_')[1]  # output id of this tensor
      tensor = (node_name, output_id)
      if not swap_info.__contains__(tensor):
        swap_info[tensor] = []
      swap_node = net[node_name]
      for fanout_node in swap_node.fanout_nodes:
        if fanout_node.node_name == k:
          continue
        input_id = -1
        for index, t in enumerate(fanout_node.fanin_tensors):
          if t.name() == vv:
            input_id = index
            break
        if input_id == -1:
          print("Error input_id, node: %s, fanin_tensor: %s" % (fanout_node.node_name, vv))
          raise ValueError

        swap_info[tensor].append((fanout_node.node_name, input_id))

      # Get fanout_nodes of conv node
      conv_node = net[k]
      assert (not conv_fanout_nodes.__contains__((tensor)))
      conv_fanout_nodes[tensor] = []
      for fanout_node in conv_node.fanout_nodes:
        conv_fanout_nodes[tensor].append(fanout_node.node_name)
        logging.info("%s" % fanout_node.node_name)

  # TODO: choose the in_trigger_node for each swapped out tensor
  print("swap tensor num: %d\n" % len(swap_info))
  in_trigger_log = False
  # if os.path.exists(data_dir+"in_trigger_node.log"):
  #   in_trigger_log = True
  
  if in_trigger_log:
    in_trigger_dict = dict()
    in_trigger_fanouts = dict()
    
    with open(data_dir+"in_trigger_node.log") as fin:
      for line in fin:
        tmp = line.split()
        assert(len(tmp) == 4)
        tensor = (tmp[0], tmp[1])
        assert(not in_trigger_dict.__contains__(tensor))
        in_trigger_dict[tensor] = tmp[2]

        # the backward computation node which overlap the swap-in
        node = net[tmp[3]]
        assert (not in_trigger_fanouts.__contains__(tensor))
        in_trigger_fanouts[tensor] = []
        for fanout_node in node.fanout_nodes:
          in_trigger_fanouts[tensor].append(fanout_node.node_name)


  with open(data_dir+"vdnn_swap_info.log", 'w') as fout:
    for k,v in swap_info.items():
      fout.write(k[0]+'\t'+k[1]+'\t'+str(len(v))+'\n')
      for vv in v:
        fout.write(vv[0]+'\t'+str(vv[1])+'\n')
      assert (k in conv_fanout_nodes.keys())
      fout.write(str(len(conv_fanout_nodes[k]))+'\n')
      for conv_fanout in conv_fanout_nodes[k]:
        fout.write(conv_fanout+'\n')

      # ignore in_trigger at first
      if in_trigger_log:
        assert (k in in_trigger_dict.keys())
        fout.write(in_trigger_dict[k]+'\n')
        fout.write(str(len(in_trigger_fanouts[k]))+'\n')
        for in_trigger_fanout in in_trigger_fanouts[k]:
          fout.write(in_trigger_fanout+'\n')





  # return forwardkey, backwardkey