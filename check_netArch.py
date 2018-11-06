import os

def Check_netArch(data_dir, net, log2file):
  layerkeys = ["conv"]
  # filter the conv layer with the kernel keyword as it's the weights
  filterconvkeys = ["kernel", "bias", "Bias"]
  # layerkeys = ["conv", "mpool", "apool"]
  filterkeys = ["batchnorm", "Relu", "gradient", "Gradient"]

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
    if "Back" in node.node_name:
      backPropa = True

    # if not backpropagation, filter the keywords we didn't need
    passflag = False
    if not backPropa:
      for filterkey in filterkeys:
        if filterkey in node.node_name:
          passflag = True
          break
    if passflag:
      continue

    # Extract the layerkey, like "conv0, mpool0"
    kn = ""
    for layerkey in layerkeys:
      if layerkey in node.node_name:
        kn = layerkey
        break

    if kn == "":
      continue

    tmp = node.node_name.split('/')
    for ttmp in tmp:
      if kn in ttmp:
        kn = ttmp
        break
    
    if "conv" in kn:
      convfilterflag = False
      for filterconvkey in filterconvkeys:
        if filterconvkey in node.node_name:
          convfilterflag = True
          break
      if convfilterflag:
        continue

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
      node = net[vv]      
      for fanin_tensor in node.fanin_tensors:
        if filterconvkey in fanin_tensor.name():
          continue
        if not swapped_tensors.__contains__(vv):
          swapped_tensors[vv] = []
        swapped_tensors[vv].append(fanin_tensor.name())

  swap_info = dict()
  for k,v in swapped_tensors.items():
    # for each tensor
    for vv in v:
      index_ = vv.rfind('_')
      node_name = vv[:index_]
      output_id = vv[index_+1:]
      # node_name = vv.split('_')[0]  # swap out tensor's node name
      # output_id = vv.split('_')[1]  # output id of this tensor
      if not swap_info.__contains__((node_name, output_id)):
        swap_info[(node_name, output_id)] = []
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
          print("Error input_id, node: %d, fanin_tensor: %s" % (fanout_node.node_name, vv))
          raise ValueError

        swap_info[(node_name, output_id)].append((fanout_node.node_name, input_id))

  # TODO: choose the in_trigger_node for each swapped out tensor

  # with open("swap_info.log", 'w') as fout:
  #   for k,v in swap_info.items():
  #     fout.write(k[0]+'\t'+k[1]+'\t'+str(len(v))+'\n')
  #     for vv in v:
  #       fout.write(vv[0]+'\t'+str(vv[1])+'\n')

        
      


  # return forwardkey, backwardkey