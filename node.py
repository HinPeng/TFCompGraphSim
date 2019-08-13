from tensor import Tensor

class Node():
  def __init__(self,
               node_name,
               start_time=0,
               end_time=0,
               gpu_time=False):
    self.node_name = node_name
    self.depth = -1
    # Can be delayed due to other operation. such as swap in
    self.logic_time = 0
    self.tmp_time = []

    # Init by the run_metadata
    # start_time = node.all_start_micros
    # end_time = start_time + node.all_end_rel_micros
    self.start_time = start_time
    self.end_time = end_time

    self.pending_count = -1
    self.fanout_nodes = []  # store the fanout nodes (Can be multiple same node)
    self.fanin_tensors = [] # store fanin tensors to this node

    self.ok_fanin_tensors = []

    self.outputs_num = 0
    self.outputs = []       # store the output tensors
    self.no_use_outputs = []# store some no use output tensors # TODO: be deprecated

    # for memory size transfer
    self.metric = 1 << 20

    self.gpu_allocator_name = "GPU_0_bfc"

    self.gpu_mem_allocated = 0
    self.gpu_mem_requested = 0

    self.access_id = 0

    self.gpu_time = gpu_time # mark the nodetime init by gpu_0 or gpu_0_stream_all

  def __cmp__(self, other):
    if self.pending_count == other.pending_count:
      return self.start_time > other.start_time

    return self.pending_count > other.pending_count

  def GetExecTime(self):
    return (self.end_time - self.start_time)

  def InitTensorRPInfo(self):
    for tensor in self.outputs:
      for fanin_tensor in self.fanin_tensors:
        tensor.inputs.append(fanin_tensor)

  def GetNodeTotalSize(self):
    # Inputs size
    total_node_size = 0
    for t in self.fanin_tensors:
      total_node_size += t.gpu_mem_allocated
    
    # Outputs size
    for t in self.outputs:
      total_node_size += t.gpu_mem_allocated

    # No use tensors' size
    for t in self.no_use_outputs:
      t.MemAllocated()
      total_node_size += t.gpu_mem_allocated
    
    return total_node_size

  def GPUMemAllocated(self):
    if (len(self.outputs) == 0) and (len(self.no_use_outputs) == 0):
      self.gpu_mem_allocated = 0
    else:
      sum_allocated_bytes = 0
      for t in self.outputs:
        if t.allocator_name == self.gpu_allocator_name:
          sum_allocated_bytes += t.allocated_bytes

      for t in self.no_use_outputs:
        if t.allocator_name == self.gpu_allocator_name:
          sum_allocated_bytes += t.allocated_bytes

      self.gpu_mem_allocated = float(sum_allocated_bytes) / self.metric

  def GPUMemRequested(self):
    if (len(self.outputs) == 0) and (len(self.no_use_outputs) == 0):
      self.gpu_mem_requested = 0
    else:
      sum_requested_bytes = 0
      for t in self.outputs:
        if t.allocator_name == self.gpu_allocator_name:
          sum_requested_bytes += t.requested_bytes

      for t in self.no_use_outputs:
        if t.allocator_name == self.gpu_allocator_name:
          sum_requested_bytes += t.requested_bytes

      self.gpu_mem_requested = float(sum_requested_bytes) / self.metric

  def GetNOkfanintensors(self):
    if len(self.fanin_tensors) == 0:
      return None
    # assert len(self.fanin_tensors) != 0
    # assert len(self.ok_fanin_tensors) !
    Nok_tensors = []
    for t in self.fanin_tensors:
      if t in self.ok_fanin_tensors:
        pass
      else:
        Nok_tensors.append(t.name())

    if len(Nok_tensors) == 0:
      return None
    else:
      return Nok_tensors
  
  def updateNodeDepth(self, nodes):
    if self.depth != -1:
      return
    if len(self.fanin_tensors) == 0:
      self.depth = 0
    for in_tensor in self.fanin_tensors:
      in_node_name = in_tensor.node_name  
      in_node = nodes[in_node_name]
      if in_node.depth == -1:
        in_node.updateNodeDepth(nodes)
      if self.depth < in_node.depth+1:
        self.depth = in_node.depth+1
    
    
  def getInputSlot(self, tensor_name):
    for i in range(len(self.fanin_tensors)):
      if tensor_name == self.fanin_tensors[i].name():
        return i
    return -1

# class Port():
#   def __init__(self,
#                node=None,
#                tid=0,
#                tensor=None):
#     self.node = node
#     self.tid = tid
#     self.tensor = tensor
