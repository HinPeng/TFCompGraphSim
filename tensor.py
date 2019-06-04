
class Tensor():
  def __init__(self,
               node_name,
               tid=-1,
               requested_bytes=0,
               allocator_name=None,
               allocated_bytes=0):
    self.node_name = node_name
    self.tid = tid  # the port of outputs of this node

    self.allocator_name = allocator_name
    self.requested_bytes = requested_bytes
    self.allocated_bytes = allocated_bytes
    self.allocated_time = 0  # init by node start_time

    # to calculate peak memory usage
    # current reference count according to nodes' fanin_tensors (at nodes' start)
    self.ref_count = -1

    self.gpu_mem_allocated = 0
    self.gpu_mem_requested = 0

    self.metric = 1 << 20

    # Swapping related info
    # indicate whether this tensos is in swapping, when it's been triggered to be swapped in, field will be set to False
    self.swapping = False
    self.swapping_ref_count = 0 # the ref count where start swapping
    self.blocking_nodes = []
    # self.node_seq = []

    self.swapout_time = 0
    self.swapin_time = 0

    # reference count according at the timing of pending_count is zero  (at nodes' end)
    self.ref_count_ = 0

    # info for re-computation
    self.inputs = []    # store tensors which can produce this tensor
    # self.func = self.node_name  # how this tensor be produced by inputs
    # May we do not need to consider the kernel?


  def name(self):
    return self.node_name+':'+str(self.tid)

  def MemAllocated(self):
    if self.allocator_name == "GPU_0_bfc":
      self.gpu_mem_allocated = float(self.allocated_bytes) / self.metric
    else:
      self.gpu_mem_allocated = 0

  def MemRequested(self):
    if self.allocator_name == "GPU_0_bfc":
      self.gpu_mem_requested = float(self.requested_bytes) / self.metric
    else:
      self.gpu_mem_requested = 0
