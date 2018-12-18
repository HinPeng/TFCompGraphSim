try:
  import Queue as q
except ImportError:
  import queue as q

class SwapInfo():
  def __init__(self, 
               tensor_name,
               allocated_time=0,
               allocated_bytes=0):
    self.tensor_name = tensor_name
    self.allocated_time = allocated_time
    self.deallocate_time = 0
    self.allocated_bytes = allocated_bytes
    self.access_list = [] #(access_index, access_time)
    self.swap_start = -1  # at which index to swap this tensor out
    self.max_access_interval = -1

  def DeallocateTime(self):
    # access_time = [v for _,v in self.access_list]
    # self.deallocate_time = max(access_time)

    # access_list has been sorted when this func is invoked
    self.deallocate_time = self.access_list[-1][1]

  def GetSwappingTime(self, pcie_bw):
    # make this microseconds
    return float(self.allocated_bytes) / pcie_bw

  def GetFirstUseIndexAfterSwap(self):
    return self.access_list[self.swap_start+1][0]
    
  def GetFirstUseTimeAfterSwap(self):
    return self.access_list[self.swap_start+1][1]

  def GetSwapoutRc(self):
    return (len(self.access_list) - self.swap_start - 1)

  def __cmp__(self, other):
    if self.max_access_interval == other.max_access_interval:
      if self.swap_start == other.swap_start:
        return 0
      elif self.swap_start > other.swap_start:
        return 1
      else:
        return -1
    elif self.max_access_interval > other.max_access_interval:
      return -1
    else:
      return 1
    # if self.max_access_interval == other.max_access_interval:
    #   return self.swap_start > other.swap_start

    # return self.max_access_interval > other.max_access_interval

class MemInfo():
  def __init__(self,
               tensor_name,
               time=-1,
               allocated_bytes=0):
    self.tensor_name = tensor_name
    self.time = time
    self.allocated_bytes = allocated_bytes

  def IsDeallocate(self):
    return self.allocated_bytes < 0

  def __cmp__(self, other):
    if self.time == other.time:
      return 0
    elif self.time > other.time:
      return 1
    else:
      return -1

class PeakMemory():
  def __init__(self):
    self.meminfos = q.PriorityQueue()
    # record the live tensors at peak memory usage
    self.peakmem_tensors_collec = []
    # record which tensor is been deallocated so far
    self.curr_deallocate_ = []
    self.mem_metric = 1 << 20
    # pass

  def InitFromSwapInfo(self, swapinfos):
    for swapinfo in swapinfos:
      meminfo_a = MemInfo(swapinfo.tensor_name,
                          time=swapinfo.allocated_time,
                          allocated_bytes=swapinfo.allocated_bytes)
      meminfo_d = MemInfo(swapinfo.tensor_name,
                          time=swapinfo.deallocate_time,
                          allocated_bytes=-swapinfo.allocated_bytes)

      self.meminfos.put(meminfo_a)
      self.meminfos.put(meminfo_d)

    # pass

  def GetPeakMemory(self):
    peak_mem = 0
    total_mem = 0
    while not self.meminfos.empty():
      meminfo = self.meminfos.get()
      total_mem += meminfo.allocated_bytes
      
      if meminfo.IsDeallocate():
        self.curr_deallocate_.append(meminfo.tensor_name)

      if total_mem > peak_mem:
        assert (meminfo.IsDeallocate() == False)
        peak_mem = total_mem
        self.peakmem_tensors_collec.append(meminfo.tensor_name)
        # remove the tensor which has been deallocated as it's not
        # at peak memory usage
        if (len(self.curr_deallocate_) > 0):
          for name in self.curr_deallocate_:
            assert name in self.peakmem_tensors_collec
            self.peakmem_tensors_collec.remove(name)

          del self.curr_deallocate_[:]

      self.meminfos.task_done()

    return (peak_mem/self.mem_metric)
