try:
  import Queue as q
except ImportError:
  import queue as q

import logger
import logging

pcie_bw = 12 * (1 << 10)

class SwapOutTimeInfo():
  def __init__(self):
    self.start_time = 0
    self.end_time = 0

class SwapInTimeInfo():
  def __init__(self):
    self.start_time = 0
    self.end_time = 0

# TODO: update this to Swap&Re-compute info
# not only just SwapInfo
class SwapInfo():
  def __init__(self,
               tensor_name,
               allocated_time=0,
               allocated_bytes=0):
    self.tensor_name = tensor_name
    self.allocated_time = allocated_time
    self.deallocate_time = 0
    self.allocated_bytes = float(allocated_bytes) / (1<<20)
    self.access_list = [] # (access_index, access_time)
    self.swap_start = -1  # at which index to swap this tensor out
    self.max_access_interval = -1   # to be deprecated
    # be used for ordering
    # needed to be re-computed when choosing a swapping candidate    

    # Swap time info
    # static swap time info is fixed for each tensor
    # No need to store this info
    # self.static_swapout_time_info = SwapOutTimeInfo()
    # self.static_swapin_time_info = SwapInTimeInfo()

    # curr swap time info can be affected by other tensor's swap time info
    # due to pci-e interference
    # Updated when deciding to add a tensor to already_queue
    self.curr_swapout_info = SwapOutTimeInfo()
    self.curr_swapin_info = SwapInTimeInfo()
    self.curr_swap_free_time = -1 # seems uesless

    # swap time info when a new candidate being added to already_queue
    self.swapout_info = SwapOutTimeInfo()
    self.swapin_info = SwapInTimeInfo()
    self.swap_free_time = -1

    # time of
    self.swap_time = float(self.allocated_bytes) / pcie_bw * 1000000
    # Time on PCI-e bus, should be fixed as it's pinned memory
    # self.swapout_start_time = 0 # Time when swapping out starts
    # self.swapout_end_time = 0       # Time when swapping out finishes
    # self.swapin_start_time = 0  #
    # self.swapin_end_time = 0    #

  def InitSwapInfo(self):
    # TODO: init swap_time first
    self.curr_swapout_info.start_time = self.access_list[self.swap_start][1]
    self.curr_swapout_info.end_time = self.curr_swapout_info.start_time + self.swap_time
    self.curr_swapin_info.end_time = self.access_list[self.swap_start+1][1]
    self.curr_swapin_info.start_time = self.curr_swapin_info.end_time - self.swap_time
    self.swapout_info.start_time = self.curr_swapout_info.start_time
    self.swapout_info.end_time = self.curr_swapout_info.end_time
    self.swapin_info.start_time = self.curr_swapin_info.start_time
    self.swapin_info.end_time = self.curr_swapin_info.end_time
    self.swap_free_time = self.swapin_info.start_time - self.swapout_info.end_time
    
  def UpdateCurrSwapInfo(self):
    self.curr_swapout_info.start_time = self.swapout_info.start_time
    self.curr_swapout_info.end_time = self.swapout_info.end_time
    self.curr_swapin_info.start_time = self.swapin_info.start_time
    self.curr_swapin_info.end_time = self.swapin_info.end_time
    self.curr_swap_free_time = self.swap_free_time

  def UpdateSwapOutTime(self, time_interval):
    # this operation do not add time_interval to the original value
    # as it will be changed when being added to already_queue as a candidate                 
    self.swapout_info.start_time = self.curr_swapout_info.start_time + time_interval
    self.swapout_info.end_time = self.curr_swapout_info.end_time + time_interval
    # self.swap_free_time = self.swapin_info.start_time - self.swapout_info.end_time
  
  def UpdateSwapInTime(self, time_interval):
    self.swapin_info.end_time = self.curr_swapin_info.end_time + time_interval
    self.swapin_info.start_time = self.curr_swapin_info.start_time + time_interval
    # also update swap_free_time here
    self.swap_free_time = self.swapin_info.start_time - \
                          self.swapout_info.end_time
  # Updating SwapTimeInfo needs current swapping decision
  # def UpdateSwapInfo(self):
    # this time would be delayed by other pinned memory data transfer
    # self.swapout_start_time = self.access_list[self.swap_start][1]
    # #
    # self.swapout_end_time = self.swapout_start_time + self.swap_time
    # self.swapin_end_time = self.swapin_start_time + self.swap_time
    # self.swap_free_time = self.swapin_start_time - self.swapout_end_time


  def DeallocateTime(self):
    # access_time = [v for _,v in self.access_list]
    # self.deallocate_time = max(access_time)

    # access_list has been sorted when this func is invoked
    # this deallocation_time is not accurate as it's been used now
    # need to add this node's execution time
    self.deallocate_time = self.access_list[-1][1]

  # def SetSwappingTime(self, pcie_bw):
  #   # microseconds
  #   self.swap_time = float(self.allocated_bytes) / pcie_bw * 1000000

  def GetFirstUseIndexAfterSwap(self):
    return self.access_list[self.swap_start+1][0]

  def GetFirstUseTimeAfterSwap(self):
    return self.access_list[self.swap_start+1][1]

  def GetSwapoutRc(self):
    return (len(self.access_list) - self.swap_start - 1)

  # When max_access_interval is equal, then the bigger swap_start gets high priority
  # or, the smaller max_access_interval gets high priority
  # TODO: should also consider this tensor size
  # def __cmp__(self, other):
  #   if self.max_access_interval == other.max_access_interval:
  #     if self.swap_start == other.swap_start:
  #       return 0
  #     elif self.swap_start > other.swap_start:
  #       return 1
  #     else:
  #       return -1
  #   elif self.max_access_interval > other.max_access_interval:
  #     return -1
  #   else:
  #     return 1
  #   # if self.max_access_interval == other.max_access_interval:
  #   #   return self.swap_start > other.swap_start

  #   # return self.max_access_interval > other.max_access_interval

  # Priority: first order is swap_free_time   (descending)
  #           second order is swap_start_time (ascending)
  def __cmp__(self, other):
    if self.swap_free_time == other.swap_free_time:
      if self.swapin_info.start_time == \
        other.swapin_info.start_time:
        return 0
      # if self.swapout_start_time == other.swapout_start_time:
      #   return 0
      elif self.swapin_info.start_time > \
          other.swapin_info.start_time:
          return 1
      else:
        return -1
      # elif self.swapout_start_time > other.swapout_start_time:
      #   return 1
      # else:
      #   return -1
    elif self.swap_free_time > other.swap_free_time:
      return -1
    else:
      return 1

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
    
    self.meminfos_dict = dict()

    self.peak_mem = 0

    self.left_peak_time = -1
    self.right_peak_time = -1
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
      # logging.debug("%s: %d, %d" % (swapinfo.tensor_name,
      #                               swapinfo.allocated_time,
      #                               swapinfo.deallocate_time))
      self.meminfos_dict[swapinfo.tensor_name] = (swapinfo.allocated_time, swapinfo.deallocate_time)

    # pass

  def GetPeakMemory(self):
    peak_mem = 0
    total_mem = 0
    tmp_ = []
    while not self.meminfos.empty():
      meminfo = self.meminfos.get()
      total_mem += meminfo.allocated_bytes
      
      # logging.debug("%s: %d" % (meminfo.tensor_name, meminfo.allocated_bytes))
      if meminfo.IsDeallocate():
        self.curr_deallocate_.append(meminfo.tensor_name)
      else:
        tmp_.append(meminfo.tensor_name)
        # logging.debug("%s is in current deallocation" % meminfo.tensor_name)

      if total_mem > peak_mem:
        assert (meminfo.IsDeallocate() == False)
        peak_mem = total_mem
        self.peakmem_tensors_collec += tmp_
        del tmp_[:]
        # logging.debug("%s enter into peakmem collection" % meminfo.tensor_name)
        # remove the tensor which has been deallocated as it's not
        # at peak memory usage
        if (len(self.curr_deallocate_) > 0):
          for name in self.curr_deallocate_:
            # TODO: a bug here when meeting resnet50
            try:
              assert name in self.peakmem_tensors_collec
            except AssertionError:
              logging.error("Error when init peak memory")
              logging.debug("Error name: %s" % name)
              # logging.debug("Current peakmem tensors collection:")
              # for t_name in self.peakmem_tensors_collec:
              #   logging.debug("%s" % t_name)
              logging.debug("Allocation time: %d" % self.meminfos_dict[name][0])
              logging.debug("Deallocation time: %d" % self.meminfos_dict[name][1])
              exit(1)
            self.peakmem_tensors_collec.remove(name)

          del self.curr_deallocate_[:]

      self.meminfos.task_done()

    # the earliest time of peak memory: earliest allocation time in peakmem_tensors_collec
    # the lastest time of peak memory: lastest deallocation time in peakmem_tensors_collec
    logging.debug("Peak Memory: %d, %d" % (total_mem, peak_mem))
    l_times = []
    r_times = []
    for t_name in self.peakmem_tensors_collec:
      # logging.debug("%s: %d, %d" % (t_name, self.meminfos_dict[t_name][0], self.meminfos_dict[t_name][1]))
      assert t_name in self.meminfos_dict.keys()
      l_times.append(self.meminfos_dict[t_name][0])
      r_times.append(self.meminfos_dict[t_name][1])

    self.left_peak_time = max(l_times)
    self.right_peak_time = min(r_times)
    self.peak_mem = peak_mem

    return peak_mem
