class SwapInfo():
  def __init__(self, tensor_name, allocated_time=0):
    self.tensor_name = tensor_name
    self.allocated_time = allocated_time
    self.deallocate_time = 0
    self.access_list = [] #(access_index, access_time)
    self.swap_start = -1
    self.max_access_interval = -1

  def DeallocateTime(self):
    # access_time = [v for _,v in self.access_list]
    # self.deallocate_time = max(access_time)

    # access_list has been sorted when this func is invoked
    self.deallocate_time = self.access_list[-1][1]

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