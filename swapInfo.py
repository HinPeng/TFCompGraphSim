class SwapInfo():
  def __init__(self, tensor_name):
    self.tensor_name = tensor_name
    self.access_list = []
    self.swap_start = -1
    self.max_access_interval = -1

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