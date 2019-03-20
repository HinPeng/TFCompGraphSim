import logger
import logging

# import collections

# TODO: not use recomp_depth_limit to constrain the recomp
# maybe use the max access time is this subrecomp more reasonable
recomp_depth_limit = 0

recomp_ratio = 0.8

# class Chain():
#   def __init__(self):
#     self.head = None  # recomp
#     self.tail = None  # recomp
#     self.member = []  # sub_recomp

#   def add(self, subrp):
#     self.member.append(subrp)

#   def addchain(self, other):
#     for subrp in other.member:
#       self.add(subrp)

#   def length(self):
#     return len(self.member)

#   def name(self):
#     return self.head.name()

#   def IsPrev(self, recomp):
#     pass


# For two use
# 1. recomps in sub_recomp will only recompute root_, the other can be recomputed by other recomp is this sub_recomp
# 2. recomps in sub_recomp will all be recomputed, so we need to set right inputs for each recomp (Current)
class SubReComp():
  def __init__(self, recomp):
    self.root_ = recomp
    # recomputation targets
    self.coll = dict()
    self.coll[recomp.name()] = recomp
    self.max_rank_var = 0
    # store current inputs which are necessary to recompute
    self.src_inputs = []
    for input_ in recomp.srcs:
      self.src_inputs.append(input_[1])
    # for input_ in recomp.srcs:
    #   # we don't need to store the index of inputs
    #   # as the input in this sub_recomp won't appear in here
    #   # but distiguish var, root
    #   # filter the same input for different rp
    #   if input_ in recomp.srcs:
    #     continue
    #   self.src_inputs.append(input_)

    # the earliest time to access this sub_recomp's tensor
    # TODO
    self.access_info = None
    # the time to recompute this sub recomputation
    self.recomp_time = 0

    # total bytes we can save: only tensors in coll
    self.saving_bytes = 0
    # Total bytes in MB, cover the inputs bytes
    self.total_bytes = 0
    self.metric = -1

    # recomputation in_trigger info
    self.recomp_ratio = 0.5

    # tuple: in_trigger_name, swapin_rc, swapin_total_rc
    self.in_trigger = None

    # name: (swapout_rc, swapout_total_rc)
    self.out_triggers = dict()

  def name(self):
    return self.root_.name()

  # recomp is prev of this sub_recomp
  def IsInSrc(self, recomp):
    if recomp.name() in self.src_inputs:
      return True
    return False

  # recomp is succ of this sub_recomp
  def IsSrc(self, recomp):
    # srcs = []
    # name_coll = [rp.name() for rp in self.coll]
    for input_ in recomp.srcs:
      if input_[1] in self.coll.keys():
        # srcs.append(rp.name())
        return True
    return False

    # if len(srcs) != 0:
    #   return srcs
    # else:
    #   return None

  def MergeSucc(self, recomp):
    self.coll[recomp.name()] = recomp
    rv = recomp.rank - self.root_.rank
    if rv > self.max_rank_var:
      self.max_rank_var = rv
    for src in recomp.srcs:
      if src[1] in self.coll.keys():
        recomp.inputs.remove(src[1])
        recomp.inputs += self.coll[src[1]].inputs        
      else:
        # recomp.inputs.append(src[1])
        if src[1] not in self.src_inputs:
          self.src_inputs.append(src[1])

  def MergePrev(self, recomp):
    rvs = [abs(rp.rank-recomp.rank) for rp in self.coll.values()]
    self.max_rank_var = max(rvs)
    self.coll[recomp.name()] = recomp
    self.src_inputs.remove(recomp.name())
    self.src_inputs += recomp.inputs
    if recomp.rank < self.root_.rank:
      self.root_ = recomp
    # recomp.inputs = [src[1] for src in recomp.srcs]
    for rp in self.coll.values():
      if recomp.name() in rp.inputs:
        rp.inputs.remove(recomp.name())
        rp.inputs += recomp.inputs

  def TryMerge(self, recomp):
    rp_srcs = self.IsSrc(recomp)
    if rp_srcs != None:
      if self.CheckRank(self.root_.rank-recomp.rank):
        # can be merged in
        # need to change this recomp's inputs
        self.coll[recomp.name()] = recomp
        for src in recomp.srcs:
          if src[1] in rp_srcs:
            recomp.inputs += self.coll[src[1]].inputs
          else:
            recomp.inputs.append(src[1])
            # check this input if in this sub_recomp's inputs
            if src[1] not in self.src_inputs:
              self.src_inputs.append(src[1])
        return True
      else:
        return False

    # if we need to check recomp's inputs
    elif self.IsInSrc(recomp):
      pass




  def SetInTrigger(self, tac, graphsim, r_peak_time):
    # choose in_trigger according to tensor accesses
    # set a ratio not total recomp time as multistream comp of GPU
    min_acc_index = self.root_.access_info[0]
    min_acc_time = self.root_.access_info[1]
    
    for rp in self.coll:
      if rp.access_info[1] < min_acc_time:
        min_acc_time = rp.access_info[1]
        min_acc_index = rp.access_info[0]

    self.access_info = (min_acc_index, min_acc_time)

    in_trigger_index = min_acc_index
    # logging.debug("Initial in trigger index: %d" % in_trigger_index)
    while True:
      in_trigger_index -= 1
      t_name = graphsim.tf_tensor_access[in_trigger_index][1]
      if t_name in tac.keys():
        if graphsim.IsWeights(t_name):
          pass
        elif graphsim.IsSize(t_name):
          pass
        else:
          # ignore the tensor which is possible to be recomp or swapping
          continue
      # logging.debug("Current in trigger index: %d, time: %d" % (in_trigger_index,
      #                               graphsim.tf_tensor_access[in_trigger_index][0]))
      if graphsim.tf_tensor_access[in_trigger_index][0] < r_peak_time:
        # logging.debug("Not enough time for %s recomputation in" % self.name())
        # logging.debug("Recomputation time: %d" % self.recomp_time)
        # logging.debug("Access time: %d, r_peak_time: %d" % (self.access_info[1], r_peak_time))
        # exit(1)
        return False

      d_time = min_acc_time - graphsim.tf_tensor_access[in_trigger_index][0]
      if d_time > self.recomp_time * self.recomp_ratio:
        in_trigger_name = graphsim.tf_tensor_access[in_trigger_index][1]
        break

    # Only one in_trigger for a sub_recomputation
    # but out trigger is for each rp in self.coll
    for rp in self.coll:
      swapinfo = tac[rp.name()]
      swapout_rc = swapinfo.GetSwapoutRc()
      swapout_total_rc = len(swapinfo.access_list)
      assert rp.name() not in self.out_triggers.keys()
      self.out_triggers[rp.name()] = (swapout_total_rc, swapout_total_rc-swapout_rc)
      # record recomp out trigger

    swapin_rc = 0
    swapin_total_rc = 1
    if in_trigger_name in tac.keys():
      access_indicies = [v for v,_ in tac[in_trigger_name].access_list]
      assert in_trigger_index in access_indicies
      swapin_rc = len(access_indicies) - access_indicies.index(in_trigger_index) - 1
      swapin_total_rc = len(access_indicies)
    elif in_trigger_name in graphsim.ngpu_tensor_access.keys():
      access_indicies = graphsim.ngpu_tensor_access[in_trigger_name]
      assert in_trigger_index in access_indicies
      swapin_rc = len(access_indicies) - access_indicies.index(in_trigger_index) - 1
      swapin_total_rc = len(access_indicies)
    else:
      pass

    self.in_trigger = (in_trigger_name, swapin_total_rc, swapin_total_rc-swapin_rc)

    return True
    

      

    # pass

  # Src inputs can be different if this src is also chosen to be recomputed
  # this is the initial src inputs
  def SetSrcs(self):
    for rp in self.coll:
      for src in rp.srcs:
        if src in self.src_inputs:
          continue
        self.src_inputs.append(src)

  def SetTotalBytes(self, tensors):
    total_bytes = 0
    assert self.saving_bytes == 0
    for rp in self.coll:
      self.saving_bytes += rp.alloc_bytes
      
    total_bytes += self.saving_bytes
    for src in self.src_inputs:
      assert src[1] in tensors.keys()
      total_bytes += tensors[src[1]].gpu_mem_allocated

    self.total_bytes = total_bytes

  def SetMetric(self):
    if self.saving_bytes == 0:
      self.metric = -1
    else:
      assert self.recomp_time != 0
      self.metric = float(self.saving_bytes)/self.recomp_time

  def GetRecompTime(self):
    # from the coll to srcs?
    # 1. consider the same computation produce multiple outputs in coll
    # filter the same recomputation (no repeated computation)
    # Or should we put the tensor with same recomputation into same subrecomp?
    # same inputs
    curr_queue = [self.root_]
    left_queue = [rp for rp in self.coll]
    total_eva_time = self.root_.eva_time
    while len(curr_queue) > 0:
      if len(left_queue) == 0:
        break
      curr_rp = curr_queue.pop()
      try:
        assert curr_rp in left_queue
      except AssertionError:
        logging.error("%s" % curr_rp.name())
        exit(1)
      left_queue.remove(curr_rp)
      for sc in curr_rp.succ:
        if sc not in self.coll:
          continue
        else:
          total_eva_time += sc.eva_time
          curr_queue.append(sc)

    # try:
    #   assert len(left_queue) == 0
    # except AssertionError:
    #   logging.error("Error recomp time")
    #   exit(1)
    self.recomp_time = total_eva_time

  def CheckRank(self, rank):
    if rank <= recomp_depth_limit:
      return True
    else:
      return False

  def AddRP(self, rp):
    if rp.rank < self.root_.rank:
      # pass
      if self.IsPPrev(rp, self.root_):
        # this would be final rv if not exceeding
        rv = self.root_.rank - rp.rank + self.max_rank_var
        if self.CheckRank(rv):
          self.max_rank_var = rv
          # logging.debug("Add %s as a root of %s subrecomp" % (rp.name(), self.root_.name()))
          self.root_ = rp
          self.coll[rp.name()] =  rp
          return True
        else:
          # logging.debug("Add %s failed due to rank: %d" % (rp.name(), rv))
          return False
      elif self.IsPPrev(self.root_, rp):
        logging.error("%s(%d) is Prev of %s(%d)" % (self.root_.name(),
                                                    self.root_.rank,
                                                    rp.name(),
                                                    rp.rank))
        exit(1)
      else:
        # rp and root
        # logging.debug("Meet a unimplemented situation!")
        # logging.debug("When comparing %s and %s" % (rp.name(), self.root_.name()))
        # exit(1)
        return False
    elif rp.rank == self.root_.rank:
      # logging.debug("Meet same rank")
      # logging.debug("When comparing %s and %s" % (rp.name(), self.root_.name()))
      # exit(1)
      # pass
      # self.coll.append(rp)
      # logging.debug("Ignore a equal rank of %s" % rp.name())
      return False
    else:
      if self.IsPPrev(self.root_, rp):
        rv = rp.rank - self.root_.rank
        if self.CheckRank(rv):
          if rv > self.max_rank_var:
            self.max_rank_var = rv
          self.coll[rp.name()] = rp
          # logging.debug("Add %s as a succ of %s subrecomp" % (rp.name(), self.root_.name()))
          return True
        else:
          # logging.debug("Add %s failed due to rank: %d" % (rp.name(), rv))
          return False
      elif self.IsPPrev(rp, self.root_):
        logging.error("%s(%d) is Prev of %s(%d)" % (rp.name(),
                                                    rp.rank,
                                                    self.root_.name(),
                                                    self.root_.rank))
        exit(1)
      else:
        # logging.debug("Meet a unimplemented situation!")
        # logging.debug("When comparing %s and %s" % (rp.name(), self.root_.name()))
        # exit(1)
        return False


  # def AddSucc(self, rp):
  #   # judge the rank_variation
  #   rv = abs(rp.rank - self.root_.rank)
  #   if rv > recomp_depth_limit:
  #     logging.debug("Exceed the max recomputation depth!")
  #     logging.debug("Root: %s, Succ: %s, rank variation: %d" % (
  #                   self.root_.name(), rp.name(), rv))
  #     return False
  #   # this won't affect the root
  #   self.coll.append(rp)
  #   # update max rank variation
  #   if rv > self.max_rank_var:
  #     self.max_rank_var = rv
  #   return True

  # def AddPrev(self, rp):
  #   # judge rp and self.root_
  #   # can we simply judge the rank?
  #   # maybe work for linear neural network
  #   if self.IsPPrev(rp, self.root_):
  #     # judge weight of recomp
  #     try:
  #       assert rp.rank < self.root_.rank
  #     except AssertionError:
  #       logging.error("%s(%d) is prev of %s(%d)" % (rp.name(),
  #                                                   rp.rank,
  #                                                   self.root_.name(),
  #                                                   self.root_.rank))
  #       exit(1)
  #     rv = rp.rank-self.root_.rank + self.max_rank_var
  #     if rv > recomp_depth_limit:
  #       logging.debug("Exceed the max recomputation depth!")
  #       logging.debug("Prev: %s, Root: %s, rank variation: %d" % (
  #                     self.root_.name(), rp.name(), rv))
  #       return False
  #     if rv > self.max_rank_var:
  #       self.max_rank_var = rv
  #     self.coll.append(rp)
  #     self.root_ = rp
  #     return True
  #   elif self.IsPPrev(self.root_, rp):
  #     try:
  #       assert rp.rank > self.root_.rank
  #     except AssertionError:
  #       logging.error("%s(%d) is prev of %s(%d)" % (self.root_.name(),
  #                                                   self.root_.rank,
  #                                                   rp.name(),
  #                                                   rp.rank))
  #       exit(1)
  #     rv = rp.rank-self.root_.rank
  #     if rv > recomp_depth_limit:
  #       logging.debug("Exceed the max recomputation depth!")
  #       logging.debug("Prev: %s, Root: %s, rank variation: %d" % (
  #                     self.root_.name(), rp.name(), rv))
  #       return False
  #     if rv > self.max_rank_var:
  #       self.max_rank_var = rv
  #     return True
  #   else:
  #     pass
    # if rp and root have no connection
    # we should consider the weight of this subrecomp
    # not just simply add it to current collection


  def IsPPrev(self, rp_src, rp):
    # rp_src, rp: recomp
    # search prev of rp recurrsively
    # if meet rp_src
    rp_queue = [p for p in rp.prev]

    while len(rp_queue) > 0:
      rp_ = rp_queue.pop()
      if rp_ == rp_src:
        return True
      else:
        rp_queue += rp_.prev

    return False






class ReCompColl():
  def __init__(self):
    # root swapinfo where other tensors can be computed from
    self.root_ = None
    self.collection = dict() # ReComp collection
    # store sub re-computation info
    self.sub_rp = []
    # pass

  def Init(self, rp_coll):
    pass

  def GetRoot(self):
    return self.root_

  def IsRoot(self, recomp):
    return recomp == self.GetRoot()

  def SetRoot(self, new_root):
    self.root_ = new_root


  def IsInColl(self, tensor):
    if tensor in self.collection:
      return True

  # No tensor with the same input srcs
  def RepeatedRP(self):
    rp_list = self.collection.values()
    length = len(rp_list)
    # for rp in rp_list:
    for i in range(length):
      t_rp = rp_list[i]
      if t_rp.same_src_flag == 1 and t_rp.same_src_root != None:
        continue
      for j in range(i+1, length):
        curr_rp = rp_list[j]
        if t_rp.IsSameSrcs(curr_rp):
          t_rp.SetSameSrcs(curr_rp)

    # for debug log
    for rp in self.collection.values():
      if rp.same_src_flag == 0:
        continue
      if rp == rp.GetSrcRoot():
        logging.debug("Same input srcs root: %s, total: %d" %
                            rp.name(),
                            len(rp.same_src_root))
        for same_src_rp in rp.same_src_rp:
          logging.debug("%s" % same_src_rp.name())




  def InitRPConnection(self):
    curr_queue = []
    curr_queue.append(self.root_)
    left_queue = [i for i in self.collection.values()]
    left_queue.remove(self.root_)
    # logging.debug("Start from %s" % self.root_.name())
    self.root_.rank = 0
    # traverse from root_ to set each rp's rank
    while len(curr_queue) > 0:
      t_recomp = curr_queue.pop()
      for recomp in left_queue:
        assert recomp != t_recomp

        # check if t_recomp in recomp's inputs
        if recomp.IsInSrcs(t_recomp):
          if t_recomp.IsInSrcs(recomp):
            logging.error("Meet a loop, %s and %s" % (recomp.name(), t_recomp.name()))
            exit(1)
          # logging.debug("P: %s, S: %s" % (recomp.name(), t_recomp.name()))
          recomp.AddPrev(t_recomp)
          t_recomp.AddSucc(recomp)
          if recomp.IsUnsetRank():
            recomp.rank = t_recomp.rank+1
          else:
            if recomp.rank < t_recomp.rank+1:
              recomp.rank = t_recomp.rank+1
          curr_queue.append(recomp)
          left_queue.remove(recomp)

    # for debug log
    # recomps_ = sorted(self.collection.values(), key=lambda x: x.rank)
    # for recomp in recomps_:
    #   logging.debug("%s, rank: %d" % (recomp.name(), recomp.rank))
      # logging.debug("prev: %d, succ: %d" % (len(recomp.prev), len(recomp.succ)))

   # def InitRank(self, rp):
  #   for input_t in rp.tensor.inputs:
  #     if input_t == self.root_.tensor:

  # def InitRank(self,
  #              root):
  #   self.root_.rank = 0
  #   curr_queue = [k for k in self.collection.keys()]
  #   active_queue = []
  #   active_queue.append(self.root_)
  #   while True:
  #     if len(curr_queue) == 0:
  #       break

  #     for rp in active_queue:

    # for rp in self.collection:
    #   if rp.prev == root:
    #     if rp.rank != -1:
    #       if rp.rank < root.rank+1:
    #         pass
    #       rp.rank = root.rank+1
    #     return self.InitRank(rp)


class ReComp():
  def __init__(self,
               tensor,
               access_info=None):
    self.tensor = tensor
    self.rank = -1  # rank in curr collection

    # the time which needs this tensor in memory
    self.access_info = access_info

    self.prev = []
    self.succ = []
    # if prev != None:
    #   self.prev.append(prev)
    # if succ != None:
    #   self.succ.append(succ)

    # closest inputs to recompute this tensor, num:tensor_name
    # 0: Candidate input: tensor in candidate
    # 1: OutCandidate input: more than once occurrence, but not candidate
    # 2: Var input: variable tensor
    # 3: Root input: which has no inputs
    self.srcs = []

    # the inputs can be changed due to other recomps been chosen
    self.inputs = []

    # 0: not set yet
    # 1. been set already
    # if 1 && same_src_root not None, no need to traverse
    # this rp again
    self.same_src_flag = 0
    self.same_src_rp = []
    self.same_src_root = None

    self.alloc_bytes = self.tensor.gpu_mem_allocated
    self.recomp_bytes = 0   # which needed in this recomputation
    # recompute evaluation, generated by access interval
    # TODO: HACK relu recomp time as some relu access time interval is very big
    self.eva_time = -1
    self.metric = 0

    self.sub_rp = None

    self.out_trigger = None
    self.in_trigger = None


  def name(self):
    return self.tensor.name()

  def IsPrev(self, rp):
    if rp in self.prev:
      return True
    return False

  def IsSucc(self, rp):
    if rp in self.succ:
      return True
    return False

  def IsSameSrcs(self, recomp):
    flag = True
    for input_ in self.srcs:
      if input_[1] in recomp.srcs:
        continue
      else:
        flag = False
        break
    
    return flag

  def SetTrigger(self, tac, graphsim, r_peak_time, on_demand=False):
    if on_demand:
      self.in_trigger = ("fxxxxxxxxxxxxk", 0, 0)
      return True
      
    acc_index = self.access_info[0]
    acc_time = self.access_info[1]

    in_trigger_index = acc_index
    while True:
      in_trigger_index -= 1
      t_name = graphsim.tf_tensor_access[in_trigger_index][1]
      if graphsim.tf_tensor_access[in_trigger_index][0] < r_peak_time:
        return False

      d_time = acc_time - graphsim.tf_tensor_access[in_trigger_index][0]
      if d_time > self.eva_time * recomp_ratio:
        in_trigger_name = graphsim.tf_tensor_access[in_trigger_index][1]
        break

    swapinfo = tac[self.name()]
    swapout_rc = swapinfo.GetSwapoutRc()
    swapout_total_rc = len(swapinfo.access_list)
    self.out_trigger = (swapout_total_rc, swapout_total_rc-swapout_rc)

    swapin_rc = 0
    swapin_total_rc = 1
    if in_trigger_name in tac.keys():
      access_indicies =[v for v, _ in tac[in_trigger_name].access_list]
      assert in_trigger_index in access_indicies
      swapin_rc = len(access_indicies) - access_indicies.index(in_trigger_index) - 1
      swapin_total_rc = len(access_indicies)
    elif in_trigger_name in graphsim.ngpu_tensor_access.keys():
      access_indicies = graphsim.ngpu_tensor_access[in_trigger_name]
      assert in_trigger_index in access_indicies
      swapin_rc = len(access_indicies) - access_indicies.index(in_trigger_index) - 1
      swapin_total_rc = len(access_indicies)
    else:
      pass

    self.in_trigger = (in_trigger_name, swapin_total_rc, swapin_total_rc-swapin_rc)

    return True

    

  def GetSrcRoot(self):
    if self.same_src_flag == 0:
      logging.warning("%s: Not set same src yet!" % self.name())
      return None
    else:
      if self.same_src_root == None:
        # self will be the root
        assert len(self.same_src_rp) > 0
        return self
      else:
        return self.same_src_root

  # Get total bytes to recompute this tensor
  def SetRecompBytes(self, tensors):
    assert self.recomp_bytes == 0
    for src in self.srcs:
      try:
        assert src[1] in tensors.keys()
      except AssertionError:
        logging.debug("%s" % src[1])
        continue
      self.recomp_bytes += tensors[src[1]].gpu_mem_allocated

    self.recomp_bytes += self.alloc_bytes

  def SetSameSrcs(self, recomp):
    if self.same_src_flag == 0:
      if self.same_src_root == None:
        self.same_src_flag = 1
        # only record in root
        self.same_src_rp.append(recomp)
        recomp.same_src_flag = 1
        recomp.same_src_root = self
      else:
        logging.error("Error!")
        exit(1)
    else:
      root_ = self.GetSrcRoot()
      # root_ same_src_flag must be set
      root_.same_src_rp.append(recomp)
      recomp.same_src_flag = 1
      recomp.same_src_root = root_


  def SetEvaTime(self, eva_time):
    self.eva_time = eva_time
    assert self.alloc_bytes > 0
    self.metric = float(self.alloc_bytes)/self.eva_time

  # def __cmp__(self, other)
  def IsUnsetRank(self):
    return self.rank == -1

  def IsEmptyPrev(self):
    if self.prev == []:
      return True
    else:
      return False

  def IsInSrcs(self, recomp):
    for _, name in self.srcs:
      if recomp.name() == name:
        return True
    return False

  def AddPrev(self, prev):
    self.prev.append(prev)

  def AddSucc(self, succ):
    self.succ.append(succ)

  def AddSrc(self, src):
    self.srcs.append(src)

  def PrintSrc(self):
    for num, name in self.srcs:
      if num == 0:
        logging.info("Candidate input: %s" % name)
      elif num == 1:
        logging.info("OutCandidate input: %s" % name)
      elif num == 2:
        logging.info("Variable input: %s" % name)
      elif num == 3:
        logging.info("Root input: %s" % name)
      else:
        logging.error("Unknow input type: %s" % name)
        exit(1)

  def IsRoot(self):
    self.srcs.sort(key=lambda x: x[0])
    # Only var or root input
    if self.srcs[0][0] >= 2:
      return True
    else:
      return False

  # def GetRecompSrcInputs(self):
  #   recomp_src = []
  #   for input_ in self.tensor.inputs:
  #     if IsVar(input_.name()):
  #       recomp_src.append(input_.name())
  #     else:
