try:
  import Queue as q
except ImportError:
  import queue as q

import os

import collections
import copy
import numpy as np

import swapInfo
import recomp_info
import draw
import logger
import logging

from check_netArch import Check_netArch

# from tomorrow import threads

from node import Node
# from node import Port

from tensor import Tensor


def node_cmp(x, y):
  if x.start_time ==  y.start_time:
    return x.access_id > y.access_id

  return x.start_time > y.start_time

def mem_cmp(x, y):
  if x[0] == y[0]:
    return x[1] > y[1]

  return x[0] > y[1]

class GraphSim():
  def __init__(self):
    # self.logic_time = 0
    # Node number in nodeInfo is a bit fewer than nodeCon

    # self.metadata_dir = "./alexnet_256_k40/"
    # self.metadata_dir = "./inception3_115_k40/"
    self.metadata_dir = "./inception3_160_p100/"
    # self.metadata_dir = "./vgg16_226_p100/"
    # self.metadata_dir = "./resnet50_190_p100/"

    self.nodeInfo_filename = "gpu_0_nodetime.txt"  # Initiate node execution time
    # self.nodeCon_filename = "1.log"

    self.tensorsize_filename = "gpu_0_outputs.txt"  # Initiate the requested bytes and allocated bytes of a tensor
    self.innode_filename = "1_innodes.txt"  # Initiate input tensor information of a node
    self.outnode_filename = "1_outnodes.txt"# Initiate outnodes of each node
    self.finish_time = 0.0
    # self.batch_size = 115
    self.batch_size = int(self.metadata_dir.split('_')[1])
    self.events = q.PriorityQueue()

    self.nodes = dict()
    self.tensors = dict()
    # self.faninNodes = dict()

    self.finish_nodes = dict()
    self.error_nodes = dict()

    # record useless tensors
    self.useless_tensors = dict()


    self.log_node_time = False

    self.log_mem = False

    self.debug_node = False

    self.debug_ref_count = False
    self.debug_rc_target = None

    self.peak_memory = 0
    self.mem_usage = []

    self.mem_limit = 6 * (1 << 10)
    self.required_saving = 0
    self.pcie_bandwidth = 12 * (1 << 10)

    self.time_metric = 1000000

    # self.node_access = []     # store the access of node in one iteration
    self.using_tf_tensor_access = True
    self.tensor_accesses = []   # store the access of tensor in one iteration, include timestamp
    self.tensor_access = []
    self.tf_tensor_access = []  # store the tensor access of tensorflow real running
    self.ngpu_tensor_access = dict()

    self.swapping_test = True
    self.swapped_tensors = dict() # ((swapped_out_tensor_name, ref_count), (swapin_trigger_tensor_name, ref_count))
    self.swapping_log = "swapping_decision.log"
    self.recomp_log = "recompute.log"

    self.swapping_debug = True
    # self.swapping_debug_log = "swapping_debug.log"


    self.swapin_trigger_distance = 20

    self.swap = False

    self.swap_time = True

    # Ignore the weights tensors when making swapping decision
    self.keys_filter = ["kernel", "beta", "grad", "bias", "batchnorm", "learning_rate", "weight"]
    self.variables = ['kernel', 'bias', 'read', 'const']
    self.v_filters = ['biasadd']
    # 'relu' should prioritize 'conv' as it's node name contains 'conv'
    self.layer_names = ["relu", "conv", "maxpool", "avgpool"]

    if self.swap:
      self.swapping_test = True
      self.swapping_debug = True
    else:
      self.swapping_test = False
      self.swapping_debug = False

    self.checkNetArch = False

    self.draw_access = False

    self.peakmem_util = swapInfo.PeakMemory()
    # some weights tensor size
    self.nouse_mem = 0

    # mm decision
    # 0: swapping
    # 1: recomputation
    self.mm_decision = dict()

    self.multargets_rp = False

    self.mm_candidates = dict()

    # recomputation info
    self.recomp_colle = recomp_info.ReCompColl()
    # the memory saving ratio from recomputation
    self.recomp_ratio = 1.0


  def EventsEngine(self):

    access_id = 0

    error_swapin = 0

    while not self.events.empty():
      e = self.events.get()

      try:
        assert(e.pending_count == 0)
      except AssertionError:
        logging.error("%s pending_count: %d" % (e.node_name, e.pending_count))
        raise AssertionError

      # Increase the fanin_tensors ref_count
      # ERROR not init here
      # for t in e.fanin_tensors:
      #   t.ref_count += 1

      # record the node processing id
      e.access_id = access_id

      access_id += 1

      if self.debug_node:
        logging.debug("Start to process %s" % e.node_name)

      # check the fanin_tensors to release the memory when the ref_count is 0
      for t in e.fanin_tensors:
        t.ref_count -= 1
        # log the specific tensor ref count for debugging
        if self.debug_ref_count:
          if t.name() == self.debug_rc_target:
            logging.debug("%s ref count decrease by one: %d" % (t.name(), t.ref_count))


        for st in self.swapped_tensors.keys():
        # t is in swapped_tensors collection
          if t.name() == st[0]:
            earliest_execution_time = max(e.logic_time, e.start_time)
            if t.ref_count > st[1]:
              # not reach out_trigger ref count
              pass
            elif t.ref_count == st[1]:
              # swap out the tensor now
              swapout_time = t.gpu_mem_requested / self.pcie_bandwidth * self.time_metric

              # TODO: may can not release the memory ASA the swap out operation finish, need CHECK this node's end time
              # TODO: check the end_time of nodes which use this tenesor before the swap out
              t.swapout_time  = earliest_execution_time + swapout_time
              self.mem_usage.append((t.swapout_time, -t.gpu_mem_requested))

              if self.swapping_debug:
                logging.debug("Start Swap out %s at %d, finish at %d" %
                             (t.name(), earliest_execution_time, t.swapout_time))

              # Add the pending_count for those nodes which need this tensor but not access it yet
              # Not here cause the node is possible to be put into queue but just not access to it yet
              # for node in self.nodes.values():
              #   if node in t.node_seq:
              #     continue

              #   # TODO: the node whose pending_count has not came to zero but got a earlier start time?
              #   # Add pending count as the tensors has been swapped out, decrease it when tensor has been swapped in
              #   if t in node.fanin_tensors:
              #     node.pending_count += 1



              # TODO: decrease the tensor memory in curr_mem here?

            else:
              # the tensor has been swapped out before, check whether has been swapped in yet?
              # swapin must be trigger before
              try:
                assert t.swapin_time != 0
              except AssertionError:
                error_swapin += 1
                logging.error("%s has not been triggered to be swapped in yet" % t.name())
                # raise AssertionError

              if t.swapin_time > earliest_execution_time:
                logging.debug("swap in overhead here %s : %f" % (t.name(),
                                                                (t.swapin_time-earliest_execution_time)))
                e.logic_time = t.swapin_time
                if self.swapping_debug:
                  logging.debug("swap in overhead here %s : %f" % (t.name(),
                                                                  (t.swapin_time-earliest_execution_time)))

        # t is in swap_in_trigger collection
        for k,v in self.swapped_tensors.items():
          # for now, if there are multiple swapped_tensors using the same swap_in_trigger
          # not consider the pcie interference yet
          if t.name() == v[0]:
            earliest_execution_time = max(e.logic_time, e.start_time)
            if t.ref_count > v[1]:
              # not reach a in_trigger ref count
              pass
            elif t.ref_count == v[1]:
              assert (k[0] in self.tensors.keys())
              swapout_tensor = self.tensors[k[0]]
              if swapout_tensor.swapout_time > earliest_execution_time:
                # in case that this tensor has not been swpped out, but in this case, seems
                # there is no difference to do this swapping
                logging.error("not enough time to swap out: %s" % swapout_tensor.name())
                earliest_execution_time = swapout_tensor.swapout_time
              swapin_time = swapout_tensor.gpu_mem_requested / self.pcie_bandwidth * self.time_metric
              swapout_tensor.swapin_time = earliest_execution_time + swapin_time

              swapout_tensor.swapping = False

              if self.swapping_debug:
                logging.debug("Start Swap in %s at %d, finish at %d" % (swapout_tensor.name(),
                                                                        earliest_execution_time,
                                                                        swapout_tensor.swapin_time))

              # Can decrease the pending_count here despite the tensor has not beed swapped in yet (but been triggered)
              for node in swapout_tensor.blocking_nodes:
                assert swapout_tensor not in node.ok_fanin_tensors
                node.ok_fanin_tensors.append(swapout_tensor)
                node.pending_count -= 1
                node.tmp_time.append(swapout_tensor.swapin_time)
                if self.swapping_debug:
                  logging.debug("remove a ref count of %s to %d" % (node.node_name, node.pending_count))

                if node.pending_count == 0:
                  node.logic_time = max(node.tmp_time)
                  self.events.put(node)
                  if self.swapping_debug:
                    logging.debug("Trigger %s to start" % node.node_name)

              self.mem_usage.append((earliest_execution_time, swapout_tensor.gpu_mem_requested))


      if e.logic_time > e.start_time:
        e.end_time = e.logic_time - e.start_time + e.end_time
        e.start_time = e.logic_time

        # if self.debug:
        #   fout_debug.write("[INFO] %s start at %d, end at %d\n" % (e.name, e.start_time, e.end_time))

      self.mem_usage.append((e.start_time, e.gpu_mem_allocated))


      # remove multiple same name fanout_node
      done_nodes = dict()
      done_nodes.clear()

      for node in e.fanout_nodes:
        if node.node_name in done_nodes.keys():
          pass
        else:
          done_nodes[node.node_name] = node

      # if self.debug_node:
      #   logging.debug("Node number after remove redundant nodes: %d" % len(done_nodes))

      # Clear input val when input_tensors ref count came to zero
      for t in e.fanin_tensors:
        try:
          assert t.ref_count >= 0
        except AssertionError:
          logging.error("Error ref count %s: %d" % (t.name(), t.ref_count))
          raise AssertionError
        if t.ref_count == 0:
          self.mem_usage.append((e.end_time, -t.gpu_mem_allocated))

      # ProcessOutputs
      # Init output_tensors of this node initial ref count

      if self.debug_node:
        logging.debug("Start init %s outputs ref count, number: %d" % (e.node_name, len(e.outputs)))
      for ut in e.outputs:
        if self.debug_node:
          if e.node_name == "cond/Switch_2":
            logging.debug("start init %s ref count" % ut.name())
            logging.debug("total fanout nodes: %d" % len(done_nodes))
        for fanout_node in done_nodes.values():

        # for fanout_node in e.fanout_nodes:
          if ut in fanout_node.fanin_tensors:
            # Initiate the ref count iff it's not been set to zero yet
            if ut.ref_count == -1:
              if self.debug_ref_count:
                if ut.name() == self.debug_rc_target:
                  logging.debug("%s increase ref count: %d" %
                               (ut.name(), ut.ref_count))
              ut.ref_count = 0
            ut.ref_count += 1
            # if ut.name() == "v/tower_0/cg/incept_v3_e0_1/conv85/batchnorm85/Const_1_0":
            #   # print("[HINT] Init ref count %d of %s\n" % (ut.ref_count, fanout_node.node_name))
            #   fout_debug.write("Init "+str(ut.ref_count)+" "+fanout_node.node_name+'\n')
            fanout_node.ok_fanin_tensors.append(ut)
            # if self.log_ref_count:
            #   fout_ref.write("%s ref count increase: %d\n" % (ut.name(), ut.ref_count))
            # print ("%s ref count increase\n" % ut.name())

      # self.node_access.append(e)
      for t in e.fanin_tensors:
        self.tensor_accesses.append((e.start_time, t))

      for node_ in e.fanout_nodes:
        node_.pending_count -= 1
        node_.tmp_time.append(e.end_time)
        if node_.pending_count == 0:
          swapping_flag = False
          for ft in node_.fanin_tensors:
            if ft.swapping == True:
              ft.ref_count_ -= 1
              # to see this tensor if been swapped-out
              if ft.ref_count_ < ft.swapping_ref_count:
                # if node_.node_name == "v/tower_0/cg/incept_v3_e0/conv80/conv2d/Conv2D":
                #   pass
                # else:
                assert ft in node_.ok_fanin_tensors
                node_.ok_fanin_tensors.remove(ft)
                node_.pending_count += 1  # wait for swapped_tensor to be swapped in
                ft.blocking_nodes.append(node_)
                swapping_flag = True

                if self.swapping_debug:
                  logging.debug("%s was blocked by %s" % (node_.node_name, ft.name()))
              # break
          if swapping_flag == True:
            continue
          node_.logic_time = max(node_.tmp_time)
          self.events.put(node_)


      # assert
      self.finish_nodes[e.node_name] = e

      if self.log_node_time:
        logging.debug("%s\t%d\t%d" % (e.node_name, e.start_time, e.end_time))


      if e.node_name == '_SINK':
        self.finish_time = e.end_time
        logging.info("finish time is %f s" % (float(self.finish_time)/1000000))

      self.events.task_done()


    # reorder the memory access

    self.tensor_accesses.sort(key=lambda x: x[0])
    self.tensor_access = [tensor for _, tensor in self.tensor_accesses]

    logging.info("Error swap in number: %d" % error_swapin)



  def GetUselessMemory(self):
    nouse_mem_size = 0
    metric = 1 << 20
    for t in self.useless_tensors.values():
      if t.allocator_name == "GPU_0_bfc":
        nouse_mem_size += float(t.allocated_bytes) / metric

    return nouse_mem_size


  def GetPeakMemory(self):
    mem_u = sorted(self.mem_usage, mem_cmp)

    total_mem = 0
    peak_mem = 0
    for _, m in mem_u:
      total_mem += m
      peak_mem = max(peak_mem, total_mem)

    self.peak_memory = peak_mem
    logging.info("Peak memory is %f MB" % peak_mem)


  # Analysis
  def access_analysis(self, tensor_access):
    tac = dict()

    if self.using_tf_tensor_access:
      # Init tensor (include cpu & gpu side) access info
      # shift this time to rel_time
      # this access info is already sorted by timestamp
      with open(self.metadata_dir+"tensor_access.txt") as fin:
        line_num = -1
        min_abs_time = -1
        max_access_time = -1
        for line in fin:
          # ignore line start with '#'
          if line[0] == '#':
            continue
          tmp = line.split()
          ac_index = 1
          if len(tmp) == 3:
            # include requested_bytes
            ac_index = 2
          tensor_name = tmp[0]
          access_time = int(tmp[ac_index])
          # tensor_name = line.split()[0]
          # requested_bytes = int(line.split()[1])
          # access_time = int(line.split()[1])
          line_num += 1
          # shift to reletive time
          if line_num == 0:
            min_abs_time = self.tensors[tensor_name].allocated_time
            try:
              assert access_time > min_abs_time
            except AssertionError:
              logging.error("%d v.s %d" % (min_abs_time, access_time))
              exit(1)
            access_time -= min_abs_time
          else:
            assert access_time >= min_abs_time
            access_time -= min_abs_time

          if access_time > max_access_time:
            max_access_time = access_time

          self.tf_tensor_access.append((access_time, tensor_name))
          # logging.debug("%s access time: %d" % (tensor_name, access_time))
          if not self.tensors.__contains__(tensor_name):
            logging.debug("tf tensor not found in simulator: %s" % tensor_name)
            if not self.ngpu_tensor_access.__contains__(tensor_name):
              self.ngpu_tensor_access[tensor_name] = []
            self.ngpu_tensor_access[tensor_name].append(line_num)
            # tensor's allocator is on CPU-side
          else:
            if self.tensors[tensor_name].allocator_name != "GPU_0_bfc":
              if not self.ngpu_tensor_access.__contains__(tensor_name):
                self.ngpu_tensor_access[tensor_name] = []
              self.ngpu_tensor_access[tensor_name].append(line_num)
            else:
              if not tac.__contains__(tensor_name):
                if self.swap_time:
                  # take time into account when making swapping decision
                  allocated_time = self.tensors[tensor_name].allocated_time
                  try:
                    assert allocated_time >= min_abs_time
                  except AssertionError:
                    logging.error("Tensor name: %s" % tensor_name)
                    logging.error("Allocation time: %d, min_ac_time: %d" % (allocated_time, min_abs_time))
                    exit(1)

                  allocated_time -= min_abs_time
                  allocated_bytes = self.tensors[tensor_name].allocated_bytes
                  tac[tensor_name] = swapInfo.SwapInfo(tensor_name,
                                              allocated_time=allocated_time,
                                              allocated_bytes=allocated_bytes)
                else:
                  tac[tensor_name] = []
              if self.swap_time:
                tac[tensor_name].access_list.append((line_num, access_time))
              else:
                tac[tensor_name].append(line_num)

        logging.info("Max access time from tf tensor access is %f s" % (max_access_time/1e6))

    else:
      # tensor_accesses = [tensor for _, tensor in tensor_access]
      for index, t in enumerate(tensor_access):
        # Ignore the tensor not in GPU
        if t.allocator_name != "GPU_0_bfc":
          # record the index of tensor which is not been allocated by GPU_BFC
          if not self.ngpu_tensor_access.__contains__(t.name()):
            self.ngpu_tensor_access[t.name()] = []
          self.ngpu_tensor_access[t.name()].append(index)
          continue


        if not tac.__contains__(t.name()):
          tac[t.name()] = []
        tac[t.name()].append(index)

    # filter the gpu tensors only show up once: deprecated
    # move to inner filter
    # Init swapinfo
    for swapinfo in tac.values():
      if len(swapinfo.access_list) > 1:
        swapinfo.access_list.sort(key=lambda x: x[1])
        swapinfo.DeallocateTime()
        self.GetMaxAccessInterval(swapinfo)

    # include len(ac) == 1 and weight tensors
    if self.draw_access:
      self.GetAccessInterval(tac)




    if self.swap_time:
      self.mm_candidates = {k:v for k,v in tac.items() if len(v.access_list) > 1}
      # peakmem_util = swapInfo.PeakMemory()
      self.peakmem_util.InitFromSwapInfo(self.mm_candidates.values())
      peak_mem = self.peakmem_util.GetPeakMemory()
      # peak_mem_col = peakmem_util.peakmem_tensors_collec
      # for t_name in peak_mem_col:
      #   assert t_name in tac_f.keys()
      #   s_list = tac_f[t_name].access_list
      #   logging.debug("%s: %s" % (t_name, str(s_list)))
      # left and right margin of peak memory usage time
      # l_peak_time = self.peakmem_util.left_peak_time
      # r_peak_time = self.peakmem_util.right_peak_time

      self.nouse_mem = self.GetUselessMemory()   # Seems that this useless memory is equal model size

      self.required_saving = peak_mem - self.mem_limit

      logging.info("Peak memory usage from tensor access is %f MB" % (peak_mem+self.nouse_mem))
      logging.info("Peak memory usage live tensors number is %d" % len(self.peakmem_util.peakmem_tensors_collec))

      self.GetMaxSavingMemory()
      # for name in peakmem_util.peakmem_tensors_collec:
      #   print("[INFO] Peak memory tensor: %s" % name)
      # tac_ff = sorted(tac_f)
      # tac_f.sort()

      # logging.debug("Candidates before filtering: %d" % len(tac_f))
      # for k,_ in tac_f.items():
      #   logging.debug("%s" % k)
      self.InitRecomp(self.mm_candidates)
      # self.FilterCandidates(tac_f, filter_size=False)
      # when considering re-computation, we should not throw little tensor?
      # self.InitRecomp(tac_f.values())
      self.swapping_decisionTime(self.mm_candidates)
    else:
      tac_f = {k:v for k,v in tac.items() if len(v) > 1}
      self.FilterCandidates(tac_f)
      self.swapping_decision(tac_f)

    # Get the tensor used multiple times and sorted by index distance
    # This is no use now as the swapping_index decision is made later
    # tac_ff = sorted(tac_f.items(), key=lambda x: x[1][-1] - x[1][0], reverse=True)
    # with open("candidates.log", 'w') as fout1:
    #   for k,v in tac_ff:
    #     fout1.write(k+': ')
    #     for vv in v:
    #       fout1.write(str(vv)+'\t')
    #     fout1.write('\n')


    # for taking time into account

    # self.swapping_decisionTime(tac_f)

    # for not taking time into account
    # self.swapping_decision(tac_f)
    # for k,v in tac_ff:
    #   print(k,v)

  def GetMaxSavingMemory(self):
    # To see the extreme memory we can save
    total_mem_saving = 0
    total_ts_saving = 0
    max_node_name = ""
    max_node_size = 0   # max memory size where tensors exist at the same time
    # max_ts_name = ""
    # max_ts_size = 0
    log2file = False
    if log2file:
      with open (self.metadata_dir+"total_swap_tensors.log", 'w') as fout1:
        fout1.truncate()
    for swapinfo in self.mm_candidates.values():
      # TODO: if we need to filter the weights node
      if swapinfo.tensor_name not in self.peakmem_util.peakmem_tensors_collec:
        continue

      swapped_out = self.tensors[swapinfo.tensor_name]
      curr_node_name = swapped_out.node_name
      curr_node = self.nodes[curr_node_name]
      curr_node_size = curr_node.GetNodeTotalSize()
      if curr_node_size > max_node_size:
        max_node_size = curr_node_size
        max_node_name = curr_node_name
      # if swapped_out.gpu_mem_allocated > max_ts_size:
      #   max_ts_name = swapinfo.tensor_name
      #   max_ts_size = swapped_out.gpu_mem_allocated
      total_mem_saving += swapped_out.gpu_mem_allocated
      if log2file:
        with open (self.metadata_dir+"total_swap_tensors.log", 'a') as fout1:
          fout1.write("%s\t%f\n" % (swapinfo.tensor_name, swapped_out.gpu_mem_allocated))
      total_ts_saving += 1

    total_mem_saving -= max_node_size   # remove the largest node memory
    total_mem_saving -= self.nouse_mem       # remove the weights memory
    logging.info("Maximum node is %s with %f MB" % (max_node_name, max_node_size))
    logging.info("Can swap out %d tensors, total memory we can save is %f MB" %
                (total_ts_saving, total_mem_saving))


  def FilterCandidates(self,
                       tac_f,
                       filter_weight=True,
                       filter_size=True,
                       size=2):
    """ tac_f: dict """
    for k,_ in tac_f.items():
      # check this tensor is not weights
      if filter_weight:
        if self.IsWeights(k):
          tac_f.pop(k)
          continue
      if filter_size:
        if self.IsSize(k):
          tac_f.pop(k)
        # tensor = self.tensors[k]
        # if tensor.gpu_mem_allocated < size:
        #   tac_f.pop(k)

      # weights_flag = False
      # for key_f in self.keys_filter:
      #   if key_f in k:
      #     weights_flag = True
      #     break
      # if weights_flag:
      #   tac_f.pop(k)

  def IsSize(self, name, s_size=2):
    try:
      assert name in self.tensors.keys()
    except AssertionError:
      logging.error("Error tensor name: %s" % name)
      exit(1)
    tensor = self.tensors[name]
    if tensor.gpu_mem_allocated < s_size:
      return True
    return False

  def IsWeights(self, name):
    weights_flag = False
    l_name = name.lower()
    for key_f in self.keys_filter:
      if key_f in l_name:
        weights_flag = True
        break
    return weights_flag

  def GetMaxAccessInterval(self, swapinfo):
    prev_t = swapinfo.access_list[0][1]
    curr_t = 0
    max_interval = -1
    max_index = 0

    for i in range(1, len(swapinfo.access_list)):
      curr_t = swapinfo.access_list[i][1]
      if (curr_t - prev_t) > max_interval:
        max_interval = curr_t - prev_t
        max_index = i - 1
      prev_t = curr_t

    swapinfo.swap_start = max_index
    swapinfo.max_access_interval = max_interval


  # already_queen: [SwapInfo]
  # candidate: SwapInfo
  # use curr_swap_info for comparison
  def UpdateSwapTimeInfo(self, already_queue, candidate):
    if len(already_queue) == 0:
      # use own curr_swap_info
      return
    # update candidate swapout info
    # TODO: move these two cmp out here
    def out_cmp(x1, x2):
      if x1.curr_swapout_info.start_time == \
         x2.curr_swapout_info.start_time:
         return x1.swap_time > x2.swap_time
      return x1.curr_swapout_info.start_time > x2.curr_swapout_info.start_time

    def in_cmp(x1, x2):
      if x1.curr_swapin_info.start_time == \
         x2.curr_swapin_info.start_time:
         return x1.swap_time > x2.swap_time
      return x1.curr_swapin_info.start_time < x2.curr_swapin_info.start_time

    temp_ = already_queue + [candidate]
    temp_.sort(cmp=out_cmp)
    c_index = temp_.index(candidate)
    # the swapinfos before candidate are fine
    # so start from the swapinfo before candidate to update swapinfo
    for i in range(c_index-1, len(temp_)-1):
      si1 = temp_[i]
      si2 = temp_[i+1]
      # update swapout_info of si2 as si1's swapout_info is fixed
      if si1.curr_swapout_info.end_time > \
         si2.curr_swapout_info.start_time:
         time_int = si1.curr_swapout_info.end_time - \
                    si2.curr_swapout_info.start_time
         si2.UpdateSwapOutTime(time_int)
        # only update swap_info not curr_swap_info

    temp_.sort(cmp=in_cmp)
    c_index = temp_.index(candidate)
    for i in range(c_index-1, len(temp_)-1):
      si1 = temp_[i]
      si2 = temp_[i+1]
      # update swapin_info of si2
      # and it's related to the choice of in_trigger
      # maybe we can ignore the choice of in_trigger here
      if si1.curr_swapin_info.start_time < \
         si2.curr_swapin_info.end_time:
         time_int = si1.curr_swapin_info.start_time - \
                    si2.curr_swapin_info.end_time
         si2.UpdateSwapInTime(time_int)

    # Update candidate's swap_free_time to already_queue's smallest?
    min_swap_free_time = temp_[0].swap_free_time
    for i in range(1, len(temp_)):
      sft_ = temp_[i].swap_free_time
      if sft_ < min_swap_free_time:
        min_swap_free_time = sft_

    candidate.swap_free_time = min_swap_free_time


  def gettime(self,
              t_ac,
              i_ac,
              t_index=0,
              i_index=0):
    if len(t_ac) <= 3:
      assert t_index == 0
    # we can get accurate time here?
    t_time = t_ac[t_index][1]

    if len(i_ac) <= 3:
      assert t_time > i_ac[0][1]
      return (t_time - i_ac[0][1])
    else:
      for i in range(len(i_ac)-1, -1, -1):
        if i_ac[i][1] > t_time:
          continue
        else:
          return (t_time-i_ac[i][1])

    return -1


  def GetAccessInterval(self, tac):
    """ tac: all gpu tensors swapinfo """
    def gettime(t_ac, i_ac, t_index=0):
      # must access in forward and backward
      # but backward usually needs 3 times except the input
      if len(t_ac) <= 3:
        assert t_index == 0
      t_time = t_ac[t_index][1]

      if len(i_ac) == 2:
        return (t_time-i_ac[0][1])
        # print("[DEBUG] access interval: %d" % (t_time-i_ac[0][1]))
      else:
        for i in range(len(i_ac)-1, -1, -1):
          if i_ac[i][1] > t_time:
            # print("[DEBUG] error time interval: %d" % (t_time-i_ac[i][1]))
            continue
          else:
            # print("[DEBUG] access interval: %d" % (t_time-i_ac[i][1]))
            # always get the closest access
            # break
            return (t_time-i_ac[i][1])

      # here we can't get the right access interval
      return -1

    node_execs = []
    ac_ints = []

    for swapinfo in tac.values():
      # filter tensor only shows up once
      if len(swapinfo.access_list) == 1:
        continue
      # filter tensor which is weight
      if (self.IsWeights(swapinfo.tensor_name)):
        continue

      tensor_name = swapinfo.tensor_name
      tensor = self.tensors[tensor_name]
      node_exec_time = self.nodes[tensor.node_name].GetExecTime()

      # get tensor access interval
      t_ac = swapinfo.access_list
      # TODO: t_index should be 0 as the first time it's being produced?
      t_index = swapinfo.swap_start

      i_ac = []
      curr_ac = []

      # logging.debug("%s inputs number: %d" % (tensor.name(), len(tensor.inputs)))
      # HACK for some tensors without input
      # as we remove some error node from metadata
      if len(tensor.inputs) == 0:
        continue
      for input_ in tensor.inputs:
        if not tac.__contains__(input_.name()):
          continue
        i_swapinfo = tac[input_.name()]
        curr_ac = i_swapinfo.access_list
        # gettime(t_ac, curr_ac, t_index)
        # print("\n")
        if len(i_ac) == 0:
          i_ac = curr_ac
        if len(curr_ac) < len(i_ac):
          i_ac = curr_ac

      # HACK for some tensors without access info
      if len(i_ac) == 0:
        continue

      ac_int = gettime(t_ac, i_ac, t_index)

      if ac_int == -1:
        logging.error("Error access interval for %s" % tensor_name)
        logging.debug("t_ac: %s" % str(t_ac))
        logging.debug("i_ac: %s" % str(i_ac))
        # print("[DEBUG] Error access interval for %s\n" % tensor_name)
        exit(1)
      node_execs.append(node_exec_time)
      ac_ints.append(ac_int)

    draw.plotting_actime(node_execs, ac_ints, title=self.metadata_dir[2:-1])


  # deprecated: no use now
  def InColle(self, target, t_col):
    if len(set(target.inputs).intersection(set(t_col))) != 0:
      return True

    t_col_inputs = [tt for t in t_col for tt in t.inputs]
    if len(set(t_col_inputs).intersection(set(target))) != 0:
      return True
    # name_coll = [s_info.tensor_name for s_info in coll]
    # tensor = self.tensors[swapinfo.tensor_name]
    # input_names = [t.name() for t in tensor.inputs]
    # if len(set(input_names).intersection(name_coll)) == 0:
    #   return

  # deprecated: no use now
  def BackTraverse(self, recomps, recomp, start_from, colle):
    if len(start_from.inputs) == 0:
      return None
    for input_t in start_from.inputs:
      if input_t in colle:
        # return input_t
        recomp.AddPrev(recomps[input_t.name()])
      else:
        return self.BackTraverse(recomps, recomp, input_t, colle)

  def IsVar(self, name):
    l_name = name.lower()
    for v in self.v_filters:
      if v in l_name:
        return False

    for v in self.variables:
      if v in l_name:
        return True

    return False


  def GetRecompSrc(self, recomp, t, c_coll, a_coll):
    """ Get recomputation inputs for t
    recomp: ReComp of target tensor
    t:      target tensor
    c_coll: candidates tensors collection
    a_coll: all tensors collection which show up more than once
    """
    if len(t.inputs) == 0:
      # logging.debug("Root input: %s" % t.name())
      recomp.AddSrc((3, t.name()))
      # return None
    for input_ in t.inputs:
      if self.IsVar(input_.name()):
        # logging.debug("Var input: %s" % input_.name())
        recomp.AddSrc((2, input_.name()))
      elif input_ in c_coll:
        # logging.debug("Candidate input: %s" % input_.name())
        recomp.AddSrc((0, input_.name()))
      elif input_.name() in a_coll.keys():
        # TODO: see if this input_ occurs after this tensor's access
        i_swapinfo = a_coll[input_.name()]
        t_swapinfo = a_coll[recomp.tensor.name()]
        i_lastac = i_swapinfo.access_list[-1][1]
        t_lastac = t_swapinfo.access_list[-1][1]
        if i_lastac > t_lastac:
          # logging.debug("Right input: %s" % input_.name())
          pass
        else:
          logging.error("Error input: %s" % input_.name())
          logging.error("Error time: %d vs. %d" % (t_lastac, i_lastac))
        recomp.AddSrc((1, input_.name()))
      else:
        # only show up once and not variable tensor
        # search recurrsively
        return self.GetRecompSrc(recomp, input_, c_coll, a_coll)

  # Calculate eva_time of each rp
  def EvaluateRP(self, recomp, tac_f):
    t_name = recomp.name()
    i_name = None
    # recomp.srcs.sort(key=lambda x: x[0])
    for num, name in recomp.srcs:
      if num == 0:
        i_name = name
        break
      elif num == 1:
        i_name = name
        break
      elif num == 2:
        logging.debug("only var input")
      else:
        logging.debug("only root input")

    t_ac = tac_f[t_name].access_list
    if i_name == None:
      logging.error("Got no input name for %s" % t_name)
    i_ac = tac_f[i_name].access_list

    eva_time = self.gettime(t_ac, i_ac)
    if eva_time == -1:
      logging.error("Can not evaluate time for %s" % t_name)
    # recomp.eva_time = eva_time
    recomp.SetEvaTime(eva_time)
    # logging.info("%s: evaluate time, %d" % (t_name, eva_time))

  # deprecated: no use now
  def MergeRP(self, rp1, rp2, recomp_coll):
    if rp1 in rp2.prev:
      # rp1 -> rp2
      if rp2.sub_rp == None:
        if rp1.sub_rp == None:
          subrecomp = recomp_info.SubReComp(rp1)
          subrecomp.AddRP(rp2)
        else:
          rp1.sub_rp.AddRP(rp2)
      else:
        if rp1.sub_rp == None:
          rp2.sub_rp.AddRP(rp1)
        else:
          pass
          logging.debug("Ignore merging two sub_recomp for now!")
          # try merge two sub_recomp

  def GetChain(self, recomp_coll, chains, candidate):
    # candidate: sub_recomp
    # chains
    # tmp_chain = recomp_info.Chain()
    # tmp_chain.add(candidate)
    # tmp_chain.head = candidate.root_

    # candidate be the head of current chains
    heads = []
    # candidate be the tail of current chains
    tails = []

    for chain in chains:
      # check if candidate can be head or tail of current chain
      # if candidate.root_.IsSucc(chain.head):
      if chain.IsPrev(candidate):
      # if recomp_coll.IsPPrev(candidate, chain.head):
        # candidate is the head
        heads.append(chain)

        # tmp_chain.addchain(chain)
        # tmp_chain.tail = chain.tail

      # elif candidate.root_.IsPrev(chain.tail):
      elif recomp_coll.IsPPrev(chain):
        # candidate is the tail
        tails.append(chain)
        # tmp_chain.addchain(chain)
        # tmp_chain.head = chain.head
        # tmp_chain.tail = chain.tail



  # def GetChain(self, already_queue, candidate):
  #   # check the already_queue if candidate is in src_inputs
  #   def GetSubrp(name, q):
  #     for p in q:
  #       if name == p.name():
  #         return p
  #     return None

  #   chain = recomp_info.Chain()
  #   chain.set_root(candidate)
  #   for subrp in already_queue:
  #     src_inputs = [src[1] for src in subrp.src_inputs]
  #     if candidate.name() in src_inputs:
  #       # candidate is prev of subrp
  #       chain.add(subrp)

  #   q_names = [srp.name() for srp in already_queue]
  #   for input_ in candidate.src_inputs:
  #     if input_[1] in q_names:
  #       # input_[1] is prev of candidate
  #       sub_rp = GetSubrp(input_[1], already_queue)
  #       # check current root?
  #       if
  #       chain.set_root(sub_rp)

  def GetOrCreateSubRP(self, sub_recomps, recomp):
    # for sub_recomp in sub_recomps:
    pass


  def CheckAvai(self, recomp):
    # check recomp tensor's direct inputs
    for t in recomp.tensor.inputs:
      # whether its inputs are swapping or recomputation
      if t.name() in self.mm_decision.keys():
        return False

    # check recomp if is prev
    for t_name in self.mm_decision.keys():
      t_ = self.tensors[t_name]
      if recomp.tensor in t_.inputs:
        if self.mm_decision[t_name] == 1:
          # ok if can recompute multi-target
          # so they must in the same sub_recomputation
          if self.multargets_rp:
            continue
          else:
            return False
        else:
          # swapping input
          continue

    return True
    
      

  def InitRecomp(self, tac_f):
    """ tac_f: all tensors which show up more than one """

    t_colls = dict()
    for swapinfo in tac_f.values():
      tensor = self.tensors[swapinfo.tensor_name]
      if self.IsWeights(swapinfo.tensor_name):
        continue
      if self.IsSize(swapinfo.tensor_name):
        continue
      t_colls[swapinfo.tensor_name] = tensor
      # logging.debug("%s: %d" % (tensor.name(), tensor.gpu_mem_allocated))
      # logging.debug("%s" % swapinfo.tensor_name)
    logging.debug("Candidates number: %d" % len(t_colls))

    # recomp_colle = recomp_info.ReCompColl()
    recomps = self.recomp_colle.collection
    # t_sets = set(t_colls.values())

    # Create recomp objects
    for k,v in t_colls.items():
      swapinfo = tac_f[k]
      acc_time = swapinfo.GetFirstUseTimeAfterSwap()
      acc_index = swapinfo.GetFirstUseIndexAfterSwap()
      recomp = recomp_info.ReComp(v, (acc_index, acc_time))
      recomps[k] = recomp

    # Assert single chain now?
    # for k,v in t_colls.items():
    #   self.BackTraverse(recomps, recomps[k], v, t_colls.values())
      # for i in v.inputs:
      #   if recomp_colle.IsInColl(i)
      # for i in v.inputs:
      #   if i in t_colls.values():
      #     logging.debug("%s needs %s" % (k, i.name()))
      #     recomps[k].AddPrev(recomps[i.name()])
      #   elif i.name() in tac_f.keys():
      #     logging.debug("%s needs %s which is not candidate" % (k, i.name()))
      #   else:
      #     logging.debug("%s needs %s which occurs once" % (k, i.name()))

    # Get input srcs of rp
    for k, v in self.recomp_colle.collection.items():
      # logging.debug("%s recomputation src:" % k)
      # prev_t = [t.tensor for t in v.prev]
      self.GetRecompSrc(v, v.tensor, t_colls.values(), tac_f)
      v.inputs = [src[1] for src in v.srcs]
      if v.IsRoot():
        if self.recomp_colle.root_ == None:
          self.recomp_colle.SetRoot(v)
        else:
          logging.warning("Got multiple root: %s" % k)

    # Init prev & succ of each rp
    # Init rank of each rp
    self.recomp_colle.InitRPConnection()
    # recomp_colle.RepeatedRP()

    # Evaluate re-computation time of each rp
    for recomp_ in recomps.values():
      recomp_.SetRecompBytes(self.tensors)
      # logging.debug("%s: recomp bytes: %d MB" % (recomp_.name(), recomp_.recomp_bytes))
      if self.recomp_colle.IsRoot(recomp_):
        continue
      self.EvaluateRP(recomp_, tac_f)
    # recomps_ = sorted(recomps.values(), key=lambda x: x.alloc_bytes, reverse=True)
    # not reverse here due to the following pop() operation
    recomps_ = sorted(recomps.values(), key=lambda x: x.metric)
    # for recomp_ in recomps_:
    #   if recomp_colle.IsRoot(recomp_):
    #     logging.debug("Ignore root: %s" % recomp_.tensor.name())
    #     continue
    #   self.EvaluateRP(recomp_, tac_f)
    # recomps.sort(key=lambda x: x.alloc_bytes)
    # candidates = []

    r_peak_time = self.peakmem_util.right_peak_time
    required_saving = self.required_saving * self.recomp_ratio

    fout = open(self.metadata_dir+self.recomp_log, 'w')

    sub_recomps = []
    # for now only one target at a recomputation
    # already_queue = []
    recomp_depth = 2
    left_queue = [recomp_ for recomp_ in recomps_]
    while True:
    # for recomp in recomps_:
      # if this recomp connect two sub_recomps, we shouldn't choose this one
      # we need to make decision when visiting all sub_recomps
      if len(left_queue) == 0:
        break

      succ = [] # sub_recomp is succ of rp
      prev = [] # sub_recomp is prev of rp
      recomp = left_queue.pop()
      # check if this rp is direct input or output of mm_decision
      if self.CheckAvai(recomp):
        pass
      else:
        logging.debug("%s is rejected" % recomp.name())
        continue
      flag = True
      for sub_recomp in sub_recomps:
        if sub_recomp.IsSrc(recomp):
          if recomp.rank - sub_recomp.root_.rank <= recomp_depth:
            prev.append(sub_recomp)
          else:
            # one failed means all failed
            flag = False
            break
        if sub_recomp.IsInSrc(recomp):
          assert sub_recomp not in prev
          if (sub_recomp.max_rank_var+sub_recomp.root_.rank-recomp.rank) <= recomp_depth:
            succ.append(sub_recomp)
          else:
            flag = False
            break

      logging.debug("%s: prev: %d, succ: %d" % (recomp.name(), len(prev), len(succ)))
      if len(prev) > 1:
        logging.error("%s prev: %d" % (recomp.name(), len(prev)))
      if len(succ) > 1:
        logging.error("%s succ: %d" % (recomp.name(), len(prev)))
      if not flag:
        continue


      if len(prev) != 0 and len(succ) != 0:
        logging.debug("%s will connect two recomps, so we ignore this" % (recomp.name()))
        continue

      s = recomp.SetTrigger(tac_f, self, r_peak_time)
      if s == False:
        logging.debug("Can not set in_trigger for %s" % recomp.name())
        continue

      if len(prev) != 0:
        if len(succ) != 0:
          logging.debug("%s will connect two recomps, so we ignore this" % (recomp.name()))
        else:
          prev_ = prev[0]
          logging.debug("%s merge succ: %s" % (prev_.name(), recomp.name()))
          prev_.MergeSucc(recomp)
          # pass
      else:
        if len(succ) != 0:
          succ_ = succ[0]
          logging.debug("%s merge prev: %s" % (succ_.name(), recomp.name()))
          succ_.MergePrev(recomp)
          # pass
        else:
          # new a sub_recomp for this recomp
          new_subrp = recomp_info.SubReComp(recomp)
          sub_recomps.append(new_subrp)

      logging.debug("Choose %s, bytes: %d MB, metric: %f" % (recomp.name(), recomp.alloc_bytes, recomp.metric))
      logging.debug("in_trigger info: %s" % str(recomp.in_trigger))

      required_saving -= recomp.alloc_bytes
      self.mm_decision[recomp.name()] = 1
      if required_saving <= 0:
        logging.info("Already choose right tensors from recomputation!")
        break

    for sub_recomp in sub_recomps:
      for recomp in sub_recomp.coll.values():
        fout.write("%s\t%d\t%d\t%s\t%d\t%d\t" % (recomp.name(),
                                                 recomp.out_trigger[0],
                                                 recomp.out_trigger[1],
                                                 recomp.in_trigger[0],
                                                 recomp.in_trigger[1],
                                                 recomp.in_trigger[2]))

        for input_ in recomp.inputs:
          fout.write("%s\t" % input_)
        fout.write("\n")




    # TODO: remove this sub_recomp as it needs to be decided at runtime
    # sub_recomps = self.recomp_colle.sub_rp
    # for recomp_ in recomps_:
    #   flag = False
    #   # TODO: how to sort these sub_recomp
    #   for sub_recomp in sub_recomps:
    #     if sub_recomp.AddRP(recomp_):
    #       flag = True
    #       break

    #   if flag:
    #     # logging.debug("Adding %s success" % recomp_.name())
    #     continue
    #   else:
    #     # logging.debug("New a sub_recomp for %s" % recomp_.name())
    #     new_sub_recomp = recomp_info.SubReComp(recomp_)
    #     sub_recomps.append(new_sub_recomp)

    #   # logging.debug("%s : %f MB, %f" %
    #   #   (recomp_.name(), recomp_.alloc_bytes, recomp_.metric))

    # # for log debug
    # logging.debug("Total %d sub recomputation" % len(sub_recomps))
    # for sub_recomp in sub_recomps:
    #   # logging.debug("Root: %s, size: %d" % (sub_recomp.root_.name(), len(sub_recomp.coll)))
    #   sub_recomp.GetRecompTime()
    #   sub_recomp.SetSrcs()
    #   sub_recomp.SetTotalBytes(self.tensors)
    #   # TODO: this metric should not include inputs size
    #   sub_recomp.SetMetric()
    #   # logging.debug("Recomputation time: %d" % sub_recomp.recomp_time)
    #   # for rp in sub_recomp.coll:
    #   #   logging.debug("%s, rank: %d" % (rp.name(), rp.rank))


    # r_peak_time = self.peakmem_util.right_peak_time
    # sub_recomps_ = sorted(sub_recomps, key=lambda x: x.metric, reverse=True)

    # total_rp_bytes = 0
    # max_rp_bytes = 0

    # fout = open(self.metadata_dir+self.recomp_log, 'w')


    # required_saving = self.required_saving * self.recomp_ratio
    # logging.debug("Required saving from recomputation is %f MB" % required_saving)
    # curr_num = 1
    # # TODO: add a candidate queue to see the new one is a prev of succ of a recomp in queue
    # candidate_queue = []
    # chains = []
    # for sub_recomp_ in sub_recomps_:

    #   logging.debug("Curr number: %d, Total size: %d, metric: %f" % (curr_num, len(sub_recomp_.coll), sub_recomp_.metric))
    #   s = sub_recomp_.SetInTrigger(tac_f, self, r_peak_time)
    #   if s == False:
    #     logging.debug("Can not set in_trigger for %s" % sub_recomp_.name())
    #   else:
    #     total_rp_bytes += sub_recomp_.saving_bytes
    #     if sub_recomp_.saving_bytes > max_rp_bytes:
    #       max_rp_bytes = sub_recomp_.saving_bytes
    #     logging.debug("Recomputation bytes: %d MB" % sub_recomp_.saving_bytes)
    #     logging.debug("Total bytes: %d MB" % sub_recomp_.total_bytes)
    #     logging.debug("%s: in_trigger info: %s" % (sub_recomp_.name(), str(sub_recomp_.in_trigger)))

    #     for rp in sub_recomp_.coll:
    #       assert rp.name() not in self.mm_decision.keys()
    #       self.mm_decision[rp.name()] = 1


    #     # NOTE: assert out trigger is one now
    #     fout.write("%s\t%d\t%d\t%s\t%d\t%d\t" % (sub_recomp_.out_triggers.keys()[0],
    #                                              sub_recomp_.out_triggers.values()[0][0],
    #                                              sub_recomp_.out_triggers.values()[0][1],
    #                                              sub_recomp_.in_trigger[0],
    #                                              sub_recomp_.in_trigger[1],
    #                                              sub_recomp_.in_trigger[2]))

    #     for src in sub_recomp_.src_inputs:
    #       fout.write("%s\t" % src[1])

    #     fout.write("\n")

    #     required_saving -= sub_recomp_.saving_bytes

    #     if required_saving <= 0:
    #       logging.info("Already choose right tensors from recomputation!")
    #       break

    #   curr_num += 1

    # fout.close()
    # if required_saving > 0:
    #   logging.info("Already choose %d MB" % total_rp_bytes)
    #   logging.error("No enough tensors")
    #   exit(1)

    # logging.debug("Total recomputation saving bytes: %d MB" % total_rp_bytes)
    # logging.debug("Max recomputation saving bytes: %d MB" % max_rp_bytes)



    # p_recomp = recomps_[0]
    # logging.info("Choose %s for recomputation, %f MB, %f us" %
    #             (p_recomp.tensor.name(),
    #              p_recomp.alloc_bytes,
    #              p_recomp.eva_time))
    # logging.info("Src inputs:")
    # p_recomp.PrintSrc()

    # for k,v in recomp_colle.collection.items():
    #   if v.IsEmptyPrev():
    #     logging.debug("%s can be re-computed directly" % k)
    #     v.depth = 0
    #     # this tensor's inputs only show up once in tensor access
    #     # which means the inputs tensors will be deallocated once finishing
    #   else:
    #     depth = 0
    #     curr_queue = [p for p in v.prev]
    #     while True:
    #       if curr_queue == []:
    #         break
    #       tmp = []
    #       depth += 1
    #       for rp in curr_queue:
    #         if rp.IsEmptyPrev():
    #           pass
    #         else:
    #           tmp += [p_ for p_ in rp.prev]
    #       curr_queue = tmp
    #     logging.debug("%s re-compute depth: %d" % (k, depth))
    #     v.rank = depth

      # c_prev = self.BackTraverse(v, t_colls.values())
      # if c_prev == None:
      #   # logging.debug("Not find prev for %s in current collection" % k)
      #   # current tensor should be the root tensor
      #   if recomp_colle.GetRoot() != None:
      #     logging.error("Find multiple root tensor!")
      #     exit(1)
      #   recomp_colle.SetRoot(recomps[v.name()])
      # else:
      #   # what if more than one tensor in collection
      #   # we need to know
      #   recomps[k].prev = recomps[c_prev.name()]
      #   recomps[c_prev.name()].succ = recomps[k]
        # logging.debug("Find closest prev %s for %s" % (c_prev.name(), k))

    # debug log
    # logging.debug("Root tensor: %s" % recomp_colle.root_.tensor.name())

    # for k,v in recomp_colle.collection.items():
    #   if v != recomp_colle.root_:
    #     assert v.prev != None
    #   if v.succ == None:
    #     logging.debug("No succeeding for %s" % k)



      # t_filter = list(set(v.inputs).intersection(t_sets))
      # if len(t_filter) != 0:
      #   try:
      #     assert len(t_filter) == 1
      #   except AssertionError:
      #     logging.debug("%s with multiple inputs in collection" % k)
      #     # exit(1)

      #   for t in t_filter:
      #     prev = recomps[t.name()]
      #     recomps[k].AddPrev(prev)
      #     prev.AddSucc(recomps[k])
      # else:
      #   logging.debug("%s with no input in collection" % k)

    # debug log
    # for k,v in recomps.items():
    #   logging.debug("%s, prev: %d, succ: %d" % (k, len(v.prev), len(v.succ)))
    #   for p in v.prev:
    #     logging.debug("Prev: %s" % p.tensor.name())
    #   for s in v.succ:
    #     logging.debug("Succ: %s" % s.tensor.name())






  def recomputation(self,
                    tac_f,
                    l_peak_time=-1,
                    r_peak_time=-1):
    # pass
    # to get longest re-computation chain?
    # for swapinfo in tac_f:
    pass



  def swapping_decisionTime(self, tac_f):
    """
    tac_f: {tensor_name: swapinfo}
    already filtered useless tensors (weights)
    """
    # def GetNodeSize(node_name):

    #   return 0

    # Init deallocation time, swap_start, max_interval
    # for swapinfo in tac_f.values():
    #   swapinfo.access_list.sort(key=lambda x : x[1])
    #   swapinfo.DeallocateTime()
    #   self.GetMaxAccessInterval(swapinfo)


    # NOTE: for swap_free_time algorithm, no need to ordered here
    # tac_ff = sorted(tac_f.values())


    # NOTE: use which one to set required saving
    # required_saving = self.peakmem_util.peak_mem - self.mem_limit

    required_saving = self.required_saving * (1-self.recomp_ratio)
    logging.info("Required saving for swapping is %f MB" % required_saving)

    if required_saving == 0:
      return

    fout = open(self.metadata_dir+self.swapping_log, 'w')

    # To see the extreme memory we can save

    # MM Algorithm: swap_free_time
    # Assumption: won't remove candidates from already_queue
    mm_already_queue = []
    mm_left_queue = []
    # filter useless swapping
    for swapinfo in tac_f.values():
      if self.IsWeights(swapinfo.tensor_name):
        continue
      if self.IsSize(swapinfo.tensor_name):
        continue

      # ignore already in mm deicison
      if swapinfo.tensor_name in self.mm_decision.keys():
        continue

      # Check this tensor if in the peak memory usage time
      if swapinfo.tensor_name not in self.peakmem_util.peakmem_tensors_collec:
        # logging.debug("%s not in peak memory usage time" % swapinfo.tensor_name)
        # skipped_list.append(swapinfo.tensor_name)
        continue

      # can be added then
      mm_left_queue.append(swapinfo)


    l_peak_time = self.peakmem_util.left_peak_time
    r_peak_time = self.peakmem_util.right_peak_time
    logging.debug("Initial left_queue length: %d" % len(mm_left_queue))
    logging.debug("Left peak memory time: %d" % l_peak_time)
    logging.debug("Right peak memory time: %d" % r_peak_time)

    # Init swap out/in time info
    for swapinfo_ in mm_left_queue:
      swapinfo_.InitSwapInfo()

    total_swap_size = 0
    while True:
      # res = []
      if len(mm_left_queue) == 0:
        # TODO: add debug info
        break
      for swapinfo_ in mm_left_queue:
        # Update swapinfo_ according to current queue
        # And we need to use this to order the swapinfo_ in mm_left_queue
        # (swap_free_time, swapout_start_time)
        # only update temp swap_info
        self.UpdateSwapTimeInfo(mm_already_queue, swapinfo_)
        # res.append(self.UpdateSwapTimeInfo(mm_already_queue, swapinfo_))
      # res.sort()
      # swapinfo = res[0]


      mm_left_queue.sort()
      # No need to sort mm_already_queue here
      swapinfo = mm_left_queue[0]
      # check the swap-out end_time if exceeds the left_peak_time
      swapout_end_time = swapinfo.swapout_info.end_time
      if swapout_end_time > l_peak_time:
        logging.debug("Not enough time for %s swapping out" % swapinfo.tensor_name)
        logging.debug("Swap-out end time: %d" % swapout_end_time)
        mm_left_queue.remove(swapinfo)
        continue
      # we decide to use this candidate?
      # here we can get the latest time to start swapping-in
      # Find appropriate in_trigger tensor

      n_index = swapinfo.GetFirstUseIndexAfterSwap()
      s_time = swapinfo.swapin_info.start_time
      # e_time = swapinfo.swapin_info.end_time

      # n_time = swapinfo.GetFirstUseTimeAfterSwap()
      in_trigger_flag = True
      in_trigger_index = n_index
      while True:
        # TODO: check if this in_trigger to early that in peak memory
        in_trigger_index -= 1
        if self.tf_tensor_access[in_trigger_index][0] < r_peak_time:
          logging.debug("Not enough time for %s swapping in" % swapinfo.tensor_name)
          in_trigger_flag = False
          break
          # exit(1)
        if self.tf_tensor_access[in_trigger_index][0] < s_time:
          in_trigger_name = self.tf_tensor_access[in_trigger_index][1]
          # print("[DEBUG] %s in_trigger index distances: %d" % (swapinfo.tensor_name, n_index-in_trigger_index))
          break

      if not in_trigger_flag:
        mm_left_queue.remove(swapinfo)
        continue

      swapout_rc = swapinfo.GetSwapoutRc()
      swapout_total_rc = len(swapinfo.access_list)
      swapin_rc = 0
      swapin_total_rc = 1
      if in_trigger_name in tac_f.keys():
        access_indicies = [v for v,_ in tac_f[in_trigger_name].access_list]
        assert (in_trigger_index in access_indicies)
        swapin_rc = len(access_indicies) - access_indicies.index(in_trigger_index) - 1
        swapin_total_rc = len(access_indicies)
      elif in_trigger_name in self.ngpu_tensor_access.keys():
        access_indicies = self.ngpu_tensor_access[in_trigger_name]
        assert (in_trigger_index in access_indicies)
        swapin_rc = len(access_indicies) - access_indicies.index(in_trigger_index) - 1
        swapin_total_rc = len(access_indicies)
      else:
        pass


      # swapinfo.UpdateCurrSwapInfo()

      mm_already_queue.append(swapinfo)
      for swapinfo_ in mm_already_queue:
        swapinfo_.UpdateCurrSwapInfo()
      mm_left_queue.remove(swapinfo)

      total_swap_size += swapinfo.allocated_bytes

      # logging.debug("choose %s, size: %d, in_trigger: %s" % (swapinfo.tensor_name, swapinfo.allocated_bytes, in_trigger_name))

      fout.write("%s\t%d\t%d\t%s\t%d\t%d\n" % (swapinfo.tensor_name,
                                               swapout_total_rc,
                                               swapout_rc,
                                               in_trigger_name,
                                               swapin_total_rc,
                                               swapin_rc))

      required_saving -= swapinfo.allocated_bytes
      # print("[DEBUG] Max access interval is %d, allocated MB: %f\n" % (swapinfo.max_access_interval, swapinfo.allocated_bytes))
      if required_saving <= 0:
        logging.info("Already choose proper swapped out tensors")
        break

    fout.close()
    if required_saving > 0:
      logging.info("Already choose %d MB" % total_swap_size)
      logging.error("No enough tensors")
      exit(1)





    # MM Algorithm: max_access_interval
    """ for swapinfo in tac_ff:
      # check this tensor is not weights
      weights_flag = False
      for key_f in self.keys_filter:
        if key_f in swapinfo.tensor_name:
          weights_flag = True
          break
      if weights_flag:
        continue

      # Check this tensor if in the peak memory usage time
      if swapinfo.tensor_name not in peakmem_util.peakmem_tensors_collec:
        print("[DEBUG] %s not in peak memory usage time\n" % swapinfo.tensor_name)
        # skipped_list.append(swapinfo.tensor_name)
        continue

      swapped_out = self.tensors[swapinfo.tensor_name]
      if swapped_out.gpu_mem_allocated < swapping_threshold:
        # skipped_list.append(swapinfo.tensor_name)
        continue

      # Find appropriate in_trigger tensor
      n_index = swapinfo.GetFirstUseIndexAfterSwap()
      n_time = swapinfo.GetFirstUseTimeAfterSwap()
      swap_time = swapinfo.GetSwappingTime(self.pcie_bandwidth)


      # TODO: try not to choose the same in_trigger
      # TODO: check the in_trigger time is not bigger than swap_out_time
      # check the swap_in_time is not at peak memory time
      in_trigger_index = n_index
      while True:
        in_trigger_index -= 1
        if (n_time-self.tf_tensor_access[in_trigger_index][0]) > swap_time:
          if in_trigger_index in skipped_list:
            continue
          else:
            in_trigger_name = self.tf_tensor_access[in_trigger_index][1]
            print("[DEBUG] %s in_trigger index distances: %d" % (swapinfo.tensor_name, n_index-in_trigger_index))
            skipped_list.append(in_trigger_index)
            break

      swapout_rc = swapinfo.GetSwapoutRc()
      swapout_total_rc = len(swapinfo.access_list)
      swapin_rc = 0
      swapin_total_rc = 1
      if in_trigger_name in tac_f.keys():
        access_indicies = [v for v,_ in tac_f[in_trigger_name].access_list]
        assert (in_trigger_index in access_indicies)
        swapin_rc = len(access_indicies) - access_indicies.index(in_trigger_index) - 1
        swapin_total_rc = len(access_indicies)
      elif in_trigger_name in self.ngpu_tensor_access.keys():
        access_indicies = self.ngpu_tensor_access[in_trigger_name]
        assert (in_trigger_index in access_indicies)
        swapin_rc = len(access_indicies) - access_indicies.index(in_trigger_index) - 1
        swapin_total_rc = len(access_indicies)
      else:
        pass

      fout.write("%s\t%d\t%d\t%s\t%d\t%d\n" % (swapinfo.tensor_name,
                                              swapout_total_rc,
                                              swapout_rc,
                                              in_trigger_name,
                                              swapin_total_rc,
                                              swapin_rc))

      required_saving -= swapinfo.allocated_bytes
      # print("[DEBUG] Max access interval is %d, allocated MB: %f\n" % (swapinfo.max_access_interval, swapinfo.allocated_bytes))
      if required_saving <= 0:
        print("[INFO] Already choose proper swapped out tensors\n")
        # print("[INFO] Total swapping memory : %d\n" % total_swapping)
        break

    fout.close()
    if required_saving > 0:
      print("[ERROR] No enough tensors\n")
      exit(1) """





  def swapping_decision(self, tac_f):
    """
    tac: dict (tensor_name, [occured_indices])
    """

    # Decide the swapping index for multiple occurrences
    swapping_indicies = dict()
    # for swapinfo in tac_f:
    #   self.GetMaxAccessInterval(swapinfo)

    # tac_ff = sorted(tac_f)

    # for swapinfo in tac_ff:
    #   print(swapinfo.tensor_name, swapinfo.swap_start, swapinfo.max_access_interval)


    # for swapinfo in tac_f:
    #   v = swapinfo.access_list
    #   if len(v) == 2:
    #     swapinfo.swap_start = 0
    #     continue

    #   v_std = v[1] - v[0]
    #   v_div = []
    #   v_div.append(1)
    #   prev = v[1]
    #   curr = 0

    #   for i in range(2, len(v)):
    #     curr = v[i]
    #     v_div.append((curr-prev)/v_std)
    #     prev = curr

    #   max_index = v_div.index(max(v_div))
    #   assert(max_index+2) <= len(v)
    #   swapinfo.swap_start = max_index

    # Calculate the swapping index
    for k,v in tac_f.items():
      if len(v) == 2:
        swapping_indicies[k] = 0
        continue

      v_std = v[1] - v[0]
      v_div = []
      v_div.append(1)
      prev = v[1]
      curr = 0
      for i in range(2, len(v)):
        curr = v[i]
        v_div.append((curr-prev)/v_std)
        prev = curr

      max_index = v_div.index(max(v_div))
      assert (max_index+2) <= len(v)
      swapping_indicies[k] = max_index
      # v = v[max_index:max_index+2]
      # del v[max_index+2:]
      # del v[0:max_index]

    tac_ff = sorted(tac_f.items(), key=lambda x: x[1][swapping_indicies[x[0]]+1] -
                                           x[1][swapping_indicies[x[0]]], reverse=True)
    # with open("candidates_sorted.log", 'w') as fout1:
    #   for k,v in tac_ff:
    #     fout1.write(k+': ')
    #     for vv in v:
    #       fout1.write(str(vv)+'\t')
    #     fout1.write('\n')


    tac = collections.OrderedDict()


    for k, v in tac_ff:
      weights_flag = False
      for key_f in self.keys_filter:
        if key_f in k:
          weights_flag = True
          break
      if weights_flag:
        continue

      tac[k] = v


    fout = open(self.metadata_dir+self.swapping_log, 'w')

    required_saving = self.peak_memory - self.mem_limit
    logging.info("Required saving is %f MB" % required_saving)
    # (tensor_name, [indices])
    total_swapping = 0
    swapping_threshold = 2  # Ignore tensor size less than 2MB

    skipped_list = []                   # Store the skipped tensor

    # candidates_num = 0
    # for k,_ in tac.items():
    #   swapped_out = self.tensors[k]
    #   if swapped_out.gpu_mem_allocated < swapping_threshold:
    #     continue
    #   candidates_num += 1

    # print("[INFO] Total available candidates are %d" % candidates_num)
    # candidate_tensors_name = [k for k,v in tac]

    for k,v in tac.items():
      assert (k in self.tensors.keys())
      swapped_out = self.tensors[k]
      assert (swapped_out.gpu_mem_allocated != 0)
      if swapped_out.gpu_mem_allocated < swapping_threshold:
        # del tac[k]
        skipped_list.append(k)
        continue


      k_nindex = v[swapping_indicies[k]+1]            # the index in tensor_access of next use
      assert (k_nindex > self.swapin_trigger_distance)

      # Make sure not choose the swappedOut tensor as swapin trigger
      i = 0
      in_trigger_index = -1
      while True:
        in_trigger_index = k_nindex - self.swapin_trigger_distance - i
        if self.using_tf_tensor_access:
          swapin_trigger_name = self.tf_tensor_access[in_trigger_index][1]
        else:
          swapin_trigger = self.tensor_access[in_trigger_index]
          swapin_trigger_name = swapin_trigger.name()
        if swapin_trigger_name not in tac.keys():
          break
        elif swapin_trigger_name in skipped_list:
          break
        else:
          i += 1

      # Check iff we choose the inputs of the same node
      try:
        assert (in_trigger_index > v[swapping_indicies[k]])
        # print(in_trigger_index - v[swapping_indicies[k]])
      except AssertionError:
        logging.error(k+", swapping_indicies:"+str(v))
        logging.error("%s, swapping_index: %d, next_use_index: %d, in_trigger_index: %d" % (k, v[swapping_indicies[k]], k_nindex, in_trigger_index))
      # check if the in_trigger and swapped_in tensor is used at the same time
      if self.using_tf_tensor_access:
        if self.tf_tensor_access[k_nindex][0] == self.tf_tensor_access[in_trigger_index][0]:
          logging.error("Choose the inputs tensors of the same node!")
      else:
        if self.tensor_accesses[k_nindex][0] == self.tensor_accesses[in_trigger_index][0]:
          logging.error("Choose the inputs tensors of the same node!")

      swapout_ref_count = len(v) - swapping_indicies[k] - 1
      swapin_ref_count = 0
      swapin_total_rc = 1
      if swapin_trigger_name in tac.keys():
        try:
          assert(in_trigger_index in tac[swapin_trigger_name])
        except TypeError:
          print(in_trigger_index)
          print(tac[swapin_trigger_name])
          raise TypeError
        swapin_ref_count = len(tac[swapin_trigger_name]) - tac[swapin_trigger_name].index(in_trigger_index) - 1
        swapin_total_rc = len(tac[swapin_trigger_name])
      elif swapin_trigger_name in self.ngpu_tensor_access.keys():
        try:
          assert (in_trigger_index in self.ngpu_tensor_access[swapin_trigger_name])
        except TypeError:
          raise TypeError
        swapin_ref_count = len(self.ngpu_tensor_access[swapin_trigger_name]) - self.ngpu_tensor_access[swapin_trigger_name].index(in_trigger_index) - 1
        swapin_total_rc = len(self.ngpu_tensor_access[swapin_trigger_name])
      else:
        # print(swapin_trigger_name)
        # raise KeyError
        # in this case, the tensor is on gpu side and only shown up once
        pass

      fout.write("%s\t%d\t%d\t%s\t%d\t%s\t%f\n" % (k, len(v), swapout_ref_count,
                                           swapin_trigger_name,
                                           swapin_total_rc,
                                           swapin_ref_count,
                                           swapped_out.gpu_mem_allocated))
      # TODO: move the swap operation to the completion of a node instead of the start of a node
      required_saving -= swapped_out.gpu_mem_allocated
      total_swapping += swapped_out.gpu_mem_requested
      logging.debug("Choose %s: %f" % (k, swapped_out.gpu_mem_allocated))

      if required_saving <= 0:
        logging.info("Already choose proper swapped out tensors")
        logging.info("Total swapping memory : %d" % total_swapping)
        break

    fout.close()

    if required_saving > 0:
      logging.error("No enough tensors")
      exit(1)

  def InitSwappingDecision(self):
    # At this time, all tensors have been created
    with open (self.metadata_dir+self.swapping_log) as fin:
      for line in fin:
        tmp = line.split()
        assert (len(tmp) == 6)
        k = (tmp[0], int(tmp[2]))
        v = (tmp[3], int(tmp[5]))
        assert (k not in self.swapped_tensors.keys())

        assert (tmp[0] in self.tensors.keys())
        t = self.tensors[tmp[0]]
        t.swapping = True
        t.swapping_ref_count = int(tmp[2])
        t.ref_count_ = int(tmp[1])
        self.swapped_tensors[k] = v

  def InitNodesExecutionTime(self):
    with open(self.metadata_dir+self.nodeInfo_filename) as fin:
      for line in fin:
        tmp = line.split()
        assert (len(tmp) == 3)
        node_name = tmp[0]
        node = Node(tmp[0], int(tmp[1]), int(tmp[2]))
        assert (not self.nodes.__contains__(node_name))
        self.nodes[node_name] = node

    # HACK for '_SINK' node
    sink_node = Node("_SINK")
    assert (not self.nodes.__contains__("_SINK"))
    self.nodes["_SINK"] = sink_node
    logging.info("Total num of Nodes is %d" % len(self.nodes))


  # Initialize node fanout nodes
  def InitNodesFanout(self):
    with open(self.metadata_dir+self.outnode_filename) as fin:
      for line in fin:
        tmp = line.split()
        node_name = tmp[0]

        if not self.nodes.__contains__(node_name):
          logging.debug("Error Node name: %s" % node_name)
          error_node = Node(node_name)
          self.error_nodes[node_name] = error_node

          for i in range(2, len(tmp)):
            fanout_nodename = tmp[i]
            error_node.fanout_nodes.append(fanout_nodename)
          continue

        node = self.nodes[node_name]
        for i in range(1, len(tmp)):
          fanout_nodename = tmp[i]
          if (self.nodes.__contains__(fanout_nodename)):
            node.fanout_nodes.append(self.nodes[fanout_nodename])
          else:
            logging.debug("Node name: %s with Error Fanout Node name: %s" % (node_name, fanout_nodename))
            # continue

  # Initialize node fanin tensors
  def InitNodesFanin(self):
    with open(self.metadata_dir+self.innode_filename) as fin:
      lines = fin.readlines()
      total_length = len(lines)

      i = 0
      while i < total_length:
        tmp = lines[i].split()
        try:
          assert "SrcNode" == tmp[0]
        except AssertionError:
          logging.error("Error line %d with no SrcNode" % i)
          # print("[ERROR] Error line %d with no SrcNode\n" % i)
          raise AssertionError

        # assert
        node_name = tmp[1]
        if not self.nodes.__contains__(node_name):
          if self.error_nodes.__contains__(node_name):
            i = i + int(tmp[2]) + 1
            continue
          else:
            logging.error("Error Node name: %s" % node_name)
            # print ("[ERROR] Error Node name: %s" % node_name)
            raise ValueError

        pending_count = int(tmp[2])

        node = self.nodes[node_name]
        node.pending_count = pending_count

        # logging.debug("%s: %d" % (node_name, pending_count))
        for j in range(pending_count):
          ttmp = lines[i+j+1].split()
          try:
            assert "InputNode" == ttmp[0]
          except AssertionError:
            logging.error("Error line %d with no InputNode" % (i+j))
            raise AssertionError

          fanin_nodename = ttmp[1]
          fanin_id = int(ttmp[2])

          # TODO: hack for resnet nueral network
          # when meeting a switch node, if fanin_id is equal to 1
          # then modify it to 0

          # decrease the pending count if the fanin_node is not in the collection
          if not self.nodes.__contains__(fanin_nodename):
            if self.error_nodes.__contains__(fanin_nodename):
              node.pending_count -= 1
              logging.debug("node %s pending count decrease by one" % node_name)
              continue
            else:
              logging.error("Error Node name: %s" % fanin_nodename)
              raise ValueError

          # Add the output tensor to the corresponding node (ignore the control flow)
          if fanin_id == -1:
            continue
          t_name = fanin_nodename+'_'+str(fanin_id)
          if not self.tensors.__contains__(t_name):
            t = Tensor(fanin_nodename,
                      tid=fanin_id)
            # logging.debug("Init %s tensor" % t.name())
            node.fanin_tensors.append(t)
          # assert not self.tensors.__contains__(t.name())
            self.tensors[t.name()] = t
          else:
            t = self.tensors[t_name]
            if t in node.fanin_tensors:
              # remove redundant fanin_tensor of node
              pass
            else:
              node.fanin_tensors.append(self.tensors[t_name])

        i = i + pending_count + 1

      logging.info("Total num of Tensors is %d" % len(self.tensors))

  def InitTensorSize(self):
    with open(self.metadata_dir+self.tensorsize_filename) as fin:
      lines = fin.readlines()
      total_length = len(lines)

      i = 0
      while i < total_length:
        tmp = lines[i].split()
        try:
          assert "SrcNode" == tmp[0]
        except AssertionError:
          logging.error("Error line %d with no SrcNode" % i)
          raise AssertionError

        # assert
        node_name = tmp[1]
        if not self.nodes.__contains__(node_name):
          if self.error_nodes.__contains__(node_name):
            i = i + int(tmp[2]) + 1
            continue
          else:
            logging.error("Not find error node name: %s" % node_name)
            raise ValueError


        # This output has no control flow
        output_num = int(tmp[2])
        # logging.debug("start init %s tensor size: %d" % (node_name, output_num))
        for j in range(output_num):
          ttmp = lines[i+j+1].split()
          try:
            assert "Output" == ttmp[0]
          except AssertionError:
            logging.error("Error line %d with no Output" % (i+j))
            raise AssertionError

          output_slot = int(ttmp[1])
          # requested_bytes = long(ttmp[2])
          # allocated_bytes = long(ttmp[3])
          requested_bytes = int(ttmp[2])
          allocated_bytes = int(ttmp[3])
          try:
            allocator_name = ttmp[4]
          except IndexError:
            # HACK for some regular tensor has no allocator name
            allocator_name = None
            # print("[DEBUG] IndexError line: %d" % (i+j))
            # raise IndexError
          try:
            allocated_time = int(ttmp[5])
          except IndexError:
            # print("[ERROR] IndexError line: %d\n" % (i+j))
            # raise IndexError
            # as the allocator_name is None
            allocated_time = int(ttmp[4])

          # HACK for numbered from 1 instead of 0 of switch node
          if output_num == 1:
            if output_slot == 1:
              output_slot = 0
              # logging.debug("HACK output slot for %s" % node_name)
          t_name = node_name+'_'+str(output_slot)
          if self.tensors.__contains__(t_name):
            # logging.debug("init %s tensor info" % t_name)
            t = self.tensors[t_name]
            t.requested_bytes = requested_bytes
            t.allocated_bytes = allocated_bytes
            t.allocator_name = allocator_name
            t.allocated_time = allocated_time

            # Initiate the GPU memory usage of this tensor
            t.MemAllocated()
            t.MemRequested()

            # Add to corresponding node
            self.nodes[node_name].outputs.append(t)
          else:
            # record the useless tensor
            # if self.useless_tensors.__contains__(t_name):
            # logging.debug("init %s useless tensor info" % t_name)
            tmp_t = Tensor(node_name, tid=output_slot,
                           requested_bytes=requested_bytes,
                           allocated_bytes=allocated_bytes,
                           allocator_name=allocator_name)

            assert (not self.useless_tensors.__contains__(t_name))
            tmp_t.allocated_time = allocated_time
            self.useless_tensors[t_name] = tmp_t
            # print("[DEBUG] Useless tensor: %s, %f B\n" % (node_name, allocated_bytes))


        i = i + output_num + 1


    # Initiate node mem usage
    sum_mem_allocated = 0.0
    for node in self.nodes.values():
      node.GPUMemAllocated()
      node.GPUMemRequested()
      sum_mem_allocated += node.gpu_mem_allocated

    logging.info("Sum GPU memory allocated is %f MB" % sum_mem_allocated)
    # print("Sum GPU memory allocated is %f\n" % sum_mem_allocated)

  # Init each tensor's inputs info
  def InitReComputationInfo(self):
    for node in self.nodes.values():
      node.InitTensorRPInfo()

  def GetNodeType(self, name):
    node_type = None
    for node_t in self.layer_names:
      if node_t in name:
        node_type = node_t
        break

    return node_type

  def DrawNodeExecTime(self):
    exec_time_limit = 10000
    exec_times = dict()
    node_times = dict()
    for name, node in self.nodes.items():
      # Ignore node name with weights
      l_name = name.lower()
      if self.IsWeights(l_name):
        continue

      # get node type: relu, conv, avgpool, maxpool
      node_type = self.GetNodeType(l_name)
      if node_type == None:
        logging.debug("can not get %s node type" % name)
        continue

      if not exec_times.__contains__(node_type):
        exec_times[node_type] = []
      node_exec_time = node.GetExecTime()
      # remove some weird node_exec_time
      if node_exec_time > exec_time_limit:
        continue
      exec_times[node_type].append(node_exec_time)

      # assert not node_times.__contains__(name)
      if not node_times.__contains__(node_type):
        node_times[node_type] = []
      node_times[node_type].append((name, node_exec_time))
      # node_times[name] = node_exec_time

    draw.plottingdict(exec_times,
                      title=self.metadata_dir[2:-1],
                      save_dir=self.metadata_dir)

    for k, v in node_times.items():
      logging.debug("Node Type: %s" % k)
      logging.debug("Avg exec time: %d" % (np.mean([i[1] for i in v])))
      # for vv in v:
      #   logging.debug("%s exec time: %d" % (vv[0], vv[1]))

  # def InitNodesConnection(self):
  #   with open(self.nodeCon_filename) as fin:
  #     for line in fin:
  #       tmp = line.strip().split('\t')
  #       node_name = tmp[0]
  #       faninNode_num = int(tmp[1])

  #       if not self.nodes.__contains__(node_name):
  #         print ("Error Node name: %s" % node_name)
  #         error_node = Node(node_name)
  #         self.error_nodes.append(error_node)

  #         for i in range(2, len(tmp)):
  #           fanout_nodename = tmp[i]
  #           error_node.fanout_nodes.append(fanout_nodename)
  #         continue

  #       node = self.nodes[node_name]
  #       node.pending_count = faninNode_num
  #       for i in range(2, len(tmp)):
  #         fanout_nodename = tmp[i]
  #         if (self.nodes.__contains__(fanout_nodename)):
  #           node.fanout_nodes.append(self.nodes[fanout_nodename])
  #         else:
  #           print ("Node name: %s with Error Fanout Node name: %s" % (node_name, fanout_nodename))
  #           continue

  #         if not self.faninNodes.__contains__(fanout_nodename):
  #           self.faninNodes[fanout_nodename] = []
  #         else:
  #           try:
  #             assert (node_name not in self.faninNodes[fanout_nodename])
  #           except AssertionError:
  #             pass
  #             # print("%s faninNode: %s\n" % (fanout_nodename, node_name))
  #             # raise AssertionError

  #         self.faninNodes[fanout_nodename].append(node_name)

  #   for error_node in self.error_nodes:
  #     for nodename in error_node.fanout_nodes:
  #       if self.nodes.__contains__(nodename):
  #         self.nodes[nodename].pending_count -= 1
  #         if self.debug:
  #           print("%s pending_count decreases by one!\n" % nodename)


  #   with open("faninNodes.log", 'w') as fout:
  #     for k, v in self.faninNodes.items():
  #       if self.debug:
  #         try:
  #           assert k in self.nodes.keys()
  #         except AssertionError:
  #           print ("%s not in nodes\n" % k)

  #         try:
  #           assert (self.nodes[k].pending_count == len(v))
  #         except AssertionError:
  #           print ("%s pending_count is not equal, %d v.s. %d" % (k,
  #                                                                 self.nodes[k].pending_count,
  #                                                                 len(v)))

  #       fout.write(k+'\t')
  #       for vv in v:
  #         fout.write(vv+'\t')
  #       fout.write('\n')

# ------------------------------------------------------- #
# FOR DEBUG USE

  def CheckSwappingDecision(self):
    assert (len(self.swapped_tensors) != 0)

    for node in self.nodes.values():
      for k,v in self.swapped_tensors.items():
        assert (k[0] in self.tensors.keys())
        kt = self.tensors[k[0]]
        assert (v[0] in self.tensors.keys())
        vt = self.tensors[v[0]]

        if kt in node.fanin_tensors and vt in node.fanin_tensors:
          logging.debug("Error swapping pair %s: %s!" % (k[0], v[0]))
          # print("[DEBUG] Error swapping pair %s: %s!" % (k[0], v[0]))


  # Get Unfinished nodes which are been blocked
  def GetUFBlockedNodes(self):
    unf_nodes = []
    with open(self.metadata_dir+self.swapping_debug_log) as fin:
      for line in fin:
        if 'blocked' in line:
          blocked_node_name = line.split()[1]
          unf_nodes.append(blocked_node_name)
        elif 'Trigger' in line:
          removed_node_name = line.split()[2]
          assert removed_node_name in unf_nodes
          unf_nodes.remove(removed_node_name)
        elif 'swap out' in line:
          swap_out_name = line.split()[4][:-2]
          unf_nodes.append(swap_out_name)
        elif 'swap in' in line:
          if 'DEBUG' in line:
            continue
          swap_in_name = line.split()[4][:-2]
          try:
            assert swap_in_name in unf_nodes
          except AssertionError:
            logging.error("Error swap_in_name: %s" % swap_in_name)
            # print("[ERROR] Error swap_in_name: %s\n" % swap_in_name)
          unf_nodes.remove(swap_in_name)
        else:
          continue

    if len(unf_nodes) == 0:
      return None
    else:
      return unf_nodes

  def CheckFailedNodes(self):
    # if not os.path.exists

    unf_nodes = self.GetUFBlockedNodes()
    failed_nodes = []
    failed_nodes_dict = dict()
    with open(self.metadata_dir+"failed_nodelog.log") as fin:
      for line in fin:
        node_name = line.split(':')[0]
        failed_nodes.append(node_name)

    # Parse the failed node to see why it failed
    with open(self.metadata_dir+"failedNodes_parser.log", 'w') as fout:
      for node_name in failed_nodes:
        assert (node_name in self.nodes.keys())
        # if node is failed due to it's been swapped out or blocked by some swapped out tensors
        if node_name in unf_nodes:
          continue

        node = self.nodes[node_name]
        Nok_t = node.GetNOkfanintensors()
        # for nokt in Nok_t:
        #   nok_nodename = nokt[:-2]
        #   if nok_nodename in failed_nodes or nok_nodename in unf_nodes:
        #     Nok_t.remove(nok_nodename)
        if Nok_t == None:
          continue

        fout.write("[DEBUG] %s: " % node_name)
        for nokt in Nok_t:
          nok_nodename = nokt[:-2]
          if nok_nodename in failed_nodes or nok_nodename in unf_nodes:
            pass
          else:
            fout.write("%s, " % nokt)
        fout.write("\n")
        # if Nok_t != None:
        #   fout.write("[DEBUG] %s: %s\n" % (node_name, str(Nok_t)))
          # print("[DEBUG] "+node_name+': '+str(Nok_t)+'\n')



    # for node_name in failed_nodes:
    #   assert (node_name in self.nodes.keys())
    #   if node_name in unf_nodes:
    #     continue

    #   node = self.nodes[node_name]
    #   for t in node.fanin_tensors:
    #     fanin_node_name = t.name()[:-2]
    #     if fanin_node_name in failed_nodes or fanin_node_name in unf_nodes:
    #       pass
    #     else:
    #       if not failed_nodes_dict.__contains__(node_name):
    #         failed_nodes_dict[node_name] = []
    #       failed_nodes_dict[node_name].append(fanin_node_name)
    #       # print("[DEBUG] %s is failed due to %s which is successful finished!\n" % (node_name, fanin_node_name))

    # with open(self.metadata_dir+"failedNodes_parser.log", 'w') as fout:
    #   for k,v in failed_nodes_dict.items():
    #     fout.write(k+':'+str(v)+'\n')

# END DEBUG Related
# ------------------------------------------------------- #

  def InitEvents(self):
    for node in self.nodes.values():
      if node.pending_count == 0:
        self.events.put(node)

  def PrintResult(self):
    try:
      assert (self.finish_time != 0)
    except AssertionError:
      # log tensors curr ref count
      with open(self.metadata_dir+"failed_rclog.log", 'w') as fout:
        for t in self.tensors.values():
          fout.write(t.name()+": "+str(t.ref_count)+'\n')

      with open(self.metadata_dir+'failed_nodelog.log', 'w') as fout:
        nodes_ = sorted(self.nodes.values(), key=lambda x: x.pending_count)
        for node in nodes_:
          if node in self.finish_nodes.values():
            continue
          fout.write(node.node_name+": "+str(node.pending_count)+'\n')

    logging.info("Final Sim result: %f images/sec" % (self.batch_size * 1000000 / float(self.finish_time)))
    # print("[INFO] Final Sim result: %f images/sec" % (self.batch_size * 1000000 / float(self.finish_time)))




if __name__ == '__main__':
  graph_sim = GraphSim()

  graph_sim.InitNodesExecutionTime()
  graph_sim.InitNodesFanout()
  graph_sim.InitNodesFanin()
  graph_sim.InitTensorSize()
  # graph_sim.DrawNodeExecTime()
  graph_sim.InitReComputationInfo()


  if graph_sim.swapping_test:
    graph_sim.InitSwappingDecision()

  if graph_sim.checkNetArch:
    # for vdnn to get right tensor to be swapped out
    Check_netArch(graph_sim.metadata_dir, graph_sim.nodes, log2file=False)
    exit(0)


  # if graph_sim.debug:
  #   graph_sim.debug_nodeInfo()

  graph_sim.InitEvents()
  graph_sim.EventsEngine()

  # graph_sim.CheckFailedNodes()

  graph_sim.PrintResult()
  graph_sim.GetPeakMemory()

  if not graph_sim.swapping_test:
    graph_sim.access_analysis(graph_sim.tensor_access)
    # graph_sim.InitSwappingDecision()
    # graph_sim.CheckSwappingDecision()
