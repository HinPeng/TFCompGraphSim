try:
  import Queue as q
except ImportError:
  import queue as q

import os

import collections
import copy

import swapInfo

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


    self.log_node_time = True
    self.node_completion_time = "node_comp_time.log"

    self.log_mem = False
    self.log_mem_file = "mem_log.log"

    self.log_ref_count = False
    self.ref_count_file = "ref_count.log"

    self.peak_memory = 0
    self.mem_usage = []

    self.mem_limit = 8 * (1 << 10)
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

    self.swapping_debug = True
    self.swapping_debug_log = "swapping_debug.log"

    self.debug = True
    self.debug_file = "log_debug.log"

    self.swapin_trigger_distance = 20

    self.swap = False

    self.swap_time = True

    # Ignore the weights tensors when making swapping decision
    self.keys_filter = ["kernel"]

    if self.swap:
      self.swapping_test = True
      self.swapping_debug = True
    else:
      self.swapping_test = False
      self.swapping_debug = False

    self.checkNetArch = False

  def EventsEngine(self):

    # Init log file pipe
    if self.debug:
      fout_debug = open(self.metadata_dir+self.debug_file, 'w')

    if self.log_node_time:
      fout_nt = open(self.metadata_dir+self.node_completion_time, 'w')

    if self.log_mem:
      fout_mem = open(self.metadata_dir+self.log_mem_file, 'w')

    if self.log_ref_count:
      fout_ref = open(self.metadata_dir+self.ref_count_file, 'w')

    # swapin_trigger_collection = []
    if self.swapping_test:
      if self.swapping_debug:
        fout_swap = open(self.metadata_dir+self.swapping_debug_log, 'w')

        # assert(len(self.swapped_tensors) != 0)
        # # remove the number of outputs
        # swapin_trigger_collection = [k[0][:-2] for k in self.swapped_tensors.values()]

    access_id = 0

    error_swapin = 0

    while not self.events.empty():
      e = self.events.get()


      # if self.debug:
      #   print("%s with pending_count: %d\n" % (e.node_name, e.pending_count))
      # assert(self.logic_time <= e.start_time)
      try:
        assert(e.pending_count == 0)
      except AssertionError:
        print ("[ERROR] %s pending_count: %d\n" % (e.node_name, e.pending_count))
        raise AssertionError

      # Increase the fanin_tensors ref_count
      # ERROR not init here
      # for t in e.fanin_tensors:
      #   t.ref_count += 1

      # record the node processing id
      e.access_id = access_id

      access_id += 1


      # Only increase the real memory usage on one node's outputs
      # self.mem_usage.append(())
      # curr_mem += e.gpu_mem_allocated
      # if self.log_mem:
      #   fout_mem.write("[DEBUG] curr memory usage is %f\n" % curr_mem)
      #   # print("[DEBUG] curr memory usage is %f" % curr_mem)
      # if curr_mem > self.peak_memory:
      #   self.peak_memory = curr_mem
        # if self.log_mem:
        #   print("[DEBUG] peak memory usage is %f" % self.peak_memory)

        # t_node_name = e.node_name
        # assert t_node_name in self.faninNodes.keys()
        # flag = True
        # with open(self.debug_file, 'w') as fout:
        #   fout.write(t_node_name+" pending_count: "+str(e.pending_count)+'\n')
        #   while (flag):
        #     for faninnode_name in self.faninNodes[t_node_name]:
        #       if not faninnode_name in self.finish_nodes.keys():
        #         pending_count = self.nodes[faninnode_name].pending_count
        #         fout.write(faninnode_name+" pending_count: "+str(pending_count)+'\n')

        #         t_node_name = faninnode_name
        #         flag = False
        #         break

        #     flag = False
        # if self.log_node_time:
        #   fout.close()
        # raise AssertionError

      # check the fanin_tensors to release the memory when the ref_count is 0
      fout_debug.write("-------------------\n")
      for t in e.fanin_tensors:
        t.ref_count -= 1

        for st in self.swapped_tensors.keys():
        # t is in swapped_tensors collection
          if t.name() == st[0]:
            earliest_execution_time = max(e.logic_time, e.start_time)
            if t.ref_count > st[1]:
              # t.node_seq.append(e)
              pass
            elif t.ref_count == st[1]:
              # swap out the tensor now
              swapout_time = t.gpu_mem_requested / self.pcie_bandwidth * self.time_metric

              # TODO: may can not release the memory ASA the swap out operation finish, need CHECK this node's end time
              # TODO: check the end_time of nodes which use this tenesor before the swap out
              t.swapout_time  = earliest_execution_time + swapout_time
              self.mem_usage.append((t.swapout_time, -t.gpu_mem_requested))

              if self.swapping_debug:
                fout_swap.write("[INFO] Start Swap out %s at %d, finish at %d\n" % (t.name(), earliest_execution_time, t.swapout_time))

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
                print("[ERROR] %s has not been triggered to be swapped in yet\n" % t.name())
                # raise AssertionError

              if t.swapin_time > earliest_execution_time:
                print("[DEBUG] swap in overhead here %s : %f\n" % (t.name(),
                                                                  (t.swapin_time-earliest_execution_time)))
                e.logic_time = t.swapin_time
                if self.swapping_debug:
                  fout_swap.write("[DEBUG] swap in overhead here %s : %f\n" % (t.name(),
                                                                  (t.swapin_time-earliest_execution_time)))
        # if t in self.swapped_tensors.keys():
        #   # Process the swapout operation
        #   earliest_execution_time = max(e.logic_time, e.start_time)
        #   if t.ref_count == t.swapout_ref_count:
        #     # swap out the tensor now
        #     swapout_time = t.gpu_mem_requested / self.pcie_bandwidth
        #     t.swapout_time = earliest_execution_time + swapout_time

        #     # TODO: decrease the tensor memory in curr_mem here?
        #     # pass
        #   else:
        #     # the tensor has been swapped out before, check whether has been swapped in yet?
        #     if t.swapin_time > earliest_execution_time:
        #       print("[DEBUG] swap in overhead here %s : %f\n" % (t.name(),
        #                                                          (t.swapin_time-earliest_execution_time)))
        #       e.start_time = t.swapin_time

        # t is in swap_in_trigger collection
        for k,v in self.swapped_tensors.items():
          # for now, if there are multiple swapped_tensors using the same swap_in_trigger
          # not consider the pcie interference yet
          if t.name() == v[0]:
            earliest_execution_time = max(e.logic_time, e.start_time)
            if t.ref_count > v[1]:
              # t.node_seq.append(e)
              pass
            elif t.ref_count == v[1]:
              assert (k[0] in self.tensors.keys())
              swapout_tensor = self.tensors[k[0]]
              if swapout_tensor.swapout_time > earliest_execution_time:
                # in case that this tensor has not been swpped out, but in this case, seems
                # there is no difference to do this swapping
                print("[ERROR] not enough time to swap out: %s\n" % swapout_tensor.name())
                earliest_execution_time = swapout_tensor.swapout_time
              swapin_time = swapout_tensor.gpu_mem_requested / self.pcie_bandwidth * self.time_metric
              swapout_tensor.swapin_time = earliest_execution_time + swapin_time

              swapout_tensor.swapping = False

              if self.swapping_debug:
                fout_swap.write("[INFO] Start Swap in %s at %d, finish at %d\n" % (swapout_tensor.name(),
                                                                                    earliest_execution_time,
                                                                                    swapout_tensor.swapin_time))

              # Can decrease the pending_count here despite the tensor has not beed swapped in yet (but been triggered)
              for node in swapout_tensor.blocking_nodes:
                assert swapout_tensor not in node.ok_fanin_tensors
                node.ok_fanin_tensors.append(swapout_tensor)
                node.pending_count -= 1
                node.tmp_time.append(swapout_tensor.swapin_time)
                if self.swapping_debug:
                  fout_swap.write("[INFO] remove a ref count of %s to %d\n" % (node.node_name, node.pending_count))
                # if self.swapping_debug:
                if node.pending_count == 0:
                  node.logic_time = max(node.tmp_time)
                  self.events.put(node)
                  if self.swapping_debug:
                    fout_swap.write("[INFO] Trigger %s to start\n" % node.node_name)
              # for node in self.nodes.values():
              #   if node in swapout_tensor.node_seq:
              #     continue

              #   if swapout_tensor in node.fanin_tensors:
              #     node.pending_count -= 1
              #     node.tmp_time.append(swapout_tensor.swapin_time)
              #     if node.pending_count == 0:
              #       node.logic_time = max(node.tmp_time)
              #       self.events.put(node)

              self.mem_usage.append((earliest_execution_time, swapout_tensor.gpu_mem_requested))



        # assert t.ref_count >= 0
        # if t.ref_count == 0:
        #   curr_mem -= t.gpu_mem_allocated


      if e.logic_time > e.start_time:
        e.end_time = e.logic_time - e.start_time + e.end_time
        e.start_time = e.logic_time

        # if self.debug:
        #   fout_debug.write("[INFO] %s start at %d, end at %d\n" % (e.name, e.start_time, e.end_time))

      self.mem_usage.append((e.start_time, e.gpu_mem_allocated))


      done_nodes = dict()
      done_nodes.clear()
      # with open(self.metadata_dir+"222.log", 'a') as fout:
      for node in e.fanout_nodes:
        # if node.node_name == "v/tower_0/gradients/v/tower_0/L2Loss_38_grad/mul":
        if node.node_name in done_nodes.keys():
          pass
        else:
          done_nodes[node.node_name] = node

      # if e.node_name == "v/tower_0/gradients/v/tower_0/mul_grad/Mul_1":
      #   for node in done_nodes.values():
      #     if node.node_name == "v/tower_0/gradients/v/tower_0/L2Loss_38_grad/mul":
      #       print("fxxxxxxk")
      # Clear input val when input_tensors ref count came to zero
      for t in e.fanin_tensors:
        try:
          assert t.ref_count >= 0
        except AssertionError:
          print(t.name(), t.ref_count)
          raise AssertionError
        if t.ref_count == 0:
          self.mem_usage.append((e.end_time, -t.gpu_mem_allocated))
          # curr_mem -= t.gpu_mem_allocated

      # ProcessOutputs
      # Init output_tensors of this node initial ref count

      for ut in e.outputs:
        # TODO: remove the multiple same fanout_node

        # for i in range(len(e.fanout_nodes)):
        for fanout_node in done_nodes.values():
        # for fanout_node in e.fanout_nodes:
          # fanout_node = e.fanout_nodes[i]
          # if fanout_node.node_name == "v/tower_0/gradients/v/tower_0/cg/incept_v3_a0/conv7/batchnorm7/FusedBatchNorm_grad/FusedBatchNormGrad":
          #   print(11111)
          # if fanout_node in done_nodes:
          #   if print_flag:
          #     print("222222")
          #   continue
          # done_nodes.append(fanout_node)

        # for fanout_node in e.fanout_nodes:
          if ut in fanout_node.fanin_tensors:
            # Initiate the ref count iff it's not been set to zero yet
            if ut.ref_count == -1:
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

      # ERROR:
      for node_ in e.fanout_nodes:
        node_.pending_count -= 1
        node_.tmp_time.append(e.end_time)
        if node_.pending_count == 0:
          swapping_flag = False
          for ft in node_.fanin_tensors:
            if ft.swapping == True:
              ft.ref_count_ -= 1
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
                  fout_swap.write("[INFO] %s was blocked by %s\n" % (node_.node_name, ft.name()))
              # break
          if swapping_flag == True:
            continue
          node_.logic_time = max(node_.tmp_time)
          self.events.put(node_)



      # assert
      self.finish_nodes[e.node_name] = e

      if self.log_node_time:
        fout_nt.write("[INFO] "+e.node_name+'\t'+str(e.start_time)+'\t'+str(e.end_time)+'\n')

      # if self.events.empty():
      #   self.finish_time = e.end_time
      if e.node_name == '_SINK':
        self.finish_time = e.end_time
        print ("[INFO] finish time is %f s" % (float(self.finish_time)/1000000))
        # print ("[INFO] peak memory usage is %f MB" % self.peak_memory)

      self.events.task_done()


    # reorder the memory access
    # sorted(self.node_access, node_cmp)
    # with open("mem_access.log", 'w') as fout1:
    # for node in self.node_access:
    #   for fanin_tensor in node.fanin_tensors:
    #     self.tensor_access.append(fanin_tensor)
        # fout1.write(fanin_tensor.name()+'\n')

    self.tensor_accesses.sort(key=lambda x: x[0])
    self.tensor_access = [tensor for _, tensor in self.tensor_accesses]


    print("Error swap in number: %d\n" % error_swapin)

    # Clean file pipe
    if self.debug:
      fout_debug.close()

    if self.log_node_time:
      fout_nt.close()

    if self.log_mem:
      fout_mem.close()

    if self.log_ref_count:
      fout_ref.close()

    if self.swapping_test:
      if self.swapping_debug:
        fout_swap.close()


  def GetPeakMemory(self):
    mem_u = sorted(self.mem_usage, mem_cmp)

    total_mem = 0
    peak_mem = 0
    for _, m in mem_u:
      total_mem += m
      peak_mem = max(peak_mem, total_mem)

    self.peak_memory = peak_mem
    print("[INFO] Peak memory is %f MB\n" % peak_mem)

  # Analysis
  def access_analysis(self, tensor_access):
    tac = dict()

    if self.using_tf_tensor_access:
      # Init tensor (include cpu & gpu side) access info
      with open(self.metadata_dir+"tensor_access.txt") as fin:
        line_num = -1
        for line in fin:
          tensor_name = line.split()[0]
          # requested_bytes = int(line.split()[1])
          access_time = int(line.split()[2])
          line_num += 1
          self.tf_tensor_access.append((access_time, tensor_name))
          if not self.tensors.__contains__(tensor_name):
            print("[DEBUG] tf tensor not found in simulator: %s\n" % tensor_name)
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

    # filter the gpu tensors only show up once
    if self.swap_time:
      tac_f = {k:v for k,v in tac.items() if len(v.access_list) > 1}
      self.swapping_decisionTime(tac_f)
    else:
      tac_f = {k:v for k,v in tac.items() if len(v) > 1}
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


  def swapping_decisionTime(self, tac_f):
    for swapinfo in tac_f.values():
      swapinfo.access_list.sort(key=lambda x : x[1])
      swapinfo.DeallocateTime()
      self.GetMaxAccessInterval(swapinfo)


    tac_ff = sorted(tac_f.values())

    peakmem_util = swapInfo.PeakMemory()
    peakmem_util.InitFromSwapInfo(tac_ff)
    peak_mem = peakmem_util.GetPeakMemory()

    print("[INFO] Peak memory usage from tensor access is %d MB\n" % peak_mem)
    print("[INFO] Peak memory usage live tensors number is %d\n" % len(peakmem_util.peakmem_tensors_collec))
    # for name in peakmem_util.peakmem_tensors_collec:
    #   print("[INFO] Peak memory tensor: %s" % name)
    # tac_ff = sorted(tac_f)
    # tac_f.sort()
    

    # for swapinfo in tac_ff:
    #   print(swapinfo.tensor_name, len(swapinfo.access_list), swapinfo.swap_start, swapinfo.max_access_interval)

    swapping_threshold = 2
    skipped_list = [] # seems do not need this
    # tac_f_keys = [swapinfo.tensor_name for _,swapinfo in tac_f]

    required_saving = self.peak_memory - self.mem_limit
    print("[INFO] Required saving is %f\n" % required_saving)

    fout = open(self.metadata_dir+self.swapping_log, 'w')

    for swapinfo in tac_ff:
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
            print("[DEBUG] %s index distances: %d" % (swapinfo.tensor_name, n_index-in_trigger_index))
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
      if required_saving <= 0:
        print("[INFO] Already choose proper swapped out tensors\n")
        # print("[INFO] Total swapping memory : %d\n" % total_swapping)
        break
    
    fout.close()
    if required_saving > 0:
      print("[ERROR] No enough tensors\n")
      exit(1)





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
    print("[INFO] Required saving is %f\n" % required_saving)
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
      # print("%s swapin_index: %d, access timestamp: %d\n" % (k, k_nindex, self.tensor_accesses[k_nindex][0]))
      # print("%s in_trigger_index: %d, access timestamp: %d\n" % (swapin_trigger_name, in_trigger_index, self.tensor_accesses[in_trigger_index][0]))
      try:
        assert (in_trigger_index > v[swapping_indicies[k]])
        # print(in_trigger_index - v[swapping_indicies[k]])
      except AssertionError:
        print(k+", swapping_indicies:"+str(v)+'\n')
        print("%s, swapping_index: %d, next_use_index: %d, in_trigger_index: %d\n" % (k, v[swapping_indicies[k]], k_nindex, in_trigger_index))
      # check if the in_trigger and swapped_in tensor is used at the same time
      if self.using_tf_tensor_access:
        if self.tf_tensor_access[k_nindex][0] == self.tf_tensor_access[in_trigger_index][0]:
          print("[ERROR] Choose the inputs tensors of the same node!\n")
      else:
        if self.tensor_accesses[k_nindex][0] == self.tensor_accesses[in_trigger_index][0]:
          print("[ERROR] Choose the inputs tensors of the same node!\n")

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
      print("[DEBUG] Choose %s: %f" % (k, swapped_out.gpu_mem_allocated))
      if required_saving <= 0:
        print("[INFO] Already choose proper swapped out tensors\n")
        print("[INFO] Total swapping memory : %d\n" % total_swapping)
        break

    fout.close()

    if required_saving > 0:
      print("[ERROR] No enough tensors\n")
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
    print("[INFO] Total num of Nodes is %d\n" % len(self.nodes))


  def InitNodesFanout(self):
    with open(self.metadata_dir+self.outnode_filename) as fin:
      for line in fin:
        tmp = line.split()
        node_name = tmp[0]

        if not self.nodes.__contains__(node_name):
          print ("[DEBUG] Error Node name: %s" % node_name)
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
            print ("[DEBUG] Node name: %s with Error Fanout Node name: %s" % (node_name, fanout_nodename))
            # continue

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
          print("[ERROR] Error line %d with no SrcNode\n" % i)
          raise AssertionError

        # assert
        node_name = tmp[1]
        if not self.nodes.__contains__(node_name):
          if self.error_nodes.__contains__(node_name):
            i = i + int(tmp[2]) + 1
            continue
          else:
            print ("[ERROR] Error Node name: %s" % node_name)
            raise ValueError

        pending_count = int(tmp[2])

        node = self.nodes[node_name]
        node.pending_count = pending_count

        # print(node_name+': '+str(pending_count))
        for j in range(pending_count):
          ttmp = lines[i+j+1].split()
          try:
            assert "InputNode" == ttmp[0]
          except AssertionError:
            print("[ERROR] Error line %d with no InputNode" % (i+j))
            raise AssertionError

          fanin_nodename = ttmp[1]
          fanin_id = int(ttmp[2])

          # decrease the pending count if the fanin_node is not in the collection
          if not self.nodes.__contains__(fanin_nodename):
            if self.error_nodes.__contains__(fanin_nodename):
              node.pending_count -= 1
              print ("[DEBUG] node %s pending count decrease by one" % node_name)
              continue
            else:
              print ("[ERROR] Error Node name: %s" % fanin_nodename)
              raise ValueError

          # Add the output tensor to the corresponding node (Ignore the control flow)
          if fanin_id == -1:
            continue
          t_name = fanin_nodename+'_'+str(fanin_id)
          if not self.tensors.__contains__(t_name):
            t = Tensor(fanin_nodename,
                      tid=fanin_id)
            node.fanin_tensors.append(t)
          # assert not self.tensors.__contains__(t.name())
            self.tensors[t.name()] = t
          else:
            t = self.tensors[t_name]
            if t in node.fanin_tensors:
              # print("[DEBUG] %s with redundant tensor %s\n" % (node.node_name, t_name))
              pass
            else:
              node.fanin_tensors.append(self.tensors[t_name])
            # self.tensors[t_name].total_ref_count += 1
            # the tensor has shown up before
            # pass

        i = i + pending_count + 1

      print("[INFO] Total num of Tensors is %d\n" % len(self.tensors))

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
          print("[ERROR] Error line %d with no SrcNode\n" % i)
          raise AssertionError

        # assert
        node_name = tmp[1]
        if not self.nodes.__contains__(node_name):
          if self.error_nodes.__contains__(node_name):
            i = i + int(tmp[2]) + 1
            continue
          else:
            print ("[ERROR] Error Node name: %s" % node_name)
            raise ValueError

        # This output has no control flow
        output_num = int(tmp[2])
        for j in range(output_num):
          ttmp = lines[i+j+1].split()
          try:
            assert "Output" == ttmp[0]
          except AssertionError:
            print("[ERROR] Error line %d with no Output" % (i+j))
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
          t_name = node_name+'_'+str(output_slot)
          if self.tensors.__contains__(t_name):
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
            # record the no use tensor
            tmp_t = Tensor(node_name, tid=output_slot,
                           requested_bytes=requested_bytes,
                           allocated_bytes=allocated_bytes,
                           allocator_name=allocator_name)

            self.nodes[node_name].no_use_outputs.append(tmp_t)
            # print("[ERROR] Error Output line %d, Error tensor name: %s\n" % ((i+j), t_name))
            # raise ValueError

        i = i + output_num + 1


    # Initiate node mem usage
    sum_mem_allocated = 0.0
    for node in self.nodes.values():
      node.GPUMemAllocated()
      node.GPUMemRequested()
      sum_mem_allocated += node.gpu_mem_allocated

    print("Sum GPU memory allocated is %f\n" % sum_mem_allocated)

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
          print("[DEBUG] Error swapping pair %s: %s!" % (k[0], v[0]))


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
            print("[ERROR] Error swap_in_name: %s\n" % swap_in_name)
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

    print("[INFO] Final Sim result: %f images/sec" % (self.batch_size * 1000000 / float(self.finish_time)))


  # No use now
  def debug_nodeInfo(self):
    with open(self.debug_file, 'w') as fout:
      for node in self.nodes.values():
        fout.write(node.node_name+'\t'+str(node.pending_count)+'\t'+str(len(node.fanout_nodes))+'\n')


if __name__ == '__main__':
  graph_sim = GraphSim()

  graph_sim.InitNodesExecutionTime()
  graph_sim.InitNodesFanout()
  graph_sim.InitNodesFanin()
  graph_sim.InitTensorSize()

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
