import sys
sys.path.append('../')

import logger
import logging

filename = "log"
out_file = "op_count_per_iter.txt"
key = "iteration starts"
total_iters = 13

with open(filename) as fin:
  op_count = dict()
  for line in fin:
    if line[0:5] == 'entry':
      break
    op_uname = line.split()[-1]
    temp = op_uname.split('_')
    op_name = temp[0]
    op_id = int(temp[1])
    if not op_count.__contains__(op_name):
      op_count[op_name] = op_id
    else:
      if (op_id > op_count[op_name]):
        op_count[op_name] = op_id

  with open(out_file, 'w') as fout:
    for k,v in op_count.items():
      fout.write("%s\t%d\n" % (k, float(v)/total_iters))

# with open(filename) as fin:
#   curr_iter = 0
#   flag = False
#   op_iter = []
#   op_iter_ori = []
#   op_id_start = dict()
#   for line in fin:
#     if line[0:5] == 'entry':
#       break
#     if not flag:
#       if key not in line:
#         op_uname = line.split()[-1]
#         temp = op_uname.split('_')
#         op_name = temp[0]
#         op_id = int(temp[1])
#         op_id_start[op_name] = op_id
#         continue
#       else:
#         flag = True
#         assert int(line.split()[0]) == 0
#         op_iter.append(dict())
#         op_iter_ori.append(dict())
#     else:
#       if key in line:
#         assert (curr_iter+1) == int(line.split()[0])
#         curr_iter += 1
#         logging.debug("%d iteration starts" % curr_iter)
#         op_iter.append(dict())
#         op_iter_ori.append(dict())
#         continue
#       op_uname = line.split()[-1]
#       temp = op_uname.split('_')
#       try:
#         assert len(temp) == 2
#       except AssertionError:
#         logging.info(line)
#         logging.info(temp)
#         exit(1)
#       op_name = temp[0]
#       op_id = int(temp[1])
#       if curr_iter == 0:
#         assert op_id_start.__contains__(op_name)
#         v = op_id
#         op_iter[curr_iter][op_name] = v
#         op_iter_ori[curr_iter][op_name] = op_id
#       else:
#         v = op_id - op_iter_ori[curr_iter-1][op_name]
#         op_iter[curr_iter][op_name] = v
#         op_iter_ori[curr_iter][op_name] = op_id        

#   logging.info("Total iteration: %d" % curr_iter)
#   op_names = op_iter[0].keys()
#   with open(out_file, 'w') as fout:
#     for op_name in op_names:
#       if not op_iter_ori[curr_iter].__contains__(op_name):
#         fout.write("%s\t%d\n" % (op_name, float(op_iter_ori[curr_iter-1][op_name]+1)/total_iters))
#       else:
#         fout.write("%s\t%d\n" % (op_name, float(op_iter_ori[curr_iter][op_name]+1)/total_iters))

  # for op_name in op_names:
  #   logging.debug("%s: %d(0)" % (op_name, op_iter[0][op_name]-op_id_start[op_name]))
  #   num = op_iter[1][op_name]
  #   for i in range(2, curr_iter+1):
  #     try:
  #       assert op_iter[i].__contains__(op_name)
  #     except AssertionError:
  #       logging.error("[Iter%d]: %s" % (i, op_name))
  #       continue
  #     if num != op_iter[i][op_name]:
  #       logging.error("Error %s: %d(%d) %d(%d)" % (op_name,
  #                                                 num,
  #                                                 1,
  #                                                 op_iter[i][op_name],
  #                                                 i))
    # logging.debug("%s: %d" % (op_name, num))


