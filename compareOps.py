def readOps(filepath):
  op_record = dict()
  with open(filepath, 'r') as fin:
    for line in fin:
      line = line.strip('\n').split('\t')
      op_record[line[0]] = line[1:]
  return op_record

def compareOps(ops1, ops2):
  diff_t = dict()
#   diff_op = dict()
  for t in ops1.keys():
    if t not in ops2.keys():
      diff_t[t] = len(ops1[t])
      continue
    if len(ops1[t]) != len(ops2[t]):
      print "DIFF_OPS_LENGTH %s" % t      
      print "OPS1: ", str(ops1[t])
      print "OPS2: ", str(ops2[t])
    else:
      flag = False
      for op in ops1[t]:
        if op not in ops2[t]:
          if not flag:
            print "COMPARE %s" % t
            flag = True            
          print op
        #   if not diff_op.__contains__(t):
        #     diff_op[t] = [op]
        #   else:
        #     diff_op[t].append(op)   
  if len(diff_t):
    for t in diff_t.keys():
      print "%s %d" % (t, diff_t[t])

if __name__ == "__main__":
  metadata_dir = './resnet152_86_p100/'
  ops_record1 = readOps(metadata_dir+'2_ops.log')
  ops_record2 = readOps(metadata_dir+'3_ops.log')
  # compareOps(ops_record1, ops_record2)
  # print "-----------------------------------------"
  compareOps(ops_record2, ops_record1)