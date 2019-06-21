# convert innodes to outnodes graph structure

import logger
import logging

dst_dir = "./resnet50_64_eager/"
innodes_file = "1_innodes.txt"

with open(dst_dir+innodes_file) as fin:
  lines = fin.readlines
  total_length = len(lines)
  
  i = 0
  while i < total_length:
    tmp = lines[i].split()
    try:
      assert "SrcNode" == tmp[0]
    except AssertionError:
      logging.error("Error line %s with no SrcNode" % i)
      raise AssertionError

    node_name = tmp[1]
    pending_count = int(tmp[2])

    for j in range(pending_count):
      ttmp = lines[i+j+1].split()
      try:
        assert "InputNode" == tmp[0]
      except AssertionError:
        logging.error("Error line %d with no InputNode" % (i+j))
        raise AssertionError

      fanin_nodename = ttmp[1]
      fanin_id = int(ttmp[2])
    