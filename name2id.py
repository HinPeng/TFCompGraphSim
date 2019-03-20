import os

import logger
import logging

node2id_filename = "1node2id.txt"
recomp_filename = "recompute.log"
swap_filename = "swapping_decision.log"
rswap_filename = "r_swap.log"
rrecomp_filename = "r_recompute.log"


node2id = dict()

def tensorname2id(name):
  # NOTE: assert slot is less than 10
  if name[-2] != '_':
    # on-demand trigger
    return "24:0"
  node_name = name[:-2]
  slot = name[-1]
  assert node_name in node2id.keys()
  id_name = str(node2id[node_name]) + ':' + slot
  return id_name

  

def NodeToId(metadir):
  rp_trans = False
  sp_trans = False

  if os.path.exists(metadir+recomp_filename):
    rp_trans = True
  if os.path.exists(metadir+swap_filename):
    sp_trans = True
  
  with open(metadir+node2id_filename) as fin:
    for line in fin:
      tmp = line.split()
      assert len(tmp) == 2
      node_name = tmp[0]
      node_id = int(tmp[1])
      assert node_name not in node2id.keys()
      node2id[node_name] = node_id

  
  # for checking
  t_ids = []
  i_ids = []

  if sp_trans:
    fout_s = open(metadir+rswap_filename, 'w')
    with open(metadir+swap_filename) as fin:
      for line in fin:
        tmp = line.split()
        assert len(tmp) == 6
        t_idname = tensorname2id(tmp[0])
        in_tri_idname = tensorname2id(tmp[3])

        fout_s.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (t_idname,
                                                  tmp[1],
                                                  tmp[2],
                                                  in_tri_idname,
                                                  tmp[4],
                                                  tmp[5]))
      fout_s.close()

  if rp_trans:
    fout_r = open(metadir+rrecomp_filename, 'w')
    with open(metadir+recomp_filename) as fin:
      for line in fin:
        tmp = line.split()
        # index: 0, 3, 6-
        assert len(tmp) >= 6
        t_idname = tensorname2id(tmp[0])
        in_tri_idname = tensorname2id(tmp[3])

        # for check
        t_id = int(t_idname[:-2])
        # px: not accurate yet, just an indication
        for t_id_ in t_ids:
          if abs(t_id-t_id_) == 1:
            logging.info("Continuous number: %d, %d" % (t_id, t_id_))
            continue
            # break
            # exit(1)
        t_ids.append(int(t_idname[:-2]))

        fout_r.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (t_idname,
                                                  tmp[1],
                                                  tmp[2],
                                                  in_tri_idname,
                                                  tmp[4],
                                                  tmp[5]))
        for i in range(6, len(tmp)):
          fout_r.write("%s\t" % tensorname2id(tmp[i]))
          i_id = int(tensorname2id(tmp[i])[:-2])
          if i_id not in i_ids:
            i_ids.append(i_id)
        fout_r.write("\n")

    fout_r.close()

  inters = list(set(t_ids).intersection(set(i_ids)))
  if len(inters) != 0:
    print("error!\n")
    for i in inters:
      print(i)
    

if __name__ == "__main__":
  # pass
  metadir = "./vgg16_226_p100/"
  # metadir = "./inception3_160_p100/"
  NodeToId(metadir)
      

  
    