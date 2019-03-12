node2id_filename = "1node2id.txt"
target_filename = "recompute.log"
rtarget_filename = "r_recompute.log"

node2id = dict()

def tensorname2id(name):
  # NOTE: assert slot is less than 10
  assert name[-2] == '_'
  node_name = name[:-2]
  slot = name[-1]
  assert node_name in node2id.keys()
  id_name = str(node2id[node_name]) + ':' + slot
  return id_name

  

def NodeToId(metadir):
  
  with open(metadir+node2id_filename) as fin:
    for line in fin:
      tmp = line.split()
      assert len(tmp) == 2
      node_name = tmp[0]
      node_id = int(tmp[1])
      assert node_name not in node2id.keys()
      node2id[node_name] = node_id

  fout = open(metadir+rtarget_filename, 'w')
  with open(metadir+target_filename) as fin:
    for line in fin:
      tmp = line.split()
      # index: 0, 3, 6-
      assert len(tmp) >= 6
      t_idname = tensorname2id(tmp[0])
      in_tri_idname = tensorname2id(tmp[3])

      fout.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (t_idname,
                                               tmp[1],
                                               tmp[2],
                                               in_tri_idname,
                                               tmp[4],
                                               tmp[5]))
      for i in range(6, len(tmp)):
        fout.write("%s\t" % tensorname2id(tmp[i]))
      fout.write("\n")

  fout.close()

if __name__ == "__main__":
  # pass
  metadir = "./vgg16_226_p100/"
  NodeToId(metadir)
      

  
    