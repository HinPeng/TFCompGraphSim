import sys

prefix1 = "v/tower_0/cg/"
prefix2 = "v/tower_0/gradients/"
n1 = len(prefix1)
n2 = len(prefix2)

def _help():
  print("Usage: python multigpu_policy $single-gpu-policy $num_gpu")

def _add(s, num_gpu, temp):
  if s.isdigit():
    for i in range(num_gpu):
      temp[i].append(s)
  elif s.startswith(prefix1):
    for i in range(num_gpu):
      if i == 0:
        temp[i].append(s)
      else:
        prefix = ("v_%d/tower_%d/cg/" % (i, i))
        target = prefix + s[n1:]
        temp[i].append(target)
  elif s.startswith(prefix2):
    left = s[n2:]
    if left.startswith(prefix1):
      for i in range(num_gpu):
        if i == 0:
          temp[i].append(s)
        else:
          prefix = ("v_%d/tower_%d/gradients/v_%d/tower_%d/cg/" % (i,i,i,i))
          target = prefix + s[(n1+n2):]
          temp[i].append(target)
    else:
      for i in range(num_gpu):
        prefix = ("v_%d/tower_%d/gradients/" % (num_gpu-1, num_gpu-1))
        target = prefix + s[n2:]
        temp[i].append(target)
  else:
    print("Can not process: %s" % s)
    exit(1)

      

def main():
  if (len(sys.argv) < 3):
    _help()
    exit(1)

  single_gpu_filename = sys.argv[1]
  num_gpu = int(sys.argv[2])
  print("generate from %s to %d gpus" % (single_gpu_filename, num_gpu))

  pos = single_gpu_filename.find("recompute")
  target_dir = single_gpu_filename[:pos]
  s1 = single_gpu_filename[pos:]
  s1 = s1[:s1.find(".")]
  target_file = target_dir+s1+("_%d" % num_gpu-1)+".log"
  fout = open(target_file, 'w')
  with open(single_gpu_filename, 'r') as fin:
    for line in fin:
      tmp = line.split('\t')
      temp = []
      for s in tmp:
        _add(s, num_gpu, temp)
      for tmp in temp:
        fout.write(tmp)
        fout.write("/n")
  fout.close()
  
if __name__ == '__main__':
  main()

