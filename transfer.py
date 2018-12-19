medatata_dir = "./inception3_115_k40/"
swapping_dec = "swapping_decision.log"

with open(medatata_dir+"swap_policy.txt", 'w') as fout:
  with open(medatata_dir+swapping_dec) as fin:
    for line in fin:
      tmp = line.split()
      swapout_num = int(tmp[1]) - int(tmp[2])
      swapin_num = int(tmp[4]) - int(tmp[5])
      fout.write("%s\t%d\t%d\t%s\t%d\t%d\n" % (tmp[0], int(tmp[1]), swapout_num, tmp[3], int(tmp[4]), swapin_num))
