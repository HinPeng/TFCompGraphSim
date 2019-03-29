import logger
import logging
import os

invalid_swap_filename = "invalid_swap.txt"
swap_filename = "r_swap.log"
new_filename = "rr_swap.log"


def main(meta_dir):
  fout = open(meta_dir+new_filename, 'w')

  invalid_swap = []
  with open(meta_dir+invalid_swap_filename) as fin:
    for line in fin:
      logging.info("Invalid swap: %s" % line)
      invalid_swap.append(line.strip())

  with open(meta_dir+swap_filename) as fin:
    for line in fin:
      tmp = line.split()
      if tmp[0] in invalid_swap:
        logging.info("Remove %s" % tmp[0])
        continue
      logging.info("Write %s to file" % tmp[0])
      if tmp[0] in invalid_swap:
        logging.error("WTF")
      fout.write(line)

  fout.close()

if __name__ == "__main__":
  # metadir = "./vgg16_226_p100/"
  metadir = "./inception3_160_p100/"
  main(metadir)