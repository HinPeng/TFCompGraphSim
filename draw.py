import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FREQ = 1
def plotting_actime(node_exec, access_int, title=None):
  assert len(node_exec) == len(access_int)
  length = len(node_exec)
  data_x = list(map(lambda x: x*FREQ, range(length)))
  plt.plot(data_x, node_exec,  color='r', linewidth=0.5, label='NT')
  plt.plot(data_x, access_int, color='b', linewidth=0.5, label='AI')
  if title == None:
    title = "1"
  plt.title(title)
  # plt.xlabel("")
  plt.ylabel("Time (us)")
  plt.legend(loc='best')
  plt.savefig('./%s.pdf' % title, format='pdf')
  plt.clf()