import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FREQ = 1
colors = ['r', 'b', 'g', 'y']
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

def plottingdict(time_dict, title=None, save_dir=None):
  if len(time_dict) == 0:
    return

  c = 0
  for k,v in time_dict.items():
    length = len(v)
    data_x = list(map(lambda x: x*FREQ, range(length)))
    plt.plot(data_x, v, color=colors[c], linewidth=0.5, label=k)
    c += 1

  
  if title == None:
    title = "1"
  plt.title(title)
  plt.xlabel("Layer instance")
  plt.ylabel("Time (us)")
  plt.legend(loc='best')
  plt.savefig("%s%s.pdf" % (save_dir, title), format='pdf')
  plt.clf()