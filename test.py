from swapInfo import SwapInfo

class test():
  def __init__(self):
    self.swapInfos = []

    swapinfo1 = SwapInfo("a")
    swapinfo1.swap_start = 0
    swapinfo1.max_access_interval = 12

    swapinfo2 = SwapInfo("b")
    swapinfo2.swap_start = 2
    swapinfo2.max_access_interval = 123

    swapinfo3 = SwapInfo("c")
    swapinfo3.swap_start = 1
    swapinfo3.max_access_interval = 123

    swapinfo4 = SwapInfo("d")
    swapinfo4.swap_start = 1
    swapinfo4.max_access_interval = 125

    self.swapInfos.append(swapinfo1)
    self.swapInfos.append(swapinfo2)
    self.swapInfos.append(swapinfo3)
    self.swapInfos.append(swapinfo4)

  def Test(self):
    self.swapInfos.sort()
    # sis = sorted(self.swapInfos)

    # for swapinfo in sis:
    for swapinfo in self.swapInfos:
      print(swapinfo.tensor_name, swapinfo.swap_start, swapinfo.max_access_interval)

if __name__ == '__main__':
  Test = test()
  Test.Test()

    