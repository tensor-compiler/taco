import sys

def parseGFLOPS(fname):
  data = []
  with open(fname, 'r') as f:
    for line in f:
      if 'achieved GFLOPS per node:' in line:
        words = line.strip('\n').strip('.').split(' ')
        nodes = int(words[1])
        gflops = float(words[7])
        data.append((nodes, gflops))
  data.sort(key=lambda x: x[0])
  for d in data:
    print("{},{}".format(d[0], d[1]))

parseGFLOPS(sys.argv[1])
