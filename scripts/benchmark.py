import argparse
import subprocess

# Arguments specialized to lassen.
def lgCPUArgs():
    return [
      '-ll:ocpu', '1',
      '-ll:othr', '20',
      '-ll:csize', '50000',
      '-ll:util', '2',
      '-dm:replicate', '1',
    ]

def lgGPUArgs(gpus):
    return lgCPUArgs() + [
        '-ll:gpu', str(gpus),
        '-ll:fsize', '15000',
    ]

def lassenHeader(procs):
    return [
        'jsrun',
        '-b', 'none',
        '-c', 'ALL_CPUS',
        '-g', 'ALL_GPUS',
        '-r', '1',
        '-n', str(procs),
    ]

def nearestSquare(max):
    val = 1
    while True:
        sq = val * val
        if sq > max:
            return val - 1
        if sq == max:
            return val
        val += 1

def nearestCube(max):
    val = 1
    while True:
        sq = val * val * val
        if sq > max:
            return val - 1
        if sq == max:
            return val
        val += 1

# Inheritable class for matrix multiply benchmarks.
class DMMBench:
    def __init__(self, initialProblemSize):
        self.initialProblemSize = initialProblemSize

    def problemSize(self, procs):
        # Weak scaling problem size.
        size = int(self.initialProblemSize * pow(procs, 1.0 / 3.0))
        size -= (size % 2)
        return size

    def getCommand(self, procs):
        pass

class CannonBench(DMMBench):
    def getgx(self, procs):
        # Asserting that we're running on powers of 2 here.
        ns = nearestSquare(procs)
        if ns ** 2 == procs:
            return ns
        return nearestSquare(procs / 2)

    def getCommand(self, procs):
        psize = self.problemSize(procs)
        gx = self.getgx(procs)
        return lassenHeader(procs) + \
               ['bin/cannonMM', '-n', str(psize), '-gx', str(gx), '-gy', str(procs // gx)] + \
               lgCPUArgs()

class JohnsonBench(DMMBench):
    def getCommand(self, procs):
        # Assuming that we're running on perfect cubes here.
        psize = self.problemSize(procs)
        gdim = nearestCube(procs)
        return lassenHeader(procs) + \
               ['bin/johnsonMM', '-n', str(psize), '-gdim', str(gdim)] + \
               lgCPUArgs()

def executeCmd(cmd):
    cmdStr = " ".join(cmd)
    print("Executing command: {}".format(cmdStr))
    try:
        subprocess.run(cmd)
    except Exception as e:
        print("Failed with exception: {}".format(str(e)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, nargs='+', help="List of node counts to run on")
    parser.add_argument("--bench", choices=["cannon", "johnson"], type=str)
    parser.add_argument("--size", type=int, help="initial size for benchmarks")
    args = parser.parse_args()

    if args.bench == "cannon":
        bench = CannonBench(args.size)
    elif args.bench == "johnson":
        bench = JohnsonBench(args.size)
    else:
        assert(False)
    for p in args.procs:
        executeCmd(bench.getCommand())

if __name__ == '__main__':
    main()