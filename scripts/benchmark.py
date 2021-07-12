import argparse
import subprocess
import os

# Arguments specialized to lassen.
def lgCPUArgs():
    return [
      '-ll:ocpu', '2',
      '-ll:othr', '18',
      '-ll:onuma', '1',
      '-ll:csize', '5000',
      '-ll:nsize', '75000',
      '-ll:ncsize', '0',
      '-ll:util', '2',
      '-dm:replicate', '1',
    ]

def lgGPUArgs(gpus):
    return [
      '-ll:ocpu', '1',
      '-ll:othr', '10',
      '-ll:csize', '150000',
      '-ll:util', '1',
      '-dm:replicate', '1',
      '-ll:gpu', str(gpus),
      '-ll:fsize', '15000',
      '-ll:bgwork', '12',
      '-ll:bgnumapin', '1',
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
        # Weak scaling problem size. Keep the memory used per
        # node the same.
        size = int(self.initialProblemSize * pow(procs, 1.0 / 2.0))
        size -= (size % 2)
        return size

    def getCommand(self, procs):
        pass

# Inheritable class for TTMC benchmarks.
class TTMCBench:
    def __init__(self, initialProblemSize):
        self.initialProblemSize = initialProblemSize

    def problemSize(self, procs):
        # Weak scaling problem size. Keep the memory used per
        # node the same.
        size = int(self.initialProblemSize * pow(procs, 1.0 / 3.0))
        size -= (size % 2)
        return size

    def getCommand(self, procs):
        pass

# Inheritable class for TTV benchmarks.
class TTVBench:
    def __init__(self, initialProblemSize):
        self.initialProblemSize = initialProblemSize

    def getgx(self, procs):
        # Asserting that we're running on powers of 2 here.
        ns = nearestSquare(procs)
        if ns ** 2 == procs:
            return ns
        return nearestSquare(procs / 2)

    def problemSize(self, procs):
        # Weak scaling problem size. Keep the memory used per
        # node the same.
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

class SUMMABench(DMMBench):
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
               ['bin/summaMM', '-n', str(psize), '-gx', str(gx), '-gy', str(procs // gx)] + \
               lgCPUArgs()

class SUMMAGPUBench(SUMMABench):
    def __init__(self, initialProblemSize, gpus):
        super().__init__(initialProblemSize)
        self.gpus = gpus

    def getCommand(self, procs):
        psize = self.problemSize(procs)
        gx = self.getgx(procs)
        return lassenHeader(procs) + \
               ['bin/summaMM-cuda', '-n', str(psize), '-gx', str(gx), '-gy', str(procs // gx), '-dm:exact_region', '-tm:untrack_valid_regions'] + \
               lgGPUArgs(self.gpus)

class CannonGPUBench(CannonBench):
    def __init__(self, initialProblemSize, gpus):
        super().__init__(initialProblemSize)
        self.gpus = gpus

    def getCommand(self, procs):
        psize = self.problemSize(procs)
        # We swap the gx and gy here so that x gets a larger extent.
        # This has a performance impact with multiple GPUs per node.
        gy = self.getgx(procs)
        return lassenHeader(procs) + \
               ['bin/cannonMM-cuda', '-n', str(psize), '-gx', str(procs // gy), '-gy', str(gy), \
                '-dm:exact_region', '-tm:untrack_valid_regions'] + \
               lgGPUArgs(self.gpus)

class JohnsonBench(DMMBench):
    def getCommand(self, procs):
        # Assuming that we're running on perfect cubes here.
        psize = self.problemSize(procs)
        gdim = nearestCube(procs)
        return lassenHeader(procs) + \
               ['bin/johnsonMM', '-n', str(psize), '-gdim', str(gdim)] + \
               lgCPUArgs()

class COSMABench(DMMBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        envs = ['env', 'COSMA_OVERLAP_COMM_AND_COMP=ON']
        cosmaDir = os.getenv('COSMA_DIR')
        header = ['jsrun', '-b', 'rs', '-c', '1', '-r', '40', '-n', str(40 * procs)]
        assert(cosmaDir is not None)
        return envs + header + \
               [os.path.join(cosmaDir, 'build/miniapp/cosma_miniapp'), '-r', '10', '-m', psize, '-n', psize, '-k', psize, '--procs_per_node', '40']

class COSMAGPUBench(DMMBench):
    def __init__(self, initialProblemSize, gpus):
        super().__init__(initialProblemSize)
        self.gpus = gpus

    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        cosmaDir = os.getenv('COSMA_DIR')
        header = ['jsrun', '-b', 'rs', '-c', str(40 // self.gpus), '-r', str(self.gpus), '-n', str(self.gpus * procs), '-g', '1']
        assert(cosmaDir is not None)
        return header + \
               [os.path.join(cosmaDir, 'build/miniapp/cosma_miniapp'), '-r', '10', '-m', psize, '-n', psize, '-k', psize, '--procs_per_node', str(self.gpus)]

class SCALAPACKBench(SUMMABench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        gx = self.getgx(procs)
        cosmaDir = os.getenv('COSMA_SCALAPACK_DIR')
        header = ['jsrun', '-b', 'rs', '-c', '10', '-r', '4', '-n', str(4 * procs)]
        return header + \
               [os.path.join(cosmaDir, 'build/miniapp/pxgemm_miniapp'), '-r', '10', '--algorithm', 'scalapack', '-n', psize,
                '-m', psize, '-k', psize, '--block_a', '2048,2048', '--block_b', '2048,2048', '--block_c', '2048,2048',
                '-p', '{},{}'.format(2 * gx, 2 * procs // gx), '--procs_per_node', '4']

class LegateBench(DMMBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        legateDir = os.getenv('LEGATE_DIR')
        assert(legateDir is not None)
        legateNumpyDir = os.getenv('LEGATE_NUMPY_DIR')
        assert(legateNumpyDir is not None)
        return [
            os.path.join(legateDir, 'bin/legate'), os.path.join(legateNumpyDir, 'examples/gemm.py'), '-n', psize, '-p', '64', '-i', '10', '--num_nodes', str(procs),
            '--omps', '2', '--ompthreads', '18', '--nodes', str(procs), '--numamem', '30000', '--eager-alloc-percentage', '1', '--cpus', '1', '--sysmem', '10000',
            '--launcher', 'jsrun', '--cores-per-node', '40', '--verbose',
        ]

class LegateGPUBench(DMMBench):
    def __init__(self, initialProblemSize, gpus):
        super().__init__(initialProblemSize)
        self.gpus = gpus

    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        legateDir = os.getenv('LEGATE_DIR')
        assert(legateDir is not None)
        legateNumpyDir = os.getenv('LEGATE_NUMPY_DIR')
        assert(legateNumpyDir is not None)
        return [
            os.path.join(legateDir, 'bin/legate'), os.path.join(legateNumpyDir, 'examples/gemm.py'), '-n', psize, '-p', '64', '-i', '10',
            '--omps', '1', '--ompthreads', '10', '--nodes', str(procs), '--sysmem', '75000', '--eager-alloc-percentage', '1', '--fbmem', '15000', '--gpus', str(self.gpus), '--verbose',
            '--launcher', 'jsrun', '--cores-per-node', '40',
        ]

class CTFBench(DMMBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        openblasLib = os.getenv('OPENBLAS_LIB_DIR')
        assert(openblasLib is not None)
        ctfDir = os.getenv('CTF_DIR')
        assert(ctfDir is not None)
        envs = ['env', 'LD_LIBRARY_PATH=LD_LIBRARY_PATH:{}'.format(openblasLib)]
        header = ['jsrun', '-b', 'rs', '-c', '10', '-r', '4', '-n', str(4 * procs)]
        return envs + header + \
               [os.path.join(ctfDir, 'bin/matmul'), '-m', psize, '-n', psize, '-k', psize, '-niter', '10', '-sp_A', '1', '-sp_B', '1', '-sp_C', '1', '-test', '0', '--procs_per_node', '4']

class LgTTMCBench(TTMCBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        return lassenHeader(procs) + ['bin/ttmc', '-n', psize, '-pieces', str(procs * 2)] + lgCPUArgs()

class LgTTVBench(TTVBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        gx = self.getgx(procs)
        return lassenHeader(procs) + [
            # Do gx * 2 to account for multiple OMP procs per node.
            'bin/ttv', '-n', psize, '-gx', str(2 * gx), '-gy', str(procs // gx), '-tm:numa_aware_alloc'
        ] + lgCPUArgs()

def executeCmd(cmd):
    cmdStr = " ".join(cmd)
    print("Executing command: {}".format(cmdStr))
    try:
        result = subprocess.run(cmd, capture_output=True)
        print(result.stdout.decode())
        print(result.stderr.decode())
    except Exception as e:
        print("Failed with exception: {}".format(str(e)))

def main():
    benches = [
        # GEMM benchmarks.
        "cannon",
        "cannon-gpu",
        "johnson",
        "cosma",
        "cosma-gpu",
        "summa",
        "summa-gpu",
        "scalapack",
        "legate",
        "legate-gpu",
        "ctf",
        # Higher order tensor benchmarks.
        "ttmc",
        "ttv",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, nargs='+', help="List of node counts to run on")
    parser.add_argument("--bench", choices=benches, type=str)
    parser.add_argument("--size", type=int, help="initial size for benchmarks")
    parser.add_argument("--gpus", type=int, help="number of GPUs for GPU benchmarks")
    args = parser.parse_args()

    if args.bench == "cannon":
        bench = CannonBench(args.size)
    elif args.bench == "cannon-gpu":
        bench = CannonGPUBench(args.size, args.gpus)
    elif args.bench == "johnson":
        bench = JohnsonBench(args.size)
    elif args.bench == "summa":
        bench = SUMMABench(args.size)
    elif args.bench == "summa-gpu":
        bench = SUMMAGPUBench(args.size, args.gpus)
    elif args.bench == "cosma":
        bench = COSMABench(args.size)
    elif args.bench == "cosma-gpu":
        bench = COSMAGPUBench(args.size, args.gpus)
    elif args.bench == "scalapack":
        bench = SCALAPACKBench(args.size)
    elif args.bench == "legate":
        bench = LegateBench(args.size)
    elif args.bench == "legate-gpu":
        bench = LegateGPUBench(args.size, args.gpus)
    elif args.bench == "ctf":
        bench = CTFBench(args.size)
    elif args.bench == "ttmc":
        bench = LgTTMCBench(args.size)
    elif args.bench == "ttv":
        bench = LgTTVBench(args.size)
    else:
        assert(False)
    for p in args.procs:
        executeCmd(bench.getCommand(p))

if __name__ == '__main__':
    main()
