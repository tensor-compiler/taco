#!/usr/bin/env python3

import argparse
import subprocess
import os

# Arguments specialized to lassen.
def lgCPUArgs(othrs=18):
    args = [
      '-ll:ocpu', '2',
      '-ll:othr', str(othrs),
      '-ll:onuma', '1',
      '-ll:csize', '5000',
      '-ll:nsize', '75000',
      '-ll:ncsize', '0',
      '-ll:util', '2',
      '-dm:replicate', '1',
    ]
    if (othrs != 18):
        args += ['-ll:ht_sharing', '0']
    return args

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

# Inheritable class for MTTKRP benchmarks.
class MTTKRPBench:
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

class LgCOSMABench(DMMBench):
    def __init__(self, initialProblemSize):
        super().__init__(initialProblemSize)
        # Mapping from processor / "rank" count to gx,gy,gz decompositions and px,py.
        self.decomp = {
          4: (1, 2, 2, 2, 2),
          8: (2, 2, 2, 2, 4),
          16: (2, 2, 4, 4, 4),
          32: (2, 4, 4, 4, 8),
          64: (4, 4, 4, 8, 8),
          128: (4, 4, 8, 8, 16),
          256: (4, 8, 8, 16, 16),
        }

    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        # Treat each NUMA node as a processor.
        assert(2 * procs in self.decomp)
        decomp = self.decomp[2 * procs]
        return lassenHeader(procs) + \
               ['bin/cosma', '-n', psize, '-gx', str(decomp[0]), '-gy', str(decomp[1]), '-gz', str(decomp[2])] + \
               lgCPUArgs()

class LgCOSMAGPUBench(LgCOSMABench):
    def __init__(self, initialProblemSize, gpus):
        super().__init__(initialProblemSize)
        self.gpus = gpus

    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        # Treat each GPU as a processor.
        assert(self.gpus * procs in self.decomp)
        decomp = self.decomp[self.gpus * procs]
        return lassenHeader(procs) + \
               ['bin/cosma-cuda', '-n', psize, '-gx', str(decomp[0]), '-gy', str(decomp[1]), '-gz', str(decomp[2]), '-px', str(decomp[3]), '-py', str(decomp[4])] + \
               lgGPUArgs(self.gpus)

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

class LgTTMCGPUBench(TTMCBench):
    def __init__(self, initialProblemSize, gpus):
        super().__init__(initialProblemSize)
        self.gpus = gpus

    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        return lassenHeader(procs) + ['bin/ttmc-cuda', '-n', psize, '-pieces', str(procs * self.gpus)] + lgGPUArgs(self.gpus)

class CTFTTMCBench(TTMCBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        openblasLib = os.getenv('OPENBLAS_LIB_DIR')
        assert(openblasLib is not None)
        ctfDir = os.getenv('CTF_DIR')
        assert(ctfDir is not None)
        envs = ['env', 'LD_LIBRARY_PATH=LD_LIBRARY_PATH:{}'.format(openblasLib)]
        header = ['jsrun', '-b', 'rs', '-c', '10', '-r', '4', '-n', str(4 * procs)]
        return envs + header + \
               [os.path.join(ctfDir, 'bin/ttmc'), '-n', psize, '-procsPerNode', '4']

class CTFMTTKRPBench(MTTKRPBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        openblasLib = os.getenv('OPENBLAS_LIB_DIR')
        assert(openblasLib is not None)
        ctfDir = os.getenv('CTF_DIR')
        assert(ctfDir is not None)
        envs = ['env', 'LD_LIBRARY_PATH=LD_LIBRARY_PATH:{}'.format(openblasLib)]
        header = ['jsrun', '-b', 'rs', '-c', '10', '-r', '4', '-n', str(4 * procs)]
        return envs + header + \
               [os.path.join(ctfDir, 'bin/mymttkrp'), '-n', psize, '-procsPerNode', '4']

class LgMTTKRPBench(MTTKRPBench):
    def __init__(self, initialProblemSize):
        super().__init__(initialProblemSize)
        # Seems like I have to hard code the problem sizes...
        self.sizes = {
          1: (1, 1, 1),
          2: (2, 1, 1),
          4: (2, 2, 1),
          8: (2, 2, 2),
          16: (4, 2, 2),
          32: (4, 4, 2),
          64: (4, 4, 4),
          128: (8, 4, 4),
          256: (8, 8, 4),
        }

    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        assert(procs in self.sizes)
        x, y, z = self.sizes[procs]
        return lassenHeader(procs) + [
            'bin/mttkrp', '-n', psize, '-gx', str(2 * x), '-gy', str(y), '-gz', str(z), '-tm:numa_aware_alloc', '-lg:eager_alloc_percentage', '50',
        ] + lgCPUArgs()

class LgGPUMTTKRPBench(LgMTTKRPBench):
    def __init__(self, initialProblemSize, gpus):
        super().__init__(initialProblemSize)
        self.gpus = gpus

    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        assert(procs in self.sizes)
        x, y, z = self.sizes[procs]
        return lassenHeader(procs) + [
            'bin/mttkrp-cuda', '-n', psize, '-gx', str(2 * x), '-gy', str(2 * y), '-gz', str(z), '-lg:eager_alloc_percentage', '50',
        ] + lgGPUArgs(self.gpus)

class LgTTVBench(TTVBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        gx = self.getgx(procs)
        return lassenHeader(procs) + [
            # Do gx * 2 to account for multiple OMP procs per node.
            'bin/ttv', '-n', psize, '-gx', str(2 * gx), '-gy', str(procs // gx), '-tm:numa_aware_alloc'
        ] + lgCPUArgs(othrs=76) # Run with more openmp threads than normal to make use of SMT.

class LgTTVGPUBench(TTVBench):
    def __init__(self, initialProblemSize, gpus):
        super().__init__(initialProblemSize)
        self.gpus = gpus
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        gx = self.getgx(procs)
        return lassenHeader(procs) + [
            'bin/ttv-cuda', '-n', psize, '-gx', str(2 * gx), '-gy', str(2 * procs // gx),
        ] + lgGPUArgs(self.gpus)

class CTFTTVBench(TTVBench):
    def getCommand(self, procs):
        psize = str(self.problemSize(procs))
        openblasLib = os.getenv('OPENBLAS_LIB_DIR')
        assert(openblasLib is not None)
        ctfDir = os.getenv('CTF_DIR')
        assert(ctfDir is not None)
        envs = ['env', 'LD_LIBRARY_PATH=LD_LIBRARY_PATH:{}'.format(openblasLib)]
        header = ['jsrun', '-b', 'rs', '-c', '10', '-r', '4', '-n', str(4 * procs)]
        return envs + header + \
               [os.path.join(ctfDir, 'bin/ttv'), '-n', psize, '-procsPerNode', '4']

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
    # Default problem sizes.
    dmm = 8192
    dmmgpu = 20000
    ttmc = 1024
    ttmcgpu = 1500
    ttv = 2000
    ttvgpu = 1750
    mttkrp = 768
    mttkrpgpu = 1500

    benches = {
        # GEMM benchmarks.
        "cannon": dmm,
        "cannon-gpu": dmmgpu,
        "johnson": dmm,
        "cosma": dmm,
        "cosma-gpu": dmmgpu,
        "lgcosma": dmm,
        "lgcosma-gpu": dmmgpu,
        "summa": dmm,
        "summa-gpu": dmmgpu,
        "scalapack": dmm,
        "legate": dmm,
        "legate-gpu": dmmgpu,
        "ctf": dmm,
        # Higher order tensor benchmarks.
        "ttmc": ttmc,
        "ttmc-gpu": ttmcgpu,
        "ctf-ttmc": ttmc,
        "ttv": ttv,
        "ttv-gpu": ttvgpu,
        "ctf-ttv": ttv,
        "mttkrp": mttkrp,
        "mttkrp-gpu": mttkrpgpu,
        "ctf-mttkrp": mttkrp,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, nargs='+', help="List of node counts to run on", default=[1])
    parser.add_argument("--bench", choices=benches.keys(), type=str)
    parser.add_argument("--size", type=int, help="initial size for benchmarks")
    parser.add_argument("--gpus", type=int, help="number of GPUs for GPU benchmarks", default=4)
    args = parser.parse_args()

    size = args.size
    if size is None:
      size = benches[args.bench]

    if args.bench == "cannon":
        bench = CannonBench(size)
    elif args.bench == "cannon-gpu":
        bench = CannonGPUBench(size, args.gpus)
    elif args.bench == "johnson":
        bench = JohnsonBench(size)
    elif args.bench == "summa":
        bench = SUMMABench(size)
    elif args.bench == "summa-gpu":
        bench = SUMMAGPUBench(size, args.gpus)
    elif args.bench == "cosma":
        bench = COSMABench(size)
    elif args.bench == "cosma-gpu":
        bench = COSMAGPUBench(size, args.gpus)
    elif args.bench == "lgcosma":
        bench = LgCOSMABench(size)
    elif args.bench == "lgcosma-gpu":
        bench = LgCOSMAGPUBench(size, args.gpus)
    elif args.bench == "scalapack":
        bench = SCALAPACKBench(size)
    elif args.bench == "legate":
        bench = LegateBench(size)
    elif args.bench == "legate-gpu":
        bench = LegateGPUBench(size, args.gpus)
    elif args.bench == "ctf":
        bench = CTFBench(size)
    elif args.bench == "ttmc":
        bench = LgTTMCBench(size)
    elif args.bench == "ttmc-gpu":
        bench = LgTTMCGPUBench(size, args.gpus)
    elif args.bench == "ctf-ttmc":
        bench = CTFTTMCBench(size)
    elif args.bench == "ttv":
        bench = LgTTVBench(size)
    elif args.bench == "ttv-gpu":
        bench = LgTTVGPUBench(size, args.gpus)
    elif args.bench == "ctf-ttv":
        bench = CTFTTVBench(size)
    elif args.bench == "mttkrp":
        bench = LgMTTKRPBench(size)
    elif args.bench == "mttkrp-gpu":
        bench = LgGPUMTTKRPBench(size, args.gpus)
    elif args.bench == "ctf-mttkrp":
        bench = CTFMTTKRPBench(size)
    else:
        assert(False)
    for p in args.procs:
        executeCmd(bench.getCommand(p))

if __name__ == '__main__':
    main()
