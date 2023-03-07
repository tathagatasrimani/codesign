import ast
from cfg.staticfg.builder import CFGBuilder
from hls import HardwareModel
from util_sim import sim

path = '/Users/PatrickMcEwen/high_level_synthesis/venv/codesign/src/cfg/benchmarks/'
benchmark = 'simple'

def main():
    global path, benchmark
    cfg = CFGBuilder().build_from_file('main.c', path + 'nonai_models/' + benchmark + '.py')
    cfg.build_visual(path + 'pictures/' + benchmark, 'jpeg', show = True)
    print(cfg.functioncfgs)
    hw = HardwareModel(0, 0)
    hw.hw_allocated['Add'] = 2
    hw.hw_allocated['Regs'] = 3
    hw.hw_allocated['Mult'] = 1
    hw.hw_allocated['Sub'] = 1
    hw.hw_allocated['FloorDiv'] = 1
    hw.hw_allocated['Gt'] = 1
    print(sim(cfg, hw.hw_allocated, first=True))


if __name__ == "__main__":
    main()