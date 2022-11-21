import os
import argparse
import pstats

parser = argparse.ArgumentParser("time profile")
# Environment
parser.add_argument("--run-script",
                    '-r',
                    type=str,
                    help="python script and function you want to analyse")
parser.add_argument("--top_num",
                    '-t',
                    type=int,
                    default=10,
                    help="python script and function you want to analyse")

args = parser.parse_args()
script = args.run_script
os.system(f"python -m cProfile -o result.out {script}")
p = pstats.Stats("result.out")
rst = p.strip_dirs().sort_stats("time", "name").print_stats(int(args.top_num))
print(rst)
os.system("rm result.out")
