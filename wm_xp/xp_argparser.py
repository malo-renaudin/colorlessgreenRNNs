import argparse

xp_parser = argparse.ArgumentParser(add_help=False)

#General arguments
xp_parser.add_argument("--name", type=str, help="experiment name")
xp_parser.add_argument("--data", type=str, help ="path to training data")
xp_parser.add_argument("--cuda", action="store_true", help="use CUDA")

#config argument
xp_parser.add_argument("--hidden_dims", type=int, nargs="+", default=[1024, 512, 252])
xp_parser.add_argument("--num_heads", type=int, nargs="+", default=[1, 2, 4]) 
xp_parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.01, 0.001])
xp_parser.add_argument("--gumbel_softmax", type=lambda x: x.lower()=='true', nargs="+", default=[True, False])

#grid search argument
xp_parser.add_argument('--grid_search_script', type=str, default='/scratch2/mrenaudin/colorlessgreenRNNs/wm_xp/gs_eval.sh')
#eval argument
xp_parser.add_argument('--nounpp', type=str, default = "//scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt")
xp_parser.add_argument('--eval_script', type=str, default='/scratch2/mrenaudin/colorlessgreenRNNs/wm_xp/tests/tests.py',
                   help='Path to the evaluation script')