import json
import os 
import argparse

def add_training_flag(data_path, training_ratio=0.8):
    episodes_path = os.path.join(data_path, 'meta/episodes.jsonl')
    with open(episodes_path, 'r') as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    training_split = int(len(lines) * training_ratio)
    for i, line in enumerate(lines):
        if i < training_split:
            line['training'] = True
        else:
            line['training'] = False
    with open(episodes_path, 'w') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
            
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data directory",
    )
    arg_parser.add_argument(
        "--training_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data to total data",
    )
    args = arg_parser.parse_args()
    add_training_flag(args.data_path, args.training_ratio)
    