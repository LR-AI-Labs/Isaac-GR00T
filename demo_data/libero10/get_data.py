import os
import gr00t
import json
from huggingface_hub import snapshot_download

WORKDIR = os.path.dirname(os.path.dirname(gr00t.__file__))
os.chdir(WORKDIR)

if __name__ == "__main__":
    dataset_name = "IPEC-COMMUNITY/libero_10_no_noops_lerobot"
    local_dir = "demo_data/libero10"

    # Download the dataset
    dataset_path = snapshot_download(
        repo_id=dataset_name,
        local_dir=local_dir,
        repo_type="dataset",    
    )
    # Add the training flag to the episode metadata
    os.system(f"python gr00t/data/add_training_flag.py --data_path {local_dir}")
    # Add modality.json file to meta
    modality = {
        "state": {
            "state": {
                "start": 0,
                "end": 8
            }
        },
        "action": {
            "action": {
                "start": 0,
                "end": 7
            }
        },
        "video": {
            "image": {
                "original_key": "observation.images.image"
            },
            "wrist_image": {
                "original_key": "observation.images.wrist_image"
            }
        },
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        }
    }
    
    with open(os.path.join(local_dir, "meta/modality.json"), "w") as f:
        json.dump(modality, f, indent=4)