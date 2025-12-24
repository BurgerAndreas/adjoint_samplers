import os
import omegaconf
import numpy as np
import torch
from pathlib import Path
import hashlib

# e.g. inference kwargs
IGNORE_OVERRIDES = [
    "use_wandb",
]

# some stuff is not relevant for the checkpoint
# allows to load checkpoint with the same name
IGNORE_OVERRIDES_CHECKPOINT = [
    "save_freq",
    "save_ckpt",
]

REPLACE = {
    "+": "",
    "experiment=": "",
    "experiment": "",
    "temperature.": "t",
    "schedule": "",
    "linear": "lin",
    "decay_start_epoch": "start",
    "beta_": "b",
    "energy.": "",
    "newton_raphson_then_norm_outside_ts": "nrnorm",
    "newton_raphson": "nr",
    "norm_gad_outside_ts": "norm",
}

REPLACE_HUMAN = {}


def name_from_config(args: omegaconf.DictConfig, is_checkpoint_name=False) -> str:
    """Generate a name for the model based on the config.
    Name is intended to be used as a file name for saving checkpoints and outputs.
    """
    try:
        # model name format:
        mname = ""
        # override format: 'pretrain_dataset=bridge,steps=10,use_wandb=False'
        override_names = ""
        # print(f'Overrides: {args.override_dirname}')
        if args.override_dirname:
            for arg in args.override_dirname.split(","):
                # make sure we ignore some overrides
                if np.any([ignore in arg for ignore in IGNORE_OVERRIDES]):
                    continue
                # ignore some more overrides for checkpoint names
                if is_checkpoint_name:
                    if np.any(
                        [ignore in arg for ignore in IGNORE_OVERRIDES_CHECKPOINT]
                    ):
                        continue
                override_names += " " + arg
    except Exception as error:
        print("\nname_from_config() failed:", error)
        print("args:", args)
        raise error
    for key, value in REPLACE.items():
        override_names = override_names.replace(key, value)
    if is_checkpoint_name or len(override_names) > 40:
        # Use a short, stable hash for checkpoint base name
        raw = override_names.strip()
        override_names = f"ck-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"
    else:
        # Make wandb name human readable
        for key, value in REPLACE_HUMAN.items():
            override_names = override_names.replace(key, value)

    # logger.info("name_from_config() mname: %s, override_names: %s", mname, override_names)
    _name = mname + override_names
    print(f"Name{' checkpoint' if is_checkpoint_name else ''}: {_name}")
    return _name
