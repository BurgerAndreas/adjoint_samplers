# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import traceback
import hydra
from omegaconf import OmegaConf
import numpy as np
import termcolor

from pathlib import Path
import io

import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import PIL
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.train_loop import train_one_epoch
import adjoint_samplers.utils.train_utils as train_utils
import adjoint_samplers.utils.distributed_mode as distributed_mode
from adjoint_samplers.utils.eval_utils import (
    fig2img,
    build_xyz_from_positions,
    render_xyz_grid,
    render_xyz_to_png,
    plot_energy_distance_hist,
    run_frequency_analysis,
)
from adjoint_samplers.utils.clustering import (
    plot_2d_projection_rmsd,
    cluster_rmsd_hdbscan,
    cluster_mbtr,
    compute_rmsd_matrix,
    cluster_rmsd_density_peaks
)
from adjoint_samplers.utils.logging_utils import name_from_config
from adjoint_samplers.energies.scine_energy import (
    count_atoms_in_molecule,
)

import pickle
import lmdb

# PyTorch runs a short benchmark during the first iteration of your model 
# to find the most efficient convolution algorithm 
# for the specific hardware and input size.
# only use if the input remains constant
cudnn.benchmark = True

# Register Hydra resolvers for molecule-based configs
# These resolvers accept a molecule string and return the number of particles/dim
OmegaConf.register_new_resolver(
    "molecule_n_particles",
    lambda molecule: count_atoms_in_molecule(str(molecule)),
)
OmegaConf.register_new_resolver(
    "molecule_dim",
    lambda molecule: count_atoms_in_molecule(str(molecule)) * 3,
)


def red(content):
    return termcolor.colored(str(content), "red", attrs=["bold"])


def green(content):
    return termcolor.colored(str(content), "green", attrs=["bold"])


def blue(content):
    return termcolor.colored(str(content), "blue", attrs=["bold"])


def cyan(content):
    return termcolor.colored(str(content), "cyan", attrs=["bold"])


def yellow(content):
    return termcolor.colored(str(content), "yellow", attrs=["bold"])


def magenta(content):
    return termcolor.colored(str(content), "magenta", attrs=["bold"])


def load_gt_geometries_from_lmdb(gt_file: str, gt_key) -> list:
    """
    Load ground truth geometries from an LMDB file.
    
    Args:
        gt_file: Path to the LMDB file
        gt_key: Key or list of keys to extract from each entry.
                e.g., "transition_state_positions" or 
                ["reactant_positions", "product_positions"]
    
    Returns:
        List of geometry dicts, where each dict contains the requested tensors
    """
    geometries = []
    
    # Normalize gt_key to a list
    if isinstance(gt_key, str):
        keys = [gt_key]
    else:
        keys = list(gt_key)
    
    with lmdb.open(gt_file, readonly=True, subdir=False, lock=False) as env:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                item = pickle.loads(value)
                geometry = {}
                for k in keys:
                    if k in item:
                        geometry[k] = item[k]
                    else:
                        raise KeyError(f"Key '{k}' not found in LMDB entry with id={key.decode()}")
                # Also store atomic numbers if available
                if "atomic_numbers" in item:
                    geometry["atomic_numbers"] = item["atomic_numbers"]
                if "rxn_id" in item:
                    geometry["rxn_id"] = item["rxn_id"]
                geometries.append(geometry)
    
    return geometries


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg):
    
    #########################################################
    # Fix config
    #########################################################
    try:
        # Update dim and n_particles from energy.molecule if it exists
        # This handles cases where energy.molecule is overridden directly (e.g., energy.molecule=h1c1n1)
        if hasattr(cfg, "energy") and hasattr(cfg.energy, "molecule"):
            molecule_str = str(cfg.energy.molecule)
            # Only update if we have a resolved molecule string (not an interpolation)
            if molecule_str and not molecule_str.startswith("${"):
                n_particles = count_atoms_in_molecule(molecule_str)
                # Temporarily disable struct mode to allow updates
                was_struct = OmegaConf.is_struct(cfg)
                if was_struct:
                    OmegaConf.set_struct(cfg, False)
                cfg.n_particles = n_particles
                cfg.dim = n_particles * 3
                if was_struct:
                    OmegaConf.set_struct(cfg, True)
        
        # Add SLURM job ID to config if it exists in environment
        if "SLURM_JOB_ID" in os.environ:
            cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]
        print(f"SLURM job ID: {cfg.slurm_job_id}")

        print("Instantiating writer with wandb...")
        run_name = name_from_config(cfg)
        writer = train_utils.Writer(
            name=run_name,  # cfg.exp_name
            cfg=cfg,
            is_main_process=distributed_mode.is_main_process(),
        )
        
        # use the wandb run id if available
        runid = "loc"
        if wandb.run is not None:
            runid = wandb.run.id
            
        # fix paths
        if not os.path.exists(cfg.scratch_dir):
            print(f"Scratch directory {cfg.scratch_dir} does not exist, using default ./results")
            cfg.scratch_dir = str(Path("./results").resolve())
        # for checkpoints and distributed
        cfg.shared_dir = cfg.scratch_dir + "/" + runid + "_" + str(cfg.slurm_job_id) + "/" + cfg.shared_dir
        os.makedirs(cfg.shared_dir, exist_ok=True)
        # for plots and results
        cfg.output_dir = cfg.scratch_dir + "/" + runid + "_" + str(cfg.slurm_job_id) + "/" + cfg.output_dir
        os.makedirs(cfg.output_dir, exist_ok=True)
        eval_dir = Path(cfg.output_dir + "/eval_figs")
        os.makedirs(eval_dir, exist_ok=True)

        train_utils.setup(cfg)
        # print(str(cfg))

        device = "cuda"

        # fix the seed for reproducibility
        seed = cfg.seed + distributed_mode.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load ground truth geometries if specified
        gt_geometries = None
        if cfg.gt_file is not None:
            print(f"Loading ground truth geometries from {cfg.gt_file}...")
            gt_geometries = load_gt_geometries_from_lmdb(cfg.gt_file, cfg.gt_key)
            print(f"Loaded {len(gt_geometries)} ground truth geometries")

        print("Instantiating energy...")
        energy = hydra.utils.instantiate(cfg.energy, device=device)

        print("Instantiating source...")
        source = hydra.utils.instantiate(cfg.source, device=device)

        print("Instantiating model...")
        ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
        controller = hydra.utils.instantiate(cfg.controller).to(device)
        sde = ControlledSDE(ref_sde, controller).to(device)

        if "corrector" in cfg:
            print("Instantiating corrector & corrector matcher...")
            corrector = hydra.utils.instantiate(cfg.corrector).to(device)
            corrector_matcher = hydra.utils.instantiate(cfg.corrector_matcher, sde=sde)
        else:
            corrector = corrector_matcher = None

        print("Instantiating grad of costs...")
        grad_term_cost = hydra.utils.instantiate(
            cfg.term_cost,
            corrector=corrector,
            energy=energy,
            ref_sde=ref_sde,
            source=source,
        )

        print("Instantiating adjoint matcher...")
        adjoint_matcher = hydra.utils.instantiate(
            cfg.adjoint_matcher,
            grad_term_cost=grad_term_cost,
            sde=sde,
        )

        print("Instantiating optimizer...")
        if corrector is not None:
            optimizer = torch.optim.Adam(
                [
                    {"params": controller.parameters(), **cfg.adjoint_matcher.optim},
                    {"params": corrector.parameters(), **cfg.corrector_matcher.optim},
                ]
            )
        else:
            optimizer = torch.optim.Adam(
                controller.parameters(),
                **cfg.adjoint_matcher.optim,
            )

        # Instantiate learning rate scheduler
        lr_schedule = None
        if hasattr(cfg, "scheduler") and cfg.scheduler is not None:
            scheduler_type = cfg.scheduler.get("type", None)
            if scheduler_type == "cosine":
                # Cosine annealing with warmup
                warmup_epochs = cfg.scheduler.get("warmup_epochs", 0)
                min_lr = cfg.scheduler.get("min_lr", 0.0)
                T_max = cfg.num_epochs - warmup_epochs
                if warmup_epochs > 0:
                    warmup_scheduler = LinearLR(
                        optimizer,
                        start_factor=1e-8 / cfg.adjoint_matcher.optim.lr,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    )
                    cosine_scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=T_max,
                        eta_min=min_lr,
                    )
                    lr_schedule = SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, cosine_scheduler],
                        milestones=[warmup_epochs],
                    )
                else:
                    lr_schedule = CosineAnnealingLR(
                        optimizer,
                        T_max=cfg.num_epochs,
                        eta_min=min_lr,
                    )
            elif scheduler_type == "plateau":
                # Reduce on plateau
                lr_schedule = ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=cfg.scheduler.get("factor", 0.5),
                    patience=cfg.scheduler.get("patience", 10),
                    min_lr=cfg.scheduler.get("min_lr", 1e-6),
                    verbose=True,
                )

        checkpoint_path = Path(cfg.checkpoint or f"{cfg.shared_dir}/checkpoint_latest.pt")
        checkpoint_path.parent.mkdir(exist_ok=True)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = train_utils.load(
                checkpoint,
                optimizer,
                controller,
                adjoint_matcher,
                corrector=corrector,
                corrector_matcher=corrector_matcher,
            )
            # Note: Not wrapping this in a DDP since we don't differentiate through SDE simulation.
        else:
            start_epoch = 0

        if cfg.distributed:
            controller = torch.nn.parallel.DistributedDataParallel(
                controller, device_ids=[cfg.gpu], find_unused_parameters=True
            )
            if corrector is not None:
                corrector = torch.nn.parallel.DistributedDataParallel(
                    corrector, device_ids=[cfg.gpu], find_unused_parameters=True
                )

        print("Instantiating evaluator...")
        evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)

        total_batches = cfg.num_epochs * cfg.train_itr_per_epoch

        print(f"Starting from {start_epoch}/{cfg.num_epochs} epochs...")
        for epoch in range(start_epoch, cfg.num_epochs):
            stage = train_utils.determine_stage(epoch, cfg)
            global_batch_start = epoch * cfg.train_itr_per_epoch
            end_batch_idx = min(
                global_batch_start + cfg.train_itr_per_epoch - 1, total_batches - 1
            )
            beta_epoch = train_utils.get_beta(
                cfg.temperature,
                end_batch_idx,
                total_batches,
                cfg.train_itr_per_epoch,
            )

            matcher, model = {
                "adjoint": (adjoint_matcher, controller),
                "corrector": (corrector_matcher, corrector),
            }.get(stage)

            loss = train_one_epoch(
                matcher,
                model,
                source,
                optimizer,
                lr_schedule,
                epoch,
                device,
                cfg,
                global_batch_start,
                total_batches,
            )

            writer.log(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_buffer_size": len(matcher.buffer),
                    "beta": beta_epoch,
                    "epoch": epoch,
                },
                step=epoch,
            )

            print(
                "[{0} | {1}] {2}".format(
                    cyan(f"{stage:<7}"),
                    yellow(f"ep={epoch:04}"),
                    green(f"loss={loss:.4f}"),
                )
            )

            # Eval epoch according to the frequency
            # otherwise eval at the end of adjoint matching
            if "eval_freq" in cfg:
                eval_this_epoch = epoch % cfg.eval_freq == 0
            else:
                eval_this_epoch = train_utils.is_last_am_epoch(epoch, cfg)

            if distributed_mode.is_main_process() and eval_this_epoch:
                # eval only after adjoint training
                if stage == "adjoint":
                    beta_eval = train_utils.get_beta(
                        cfg.temperature,
                        end_batch_idx,
                        total_batches,
                        cfg.train_itr_per_epoch,
                    )
                    n_gen_samples = 0
                    x1_list = []
                    while n_gen_samples < cfg.num_eval_samples:
                        B = min(
                            cfg.eval_batch_size, cfg.num_eval_samples - n_gen_samples
                        )
                        x0 = source.sample(
                            [
                                B,
                            ]
                        ).to(device)
                        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)

                        # model samples
                        x0, x1 = sdeint(
                            sde,
                            x0,
                            timesteps,
                            only_boundary=True,
                        )
                        x1_list.append(x1)
                        n_gen_samples += x1.shape[0]
                        print(
                            "Generated {} samples (total: {}/{})".format(
                                x1.shape[0],
                                n_gen_samples,
                                cfg.num_eval_samples,
                            )
                        )

                    samples = torch.cat(x1_list, dim=0)

                    # Compare to reference samples
                    eval_dict = evaluator(samples)

                    if "hist_img" in eval_dict:
                        fname = eval_dir / f"gen_epoch_{epoch}.png"
                        eval_dict["hist_img"].save(fname)
                        print(f"Saved generated samples to\n {fname.resolve()}")

                    #####################
                    # Plot histograms for energy and interatomic distances
                    #####################
                    distances_full = plot_energy_distance_hist(
                        samples,
                        energy,
                        epoch,
                        eval_dir,
                        eval_dict,
                        beta=beta_eval,
                        energy_min=getattr(cfg, "energy_min", None),
                        energy_max=getattr(cfg, "energy_max", None),
                        dist_min=getattr(cfg, "dist_min", None),
                        dist_max=getattr(cfg, "dist_max", None),
                        vert_lines=getattr(cfg, "vert_lines", None),
                    )

                    #####################
                    # Cluster samples by RMSD to approximate modes
                    #####################
                    if hasattr(energy, "n_particles") and hasattr(
                        energy, "n_spatial_dim"
                    ):
                        if cfg.num_samples_clustering is None:
                            cluster_samples = samples
                        else:
                            cluster_samples = samples[: cfg.num_samples_clustering]

                        rmsd_matrix = compute_rmsd_matrix(cluster_samples, energy, cfg, eval_dir, eval_dict, tag="rmsd")
                        medoid_indices_rmsd, cluster_labels_rmsd = cluster_rmsd_hdbscan(
                            cluster_samples,
                            energy,
                            cfg,
                            eval_dir,
                            eval_dict,
                            rmsd_matrix=rmsd_matrix,
                            tag="rmsd",
                        )

                        # Plot 2D projections with HDBSCAN clusters
                        max_samples_proj = getattr(cfg, "max_samples_projection", 5000)
                        plot_2d_projection_rmsd(
                            cluster_samples,
                            energy,
                            eval_dir,
                            eval_dict,
                            cfg=cfg,
                            tag="hdbscan",
                            cluster_labels=cluster_labels_rmsd,
                            n_samples_max=max_samples_proj,
                            rmsd_matrix=rmsd_matrix,
                        )

                        # Run frequency analysis on HDBSCAN medoids
                        run_frequency_analysis(
                            medoid_indices_rmsd,
                            cluster_samples,
                            energy,
                            eval_dict,
                            tag="hdbscan",
                            beta=beta_eval,
                        )

                        # Cluster using density peaks algorithm (reuse RMSD matrix)
                        cluster_centers_dp, cluster_labels_dp = cluster_rmsd_density_peaks(
                            cluster_samples,
                            energy,
                            cfg,
                            eval_dir,
                            eval_dict,
                            tag="density_peaks",
                            beta=beta_eval,
                            rmsd_matrix=rmsd_matrix,
                        )

                        # Plot 2D projections with density peaks clusters
                        plot_2d_projection_rmsd(
                            cluster_samples,
                            energy,
                            eval_dir,
                            eval_dict,
                            cfg=cfg,
                            tag="density_peaks",
                            cluster_labels=cluster_labels_dp,
                            n_samples_max=max_samples_proj,
                            rmsd_matrix=rmsd_matrix,
                        )

                        # Run frequency analysis on density peaks cluster centers
                        run_frequency_analysis(
                            cluster_centers_dp,
                            cluster_samples,
                            energy,
                            eval_dict,
                            tag="density_peaks",
                            beta=beta_eval,
                        )

                    writer.log(eval_dict, step=epoch)

                if cfg.save_ckpt:
                    print("Saving checkpoint ... ")
                    train_utils.save(
                        epoch,
                        cfg,
                        optimizer,
                        controller,
                        adjoint_matcher,
                        corrector=corrector,
                        corrector_matcher=corrector_matcher,
                        ckpt_dir=Path(cfg.shared_dir),
                    )

    except Exception as e:
        # This way we have the full traceback in the log.  otherwise Hydra
        # will handle the exception and store only the error in a pkl file
        print(traceback.format_exc(), file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()