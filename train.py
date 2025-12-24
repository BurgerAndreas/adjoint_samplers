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

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.train_loop import train_one_epoch
import adjoint_samplers.utils.train_utils as train_utils
import adjoint_samplers.utils.distributed_mode as distributed_mode
from adjoint_samplers.utils.eval_utils import (
    interatomic_dist,
    fig2img,
    build_xyz_from_positions,
    render_xyz_grid,
    render_xyz_to_png,
    cluster_intradist,
    cluster_rmsd,
    cluster_mbtr,
    run_frequency_analysis,
    _get_atom_types_from_energy,
    plot_energy_distance_hist,
    plot_2d_projection,
)
from adjoint_samplers.utils.logging_utils import name_from_config
from adjoint_samplers.energies.scine_energy import (
    count_atoms_in_molecule,
)


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


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg):
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

        train_utils.setup(cfg)
        # print(str(cfg))

        device = "cuda"

        # fix the seed for reproducibility
        seed = cfg.seed + distributed_mode.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

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
        lr_schedule = None  # TODO(ghliu) add scheduler
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

        checkpoint_path = Path(cfg.checkpoint or "checkpoints/checkpoint_latest.pt")
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

        # Add SLURM job ID to config if it exists in environment
        if "SLURM_JOB_ID" in os.environ:
            cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]
        print(f"SLURM job ID: {cfg.slurm_job_id}")

        print("Instantiating writer...")
        run_name = name_from_config(cfg)
        writer = train_utils.Writer(
            name=run_name,  # cfg.exp_name
            cfg=cfg,
            is_main_process=distributed_mode.is_main_process(),
        )

        print("Instantiating evaluator...")
        eval_dir = Path("eval_figs")
        eval_dir.mkdir(exist_ok=True)
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
                    # Cluster samples by pairwise distance vectors to approximate modes
                    #####################
                    if hasattr(energy, "n_particles") and hasattr(
                        energy, "n_spatial_dim"
                    ):
                        if cfg.num_samples_clustering is None:
                            cluster_samples = samples
                        else:
                            cluster_samples = samples[: cfg.num_samples_clustering]
                        distances_cluster = interatomic_dist(
                            cluster_samples, energy.n_particles, energy.n_spatial_dim
                        ).detach()

                        medoid_indices_intradist, cluster_labels_intradist = (
                            cluster_intradist(
                                distances_cluster,
                                cluster_samples,
                                energy,
                                cfg,
                                eval_dir,
                                eval_dict,
                                tag="intradist",
                            )
                        )

                        # Plot 2D projections
                        max_samples_proj = getattr(cfg, "max_samples_projection", 5000)
                        plot_2d_projection(
                            cluster_samples,
                            energy,
                            eval_dir,
                            eval_dict,
                            tag="intradist",
                            cluster_labels=cluster_labels_intradist,
                            n_samples_max=max_samples_proj,
                        )

                        if getattr(cfg, "cluster_by_rmsd", False):
                            medoid_indices_ordered_rmsd = cluster_rmsd(
                                cluster_samples,
                                energy,
                                cfg,
                                eval_dir,
                                eval_dict,
                                tag="rmsd",
                            )
                            run_frequency_analysis(
                                medoid_indices_ordered_rmsd,
                                cluster_samples,
                                energy,
                                eval_dict,
                                tag="rmsd",
                                beta=beta_eval,
                            )

                        if getattr(cfg, "cluster_by_mbtr", False):
                            medoid_indices_mbtr = cluster_mbtr(
                                cluster_samples,
                                energy,
                                cfg,
                                eval_dir,
                                eval_dict,
                                tag="mbtr",
                            )
                            run_frequency_analysis(
                                medoid_indices_mbtr,
                                cluster_samples,
                                energy,
                                eval_dict,
                                tag="mbtr",
                                beta=beta_eval,
                            )

                        run_frequency_analysis(
                            medoid_indices_intradist,
                            cluster_samples,
                            energy,
                            eval_dict,
                            tag="intradist",
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
                    )

    except Exception as e:
        # This way we have the full traceback in the log.  otherwise Hydra
        # will handle the exception and store only the error in a pkl file
        print(traceback.format_exc(), file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
