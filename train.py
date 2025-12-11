# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import traceback
import hydra
import numpy as np
import termcolor
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

from pathlib import Path
import io

import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import PIL
import wandb

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.train_loop import train_one_epoch
import adjoint_samplers.utils.train_utils as train_utils
import adjoint_samplers.utils.distributed_mode as distributed_mode
from adjoint_samplers.utils.frequency_analysis import analyze_frequencies_torch
from adjoint_samplers.utils.eval_utils import (
    interatomic_dist,
    fig2img,
    build_xyz_from_positions,
    render_xyz_grid,
    render_xyz_to_png,
)


cudnn.benchmark = True


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

        print("Instantiating writer...")
        writer = train_utils.Writer(
            name=cfg.exp_name,
            cfg=cfg,
            is_main_process=distributed_mode.is_main_process(),
        )

        print("Instantiating evaluator...")
        eval_dir = Path("eval_figs")
        eval_dir.mkdir(exist_ok=True)
        evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)

        print(f"Starting from {start_epoch}/{cfg.num_epochs} epochs...")
        for epoch in range(start_epoch, cfg.num_epochs):
            stage = train_utils.determine_stage(epoch, cfg)

            matcher, model = {
                "adjoint": (adjoint_matcher, controller),
                "corrector": (corrector_matcher, corrector),
            }.get(stage)

            loss = train_one_epoch(
                matcher, model, source, optimizer, lr_schedule, epoch, device, cfg
            )

            writer.log(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_buffer_size": len(matcher.buffer),
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
                        print(f"Saved generated samples to {fname.resolve()}")

                    #####################
                    # Plot histograms for energy and interatomic distances
                    #####################
                    print("Plotting energy and distance histograms...")
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                    # Energy histogram
                    energy_values = energy.eval(samples).detach().cpu().numpy()
                    axes[0].hist(energy_values, bins=50, density=True)
                    axes[0].set_xlabel("Energy")
                    axes[0].set_ylabel("Density")
                    axes[0].set_title(f"Energy Distribution (epoch {epoch})")
                    axes[0].grid(True)

                    # Interatomic distance histogram (if applicable)
                    if hasattr(energy, "n_particles") and hasattr(
                        energy, "n_spatial_dim"
                    ):
                        distances_full = interatomic_dist(
                            samples, energy.n_particles, energy.n_spatial_dim
                        ).detach()
                        distances = distances_full.cpu().numpy().reshape(-1)
                        axes[1].hist(distances, bins=50, density=True)
                        axes[1].set_xlabel("Interatomic Distance")
                        axes[1].set_ylabel("Density")
                        axes[1].set_title(
                            f"Interatomic Distance Distribution (epoch {epoch})"
                        )
                        axes[1].grid(True)

                    else:
                        axes[1].text(
                            0.5,
                            0.5,
                            "N/A\n(Not a particle system)",
                            ha="center",
                            va="center",
                            transform=axes[1].transAxes,
                        )
                        axes[1].set_title("Interatomic Distance Distribution")

                    plt.tight_layout()
                    fig.canvas.draw()  # ensure matplotlib renders before conversion
                    energy_dist_hist_img = fig2img(fig)
                    # fname = eval_dir / f"energy_dist_hist.png"
                    # energy_dist_hist_img.save(
                    #     fname
                    # )
                    # print(f"Saved energy dist hist to {fname.resolve()}")

                    # if writer.writer is not None:
                    fname = eval_dir / "energy_dist_hist.png"
                    energy_dist_hist_img.save(fname)
                    print(f"Saved energy dist hist to {fname.resolve()}")

                    # also log the histogram image to wandb
                    eval_dict["energy_dist_hist"] = wandb.Image(energy_dist_hist_img)
                    plt.close(fig)

                    #####################
                    # Cluster samples by pairwise distance vectors to approximate modes
                    #####################
                    if hasattr(energy, "n_particles") and hasattr(
                        energy, "n_spatial_dim"
                    ):
                        # Cluster samples by pairwise distance vectors to approximate modes
                        dist_features = (
                            distances_full.cpu()
                            .numpy()
                            .reshape(distances_full.shape[0], -1)
                        )
                        dbscan = DBSCAN(
                            eps=cfg.dbscan.eps, min_samples=cfg.dbscan.min_samples
                        )
                        labels = dbscan.fit_predict(dist_features)
                        uniq, counts = np.unique(labels, return_counts=True)

                        # compute medoid indices per cluster (exclude noise label -1)
                        medoid_indices = []
                        for label in uniq:
                            if label == -1:
                                continue
                            idxs = np.where(labels == label)[0]
                            if len(idxs) == 0:
                                continue
                            sub_feat = dist_features[idxs]
                            pdists = pairwise_distances(sub_feat)
                            medoid_local = np.argmin(pdists.sum(axis=1))
                            medoid_indices.append(idxs[medoid_local])

                        # order clusters by size (desc) and cap to 9
                        label_counts = {int(k): int(v) for k, v in zip(uniq, counts)}
                        sorted_labels = [
                            lab
                            for lab, _ in sorted(
                                label_counts.items(), key=lambda kv: kv[1], reverse=True
                            )
                            if lab != -1
                        ]
                        medoid_indices_ordered = []
                        for lab in sorted_labels:
                            idxs = np.where(labels == lab)[0]
                            if len(idxs) == 0:
                                continue
                            sub_feat = dist_features[idxs]
                            pdists = pairwise_distances(sub_feat)
                            medoid_local = np.argmin(pdists.sum(axis=1))
                            medoid_indices_ordered.append(int(idxs[medoid_local]))
                            if len(medoid_indices_ordered) >= 9:
                                break
                        num_clusters = len(medoid_indices_ordered)

                        # render medoid representatives as molecules if available
                        if num_clusters > 0:
                            print(f"Rendering {num_clusters} medoid representatives...")
                            medoid_xyz = []
                            for idx in medoid_indices_ordered:
                                pos = (
                                    samples[idx]
                                    .detach()
                                    .reshape(energy.n_particles, energy.n_spatial_dim)
                                    .cpu()
                                    .numpy()
                                )
                                # if 2D positions, pad a zero z-dim for 3D visualization
                                if pos.shape[1] == 2:
                                    pos = np.concatenate(
                                        [pos, np.zeros((pos.shape[0], 1))], axis=1
                                    )
                                xyz = build_xyz_from_positions(
                                    pos, atom_type="C", center=True
                                )
                                medoid_xyz.append(xyz)

                            # Render first 3 medoids individually
                            for i, xyz_str in enumerate(medoid_xyz[:3]):
                                png_bytes = render_xyz_to_png(
                                    xyz_str, width=600, height=600
                                )
                                medoid_img = PIL.Image.open(io.BytesIO(png_bytes))
                                if medoid_img.mode != "RGB":
                                    medoid_img = medoid_img.convert("RGB")
                                fname = eval_dir / f"medoid_{i}.png"
                                medoid_img.save(fname)
                                print(f"Saved medoid {i} to {fname.resolve()}")
                                eval_dict[f"medoid_{i}"] = wandb.Image(medoid_img)

                            # render medoid grid of up to 9 medoids
                            if len(medoid_xyz) == 0:
                                print(
                                    "Warning: medoid_xyz is empty, skipping grid rendering"
                                )
                            else:
                                medoid_grid_img = render_xyz_grid(
                                    medoid_xyz,
                                    ncols=3,
                                    width=900,
                                    height=900,
                                )
                                # fname = eval_dir / f"dbscan_medoid_grid.png"
                                # medoid_grid_img.save(fname)
                                # print(f"Saved medoid grid to {fname}")
                                # if writer.writer is not None:
                                fname = eval_dir / "dbscan_medoid_grid.png"
                                medoid_grid_img.save(fname)
                                print(f"Saved medoid grid to {fname.resolve()}")
                                eval_dict["dbscan_medoid_grid"] = wandb.Image(
                                    medoid_grid_img
                                )

                            # frequency analysis
                            freq_minima = 0
                            freq_ts = 0
                            freq_other = 0
                            freq_samples = 0
                            if energy.n_spatial_dim in (2, 3):
                                for idx in medoid_indices_ordered:
                                    atoms = ["x"] * energy.n_particles
                                    hess = energy.hessian_E(
                                        samples[idx : idx + 1]
                                    ).detach()[0]
                                    if energy.n_spatial_dim == 3:
                                        freq = analyze_frequencies_torch(
                                            hessian=hess,
                                            cart_coords=samples[idx],
                                            atomsymbols=atoms,
                                            ev_thresh=-1e-6,
                                        )
                                        neg_num = int(freq["neg_num"])
                                    elif energy.n_spatial_dim == 2:
                                        h_flat = hess.reshape(
                                            samples[idx].numel(), samples[idx].numel()
                                        )
                                        h_flat = (h_flat + h_flat.T) / 2.0
                                        eigvals = torch.linalg.eigvalsh(h_flat)
                                        neg_num = int((eigvals < -1e-6).sum().item())
                                    freq_samples += 1
                                    if neg_num == 0:
                                        freq_minima += 1
                                    elif neg_num == 1:
                                        freq_ts += 1
                                    else:
                                        freq_other += 1
                                if freq_samples > 0:
                                    eval_dict["freq_minima"] = freq_minima
                                    eval_dict["freq_transition_states"] = freq_ts
                                    eval_dict["freq_other"] = freq_other
                                    denom = freq_minima if freq_minima > 0 else 1
                                    eval_dict["freq_ts_over_min_ratio"] = (
                                        freq_ts / denom
                                    )
                                    eval_dict["freq_total_samples"] = freq_samples
                            else:
                                print(
                                    f"Warning: energy.n_spatial_dim is {energy.n_spatial_dim}, skipping frequency analysis"
                                )

                        else:
                            print("No clusters found")

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
