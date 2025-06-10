"""
This module implements LaMBO2 by Gruver, Stanton et al. 2023.

LaMBO2 is an improvement on LaMBO [Stanton et al. 2022], using
guided discrete diffusion and network ensembles instead of
latent space optimization using Gaussian Processes.

In this module, we import [`cortex`](https://github.com/prescient-design/cortex)
and use the default configuration files except for the `lambo`
optimizer, which is replaced by a more conservative version.
The exact configuration file can be found alongside this file
in our repository:
https://github.com/MachineLearningLifeScience/poli-baselines/tree/main/src/poli_baselines/solvers/bayesian_optimization/lambo2/hydra_configs

:::{warning}
This optimizer only works for **protein-related** black boxes, like
- `foldx_stability`
- `foldx_sasa`
- `rasp`
- `ehrlich`
:::

"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

try:
    import edlib
    import hydra
    import lightning as L
    from beignet import farthest_first_traversal
    from omegaconf import OmegaConf
except ImportError as e:
    raise ImportError(
        "You are trying to use the Lambo2 solver. Install "
        "the relevant optional dependencies with [lambo2].\n"
        "You can do this by running e.g. \n"
        "pip install 'poli-baselines[lambo2] @ git+https://github.com/MachineLearningLifeScience/poli-baselines.git'"
    ) from e

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.mutations import add_random_mutations_to_reach_pop_size
import poli_baselines

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import pdb


# THIS_DIR = Path(__file__).parent.resolve()
# DEFAULT_CONFIG_DIR = THIS_DIR / "hydra_configs"
DEFAULT_CONFIG_DIR = Path(poli_baselines.__file__).parent / "solvers" / "bayesian_optimization" / "lambo2" / "hydra_configs"



def edit_dist(x: str, y: str):
    """
    Computes the edit distance between two strings.
    """
    return edlib.align(x, y)["editDistance"]


class LaMBO2(AbstractSolver):
    """
    LaMBO2 solver for protein-related black boxes.

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box to optimize. Must be protein-related. To ensure that the
        black box is protein-related, we verify that the `alphabet` inside the
        `info` attribute of the black box is a protein alphabet.
    x0 : np.ndarray
        The initial solutions to the black box. If not enough solutions are
        provided, the solver will generate random mutants to reach the population
        size specified in the configuration file (as cfg.num_samples).
    y0 : np.ndarray, optional
        The initial evaluations of the black box. If not provided, the solver
        will evaluate the black box on the initial solutions.
    config_dir : Path | str, optional
        The directory where the configuration files are stored. If not provided,
        the default configuration files (stored alongside this file in our
        repository) will be used. If you are interested in modifying the
        configurations, we recommend taking a look at the tutorials inside `cortex`.
    config_name : str, optional
        The name of the configuration file to use. Defaults to "generic_training".
    overrides : list[str], optional
        A list of overrides to apply to the configuration file. For example,
        ["num_samples=10", "max_epochs=5"]. To know what to override, we recommend
        taking a look at the tutorials inside `cortex`.
    seed : int, optional
        The random seed to use. If not provided, we use the seed provided in the
        configuration file. If provided, this seed will override the seed in the
        configuration file.
    max_epochs_for_retraining : int, optional
        The number of epochs to retrain the model after each step. Defaults to 1.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        config: OmegaConf | None = None,
        config_dir: Path | str | None = None,
        config_name: str = "generic_training",
        overrides: list[str] | None = None,
        seed: int | None = None,
        max_epochs_for_retraining: int = 1,
        restrict_candidate_points_to: np.ndarray | None = None,
        logger=None,
    ):
        super().__init__(black_box=black_box, x0=x0, y0=y0)
        self.experiment_id = f"{uuid4()}"[:8]
        self.max_epochs_for_retraining = max_epochs_for_retraining
        self.restrict_candidate_points_to = restrict_candidate_points_to

        if config is None:
            if config_dir is None:
                config_dir = DEFAULT_CONFIG_DIR
            with hydra.initialize_config_dir(config_dir=str(config_dir)):
                cfg = hydra.compose(config_name=config_name, overrides=overrides)
                OmegaConf.set_struct(cfg, False)
        else:
            cfg = config

        # Setting the random seed
        # We are ignoring the seed in the original config file.
        if seed is not None:
            cfg.update({"random_seed": seed})

        seed_python_numpy_and_torch(cfg.random_seed)
        L.seed_everything(seed=cfg.random_seed, workers=True)

        self.cfg = cfg
        print(OmegaConf.to_yaml(cfg))
        self.logger = logger

        if x0 is None:
            raise ValueError(
                "In the Lambo2 optimizer, it is necessary to pass at least "
                "a single solution to the solver through x0."
            )
        elif x0.shape[0] < cfg.num_samples:
            original_size = x0.shape[0]
            x0 = add_random_mutations_to_reach_pop_size(
                x0,
                alphabet=self.black_box.info.alphabet,
                population_size=cfg.num_samples,
            )

        tokenizable_x0 = np.array([" ".join(x_i) for x_i in x0])

        x0_for_black_box = np.array([seq.replace(" ", "") for seq in tokenizable_x0])

        if y0 is None:
            y0 = self.black_box(x0_for_black_box)
        elif y0.shape[0] < x0.shape[0]:
            y0 = np.vstack([y0, self.black_box(x0_for_black_box[original_size:])])
        
        ### add new lines
        # best_f = torch.tensor(y0).max(dim=0).values if isinstance(y0, torch.Tensor) else torch.tensor(y0).max(dim=0).values
        # best_f_list = best_f.tolist()
        # OmegaConf.set_struct(self.cfg.guidance_objective.static_kwargs, False)
        # self.cfg.guidance_objective.static_kwargs.best_f = best_f_list

        self.history_for_training = {
            "x": [tokenizable_x0],
            # "y": [y0.flatten()],
            # I had to change this line:
            "y": [y0.squeeze()],
            "t": [np.full(len(y0), 0)],
        }

        # pdb.set_trace()

        # Pre-training the model.
        MODEL_FOLDER = Path(cfg.data_dir) / self.experiment_id
        MODEL_FOLDER.mkdir(exist_ok=True, parents=True)
        self.model_path = MODEL_FOLDER / "ongoing_model.ckpt"
        self._train_model_with_history(
            save_checkpoint_to=self.model_path,
            max_epochs=cfg.max_epochs,
        )

    @property
    def history(self) -> dict[str, list[np.ndarray]]:
        """
        Returns the history of the black box evaluations.

        Returns
        -------
        dict[str, list[np.ndarray]]
            The history of the black box evaluations.
        """
        all_x = np.concatenate(self.history_for_training["x"], axis=0)
        all_y = np.concatenate(self.history_for_training["y"], axis=0)
        all_t = np.concatenate(self.history_for_training["t"], axis=0)

        return {
            "x": [np.array(["".join(x_i).replace(" ", "")]) for x_i in all_x],
            # "y": [np.array([[y_i]]) for y_i in all_y],
            "y": [all_y],  # I changed this line
            "t": [np.array([t_i]) for t_i in all_t],
        }

    def _train_model_with_history(
        self,
        load_checkpoint_from: Path | None = None,
        save_checkpoint_to: Path | None = None,
        max_epochs: int = 2,
    ) -> L.LightningModule:
        """
        Trains the model with the history of the black box evaluations.

        Parameters
        ----------
        load_checkpoint_from : Path, optional
            The path to the checkpoint to load. If not provided, the model will
            be trained from scratch.
        save_checkpoint_to : Path, optional
            The path to save the checkpoint. If not provided, the model will not
            be saved.
        max_epochs : int, optional
            The number of epochs to train the model. Defaults to 2.
        """

        x = np.concatenate(self.history_for_training["x"][::-1])
        y = np.concatenate(self.history_for_training["y"][::-1])
        t = np.concatenate(self.history_for_training["t"][::-1])

        t_partition = _geometric_partitioning(t)

        # is_feasible = y > -float("inf")
        # I had to change this line:
        # is_feasible = (y > -float("inf")).prod(axis=1)
        is_feasible = (y > -float("inf")).prod(axis=1).astype(bool)
        feasible_x = x[is_feasible]
        feasible_y = y[is_feasible]
        feasible_t = t[is_feasible]

        # Dynamically set outcome_cols BEFORE model instantiation
        # model_cfg = self.cfg.tree
        # model_cfg.generic_task.outcome_cols = [f"obj_{i}" for i in range(feasible_y.shape[1])]
        outcome_cols = [f"obj_{i}" for i in range(feasible_y.shape[1])]

        # pdb.set_trace()

        from omegaconf import OmegaConf

        # Set outcome_cols in the tasks config
        if self.cfg.tasks.protein_property.get('generic_task') is not None:
            self.cfg.tasks.protein_property.get('generic_task').outcome_cols = outcome_cols
        else:
            print(OmegaConf.to_yaml(self.cfg))
            raise ValueError("Expected `generic_task` in cfg but not found.")


        
        model = hydra.utils.instantiate(self.cfg.tree)
        model.build_tree(self.cfg, skip_task_setup=True)

        if load_checkpoint_from is not None and load_checkpoint_from.exists():
            model.load_state_dict(
                torch.load(
                    load_checkpoint_from,
                    map_location="cpu",
                    weights_only=False,
                )["state_dict"]
            )

        dedup_feas_x, indices = np.unique(feasible_x, return_index=True)
        dedup_feas_y = feasible_y[indices]
        dedup_feas_t = feasible_t[indices]

        print(f"Total History: {len(x)}")
        print(f"Unique Feasible Solutions: {len(dedup_feas_x)}")
        print(f"Top-5 Objective Values: {np.sort(dedup_feas_y.flatten())[-5:]}")

        # 💡 Assign outcome_cols dynamically based on feasible_y
        # model_cfg.generic_task.outcome_cols = [f"obj_{i}" for i in range(feasible_y.shape[1])]

        # I've changed the below lines:
        # Convert multi-objective feasible_y into a 1D numpy object array for DataFrame compatibility
        # obj_y = np.empty(len(feasible_y), dtype=object)
        # for idx, row in enumerate(feasible_y):
        #     obj_y[idx] = row
        # Prepare separate columns for each objective for regression
        # num_objs = feasible_y.shape[1]
        # obj_cols = {}
        # for i in range(num_objs):
        #    obj_cols[f"obj_{i}"] = feasible_y[:, i]
        obj_cols = {f"obj_{i}": feasible_y[:, i] for i in range(feasible_y.shape[1])}


        task_setup_kwargs = {
            # task_key:
            "generic_constraint": {
                "data": {
                    "tokenized_seq": x,
                    "is_feasible": is_feasible,
                    "recency": t_partition,
                },
            },
            "generic_task": {
                # dataset kwarg
                "data": {
                    "tokenized_seq": feasible_x,
                    # "generic_task": y[y >= 0] + np.random.normal(0, math.sqrt(0.01), y[y >= 0].shape),
                    # "generic_task": feasible_y,
                    # "generic_task": obj_y,  # I've changed this line
                    **obj_cols,
                    "recency": t_partition[is_feasible],
                }
            },
            "protein_seq": {
                # dataset kwarg
                "data": {
                    "tokenized_seq": dedup_feas_x,
                    "recency": dedup_feas_t,
                }
            },
        }

        for task_key, task_obj in model.task_dict.items():
            task_obj.data_module.setup(
                stage="test", dataset_kwargs=task_setup_kwargs[task_key]
            )
            task_obj.data_module.setup(
                stage="fit", dataset_kwargs=task_setup_kwargs[task_key]
            )

        # instantiate trainer, set logger
        self.trainer: L.Trainer = hydra.utils.instantiate(
            self.cfg.trainer, max_epochs=max_epochs
        )
        self.trainer.fit(
            model,
            train_dataloaders=model.get_dataloader(split="train"),
            # val_dataloaders=model.get_dataloader(split="val"),
        )

        if save_checkpoint_to:
            self.trainer.save_checkpoint(save_checkpoint_to)

        return model

    def get_candidate_points(self):
        if self.restrict_candidate_points_to is not None:
            # Let's assume that the user passes a wildtype as
            # np.array(["AAAAA"]) or np.array(["A", "A", "A", ...]).
            assert len(self.restrict_candidate_points_to.shape) == 1
            if self.restrict_candidate_points_to.shape[0] == 1:
                tokenizable_candidate_point = " ".join(
                    self.restrict_candidate_points_to
                )
                candidate_points = np.array(
                    [tokenizable_candidate_point for _ in range(self.cfg.num_samples)]
                )
            elif self.restrict_candidate_points_to.shape[0] == self.cfg.num_samples:
                candidate_points = np.array(
                    [" ".join(x_i) for x_i in self.restrict_candidate_points_to]
                )
            else:
                raise ValueError(
                    "The restrict_candidate_points_to array must be of size "
                    f"self.cfg.num_samples ({self.cfg.num_samples}) or of size 1. "
                    f"Got {len(self.restrict_candidate_points_to[0])} instead."
                )

            return candidate_points
        else:
            return self.get_candidate_points_from_history()

    def farthest_first_traversal_moo(
        self,
        library,
        distance_fn,
        ranking_scores,
        n,
        descending=True,
    ):
        """
        Multi-objective farthest-first traversal using Pareto rank as priority.
        Lower ranks are better (i.e., rank 0 = Pareto front).
    
        Parameters:
            library: List of candidate items (e.g., sequences)
            distance_fn: Function to compute pairwise distances
            ranking_scores: 1D array of Pareto ranks (ints) OR
                            2D array of objective values
            n: Number of points to select
            descending: Whether to prioritize higher or lower scores
                        (for ranks, descending=False means prefer lower ranks)
    
        Returns:
            List of selected indices
        """
    
        if isinstance(ranking_scores, torch.Tensor):
            ranking_scores = ranking_scores.cpu().numpy()
    
        if len(ranking_scores.shape) == 2:
            # convert from multi-objective scores to Pareto ranks
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            nds = NonDominatedSorting()
            _, rank = nds.do(ranking_scores, return_rank=True)
        else:
            rank = ranking_scores
    
        if descending:
            score_order = np.argsort(-rank)
        else:
            score_order = np.argsort(rank)
    
        selected = []
        selected.append(score_order[0])
    
        for _ in range(1, n):
            max_dist = -1
            best_idx = None
            for i in score_order:
                if i in selected:
                    continue
                min_dist = min(distance_fn(library[i], library[j]) for j in selected)
                if min_dist > max_dist:
                    max_dist = min_dist
                    best_idx = i
            if best_idx is not None:
                selected.append(best_idx)
    
        return selected

    def get_candidate_points_from_history(self) -> np.ndarray:
        """
        Returns the current best population (whose size is specified in the
        configuration file as cfg.num_samples) from the history of the black
        box evaluations, using EHVI for candidate selection.
        """

        x = np.concatenate(self.history_for_training["x"], axis=0)
        y = np.concatenate(self.history_for_training["y"], axis=0)

        # Prepare candidate pool
        nds = NonDominatedSorting()
        _, sorted_y0_idxs = nds.do(y, return_rank=True)
        top_k = min(len(x), self.cfg.fft_expansion_factor * self.cfg.num_samples)

        candidate_points = x[sorted_y0_idxs[:top_k]]
        candidate_scores = y[sorted_y0_idxs[:top_k]]

        # Convert candidate_points to tensors with proper formatting
        candidate_tensor = torch.tensor(
            np.array([[ord(c) for c in s.replace(" ", "")] for s in candidate_points]),
            dtype=torch.float32
        )

        # Convert full training data to tensors
        train_x = torch.tensor(
            np.array([[ord(c) for c in s.replace(" ", "")] for s in x]),
            dtype=torch.float32
        )
        train_y = torch.tensor(y, dtype=torch.float32)

        # Train separate GPs for each objective
        models = [SingleTaskGP(train_x, train_y[:, i:i+1]) for i in range(train_y.shape[1])]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        model.train()
        mll.eval()

        # EHVI acquisition
        ref_point = train_y.min(dim=0).values - 0.1
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acq_fn = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )

        # Evaluate EHVI scores for each candidate (in batch mode)
        model.eval()
        with torch.no_grad():
            acq_vals = acq_fn(candidate_tensor.unsqueeze(1))

        # pdb.set_trace()
        
        topk = torch.topk(acq_vals.squeeze(), k=self.cfg.num_samples).indices

        selected = candidate_points[topk.numpy()]
        print(candidate_scores[topk.numpy()])

        return selected

    # def get_candidate_points_from_history(self) -> np.ndarray:
    #     """
    #     Returns the current best population (whose size is specified in the
    #     configuration file as cfg.num_samples) from the history of the black
    #     box evaluations.
    #     """
    #     x = np.concatenate(self.history_for_training["x"], axis=0)
    #     y = np.concatenate(self.history_for_training["y"], axis=0)

    #     nds = NonDominatedSorting()
    #     _, sorted_y0_idxs = nds.do(y, return_rank=True)
    #     candidate_points = x[
    #         sorted_y0_idxs[
    #             : min(len(x), self.cfg.fft_expansion_factor * self.cfg.num_samples)
    #         ]
    #     ]
    #     candidate_scores = y[
    #         sorted_y0_idxs[
    #             : min(len(x), self.cfg.fft_expansion_factor * self.cfg.num_samples)
    #         ]
    #     ]
        
    #     indices = self.farthest_first_traversal_moo(
    #        library=candidate_points,
    #        distance_fn=edit_dist,
    #        ranking_scores=torch.tensor(candidate_scores, dtype=torch.float32),
    #        n=min(self.cfg.num_samples, len(candidate_points)),
    #        descending=True,
    #     )

    #     print(candidate_scores[indices])

    #     return candidate_points[indices]

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the model, runs the optimizer (LaMBO2) for the
        number of steps in the config, computes new proposal,
        evaluates on the black box and updates history.
        """
        # Load the model and optimizer
        model = self._train_model_with_history(
            load_checkpoint_from=self.model_path,
            max_epochs=self.max_epochs_for_retraining,
        )

        # best_f_tensor = torch.tensor(self.history["y"][-1]).max(dim=0).values
        # best_f = best_f_tensor.tolist()

        # OmegaConf.set_struct(self.cfg.guidance_objective.static_kwargs, False)
        # self.cfg.guidance_objective.static_kwargs.best_f = best_f

        # Builds the acquisition function
        candidate_points = self.get_candidate_points()
        acq_fn_runtime_kwargs = hydra.utils.call(
            self.cfg.guidance_objective.runtime_kwargs,
            model=model,
            candidate_points=candidate_points,
        )
        acq_fn = hydra.utils.instantiate(
            self.cfg.guidance_objective.static_kwargs, **acq_fn_runtime_kwargs
        )

        # Builds the optimizer
        tokenizer_transform = model.root_nodes["protein_seq"].eval_transform
        tokenizer = tokenizer_transform[0].tokenizer

        # Making sure the model doesn't edit length
        if not self.cfg.allow_length_change:
            tokenizer.corruption_vocab_excluded.add(
                "-"
            )  # prevent existing gap tokens from being masked
            tokenizer.sampling_vocab_excluded.add(
                "-"
            )  # prevent any gap tokens from being sampled

        tok_idxs = tokenizer_transform(candidate_points)
        is_mutable = tokenizer.get_corruptible_mask(tok_idxs)
        tok_idxs = tokenizer_transform(candidate_points)
        optimizer = hydra.utils.instantiate(
            self.cfg.optim,
            params=tok_idxs,
            is_mutable=is_mutable,
            model=model,
            objective=acq_fn,
            constraint_fn=None,
        )

        # Compute proposals using the optimizer
        for _ in range(self.cfg.num_steps):
            # Take a step on the optimizer, diffusing towards promising sequences.
            metrics = optimizer.step()
            if self.logger:
                self.logger.log_metrics(metrics)

        # Get the most promising sequences from the optimizer
        best_solutions = optimizer.get_best_solutions()
        new_designs = best_solutions["protein_seq"].values
        new_designs_for_black_box = np.array(
            [seq.replace(" ", "") for seq in new_designs]
        )

        print('new_designs_for_black_box shape: ', new_designs_for_black_box.shape)
        print('new_designs_for_black_box shape: ', new_designs_for_black_box)

        # Evaluate the black box
        new_y = self.black_box(new_designs_for_black_box)
        print(new_y)

        # Updating the history that is used for training.
        self.history_for_training["x"].append(new_designs)
        # self.history_for_training["y"].append(new_y.flatten())
        self.history_for_training["y"].append(new_y)  # I've changed this line
        last_t = self.history_for_training["t"][-1][-1]
        self.history_for_training["t"].append(np.full(len(new_y), last_t + 1))

        # print(f"\n{new_designs_for_black_box}\n")

        return new_designs_for_black_box, new_y

    def solve(self, max_iter: int = 10) -> None:
        """
        Solves the black box optimization problem for a maximum of `max_iter`
        iterations.

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations to run the solver. Defaults to 10.
        """
        for _ in range(max_iter):
            self.step()


def _geometric_partitioning(t_arr):
    """
    Given an array t with discrete integer values [0, 1, ..., n],
    assign partition index 0 if t == n, 1 if t in [n - 1, n - 2],
    2 if t in [n - 3, n - 4, n - 5, n - 6]. Stop when n - i == 0.
    """
    n = np.max(t_arr)
    result = np.zeros_like(t_arr)

    partition_index = 0
    current_n = n

    while current_n >= 0:
        if partition_index == 0:
            result[t_arr == current_n] = partition_index
            current_n -= 1
        else:
            partition_size = 2**partition_index
            start = min(current_n - partition_size + 1, 0)
            end = current_n + 1
            result[(t_arr >= start) & (t_arr < end)] = partition_index
            current_n -= partition_size

        partition_index += 1

    return result