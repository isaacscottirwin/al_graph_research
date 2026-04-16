from dataclasses import dataclass
import numpy as np
from al_graph_research.experiments.only_edges.edge_alteration_experiment import EdgeAlterationExperiment
from typing import Literal



@dataclass
class ExperimentConfig:
    n_runs: int
    num_rounds: int | Literal["max"]
    number_neighbors: int
    kernel: str
    alteration_strategy: str
    graph_construction_method: str
    starting_idx: int
    ending_idx: int
    empty_val: int


class ExperimentConfigBuilder:
    @staticmethod
    def build_experiment(cfg: ExperimentConfig, seed: int | None = None):
        if not (cfg.num_rounds == "max" or (isinstance(cfg.num_rounds, int) and cfg.num_rounds > 0)):
            raise ValueError("num_rounds must be a positive integer or 'max'.")
        rng = np.random.default_rng(seed) if seed is not None else None

        return EdgeAlterationExperiment(
            n_runs=cfg.n_runs,
            num_rounds=cfg.num_rounds,
            number_neighbors=cfg.number_neighbors,
            kernel=cfg.kernel,
            alteration_strategy=cfg.alteration_strategy,
            graph_construction_method=cfg.graph_construction_method,
            starting_idx=cfg.starting_idx,
            ending_idx=cfg.ending_idx,
            empty_val=cfg.empty_val,
            rng=rng,
        )

class ExperimentConfigurations:
    MNIST_ZERO = ExperimentConfig(
        n_runs=3,
        num_rounds=50,
        number_neighbors=30,
        kernel="uniform",
        alteration_strategy="zero",
        graph_construction_method="partner",
        starting_idx=0,
        ending_idx=3,
        empty_val=0
    )
    MNIST_NEGATE = ExperimentConfig(
        n_runs=3,
        num_rounds=50,
        number_neighbors=30,
        kernel="uniform",
        alteration_strategy="negate",
        graph_construction_method="partner",
        starting_idx=0,
        ending_idx=3,
        empty_val=0
    )
    MNIST_ZERO_MAX = ExperimentConfig(
        n_runs=1,
        num_rounds="max",
        number_neighbors=30,
        kernel="uniform",
        alteration_strategy="zero",
        graph_construction_method="partner",
        starting_idx=0,
        ending_idx=3,
        empty_val=0
    )
    MNIST_NEGATE_MAX = ExperimentConfig(
        n_runs=1,
        num_rounds="max",
        number_neighbors=30,
        kernel="uniform",
        alteration_strategy="negate",
        graph_construction_method="partner",
        starting_idx=0,
        ending_idx=3,
        empty_val=0
    )
    THREE_BLOBS_ZERO = ExperimentConfig(
        n_runs=3,
        num_rounds=20,
        number_neighbors=30,
        kernel="gaussian",
        alteration_strategy="zero",
        graph_construction_method="partner",
        starting_idx=1,
        ending_idx=3,
        empty_val=0
    )
    THREE_BLOBS_NEGATE = ExperimentConfig(
        n_runs=3,
        num_rounds=20,
        number_neighbors=30,
        kernel="gaussian",
        alteration_strategy="negate",
        graph_construction_method="partner",
        starting_idx=1,
        ending_idx=3,
        empty_val=0
    )