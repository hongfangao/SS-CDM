from abc import ABC, abstractmethod, abstractproperty
from functools import partial
from typing import Any, Optional

import numpy as np
import torch

from cd2.utils.tensors import check_flat_array

class Metric(ABC):
    def __init__(self, original_samples: np.ndarray | torch.Tensor) -> None:
        self.original_samples = check_flat_array(original_samples)
    
    @abstractmethod
    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> dict[str, Any]: ...

    @abstractproperty
    def name(self) -> str: ...

    @property
    def baseline_metrics(self) -> dict[str, float]:
        return {}

class MetricCollection:
    def __init__(
        self,
        metrics: list[Metric],
        original_samples: Optional[np.ndarray | torch.Tensor] = None,
        include_baselines: bool = True,
    ) -> None:
        metrics_time: list[Metric] = []

        for metric in metrics:
            if isinstance(metric, partial):
                assert(
                    original_samples is not None
                ), f"Original samples must be provided for metric {metric.name} to be instantiated."
                metrics_time.append(metric(original_samples=original_samples))
        self.metric_time = metrics_time
        self.include_baselines = include_baselines
    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> dict[str, Any]:
        metric_dict = {}
        for metric_time in self.metric_time:
            metric_dict.update(
                {f"time_{k}": v for k, v in metric_time(other_samples).items()}
            )
        if self.include_baselines:
            metric_dict.update(self.baseline_metrics)
        
        return dict(sorted(metric_dict.items(), key = lambda item:item[0]))
    
    @property
    def baseline_metrics(self) -> dict[str, float]:
        metric_dict = {}
        for metric_time in self.metric_time:
            metric_dict.update(
                {f"time_{k}": v for k,v in metric_time.baseline_metrics.items()}
            )
        return metric_dict