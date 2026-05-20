from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def load_model(self, device: str = "cuda") -> None:
        ...

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs):
        ...
