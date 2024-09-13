# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any, Optional

import numpy as np

from tritonoa.modeling.environment.base import Environment
from tritonoa.modeling.environment.ssp import SoundSpeedProfile

AT_EXECUTABLES = {
    "bellhop": Path("src/bin/at/Bellhop/bellhop.exe"),
    "kraken": Path("src/bin/at/Kraken/kraken.exe"),
    "krakenc": Path("src/bin/at/Kraken/krakenc.exe"),
    "scooter": Path("src/bin/at/Scooter/scooter.exe"),
}


class UnknownCommandError(Exception):
    pass


@dataclass
class SoundSpeedProfileAT(SoundSpeedProfile):
    rho: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64))
    c_s: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=np.float64))
    a_p: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=np.float64))
    a_s: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=np.float64))

    def __post_init__(self):
        super().format_data()


@dataclass
class SSPLayer:
    ssp: SoundSpeedProfileAT
    nmesh: int = 0
    sigma: float = 0

    def __post_init__(self):
        self.z_max = self.ssp.z.max()


class AcousticsToolboxModel(ABC):
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    @abstractmethod
    def run(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        *args,
        **kwargs,
    ) -> Any:
        pass

    def run_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
    ) -> int:
        if model_path is None:
            command = f"{model_name.lower()}.exe {self.environment.title}"
        else:
            command = (
                f"{str(model_path)}/{model_name.lower()}.exe {self.environment.title}"
            )

        try:
            return subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.environment.tmpdir,
            )
        except:
            raise UnknownCommandError(
                f"Unknown command: {str(model_path)}/{model_name.lower()}.exe"
            )
