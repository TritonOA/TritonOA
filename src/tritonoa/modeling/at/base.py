# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
import secrets
import subprocess
from typing import Any, Optional

import numpy as np

from tritonoa.modeling.environment.halfspace import Bottom, Top
from tritonoa.modeling.environment.ssp import SoundSpeedProfile

AT_EXECUTABLES = {
    "bellhop": Path("src/lib/at/bin/bellhop.exe"),
    "kraken": Path("src/lib/at/bin/kraken.exe"),
    "krakenc": Path("src/lib/at/bin/krakenc.exe"),
    "scooter": Path("src/lib/at/bin/scooter.exe"),
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


@dataclass
class Environment(ABC):
    title: str
    freq: float
    layers: list[SSPLayer]
    top: Top
    bottom: Bottom
    tmpdir: str | bytes | Path = "."

    def __post_init__(self):
        self.nmedia = len(self.layers)
        self.tmpdir = Path(self.tmpdir)
        self.title += str(secrets.token_hex(4))

    @abstractmethod
    def write_envfil(self) -> Any:
        pass

    def _check_tmpdir(self) -> None:
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    def _write_envfil(self) -> Path:
        self._check_tmpdir()
        envfil = self.tmpdir / f"{self.title}.env"
        with open(envfil, "w") as f:
            # Block 1 - Title
            f.write(f"'{self.title}' ! Title \r\n")
            # Block 2 - Frequency
            f.write(f"{self.freq:8.2f} \t \t \t ! Frequency (Hz) \r\n")
            # Block 3 - Number of Layers
            f.write(f"{self.nmedia:8d} \t \t \t ! NMEDIA \r\n")
            # Block 4 - Top Option
            f.write(f"'{self.top.opt}' \t \t \t ! Top Option \r\n")
            # Block 4a - Top Halfspace Properties
            if self.top.opt[1] == "A":
                f.write(
                    f"     {self.layers[0].ssp.z[0]:6.2f}"
                    + f" {self.top.c_p:6.2f}"
                    + f" {self.top.c_s:6.2f}"
                    + f" {self.top.rho:6.2f}"
                    + f" {self.top.a_p:6.2f}"
                    + f" {self.top.a_s:6.2f}"
                    + "  \t ! Upper halfspace \r\n"
                )
            # Block 5 - Sound Speed Profile
            for layer in self.layers:
                f.write(
                    f"{layer.nmesh:5d} "
                    + f"{layer.sigma:4.2f} "
                    + f"{layer.z_max:6.2f} "
                    + "\t ! N sigma max_layer_depth \r\n"
                )
                for zz in range(len(layer.ssp.z)):
                    f.write(
                        f"\t {layer.ssp.z[zz]:6.2f} "
                        + f"\t {layer.ssp.c_p[zz]:6.2f} "
                        + f"\t {layer.ssp.c_s[zz]:6.2f} "
                        + f"\t {layer.ssp.rho[zz]:6.2f} "
                        + f"\t {layer.ssp.a_p[zz]:6.2f} "
                        + f"\t {layer.ssp.a_s[zz]:6.2f} "
                        + "/ \t ! z cp cs rho ap as \r\n"
                    )
            # Block 6 - Bottom Option
            f.write(
                f"'{self.bottom.opt}' {self.bottom.sigma:6.2f}"
                + "  \t \t ! Bottom Option, sigma\r\n"
            )
            # Block 6a - Bottom Halfspace from Geoacoustic Parameters
            if self.bottom.opt[0] == "A":
                f.write(
                    f"     {self.bottom.z:6.2f}"
                    + f" {self.bottom.c_p:6.2f}"
                    + f" {self.bottom.c_s:6.2f}"
                    + f" {self.bottom.rho:6.2f}"
                    + f" {self.bottom.a_p:6.2f}"
                    + f" {self.bottom.a_s:6.2f}"
                    + "  \t ! Lower halfspace \r\n"
                )
            # Block 6b - Bottom Halfspace from Grain Size
            elif self.bottom.opt[0] == "G":
                f.write(f"   {self.bottom.z:6.2f} {self.bottom.mz:6.2f} ! zb mz\r\n")

        return envfil


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
    ) -> int:
        executable = self.get_executable(model_name)
        command = f"{executable} {self.environment.title}"

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
                f"Unknown command: `{command}`"
            )

    @staticmethod
    def get_executable(model_name: str) -> str:
        return str(AT_EXECUTABLES.get(model_name.lower()).resolve())
