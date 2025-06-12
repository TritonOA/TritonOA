from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from tritonoa.modeling.at.base import (
    AcousticsToolboxModel,
    SoundSpeedProfileAT,
    SSPLayer,
)
from tritonoa.modeling.array import Receiver, Source
from tritonoa.modeling.at.base import Environment
from tritonoa.modeling.environment.halfspace import Bottom, Top
from tritonoa.modeling.util import clean_up_files


class BellhopModelExtensions(Enum):
    # TODO: Convert to StringEnum
    ENV = "env"
    PRT = "prt"
    RAY = "ray"


class BellhopRunType(Enum):
    # TODO: Convert to StringEnum
    R = "R"  #  Generates a ray file
    E = "E"  #  Generates an eigenray file
    A = "A"  #  Generates an amplitude-delay file (ascii)
    a = "a"  #  Generates an amplitude-delay file (binary)
    C = "C"  #  Coherent TL calculation
    I = "I"  #  Incoherent TL calculation
    S = "S"  #  Semicoherent TL calculation (Lloyd mirror source pattern)


class BellhopBeamType(Enum):
    # TODO: Convert to StringEnum
    G = "G"  #  Geometric hat beams in Cartesian coordinates (default)
    g = "g"  #  Geometric hat beams in ray-centered coordinates
    B = "B"  #  Geometric Gaussian beams


class BellhopPatternFileChoice(Enum):
    # TODO: Convert to StringEnum
    READ = "*"  # read in a source beam pattern file
    DONT_READ = "O"  # don't (default)


class BellhopSourceType(Enum):
    # TODO: Convert to StringEnum
    R = "R"  # point source (cylindrical coordinates) (default)
    X = "X"  # line source (cartesian coordinates)


class BellhopGridType(Enum):
    # TODO: Convert to StringEnum
    R = "R"  # rectilinear grid (default)
    I = "I"  # irregular grid


@dataclass
class Ray:
    r: list[float]
    z: list[float]
    launch_angle: float
    num_top_bounce: int
    num_bot_bounce: int


@dataclass
class RayFan:
    rays: list[Ray]


class Beams:
    def __init__(self, source: Source, receiver: Receiver) -> None:
        self.source = source
        self.receiver = receiver

    def read_beams(self, rayfil: str) -> str:
        self.rayfil = f"{rayfil}.ray"

        with open(self.rayfil, "rb") as f:
            self.title = f.readline().decode("utf-8").strip()[1:-1].strip()
            self.freq = float(f.readline().decode("utf-8").strip())
            Nsxyz = [int(i) for i in f.readline().decode("utf-8").strip().split()]
            NBeamAngles = [int(i) for i in f.readline().decode("utf-8").strip().split()]
            self.depth_top = float(f.readline().decode("utf-8").strip())
            self.depth_bot = float(f.readline().decode("utf-8").strip())
            self.type = f.readline().decode("utf-8").strip()[1:-1].strip()
            self.num_sx = Nsxyz[0]
            self.num_sy = Nsxyz[1]
            self.num_sz = Nsxyz[2]
            self.num_beams = NBeamAngles[0]

            sources = []
            for i in range(self.num_sz):
                rays = []
                for j in range(self.num_beams):
                    raw_alpha0 = f.readline().decode("utf-8").strip()
                    if not raw_alpha0:
                        sources.append(rays)
                        break
                    alpha0 = float(raw_alpha0)
                    ray_metadata = f.readline().decode("utf-8").strip().split()
                    num_steps = int(ray_metadata[0])
                    num_top_bounce = int(ray_metadata[1])
                    num_bot_bounce = int(ray_metadata[2])
                    if self.type == "rz":
                        raw_ray_data = [f.readline() for _ in range(num_steps)]
                        ray_data = [
                            i.decode("utf-8").strip().split() for i in raw_ray_data
                        ]
                        ranges = [float(i[0]) for i in ray_data]
                        depths = [float(i[1]) for i in ray_data]
                    if self.type == "xyz":
                        pass

                    rays.append(
                        Ray(ranges, depths, alpha0, num_top_bounce, num_bot_bounce)
                    )
                else:
                    sources.append(rays)
                    continue
                break

            self.sources = sources
            return self.rayfil


@dataclass
class BeamFan:
    alpha: list[float] = field(default_factory=lambda: [-11.0, 11.0])
    nbeams: int = 51
    isingle: Optional[int] = None


@dataclass
class BellhopOptions:
    run_type: str = BellhopRunType.R.value
    beam_type: str = BellhopBeamType.G.value
    pattern_file_choice: str = BellhopPatternFileChoice.DONT_READ.value
    source_type: str = BellhopSourceType.R.value
    grid_type: str = BellhopGridType.R.value

    def __post_init__(self):
        self._check_options()
        self.opt = ""
        for field in self.__dataclass_fields__.keys():
            self.opt += getattr(self, field)

    def _check_options(self):
        if self.run_type not in [i.value for i in BellhopRunType]:
            raise ValueError(f"Invalid run type: {self.run_type}")
        if self.beam_type not in [i.value for i in BellhopBeamType]:
            raise ValueError(f"Invalid beam type: {self.beam_type}")
        if self.pattern_file_choice not in [i.value for i in BellhopPatternFileChoice]:
            raise ValueError(f"Invalid pattern file choice: {self.pattern_file_choice}")
        if self.source_type not in [i.value for i in BellhopSourceType]:
            raise ValueError(f"Invalid source type: {self.source_type}")
        if self.grid_type not in [i.value for i in BellhopGridType]:
            raise ValueError(f"Invalid grid type: {self.grid_type}")


@dataclass
class NumericalIntegrator:
    zbox: float  # The maximum depth to trace a ray (m).
    rbox: float  # The maximum range to trace a ray (km).
    step: int = (
        0  # The step size used for tracing the rays (m). (Use 0 to let BELLHOP choose the step size.)
    )


@dataclass(kw_only=True)
class BellhopEnvironment(Environment):
    source: Source
    receiver: Receiver
    options: BellhopOptions = field(default_factory=BellhopOptions)
    fan: BeamFan = field(default_factory=BeamFan)
    integrator: NumericalIntegrator

    def write_envfil(self) -> Path:
        def _check_single_trace():
            if self.top.opt[5] == "I" and self.fan.isingle is None:
                raise ValueError(
                    "Single beam index must be specified for single trace."
                )
            if self.top.opt[5] == " " and self.fan.isingle is not None:
                raise ValueError(
                    "Top option single trace not set but single trace index specified."
                )

        opt = self.options.opt
        envfil = self._write_envfil()
        with open(envfil, "a") as f:
            # Block 7 - Source/Receiver Depths and Ranges
            # Number of Source Depths
            f.write(f"{self.source.nz:5d} \t \t \t \t ! NSD")
            # Source Depths [m]
            if (self.source.nz > 1) and self.source.equally_spaced():
                f.write(f"\r\n    {self.source.z[0]:6f} {self.source.z[-1]:6f}")
            else:
                f.write(f"\r\n    ")
                for zz in self.source.z:
                    f.write(f"{zz:6f} ")
            f.write("/ \t ! SD(1)  ... (m) \r\n")
            # Number of Receiver Depths
            f.write(f"{self.receiver.nz:5d} \t \t \t \t ! NRD")
            # Receiver Depths [m]
            if (self.receiver.nz > 1) and self.receiver.equally_spaced():
                f.write(f"\r\n    {self.receiver.z[0]:6f} {self.receiver.z[-1]:6f}")
            else:
                f.write(f"\r\n    ")
                for zz in self.receiver.z:
                    f.write(f"{zz:6f} ")
            f.write("/ \t ! RD(1)  ... (m) \r\n")
            # Number of Receiver Ranges
            f.write(f"{self.receiver.nr} \t \t \t \t ! NR")
            # Receiver Ranges [km]
            f.write(f"\r\n    ")
            for rr in self.receiver.r:
                f.write(f"{rr:6f} ")
            # Block 8 - Run Type
            f.write(f"\r\n'{opt}' \t \t \t ! Run Type")
            # Block 9 - Beam Fan
            _check_single_trace()
            if self.fan.isingle is not None:
                isingle = self.fan.isingle
            else:
                isingle = ""
            f.write(f"\r\n{self.fan.nbeams} {isingle} \t \t \t \t \t ! NBEAMS")
            if len(self.fan.alpha) == 2:
                f.write(
                    f"\r\n{self.fan.alpha[0]:6.2f} {self.fan.alpha[-1]:6.2f} / \t ! ALPHA(1:NBEAMS) (degrees)"
                )
            else:
                f.write(f"\r\n    ")
                for aa in self.fan.alpha:
                    f.write(f"{aa:6.2f} ")
                f.write(f"\t ! ALPHA(1:NBEAMS) (degrees)")
            # Block 10 - Numerical Integrator
            f.write(
                f"\r\n{self.integrator.step} {self.integrator.zbox} {self.integrator.rbox} \t \t \t \t ! STEP (m)  ZBOX (m)  RBOX (km)"
            )

        return envfil


class BellhopModel(AcousticsToolboxModel):
    model_name = "bellhop"

    def __init__(self, environment: BellhopEnvironment) -> None:
        super().__init__(environment)

    def run(
        self,
        keep_files: bool = False,
    ) -> None:
        """Returns modes, pressure field, rvec, zvec"""

        _ = self.environment.write_envfil()
        self.run_model(model_name=self.model_name)
        self.beams = Beams(self.environment.source, self.environment.receiver)
        self.beams.read_beams(self.environment.tmpdir / self.environment.title)
        if not keep_files:
            clean_up_files(
                self.environment.tmpdir,
                extensions=[i.value for i in BellhopModelExtensions],
                pattern=self.environment.title,
            )


def build_bellhop_environment(parameters: dict) -> BellhopEnvironment:
    """Helper function to generate a `BellhopEnvironment` from a dictionary.

    Args:
        parameters: Dictionary with the parameters for the `BellhopEnvironment`.

    Returns:
        `BellhopEnvironment` object.
    """
    layers = [
        SSPLayer(SoundSpeedProfileAT(**layer_kwargs))
        for layer_kwargs in parameters.get("layerdata")
    ]
    return BellhopEnvironment(
        title=parameters.get("title", "Bellhop"),
        freq=parameters.get("freq", 400.0),
        layers=layers,
        top=Top(
            opt=parameters.get("top_opt", "CVF    "),
            z=parameters.get("top_z"),
            c_p=parameters.get("top_c_p"),
            c_s=parameters.get("top_c_s", 0.0),
            rho=parameters.get("top_rho"),
            a_p=parameters.get("top_a_p", 0.0),
            a_s=parameters.get("top_a_s", 0.0),
        ),
        bottom=Bottom(
            opt=parameters.get("bot_opt", "A"),
            sigma=parameters.get("bot_sigma", 0.0),
            z=parameters.get("bot_z", layers[-1].z_max),
            c_p=parameters.get("bot_c_p"),
            c_s=parameters.get("bot_c_s", 0.0),
            rho=parameters.get("bot_rho"),
            a_p=parameters.get("bot_a_p", 0.0),
            a_s=parameters.get("bot_a_s", 0.0),
            mz=parameters.get("bot_mz"),
        ),
        tmpdir=parameters.get("tmpdir", "tmp"),
        source=Source(z=parameters.get("src_z", 1.0)),
        receiver=Receiver(
            z=parameters.get("rec_z", 1.0),
            r=parameters.get("rec_r", 1.0),
            tilt=parameters.get("tilt", None),
            azimuth=parameters.get("azimuth", None),
            z_pivot=parameters.get("z_pivot", None),
        ),
        options=BellhopOptions(
            run_type=parameters.get("run_type", "R"),
            beam_type=parameters.get("beam_type", "G"),
            pattern_file_choice=parameters.get("pattern_file_choice", "O"),
            source_type=parameters.get("source_type", "R"),
            grid_type=parameters.get("grid_type", "R"),
        ),
        fan=BeamFan(
            alpha=parameters.get("alpha", [-11.0, 11.0]),
            nbeams=parameters.get("nbeams", 51),
            isingle=parameters.get("isingle", None),
        ),
        integrator=NumericalIntegrator(
            zbox=parameters.get("zbox", 1000.0),
            rbox=parameters.get("rbox", 100.0),
            step=parameters.get("step", 0),
        ),
    )


def run_bellhop(parameters: dict, keep_files=False) -> BellhopModel:
    """Run Bellhop propagation model.

    Args:
        parameters: Dictionary with the parameters for the BellhopEnvironment.

    Returns:
        `BellhopModel` object containing ray trace results.
    """
    environment = build_bellhop_environment(parameters)
    model = BellhopModel(environment)
    model.run(keep_files=keep_files)
    return model
