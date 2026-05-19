"""Output-related typed configuration blocks.

This module keeps the output contract separate from the larger
``polyfempy.api.config`` facade. ``config.py`` re-exports these names for
backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _to_plain_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _to_plain_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_value(v) for v in value]
    return value


# ============================================================================
# Output Configuration Classes
# ============================================================================

@dataclass
class OutputLog:
    """Setting for the output log."""

    level: Union[int, str] = "debug"
    file_level: Union[int, str] = "trace"
    path: str = ""
    quiet: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.level != "debug":
            result["level"] = self.level
        if self.file_level != "trace":
            result["file_level"] = self.file_level
        if self.path:
            result["path"] = self.path
        if self.quiet:
            result["quiet"] = True
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputLog":
        return cls(
            level=d.get("level", "debug"),
            file_level=d.get("file_level", "trace"),
            path=str(d.get("path", "")),
            quiet=bool(d.get("quiet", False)),
        )


@dataclass
class OutputParaviewOptions:
    """Optional fields in the Paraview output."""

    use_hdf5: bool = False
    material: bool = False
    body_ids: bool = False
    contact_forces: bool = False
    friction_forces: bool = False
    normal_adhesion_forces: bool = False
    tangential_adhesion_forces: bool = False
    velocity: bool = False
    acceleration: bool = False
    scalar_values: bool = True
    tensor_values: bool = True
    discretization_order: bool = True
    nodes: bool = True
    forces: bool = False
    force_high_order: bool = False
    jacobian_validity: bool = False

    def to_dict(self) -> Dict[str, Any]:
        defaults = type(self)()
        result: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value != getattr(defaults, key):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputParaviewOptions":
        defaults = cls()
        kwargs = {key: d.get(key, getattr(defaults, key)) for key in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class ParaviewOutput:
    """Paraview output configuration - provides IDE autocomplete support.
    
    Attributes:
        volume: Export volume data. Defaults to True.
        surface: Export surface data. Defaults to False.
        wireframe: Export wireframe. Defaults to False.
        points: Export points. Defaults to False.
        file_name: Output file name (e.g., "sim.pvd"). Defaults to None.
        options: Additional options (e.g., contact_forces, friction_forces, velocity, acceleration).
        vismesh_rel_area: Visualization mesh relative area. Defaults to None.
    
    Example:
        >>> paraview = ParaviewOutput(volume=True, surface=True, file_name="output.pvd")
    """
    volume: bool = True
    surface: bool = False
    wireframe: bool = False
    points: bool = False
    file_name: Optional[str] = None
    options: Optional[Union[OutputParaviewOptions, Dict[str, Any]]] = None
    vismesh_rel_area: Optional[float] = 1e-5
    skip_frame: Optional[int] = 1
    high_order_mesh: bool = True
    fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "volume": self.volume,
            "surface": self.surface,
            "wireframe": self.wireframe,
            "points": self.points,
            "high_order_mesh": self.high_order_mesh,
        }
        if self.file_name is not None:
            result["file_name"] = self.file_name
        if self.options is not None:
            options = _to_plain_value(self.options)
            if options:
                result["options"] = options
        if self.vismesh_rel_area is not None:
            result["vismesh_rel_area"] = self.vismesh_rel_area
        if self.skip_frame is not None:
            result["skip_frame"] = self.skip_frame
        if self.fields:
            result["fields"] = list(self.fields)
        return result

    @classmethod
    def time_sequence(
        cls,
        *,
        file_name: str = "results.pvd",
        volume: bool = True,
        surface: bool = False,
        wireframe: bool = False,
        points: bool = False,
        vismesh_rel_area: Optional[float] = 1e-5,
        skip_frame: Optional[int] = 1,
        high_order_mesh: bool = True,
        fields: Optional[List[str]] = None,
        material: bool = False,
        body_ids: bool = False,
        contact_forces: bool = False,
        friction_forces: bool = False,
        normal_adhesion_forces: bool = False,
        tangential_adhesion_forces: bool = False,
        velocity: bool = False,
        acceleration: bool = False,
        scalar_values: bool = True,
        tensor_values: bool = True,
        discretization_order: bool = True,
        nodes: bool = True,
        forces: bool = False,
        force_high_order: bool = False,
        jacobian_validity: bool = False,
    ) -> "ParaviewOutput":
        """Construct a ParaView time-sequence export with common field toggles."""
        options = OutputParaviewOptions(
            material=material,
            body_ids=body_ids,
            contact_forces=contact_forces,
            friction_forces=friction_forces,
            normal_adhesion_forces=normal_adhesion_forces,
            tangential_adhesion_forces=tangential_adhesion_forces,
            velocity=velocity,
            acceleration=acceleration,
            scalar_values=scalar_values,
            tensor_values=tensor_values,
            discretization_order=discretization_order,
            nodes=nodes,
            forces=forces,
            force_high_order=force_high_order,
            jacobian_validity=jacobian_validity,
        )
        return cls(
            volume=volume,
            surface=surface,
            wireframe=wireframe,
            points=points,
            file_name=file_name,
            options=options,
            vismesh_rel_area=vismesh_rel_area,
            skip_frame=skip_frame,
            high_order_mesh=high_order_mesh,
            fields=list(fields or []),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParaviewOutput":
        options = d.get("options")
        if isinstance(options, dict):
            options = OutputParaviewOptions.from_dict(options)
        return cls(
            volume=bool(d.get("volume", True)),
            surface=bool(d.get("surface", False)),
            wireframe=bool(d.get("wireframe", False)),
            points=bool(d.get("points", False)),
            file_name=d.get("file_name"),
            options=options,
            vismesh_rel_area=d.get("vismesh_rel_area", 1e-5),
            skip_frame=d.get("skip_frame", 1),
            high_order_mesh=bool(d.get("high_order_mesh", True)),
            fields=list(d.get("fields", [])),
        )


@dataclass
class OutputDataAdvanced:
    """Advanced text/data-output options."""

    reorder_nodes: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"reorder_nodes": True} if self.reorder_nodes else {}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputDataAdvanced":
        return cls(reorder_nodes=bool(d.get("reorder_nodes", False)))


@dataclass
class OutputData:
    """File names to write output data to."""

    solution: str = ""
    full_mat: str = ""
    stiffness_mat: str = ""
    stress_mat: str = ""
    state: str = ""
    rest_mesh: str = ""
    mises: str = ""
    nodes: str = ""
    advanced: Optional[Union[OutputDataAdvanced, Dict[str, Any]]] = None
    file_index_offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in ("solution", "full_mat", "stiffness_mat", "stress_mat", "state", "rest_mesh", "mises", "nodes"):
            value = getattr(self, key)
            if value:
                result[key] = value
        if self.advanced is not None:
            advanced = _to_plain_value(self.advanced)
            if advanced:
                result["advanced"] = advanced
        if self.file_index_offset != 0:
            result["file_index_offset"] = self.file_index_offset
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputData":
        advanced = d.get("advanced")
        if isinstance(advanced, dict):
            advanced = OutputDataAdvanced.from_dict(advanced)
        return cls(
            solution=str(d.get("solution", "")),
            full_mat=str(d.get("full_mat", "")),
            stiffness_mat=str(d.get("stiffness_mat", "")),
            stress_mat=str(d.get("stress_mat", "")),
            state=str(d.get("state", "")),
            rest_mesh=str(d.get("rest_mesh", "")),
            mises=str(d.get("mises", "")),
            nodes=str(d.get("nodes", "")),
            advanced=advanced,
            file_index_offset=int(d.get("file_index_offset", 0)),
        )


@dataclass
class OutputAdvanced:
    """Additional output options."""

    timestep_prefix: str = "step_"
    sol_on_grid: float = -1
    compute_error: bool = True
    sol_at_node: int = -1
    vis_boundary_only: bool = False
    curved_mesh_size: bool = False
    save_solve_sequence_debug: bool = False
    save_ccd_debug_meshes: bool = False
    save_time_sequence: bool = True
    save_nl_solve_sequence: bool = False
    spectrum: bool = False

    def to_dict(self) -> Dict[str, Any]:
        defaults = type(self)()
        result: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value != getattr(defaults, key):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputAdvanced":
        defaults = cls()
        kwargs = {key: d.get(key, getattr(defaults, key)) for key in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class OutputReference:
    """Reference solution/gradient output."""

    solution: List[str] = field(default_factory=list)
    gradient: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.solution:
            result["solution"] = list(self.solution)
        if self.gradient:
            result["gradient"] = list(self.gradient)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputReference":
        return cls(
            solution=list(d.get("solution", [])),
            gradient=list(d.get("gradient", [])),
        )


@dataclass
class ResultOutput:
    """Python-side result request for ``solve()``.

    This does **not** go into the PolyFEM JSON schema. It tells the Python API
    which result fields the user cares about and whether missing fields should
    be treated as an error.

    Attributes:
        fields: Requested result fields, e.g. ``["u", "stress", "von_mises"]``.
            ``None`` keeps legacy behavior and lets ``solve()`` return whatever
            is cheaply available.
        strict: If True, ``solve()`` raises when any requested field is still
            unavailable after native extraction and any configured fallbacks.
    """

    fields: Optional[List[str]] = None
    strict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.fields is not None:
            result["fields"] = list(self.fields)
        if self.strict:
            result["strict"] = True
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResultOutput":
        fields = d.get("fields")
        if fields is not None:
            fields = [str(x) for x in fields]
        return cls(fields=fields, strict=bool(d.get("strict", False)))


@dataclass
class FallbackOutput:
    """Python-side fallback policy for ``solve()`` result extraction.

    Attributes:
        sampled_vtu: Controls whether ``solve()`` may reuse user-exported VTU
            files to backfill sampled fields/history when the native result
            bundle does not provide them directly.
            - ``"never"``: do not reuse exported VTUs
            - ``"auto"``: allow exported-VTU reuse when needed
            - ``"always"``: eagerly allow exported-VTU reuse
        temp_storage: Legacy compatibility knob from the removed temporary-VTU
            probe path. It is accepted but ignored.
        keep_temp_files: Legacy compatibility knob from the removed
            temporary-VTU probe path. It is accepted but ignored.
    """

    sampled_vtu: str = "never"
    temp_storage: str = "ram"
    keep_temp_files: bool = False

    def __post_init__(self):
        sampled_vtu = str(self.sampled_vtu).strip().lower()
        if sampled_vtu not in ("never", "auto", "always"):
            raise ValueError(f"sampled_vtu must be one of never/auto/always, got {self.sampled_vtu!r}")
        self.sampled_vtu = sampled_vtu

        temp_storage = str(self.temp_storage).strip().lower()
        if temp_storage not in ("ram", "disk"):
            raise ValueError(f"temp_storage must be 'ram' or 'disk', got {self.temp_storage!r}")
        self.temp_storage = temp_storage

    def to_dict(self) -> Dict[str, Any]:
        result = {"sampled_vtu": self.sampled_vtu}
        if self.temp_storage != "ram":
            result["temp_storage"] = self.temp_storage
        if self.keep_temp_files:
            result["keep_temp_files"] = True
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FallbackOutput":
        return cls(
            sampled_vtu=str(d.get("sampled_vtu", "never")),
            temp_storage=str(d.get("temp_storage", "ram")),
            keep_temp_files=bool(d.get("keep_temp_files", False)),
        )


@dataclass
class Output:
    """Output configuration - provides IDE autocomplete support.
    
    Attributes:
        directory: Output directory. Defaults to "output".
        paraview: Paraview output configuration (optional).
        json: Export JSON results (can be bool or string filename). Defaults to True.
        log: Log configuration (level, etc.).
        advanced: Advanced output options (e.g., save_time_sequence, save_solve_sequence_debug).
        save_paraview: Python-side convenience switch. If False, ``to_dict()``
            disables Paraview sequence output without requiring the caller to
            manually touch ``advanced.save_time_sequence`` or clear
            ``paraview.file_name``.
        save_vtu: Python-side convenience switch for step-VTU export only. If
            False, ``to_dict()`` clears ``paraview.file_name`` but leaves
            ``advanced.save_time_sequence`` unchanged so in-memory history can
            still be collected.
        result: Python-side result request for ``solve()``.
        fallback: Python-side result fallback policy for ``solve()``.
    
    Example:
        >>> output = Output(directory="results", paraview=ParaviewOutput(volume=True))
    """
    directory: str = "output"
    paraview: Optional[ParaviewOutput] = None
    json: Union[bool, str] = True
    restart_json: Optional[str] = None
    log: Optional[Union[OutputLog, Dict[str, Any]]] = None
    data: Optional[Union[OutputData, Dict[str, Any]]] = None
    advanced: Optional[Union[OutputAdvanced, Dict[str, Any]]] = None
    reference: Optional[Union[OutputReference, Dict[str, Any]]] = None
    stats: bool = False
    save_paraview: Optional[bool] = None
    save_vtu: Optional[bool] = None
    result: Optional[Union[ResultOutput, Dict[str, Any]]] = None
    fallback: Optional[Union[FallbackOutput, Dict[str, Any]]] = None

    def _ensure_log(self) -> OutputLog:
        if self.log is None:
            self.log = OutputLog()
        elif isinstance(self.log, dict):
            self.log = OutputLog.from_dict(self.log)
        return self.log

    def _ensure_paraview(self) -> ParaviewOutput:
        if self.paraview is None:
            self.paraview = ParaviewOutput()
        elif isinstance(self.paraview, dict):
            self.paraview = ParaviewOutput.from_dict(self.paraview)
        return self.paraview

    def _ensure_paraview_options(self) -> OutputParaviewOptions:
        paraview = self._ensure_paraview()
        if paraview.options is None:
            paraview.options = OutputParaviewOptions()
        elif isinstance(paraview.options, dict):
            paraview.options = OutputParaviewOptions.from_dict(paraview.options)
        return paraview.options

    def _ensure_advanced(self) -> OutputAdvanced:
        if self.advanced is None:
            self.advanced = OutputAdvanced()
        elif isinstance(self.advanced, dict):
            self.advanced = OutputAdvanced.from_dict(self.advanced)
        return self.advanced
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backend compatibility)."""
        result = {
            "directory": self.directory,
        }
        if isinstance(self.json, str):
            result["json"] = self.json
        elif self.json:
            result["json"] = True
        if self.restart_json:
            result["restart_json"] = self.restart_json
        
        paraview_dict = self.paraview.to_dict() if self.paraview is not None else None
        advanced_dict = _to_plain_value(self.advanced) if self.advanced is not None else None

        if self.save_paraview is False:
            if paraview_dict is None:
                paraview_dict = {}
            paraview_dict["file_name"] = ""
            if advanced_dict is None:
                advanced_dict = {}
            advanced_dict["save_time_sequence"] = False

        if self.save_vtu is False and paraview_dict is not None:
            paraview_dict["file_name"] = ""

        if paraview_dict is not None:
            result["paraview"] = paraview_dict
        if self.log is not None:
            log_dict = _to_plain_value(self.log)
            if log_dict:
                result["log"] = log_dict
        if self.data is not None:
            data_dict = _to_plain_value(self.data)
            if data_dict:
                result["data"] = data_dict
        if advanced_dict is not None:
            if advanced_dict:
                result["advanced"] = advanced_dict
        if self.reference is not None:
            reference_dict = _to_plain_value(self.reference)
            if reference_dict:
                result["reference"] = reference_dict
        if self.stats:
            result["stats"] = True
        return result

    def runtime_options(self) -> Dict[str, Any]:
        """Return Python-only runtime output controls for ``solve()``.

        These options are intentionally excluded from ``to_dict()`` because they
        are not part of the PolyFEM JSON schema.
        """
        result_cfg = None
        if isinstance(self.result, ResultOutput):
            result_cfg = self.result.to_dict()
        elif isinstance(self.result, dict):
            result_cfg = dict(self.result)

        fallback_cfg = None
        if isinstance(self.fallback, FallbackOutput):
            fallback_cfg = self.fallback.to_dict()
        elif isinstance(self.fallback, dict):
            fallback_cfg = dict(self.fallback)

        out: Dict[str, Any] = {}
        if result_cfg:
            out["result"] = result_cfg
        if fallback_cfg:
            out["fallback"] = fallback_cfg
        return out

    def resolve_relative_paths(self, base_dir: Union[str, PathLike[str]]) -> "Output":
        """Resolve relative output targets against ``base_dir`` in place.

        This is useful for scripts that load a JSON template and then redirect
        all outputs into a run-specific workspace without manually patching
        ``output.log.path``, ``output.paraview.file_name`` and ``output.json``.
        """
        base = Path(base_dir).resolve()

        if isinstance(self.log, OutputLog):
            path = self.log.path
            if isinstance(path, str) and path and not Path(path).is_absolute():
                self.log.path = str((base / path).resolve())
        elif isinstance(self.log, dict):
            log = dict(self.log)
            path = log.get("path")
            if isinstance(path, str) and path and not Path(path).is_absolute():
                log["path"] = str((base / path).resolve())
            self.log = log

        if self.paraview is not None:
            file_name = self.paraview.file_name
            if (
                isinstance(file_name, str)
                and file_name
                and not Path(file_name).is_absolute()
            ):
                self.paraview.file_name = str((base / file_name).resolve())

        if isinstance(self.json, str) and self.json and not Path(self.json).is_absolute():
            self.json = str((base / self.json).resolve())

        return self

    def request_results(self, fields: List[str], *, strict: bool = False) -> "Output":
        """Convenience helper for ``solve()`` result requests."""
        self.result = ResultOutput(fields=list(fields), strict=bool(strict))
        return self

    def configure_fallback(
        self,
        *,
        sampled_vtu: str = "auto",
        temp_storage: str = "ram",
        keep_temp_files: bool = False,
    ) -> "Output":
        """Convenience helper for exported-VTU backfill behavior.

        ``temp_storage`` / ``keep_temp_files`` are retained for backward
        compatibility but no longer affect runtime behavior.
        """
        self.fallback = FallbackOutput(
            sampled_vtu=sampled_vtu,
            temp_storage=temp_storage,
            keep_temp_files=keep_temp_files,
        )
        return self

    def configure_vtu_export(self, enabled: bool) -> "Output":
        """Convenience helper for step-VTU export without touching history."""
        self.save_vtu = bool(enabled)
        return self

    @classmethod
    def history(
        cls,
        *,
        directory: str = "output",
        json: Union[bool, str] = True,
        restart_json: Optional[str] = None,
        pvd: str = "results.pvd",
        surface: bool = False,
        wireframe: bool = False,
        points: bool = False,
        vismesh_rel_area: Optional[float] = 1e-5,
        skip_frame: Optional[int] = 1,
        high_order_mesh: bool = True,
        timestep_prefix: str = "step_",
        save_time_sequence: bool = True,
        save_vtu: bool = True,
    ) -> "Output":
        """Build the common history/paraview skeleton, then refine it in steps."""
        output = cls(directory=directory, json=json, restart_json=restart_json)
        output.set_paraview_sequence(
            file_name=pvd,
            surface=surface,
            wireframe=wireframe,
            points=points,
            vismesh_rel_area=vismesh_rel_area,
            skip_frame=skip_frame,
            high_order_mesh=high_order_mesh,
        )
        output.set_history_sequence(
            timestep_prefix=timestep_prefix,
            save_time_sequence=save_time_sequence,
        )
        output.configure_vtu_export(save_vtu)
        return output

    def set_log(
        self,
        *,
        path: str = "polyfem.log",
        level: Union[int, str] = "debug",
        file_level: Union[int, str] = "debug",
        quiet: bool = False,
    ) -> "Output":
        """Set the standard log block."""
        self.log = OutputLog(
            level=level,
            file_level=file_level,
            path=path,
            quiet=quiet,
        )
        return self

    def set_paraview_sequence(
        self,
        *,
        file_name: str = "results.pvd",
        volume: bool = True,
        surface: bool = False,
        wireframe: bool = False,
        points: bool = False,
        vismesh_rel_area: Optional[float] = 1e-5,
        skip_frame: Optional[int] = 1,
        high_order_mesh: bool = True,
        fields: Optional[List[str]] = None,
    ) -> "Output":
        """Configure the ParaView time-sequence block without touching field toggles."""
        paraview = self._ensure_paraview()
        paraview.volume = volume
        paraview.surface = surface
        paraview.wireframe = wireframe
        paraview.points = points
        paraview.file_name = file_name
        paraview.vismesh_rel_area = vismesh_rel_area
        paraview.skip_frame = skip_frame
        paraview.high_order_mesh = high_order_mesh
        if fields is not None:
            paraview.fields = list(fields)
        return self

    def enable_paraview_fields(
        self,
        *,
        use_hdf5: Optional[bool] = None,
        material: Optional[bool] = None,
        body_ids: Optional[bool] = None,
        contact_forces: Optional[bool] = None,
        friction_forces: Optional[bool] = None,
        normal_adhesion_forces: Optional[bool] = None,
        tangential_adhesion_forces: Optional[bool] = None,
        velocity: Optional[bool] = None,
        acceleration: Optional[bool] = None,
        scalar_values: Optional[bool] = None,
        tensor_values: Optional[bool] = None,
        discretization_order: Optional[bool] = None,
        nodes: Optional[bool] = None,
        forces: Optional[bool] = None,
        force_high_order: Optional[bool] = None,
        jacobian_validity: Optional[bool] = None,
    ) -> "Output":
        """Enable or disable ParaView field toggles with IDE-friendly keywords."""
        options = self._ensure_paraview_options()
        updates = {
            "use_hdf5": use_hdf5,
            "material": material,
            "body_ids": body_ids,
            "contact_forces": contact_forces,
            "friction_forces": friction_forces,
            "normal_adhesion_forces": normal_adhesion_forces,
            "tangential_adhesion_forces": tangential_adhesion_forces,
            "velocity": velocity,
            "acceleration": acceleration,
            "scalar_values": scalar_values,
            "tensor_values": tensor_values,
            "discretization_order": discretization_order,
            "nodes": nodes,
            "forces": forces,
            "force_high_order": force_high_order,
            "jacobian_validity": jacobian_validity,
        }
        for key, value in updates.items():
            if value is not None:
                setattr(options, key, value)
        return self

    def set_history_sequence(
        self,
        *,
        timestep_prefix: str = "step_",
        save_time_sequence: bool = True,
    ) -> "Output":
        """Configure the advanced history/time-sequence output block."""
        advanced = self._ensure_advanced()
        advanced.timestep_prefix = timestep_prefix
        advanced.save_time_sequence = save_time_sequence
        return self

    @classmethod
    def history_run(
        cls,
        *,
        directory: str = "output",
        json: Union[bool, str] = True,
        restart_json: Optional[str] = None,
        log_path: str = "polyfem.log",
        log_level: Union[int, str] = "debug",
        log_file_level: Union[int, str] = "debug",
        quiet: bool = False,
        pvd: str = "results.pvd",
        surface: bool = False,
        wireframe: bool = False,
        points: bool = False,
        vismesh_rel_area: Optional[float] = 1e-5,
        skip_frame: Optional[int] = 1,
        high_order_mesh: bool = True,
        material: bool = False,
        body_ids: bool = False,
        contact_forces: bool = False,
        friction_forces: bool = False,
        normal_adhesion_forces: bool = False,
        tangential_adhesion_forces: bool = False,
        velocity: bool = False,
        acceleration: bool = False,
        scalar_values: bool = True,
        tensor_values: bool = True,
        discretization_order: bool = True,
        nodes: bool = True,
        forces: bool = False,
        force_high_order: bool = False,
        jacobian_validity: bool = False,
        timestep_prefix: str = "step_",
        save_time_sequence: bool = True,
        requested_fields: Optional[List[str]] = None,
        strict: bool = False,
        save_vtu: bool = True,
    ) -> "Output":
        """Construct the common history + VTU output stack in one call."""
        output = cls.history(
            directory=directory,
            json=json,
            restart_json=restart_json,
            pvd=pvd,
            surface=surface,
            wireframe=wireframe,
            points=points,
            vismesh_rel_area=vismesh_rel_area,
            skip_frame=skip_frame,
            high_order_mesh=high_order_mesh,
            timestep_prefix=timestep_prefix,
            save_time_sequence=save_time_sequence,
            save_vtu=save_vtu,
        )
        output.set_log(
            path=log_path,
            level=log_level,
            file_level=log_file_level,
            quiet=quiet,
        )
        output.enable_paraview_fields(
            material=material,
            body_ids=body_ids,
            contact_forces=contact_forces,
            friction_forces=friction_forces,
            normal_adhesion_forces=normal_adhesion_forces,
            tangential_adhesion_forces=tangential_adhesion_forces,
            velocity=velocity,
            acceleration=acceleration,
            scalar_values=scalar_values,
            tensor_values=tensor_values,
            discretization_order=discretization_order,
            nodes=nodes,
            forces=forces,
            force_high_order=force_high_order,
            jacobian_validity=jacobian_validity,
        )
        if requested_fields is not None:
            output.request_results(list(requested_fields), strict=strict)
        return output
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Output":
        """Create Output from dictionary (backward compatibility)."""
        paraview = None
        if "paraview" in d:
            if isinstance(d["paraview"], dict):
                paraview = ParaviewOutput.from_dict(d["paraview"])
            else:
                paraview = d["paraview"]

        log_cfg = None
        if "log" in d and isinstance(d["log"], dict):
            log_cfg = OutputLog.from_dict(d["log"])
        elif "log" in d:
            log_cfg = d["log"]

        data_cfg = None
        if "data" in d and isinstance(d["data"], dict):
            data_cfg = OutputData.from_dict(d["data"])
        elif "data" in d:
            data_cfg = d["data"]

        advanced_cfg = None
        if "advanced" in d and isinstance(d["advanced"], dict):
            advanced_cfg = OutputAdvanced.from_dict(d["advanced"])
        elif "advanced" in d:
            advanced_cfg = d["advanced"]

        reference_cfg = None
        if "reference" in d and isinstance(d["reference"], dict):
            reference_cfg = OutputReference.from_dict(d["reference"])
        elif "reference" in d:
            reference_cfg = d["reference"]
        
        result_cfg = None
        if "result" in d and isinstance(d["result"], dict):
            result_cfg = ResultOutput.from_dict(d["result"])

        fallback_cfg = None
        if "fallback" in d and isinstance(d["fallback"], dict):
            fallback_cfg = FallbackOutput.from_dict(d["fallback"])

        return cls(
            directory=d.get("directory", "output"),
            paraview=paraview,
            json=d.get("json", True),
            restart_json=d.get("restart_json"),
            log=log_cfg,
            data=data_cfg,
            advanced=advanced_cfg,
            reference=reference_cfg,
            stats=bool(d.get("stats", False)),
            save_paraview=d.get("save_paraview"),
            save_vtu=d.get("save_vtu"),
            result=result_cfg,
            fallback=fallback_cfg,
        )


__all__ = [
    "OutputLog",
    "OutputParaviewOptions",
    "ParaviewOutput",
    "OutputDataAdvanced",
    "OutputData",
    "OutputAdvanced",
    "OutputReference",
    "ResultOutput",
    "FallbackOutput",
    "Output",
]
