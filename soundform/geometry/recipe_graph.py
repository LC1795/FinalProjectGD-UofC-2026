"""Typed, locally evaluated graph operations for AI pavilion generation."""

from __future__ import annotations

import re

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator

from soundform.geometry.recipe import _number
from soundform.geometry.recipe_sections import build_section_shape

PARAMETER_PATTERN = r"^p_[A-Za-z0-9_]{1,34}$"
# Input kinds are used by both the response schema and the local evaluator.
NODES = {
    "perimeter": ("curve", dict(radius="number", variation="number", lobes="number", phase="number")),
    "rail": ("curve", dict(points="xyz_list")),
    "distribute": ("points", dict(curve="curve", count="number", entrance_bearing="number", entrance_width="number")),
    "orient": ("frames", dict(points="points", angle="number", follow_tangent="boolean")),
    "instance": ("mesh", dict(mesh="mesh", frames="frames")),
    "sample_sound": ("sound", dict(points="points")),
    "rotate_boards": ("frames", dict(sound="sound", quiet_angle="number", loud_angle="number", strength="number")),
    "boards": ("mesh", dict(frames="frames", width="number", thickness="number", height="number")),
    "path": ("paths", dict(points="xyz_list")),
    "branch_paths": ("paths", dict(stem="xyz_list", tips="xyz_list")),
    "taper": ("radii", dict(paths="paths", base_radius="number", tip_radius="number")),
    "pipe": ("mesh", dict(taper="radii")),
    "loft": ("mesh", dict(sections="sections", closed="boolean", cap="boolean")),
    "polyline": ("curve", dict(points="xyz_list", closed="boolean")),
    "ellipse": ("curve", dict(x="number", y="number", z="number", radius_x="number", radius_y="number")),
    "box": ("mesh", dict(x="number", y="number", z="number", width="number", depth="number", height="number")),
    "cylinder": ("mesh", dict(x="number", y="number", z="number", radius="number", height="number")),
    "extrude": ("mesh", dict(profile="curve", x="number", y="number", z="number", cap="boolean")),
    "loft_curves": ("mesh", dict(start="curve", end="curve", cap="boolean")),
    "sweep": ("mesh", dict(rail="curve", profile="curve", end_scale="number")),
    "move": ("mesh", dict(mesh="mesh", x="number", y="number", z="number")),
    "rotate": ("mesh", dict(mesh="mesh", angle="number", x="number", y="number")),
    "scale": ("mesh", dict(mesh="mesh", x="number", y="number", z="number")),
    "mirror": ("mesh", dict(mesh="mesh", x_plane="number")),
    "merge": ("mesh", dict(first="mesh", second="mesh")),
}
DESCRIPTIONS = {
    "perimeter": "Closed wavy plan curve. Radius and variation are metres; lobes is an integer; phase is a bearing.",
    "rail": "Open smooth path through explicit XYZ points.",
    "distribute": "Equally spaced samples by distance along a curve. Optional radial entrance gap via width (0 means none).",
    "orient": "Create placement frames from distributed points. Angle is measured from the tangent if follow_tangent is true, or from local +X otherwise.",
    "instance": "Copy any mesh into the placement frames. Its local origin is the insertion point and +X follows the frame yaw.",
    "sample_sound": "Attach measured directional energy (0–1) to each point using its bearing from the listener.",
    "rotate_boards": "Interpolate between quiet and loud yaw offsets measured from the local tangent. Strength is 0–1.",
    "boards": "Place a vertical board per frame; width is along local yaw, thickness across it, height vertical.",
    "path": "Provide one XYZ path for a tapered pipe.",
    "branch_paths": "Create a shared stem and one curved path from its top to each XYZ tip.",
    "taper": "Assign a smoothly decreasing radius from base to tip along each path.",
    "pipe": "Build round solid meshes from the tapered paths.",
    "loft": "Join inline XYZ profiles in order. Unequal vertex counts are matched by normalized distance. Closed caps require simple planar end loops.",
    "polyline": "Connect XYZ points by straight segments. Set closed to join the last point to the first.",
    "ellipse": "Closed XY ellipse with centre x,y,z and positive radii. Equal radii form a circle.",
    "box": "Solid box with base centre x,y,z. Width runs along X and depth along Y.",
    "cylinder": "Solid vertical cylinder with base centre x,y,z.",
    "extrude": "Translate a curve by vector x,y,z and join the two sections. Caps require a simple closed planar profile.",
    "loft_curves": "Join two open curves or two closed curves. Curves should have aligned starts and consistent orientation. Caps require simple planar end loops.",
    "sweep": "Carry a simple closed XY profile along an open rail. The profile XY coordinates are local offsets. Positive end_scale tapers or expands it.",
    "move": "Translate a mesh by vector x,y,z.",
    "rotate": "Rotate a mesh counterclockwise in plan about vertical axis through x,y.",
    "scale": "Scale a mesh about the local origin using positive factors along X,Y,Z.",
    "mirror": "Reflect a mesh across the vertical plane X=x_plane.",
    "merge": "Collect two mesh outputs without cutting or fusing them.",
}


def validate_graph(proposal):
    if not isinstance(proposal.get("rationale"), str):
        raise ValueError("The graph needs a rationale.")
    parameters, nodes = proposal.get("parameters"), proposal.get("nodes")
    if not isinstance(parameters, list) or not 1 <= len(parameters) <= 16:
        raise ValueError("Declare 1–16 graph parameters.")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 24:
        raise ValueError("Declare 1–24 graph nodes.")
    lookup = {}
    for p in parameters:
        name = p.get("name") if isinstance(p, dict) else None
        if not isinstance(name, str) or not re.fullmatch(PARAMETER_PATTERN, name) or name in lookup:
            raise ValueError(f"Invalid or duplicate graph parameter {name!r}.")
        if not isinstance(p.get("label"), str) or not p["label"].strip() or p.get("unit") not in ("", "m", "°", "%"):
            raise ValueError(f"Parameter {name} needs a label and supported unit.")
        low, high, value, step = (_number(p.get(key)) for key in ("min", "max", "value", "step"))
        if not low < high or not low <= value <= high or not 0.001 <= step <= high - low:
            raise ValueError(f"Invalid range, value or step for {name}.")
        lookup[name] = value
    outputs = []
    for index, node in enumerate(nodes):
        kind = node.get("op") if isinstance(node, dict) else None
        if not isinstance(kind, str) or kind not in NODES:
            raise ValueError(f"Node {index} has an unsupported operation.")
        output, inputs = NODES[kind]
        values = node.get("inputs")
        if not isinstance(values, dict) or set(values) != set(inputs):
            raise ValueError(f"Node {index} ({kind}) requires inputs {', '.join(inputs)}.")
        if not isinstance(node.get("label"), str) or not node["label"].strip():
            raise ValueError(f"Node {index} needs a display label.")
        for key, value_type in inputs.items():
            try:
                value = values[key]
                if value_type == "number":
                    _scalar(value, lookup)
                elif value_type == "xyz_list":
                    _points(value, lookup)
                elif value_type == "sections":
                    if not isinstance(value, list) or not 2 <= len(value) <= 16:
                        raise ValueError("Use 2–16 loft sections.")
                    for section in value:
                        _points(section, lookup)
                elif value_type == "boolean":
                    if type(value) is not bool:
                        raise ValueError("Use a boolean.")
                elif type(value) is not int or not 0 <= value < index or outputs[value] != value_type:
                    raise ValueError(f"Use the index of an earlier node whose output is {value_type}.")
            except ValueError as exc:
                raise ValueError(f"Node {index} ({kind}), {key}: {exc}") from exc
        outputs.append(output)
    if "mesh" not in outputs:
        raise ValueError("The graph has no mesh output.")
    return {"recipe_version": 2, "rationale": proposal["rationale"][:1200],
            "parameters": parameters, "nodes": nodes}


def _scalar(value, parameters):
    if isinstance(value, str):
        if value not in parameters:
            raise ValueError(f"{value!r} is not a declared parameter reference.")
        return parameters[value]
    return _number(value)


def _points(values, parameters):
    if not isinstance(values, list) or not 2 <= len(values) <= 24:
        raise ValueError("Provide 2–24 points.")
    if any(not isinstance(point, list) or len(point) != 3 for point in values):
        raise ValueError("Each point needs exactly three XYZ coordinates.")
    return np.array([[_scalar(coordinate, parameters) for coordinate in point] for point in values])


def _bearing(points):
    return np.degrees(np.arctan2(-points[:, 0], points[:, 1])) % 360


def _perimeter(radius, variation, lobes, phase):
    if radius <= 0 or not 0 <= variation < radius or not 1 <= lobes <= 12:
        raise ValueError("Use a positive radius, smaller nonnegative variation and 1–12 lobes.")
    angle = np.linspace(0, 360, 721)
    radii = radius + variation * np.cos(np.deg2rad(round(lobes) * (angle - phase)))
    return np.c_[-radii * np.sin(np.deg2rad(angle)), radii * np.cos(np.deg2rad(angle)), np.zeros_like(angle)]


def _distribute(curve, count, entrance_bearing, entrance_width):
    if not 2 <= count <= 240 or entrance_width < 0:
        raise ValueError("Use 2–240 samples and a nonnegative entrance width.")
    distances = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    if np.any(distances < 1e-7):
        raise ValueError("Curve has coincident consecutive points.")
    lengths = np.r_[0.0, distances.cumsum()]
    closed = np.allclose(curve[0], curve[-1])
    positions = np.linspace(0, lengths[-1], round(count), endpoint=not closed)
    points = np.column_stack([np.interp(positions, lengths, axis) for axis in curve.T])
    # Use the path tangent before removing any entrance samples.
    segments = np.clip(np.searchsorted(lengths, positions, side="right") - 1, 0, len(distances) - 1)
    tangent = curve[segments + 1] - curve[segments]
    yaw = np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0]))
    separation = np.abs((_bearing(points) - entrance_bearing + 180) % 360 - 180)
    radius = np.linalg.norm(points[:, :2], axis=1)
    gap = np.degrees(np.arcsin(np.clip(entrance_width / (2 * np.maximum(radius, 1e-8)), 0, 1)))
    keep = (separation > gap) if entrance_width else np.ones(len(points), dtype=bool)
    return points[keep], yaw[keep]


def _boards(frames, width, thickness, height):
    if min(width, thickness, height) < 0.005:
        raise ValueError("Board dimensions must be at least 5 mm.")
    points, yaw, gap_bearing, gap_width = frames
    if gap_width:
        clearance = (gap_width + np.hypot(width, thickness)) / 2
        radius = np.linalg.norm(points[:, :2], axis=1)
        delta = np.deg2rad((_bearing(points) - gap_bearing + 180) % 360 - 180)
        keep = (np.cos(delta) <= 0) | (radius * np.abs(np.sin(delta)) >= clearance)
        points, yaw = points[keep], yaw[keep]
    pieces = []
    for point, rotation in zip(points, yaw, strict=True):
        box = trimesh.creation.box(extents=(width, thickness, height))
        box.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(rotation), (0, 0, 1)))
        box.apply_translation(point + (0, 0, height / 2))
        pieces.append(box)
    return pieces


def _is_closed(curve):
    return np.linalg.norm(curve[0] - curve[-1]) < 1e-7


def _loft(sections, closed, cap):
    profiles, knots = [], []
    for profile in sections:
        if closed and _is_closed(profile):
            profile = profile[:-1]
        if len(profile) < (3 if closed else 2):
            raise ValueError("Closed profiles need at least three vertices and open profiles need two.")
        path = np.vstack((profile, profile[0])) if closed else profile
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        if np.any(distances < 1e-7):
            raise ValueError("Loft profiles contain coincident consecutive points.")
        lengths = np.r_[0.0, distances.cumsum()]
        profiles.append(path)
        knots.append(lengths / lengths[-1])
    if len({len(profile) for profile in profiles}) != 1:
        # Preserve all corners while giving each profile the same sampling positions.
        fractions = np.sort(np.concatenate(knots))
        fractions = fractions[np.r_[True, np.diff(fractions) > 1e-8]]
        profiles = [
            np.column_stack([np.interp(fractions, distances, axis) for axis in path.T])
            for path, distances in zip(profiles, knots, strict=True)
        ]
    sections = np.asarray(profiles)
    if closed:
        sections = sections[:, :-1]
    return [build_section_shape("loft", {"sections": sections, "cap": int(cap),
                                         "closed_profile": int(closed)})]


def _primitive(kind, args):
    dimensions = (args["width"], args["depth"], args["height"]) if kind == "box" else (args["radius"], args["height"])
    if min(dimensions) < 0.005:
        raise ValueError("Primitive dimensions must be at least 5 mm.")
    if kind == "box":
        mesh = trimesh.creation.box(extents=dimensions)
    else:
        mesh = trimesh.creation.cylinder(radius=args["radius"], height=args["height"], sections=48)
    mesh.apply_translation((args["x"], args["y"], args["z"] + args["height"] / 2))
    return [mesh]


def _transform(kind, args):
    matrix = np.eye(4)
    if kind == "move":
        matrix[:3, 3] = [args[axis] for axis in "xyz"]
    elif kind == "rotate":
        matrix = trimesh.transformations.rotation_matrix(
            np.deg2rad(args["angle"]), (0, 0, 1), (args["x"], args["y"], 0)
        )
    elif kind == "scale":
        factors = [args[axis] for axis in "xyz"]
        if min(factors) <= 0:
            raise ValueError("Scale factors must be positive.")
        matrix[np.arange(3), np.arange(3)] = factors
    else:
        matrix[0, 0], matrix[0, 3] = -1, 2 * args["x_plane"]
    return [piece.copy().apply_transform(matrix) for piece in args["mesh"]]


def _evaluate(kind, args, directional_energy):
    if kind == "perimeter":
        return _perimeter(**args)
    if kind == "rail":
        points = args["points"]
        lengths = np.r_[0.0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
        if np.any(np.diff(lengths) < 1e-7):
            raise ValueError("Rail has coincident consecutive points.")
        return PchipInterpolator(lengths, points, axis=0)(np.linspace(0, lengths[-1], 8 * len(points)))
    if kind == "distribute":
        points, yaw = _distribute(**args)
        return points, yaw, args["entrance_bearing"], args["entrance_width"]
    if kind == "orient":
        points, yaw, bearing, width = args["points"]
        rotation = yaw + args["angle"] if args["follow_tangent"] else np.full(len(points), args["angle"])
        return points, rotation, bearing, width
    if kind == "instance":
        points, yaw, _, _ = args["frames"]
        pieces = args["mesh"]
        if len(points) * len(pieces) > 480 or len(points) * sum(len(piece.faces) for piece in pieces) > 160_000:
            raise ValueError("The instances exceed the geometry budget. Reduce the sample count or source mesh detail.")
        copies = []
        for point, angle in zip(points, yaw, strict=True):
            matrix = trimesh.transformations.rotation_matrix(np.deg2rad(angle), (0, 0, 1))
            matrix[:3, 3] = point
            copies.extend(piece.copy().apply_transform(matrix) for piece in pieces)
        return copies
    if kind == "sample_sound":
        points, yaw, bearing, width = args["points"]
        strength = (np.interp(_bearing(points), *directional_energy, period=360)
                    if directional_energy is not None else np.zeros(len(points)))
        return points, yaw, bearing, width, strength
    if kind == "rotate_boards":
        points, yaw, bearing, width, energy = args["sound"]
        strength = args["strength"]
        if not 0 <= strength <= 1:
            raise ValueError("Sound response strength must be between 0 and 1.")
        offset = args["quiet_angle"] + strength * energy * (args["loud_angle"] - args["quiet_angle"])
        return points, yaw + offset, bearing, width
    if kind == "boards":
        return _boards(**args)
    if kind == "path":
        return [args["points"]]
    if kind == "branch_paths":
        stem = args["stem"]
        return [stem, *[np.vstack((stem[-1], tip)) for tip in args["tips"]]]
    if kind == "taper":
        if min(args["base_radius"], args["tip_radius"]) < 0.005:
            raise ValueError("Pipe radii must be at least 5 mm.")
        return [(path, np.linspace(args["base_radius"], args["tip_radius"], len(path))) for path in args["paths"]]
    if kind == "pipe":
        return [build_section_shape("pipe", {"rail": path, "radii": radii}) for path, radii in args["taper"]]
    if kind == "loft":
        return _loft(args["sections"], args["closed"], args["cap"])
    if kind == "polyline":
        points = args["points"]
        if args["closed"] and not _is_closed(points):
            points = np.vstack((points, points[0]))
        if args["closed"] and len(points) < 4:
            raise ValueError("A closed polyline needs at least three vertices.")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-7):
            raise ValueError("A polyline needs distinct consecutive points.")
        return points
    if kind == "ellipse":
        if min(args["radius_x"], args["radius_y"]) < 0.005:
            raise ValueError("Ellipse radii must be at least 5 mm.")
        angle = np.linspace(0, 2 * np.pi, 49)
        return np.column_stack((args["x"] + args["radius_x"] * np.cos(angle),
                                args["y"] + args["radius_y"] * np.sin(angle),
                                np.full(len(angle), args["z"])))
    if kind in ("box", "cylinder"):
        return _primitive(kind, args)
    if kind == "extrude":
        profile = args["profile"]
        vector = np.array([args[axis] for axis in "xyz"])
        if np.linalg.norm(vector) < 1e-7:
            raise ValueError("The extrusion vector must be nonzero.")
        return _loft([profile, profile + vector], _is_closed(profile), args["cap"])
    if kind == "loft_curves":
        start, end = args["start"], args["end"]
        closed = _is_closed(start)
        if closed != _is_closed(end):
            raise ValueError("Loft curves must both be open or both closed.")
        return _loft([start, end], closed, args["cap"])
    if kind == "sweep":
        rail, profile = args["rail"], args["profile"]
        if _is_closed(rail) or not _is_closed(profile) or not np.allclose(profile[:, 2], 0):
            raise ValueError("Sweep requires an open rail and a closed profile in the XY plane at Z=0.")
        if len(rail) > 240 or len(profile) > 97:
            raise ValueError("Sweep supports up to 240 rail and 96 profile vertices.")
        return [build_section_shape("sweep", {"rail": rail, "profile": profile[:-1, :2],
                                             "scales": np.linspace(1, args["end_scale"], len(rail))})]
    if kind in ("move", "rotate", "scale", "mirror"):
        return _transform(kind, args)
    if kind == "merge":
        return [*args["first"], *args["second"]]
    raise ValueError(f"Unsupported node {kind}.")


def build_graph(graph, params, directional_energy):
    parameters = {p["name"]: p["value"] for p in graph["parameters"]}
    values, meshes, labels = [], [], []
    used_meshes = {
        node["inputs"][key] for node in graph["nodes"]
        for key, input_type in NODES[node["op"]][1].items() if input_type == "mesh"
    }
    pieces, triangles = 0, 0
    for index, node in enumerate(graph["nodes"]):
        kind = node["op"]
        output, types = NODES[kind]
        try:
            args = {}
            for key, value_type in types.items():
                value = node["inputs"][key]
                if value_type == "number":
                    value = _scalar(value, parameters)
                elif value_type == "xyz_list":
                    value = _points(value, parameters)
                elif value_type == "sections":
                    value = [_points(section, parameters) for section in value]
                elif value_type != "boolean":
                    value = values[value]
                args[key] = value
            result = _evaluate(kind, args, directional_energy)
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(f"Node {index} ({node['label']}): {exc}") from exc
        values.append(result)
        if output == "mesh" and result:
            node_triangles = sum(len(piece.faces) for piece in result)
            if len(result) > 480 or node_triangles > 160_000:
                raise ValueError("A node exceeds the geometry budget. Reduce repetitions or profile detail.")
            if index in used_meshes:
                continue
            pieces += len(result)
            triangles += node_triangles
            if pieces > 480 or triangles > 160_000:
                raise ValueError("The graph exceeds the geometry budget. Reduce repetitions or profile detail.")
            mesh = trimesh.util.concatenate(result)
            meshes.append(mesh)
            labels.append(np.full(len(mesh.faces), f"{index + 1}. {node['label'][:60]}"))
    if not meshes:
        raise ValueError("The terminal nodes produced no geometry.")
    mesh = trimesh.util.concatenate(meshes)
    mesh.apply_translation((params.protected_center_x, params.protected_center_y, 0))
    mesh.face_attributes["pavilion_part"] = np.concatenate(labels)
    return mesh
