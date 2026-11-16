"""Evaluate model-authored geometry recipes as data, with bounded expressions."""

from __future__ import annotations

import ast
import math
import keyword
import operator
import re

import numpy as np
import trimesh

from soundform.geometry.recipe_sections import build_section_shape

MAX_INSTANCES = 480
FUNCTIONS = {
    "sin": lambda x: math.sin(math.radians(x)),
    "cos": lambda x: math.cos(math.radians(x)),
    "bearing": lambda x, y: math.degrees(math.atan2(-x, y)) % 360,
    "angle_delta": lambda a, b: abs((a - b + 180) % 360 - 180),
    "min": min, "max": max, "abs": abs, "floor": math.floor, "ceil": math.ceil,
    "sqrt": math.sqrt,
}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Mod: operator.mod}
COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
           ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
SHAPES = {"box": ("x", "y", "z", "width", "depth", "height", "yaw"),
          "cylinder": ("x", "y", "z", "radius", "height"),
          "panel": ("a", "b", "c", "d", "thickness"),
          "pipe": ("rail", "radii"),
          "sweep": ("rail", "profile", "scales"),
          "loft": ("sections", "cap")}
SECTION_SHAPES = {"pipe", "sweep", "loft"}
MAX_TRIANGLES = 160_000
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,35}")
RESERVED = {"i", "t", "n", "sound", "protected_radius", "max_height", *FUNCTIONS}


def _number(value, *, limit=10_000):
    if type(value) not in (int, float):
        raise ValueError(f"Expected a number, got {type(value).__name__}.")
    if not -limit <= value <= limit:
        raise ValueError(f"Expected a finite number within ±{limit:,}, got {value!r}.")
    return float(value)


def _expression_number(value):
    return _number(value, limit=10**12)


def _parse(value):
    if type(value) is bool:
        return ast.Constant(value)
    if type(value) in (int, float):
        return ast.Constant(_expression_number(value))
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError("Each recipe expression must be a number or a formula of at most 200 characters.")
    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid geometry formula: {value}") from exc
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
               ast.Compare, ast.IfExp, ast.Call, ast.BoolOp, ast.And, ast.Or, ast.Not,
               ast.UAdd, ast.USub, *BINARY, *COMPARE)
    nodes = list(ast.walk(tree))
    if len(nodes) > 60 or any(not isinstance(node, allowed) for node in nodes):
        raise ValueError(f"Unsupported geometry formula: {value}")
    return tree.body


def _eval(node, context, functions):
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is bool else _expression_number(node.value)
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"Unknown variable {node.id!r}. Use a declared parameter or an earlier local expression.")
        return context[node.id]
    if isinstance(node, ast.BinOp):
        return _expression_number(BINARY[type(node.op)](_eval(node.left, context, functions), _eval(node.right, context, functions)))
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, context, functions)
        if isinstance(node.op, ast.Not):
            return not value
        return _expression_number(-value if isinstance(node.op, ast.USub) else value)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, context, functions)
        for op, rhs in zip(node.ops, node.comparators, strict=True):
            right = _eval(rhs, context, functions)
            if not COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        branch = node.body if _eval(node.test, context, functions) else node.orelse
        return _eval(branch, context, functions)
    if isinstance(node, ast.BoolOp):
        values = (_eval(item, context, functions) for item in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions
            and not node.keywords):
        return _expression_number(functions[node.func.id](*(_eval(arg, context, functions) for arg in node.args)))
    raise ValueError("The recipe used an unknown variable or an unsupported operation.")


def validate_recipe(proposal):
    if not isinstance(proposal.get("rationale"), str):
        raise ValueError("The recipe needs a written rationale.")
    parameters = proposal.get("parameters")
    groups = proposal.get("groups")
    if not isinstance(parameters, list) or not 1 <= len(parameters) <= 16:
        raise ValueError("Declare 1 to 16 editable parameters.")
    if not isinstance(groups, list) or not 1 <= len(groups) <= 10:
        raise ValueError("Declare 1 to 10 geometry groups.")
    names = set(RESERVED)
    normalized_parameters = []
    for index, parameter in enumerate(parameters, 1):
        if not isinstance(parameter, dict):
            raise ValueError("Each parameter must be an object.")
        name, label, unit = (parameter.get(key) for key in ("name", "label", "unit"))
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or keyword.iskeyword(name):
            received = repr(name)[:75]
            raise ValueError(f"Parameter {index} has invalid name {received}. Use a single identifier such as branch_radius.")
        if name in RESERVED:
            raise ValueError(f"Parameter {index} uses reserved name {name!r}. Rename it and update its formulas.")
        if name in names:
            raise ValueError(f"Parameter {index} repeats name {name!r}. Use distinct names and update their formulas.")
        label = " ".join(label.split()) if isinstance(label, str) else ""
        label = (label or name.replace("_", " ").capitalize())[:60]
        unit = unit.strip() if isinstance(unit, str) else ""
        aliases = {
            "meter": "m", "meters": "m", "metre": "m", "metres": "m",
            "deg": "°", "degree": "°", "degrees": "°",
            "percent": "%", "percentage": "%",
            "unitless": "", "none": "", "n/a": "", "count": "", "ratio": "",
        }
        unit = aliases.get(unit.casefold(), unit)
        # Units are display suffixes. Formulas decide what the numbers mean.
        if len(unit) > 16 or not all(char.isalnum() or char in " °%/²³._-×" for char in unit):
            unit = ""
        low, high, value, step = (_number(parameter.get(key)) for key in ("min", "max", "value", "step"))
        if not low < high or not low <= value <= high or not 0.001 <= step <= high - low:
            raise ValueError(f"Invalid range, default, or step for {label}.")
        names.add(name)
        normalized_parameters.append({**parameter, 'label': label, 'unit': unit})
    issues = []
    for index, group in enumerate(groups, 1):
        if not isinstance(group, dict):
            issues.append(f"Group {index} must be an object.")
            continue
        label, shape = group.get("label"), group.get("shape")
        label = (" ".join(label.split())[:60] if isinstance(label, str) else "") or f"Group {index}"
        if not isinstance(shape, str) or shape not in SHAPES:
            received = repr(shape)[:80] if isinstance(shape, str) else "missing or non-text"
            issues.append(f"Group {index} ({label}): unsupported shape {received}. Use {', '.join(SHAPES)}.")
            continue
        fields = group.get("geometry")
        required = set(SHAPES[shape])
        optional = {"closed_profile"} if shape == "loft" else set()
        if not isinstance(fields, dict) or not required <= fields.keys() <= required | optional:
            issues.append(f"{label}: {shape} requires {', '.join(SHAPES[shape])}.")
            continue
        try:
            _compile_geometry(shape, fields)
        except ValueError as exc:
            issues.append(f"{label}: {exc}")
    if issues:
        raise ValueError(" ".join(issues))
    groups = [{**group, "label": (" ".join(group["label"].split())[:60]
               if isinstance(group.get("label"), str) else "") or f"Group {index}"}
              for index, group in enumerate(groups, 1)]
    return {"recipe_version": 1, "rationale": proposal["rationale"][:1200],
            "parameters": normalized_parameters, "groups": groups}


def _compile_geometry(shape, fields):
    def sequence(items, low, high, label):
        if not isinstance(items, list) or not low <= len(items) <= high:
            received = f"{len(items)} entries" if isinstance(items, list) else type(items).__name__
            raise ValueError(f"{label} must have {low}–{high} entries. Received {received}.")
        return items

    def points(items, size, minimum, maximum, label):
        return tuple(tuple(_parse(value) for value in sequence(point, size, size, "A point"))
                     for point in sequence(items, minimum, maximum, label))

    result = {}
    for key, value in fields.items():
        if shape == "panel" and key in ("a", "b", "c", "d"):
            result[key] = tuple(_parse(x) for x in sequence(value, 3, 3, "A panel corner"))
        elif key == "rail":
            result[key] = points(value, 3, 2, 24, "A rail")
        elif key == "profile":
            result[key] = points(value, 2, 3, 20, "A sweep profile")
        elif key in ("radii", "scales"):
            result[key] = tuple(_parse(x) for x in sequence(value, 2, 24, key))
        elif key == "sections":
            loops = sequence(value, 2, 16, "A loft")
            open_profile = fields.get("closed_profile") in (0, False, "0", "False")
            if "closed_profile" not in fields and fields.get("cap") in (0, False, "0", "False"):
                open_profile = all(isinstance(loop, list) and len(loop) == 2 for loop in loops)
                if open_profile:
                    result["closed_profile"] = _parse(0)
            if not open_profile:
                loops = [loop[:-1] if isinstance(loop, list) and len(loop) >= 4 and loop[0] == loop[-1]
                         else loop for loop in loops]
            sections = [points(loop, 3, 2 if open_profile else 3, 20, f"Loft section {index}")
                        for index, loop in enumerate(loops, 1)]
            if len(set(map(len, sections))) != 1:
                raise ValueError("Loft sections must have matching point counts.")
            result[key] = tuple(sections)
        else:
            result[key] = _parse(value)
    return result


def _evaluate_at(formula, context, functions, location, *, numeric=False, allow_bool=False):
    try:
        result = _eval(formula, context, functions)
        if allow_bool and type(result) is bool:
            return int(result)
        return _number(result) if numeric else result
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError(f"{location}, formula {ast.unparse(formula)[:140]!r}: {exc}") from exc


def _geometry_value(formula, context, functions, location, *, allow_bool=False):
    if isinstance(formula, tuple):
        return np.asarray([_geometry_value(item, context, functions, f"{location}[{index}]")
                           for index, item in enumerate(formula)])
    return _evaluate_at(formula, context, functions, location, numeric=True, allow_bool=allow_bool)


def _make_shape(shape, geometry):
    if shape in SECTION_SHAPES:
        return build_section_shape(shape, geometry)
    if shape == "panel":
        corners = np.array([geometry[key] for key in ("a", "b", "c", "d")])
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        length = np.linalg.norm(normal)
        if length < 1e-7 or geometry["thickness"] < 0.005:
            raise ValueError("A panel needs non-collinear corners and at least 5 mm thickness.")
        backing = corners + normal / length * geometry["thickness"]
        faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
        for i in range(4):
            j = (i + 1) % 4
            faces.extend(([i, j, j + 4], [i, j + 4, i + 4]))
        mesh = trimesh.Trimesh(vertices=np.vstack([corners, backing]), faces=faces, process=False)
        mesh.fix_normals()
        if mesh.volume <= 1e-8 or np.min(mesh.area_faces) <= 1e-8:
            raise ValueError("A panel has a degenerate or crossed footprint.")
        return mesh
    dimensions = [geometry[key] for key in (("width", "depth", "height") if shape == "box" else ("radius", "height"))]
    if min(dimensions) < 0.005:
        raise ValueError("Every solid dimension must be at least 5 mm.")
    if shape == "box":
        mesh = trimesh.creation.box(extents=dimensions)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(geometry["yaw"]), (0, 0, 1)))
    else:
        mesh = trimesh.creation.cylinder(radius=geometry["radius"], height=geometry["height"], sections=16)
    mesh.apply_translation((geometry["x"], geometry["y"], geometry["z"] + geometry["height"] / 2))
    return mesh


def build_recipe(proposal, params, directional_energy):
    context = {parameter["name"]: parameter["value"] for parameter in proposal["parameters"]}
    context.update(protected_radius=params.protected_radius, max_height=params.max_height)

    def sound(angle):
        if directional_energy is None:
            return 0.0
        bearings, values = directional_energy
        return float(np.interp(angle, bearings, values, period=360))

    functions = {**FUNCTIONS, "sound": sound}
    meshes, labels = [], []
    budget = MAX_INSTANCES
    triangle_budget = MAX_TRIANGLES
    for index, group in enumerate(proposal["groups"], 1):
        try:
            count = int(_evaluate_at(_parse(group.get("count", 1)), context, functions, "count", numeric=True))
            if not 1 <= count <= budget:
                raise ValueError(f"A recipe may budget at most {MAX_INSTANCES} instances across its groups.")
            budget -= count
            shape = group["shape"]
            geometry = _compile_geometry(shape, group["geometry"])
            variables = group.get("let", {})
            if not isinstance(variables, dict) or len(variables) > 12:
                raise ValueError("A group may declare up to 12 local expressions.")
            locals_ = {}
            for name, value in variables.items():
                if not IDENTIFIER.fullmatch(name) or keyword.iskeyword(name) or name in context or name in RESERVED:
                    raise ValueError(f"Invalid local name: {name}")
                locals_[name] = _parse(value)
            condition = _parse(group.get("when", "True"))
            group_meshes = []
            for i in range(count):
                values = {**context, "i": i, "n": count, "t": i / max(1, count - 1)}
                for name, formula in locals_.items():
                    values[name] = _evaluate_at(formula, values, functions, f"instance {i + 1}, let.{name}")
                if not _evaluate_at(condition, values, functions, f"instance {i + 1}, when"):
                    continue
                dimensions = {
                    key: _geometry_value(formula, values, functions, f"instance {i + 1}, geometry.{key}",
                                         allow_bool=shape == "loft" and key in ("cap", "closed_profile"))
                    for key, formula in geometry.items()
                }
                piece = _make_shape(shape, dimensions)
                triangle_budget -= len(piece.faces)
                if triangle_budget < 0:
                    raise ValueError(f"Reduce section detail or repetitions to stay within {MAX_TRIANGLES:,} triangles.")
                group_meshes.append(piece)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise ValueError(f"{group['label']}: {exc}") from exc
        if group_meshes:
            mesh = trimesh.util.concatenate(group_meshes)
            meshes.append(mesh)
            labels.append(np.full(len(mesh.faces), f"{index}. {group['label']}"))
    if not meshes:
        raise ValueError("The recipe produced no geometry. Check the group conditions.")
    mesh = trimesh.util.concatenate(meshes)
    mesh.apply_translation((params.protected_center_x, params.protected_center_y, 0))
    mesh.face_attributes["pavilion_part"] = np.concatenate(labels)
    return mesh
