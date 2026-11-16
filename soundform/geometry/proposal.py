"""Build and evaluate pavilion proposals from declarative geometry instructions."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from soundform.acoustics.protection_estimator import estimate_protection
from soundform.acoustics.response import compute_response
from soundform.errors import GeometryError
from soundform.geometry.design_space import EAR_HEIGHT_M, prepare_parameters
from soundform.geometry.generate import generate_architecture
from soundform.geometry.mesh_utils import as_mesh
from soundform.geometry.proposal_walls import wall_grid, wall_mesh
from soundform.geometry.recipe import build_recipe, validate_recipe
from soundform.geometry.recipe_contract import concept_response_format, plan_response_format, recipe_response_format
from soundform.geometry.recipe_graph import DESCRIPTIONS, NODES, build_graph, validate_graph

_MAX_SITE_FIT_M = 0.15

# Label, minimum, maximum, units. Angles use the design frame.
OPERATIONS = {
    "arc_wall": {
        "bearing": ("Wall direction", 0, 360, "°"),
        "sweep": ("Wall sweep", 10, 359, "°"),
        "offset": ("Offset from occupant zone", 0.4, 15, "m"),
        "height": ("Screen height", 1.0, 12, "m"),
    },
    "straight_screen": {
        "bearing": ("Screen direction", 0, 360, "°"),
        "length": ("Screen length", 0.5, 30, "m"),
        "offset": ("Offset from occupant zone", 0.4, 15, "m"),
        "height": ("Screen height", 1.0, 12, "m"),
    },
    "shed_roof": {
        "bearing": ("Direction of roof rise", 0, 360, "°"),
        "depth": ("Roof depth", 1, 45, "m"),
        "width": ("Roof width", 1, 45, "m"),
        "height": ("Low roof edge", 2.2, 12, "m"),
        "rise": ("Roof rise", 0, 5, "m"),
    },
}


WALL_DEFAULTS = {"thickness": 0.25, "perforation": 0, "sound_rise": 0}
OPERATION_DEFAULTS = {
    "arc_wall": {**WALL_DEFAULTS, "entrance_bearing": 45, "entrance_width": 0},
    "straight_screen": WALL_DEFAULTS,
}
for kind in OPERATION_DEFAULTS:
    OPERATIONS[kind].update({
        "thickness": ("Wall thickness", 0.08, 0.8, "m"),
        "perforation": ("Perforation width fraction", 0, 0.8, ""),
        "sound_rise": ("Extra height toward louder bearings", 0, 4, "m"),
    })
OPERATIONS["arc_wall"].update({
    "entrance_bearing": ("Entrance bearing", 0, 360, "°"),
    "entrance_width": ("Entrance width", 0, 8, "m"),
})


def threshold_maps(field, threshold):
    """Select recording windows by broadband level relative to the loudest window."""
    selected = (field.frame_level_db >= threshold) & (not field.silent)
    return {name: getattr(field, name)[selected].sum(axis=0)
            for name in ("broadband", "low", "mid", "high")}, selected


def _local_site_boundary(params):
    x, y = params.protected_center_x, params.protected_center_y
    if params.site_shape == "circle":
        return {"shape": "circle", "center_xy_m": [-x, -y], "radius_m": params.site_radius}
    half_x, half_y = params.site_width / 2, params.site_depth / 2
    return {"shape": "rectangle", "x_m": [-half_x - x, half_x - x],
            "y_m": [-half_y - y, half_y - y]}


def sound_summary(field, maps, selected, params):
    """Represent the full circle as twelve design-frame sectors."""
    angles = (field.azimuth_deg + params.rotation_offset_deg) % 360
    sectors = ((angles + 15) % 360 // 30).astype(int)
    total = float(maps["broadband"].sum())
    local_bounds = _local_site_boundary(params)
    if local_bounds["shape"] == "circle":
        radius = local_bounds["radius_m"] - float(np.hypot(*local_bounds["center_xy_m"]))
    else:
        radius = min(-local_bounds["x_m"][0], local_bounds["x_m"][1],
                     -local_bounds["y_m"][0], local_bounds["y_m"][1])
    return {
        "recipe_frame": {
            "origin": "Occupant center at ground level (0,0,0)",
            "site_boundary_in_local_coordinates": local_bounds,
            "max_centered_radius_m": round(max(0, radius), 3),
            "max_height_m": params.max_height,
            "occupied_radius_m": params.protected_radius,
            "measured_bearing_convention": "0 north, 45 northwest, 90 west, 180 south, 270 east",
            "has_above_threshold_energy": total > 0,
        },
        "selected_windows": int(selected.sum()),
        "total_windows": int(selected.size),
        "protected_radius_m": params.protected_radius,
        "site": {key: getattr(params, key) for key in (
            "site_shape", "site_radius", "site_width", "site_depth",
            "protected_center_x", "protected_center_y", "max_height",
        )},
        "sectors": [
            {"bearing_deg": index * 30, **{
                name: round(float(values[:, sectors == index].sum()) / max(total, 1e-30), 4)
                for name, values in maps.items()
            }} for index in range(12)
        ],
    }


def _claude_error(exc: HTTPError, key: str) -> str:
    try:
        payload = json.loads(exc.read(16384))
    except (ValueError, OSError):
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    error = error if isinstance(error, dict) else {}
    code = error.get("type")
    code = code if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else None
    headline = f"Claude request failed (HTTP {exc.code}" + (f", {code})." if code else ").")
    if exc.code == 401:
        return headline + " Check ANTHROPIC_API_KEY and relaunch Soundform from that same window."
    if exc.code == 429 or code == "rate_limit_error":
        return headline + " Wait briefly, then try Generate proposal again."
    if code in {"credit_balance_exhausted", "billing_error"} or exc.code == 402:
        return headline + " Check the credit balance for this Claude API key."
    message = error.get("message")
    if isinstance(message, str):
        message = re.sub(r"sk-ant-[^\s\x27\x22,)]+", "[redacted]", message.replace(key, "[redacted]"))
        return headline + " " + " ".join(message.split())[:350]
    return headline


def _image_block(data_url: str) -> dict:
    header, _, encoded = data_url.partition(",")
    media = header.removeprefix("data:").split(";", 1)[0] or "image/png"
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": encoded}}


def _json_object(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def request_proposal(prompt, sketches, summary, threshold, *, planning=False, plan=None, repair=None, review=None):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY in the environment and relaunch Soundform.")
    catalog = {
        name: {"output": output, "inputs": inputs, "description": DESCRIPTIONS[name]}
        for name, (output, inputs) in NODES.items()
    }
    instructions = (
        "Translate the current design brief, reference images and directional sound data into a "
        "parametric geometry graph. The brief and references determine the form and strategy. "
        "In the rationale, explain the design decisions, how the sound data influenced them, "
        "and any requested details the graph approximates or omits. Describe only features that "
        "the returned graph actually implements. A node label alone does not implement a feature. "
        "Return recipe_version 2, a rationale, editable parameters and ordered nodes. "
        "Each node has op, label and inputs. Use the exact input names from the catalog. "
        "Connected inputs are zero-based indices of preceding nodes with the required output type. "
        "xyz_list and sections are inline XYZ coordinates. A number is a JSON number or the exact "
        "name of a declared p_ parameter. Formula strings are unsupported. Give editable controls "
        "meaningful labels and ranges. The unit field must be exactly \"\", \"m\", \"°\", or \"%\". "
        "Use \"°\" for angles and \"m\" for lengths. "
        "Local +Y is north and +X is east. Bearing 45 is northwest. The local origin is the listener "
        "at ground level. Keep complete components within the recipe_frame site bounds and maximum height. "
        "Include a terminal mesh node. Mesh nodes feeding another mesh node are intermediate geometry. "
        "Disconnected terminal meshes are displayed as separate components. Merge groups components "
        "without performing a Boolean union. Closed loft profiles use consistent start points and winding. "
        "Caps require simple planar closed loops. "
        f"Operation catalog: {json.dumps(catalog)}"
    )
    design_input = {
        "prompt": prompt, "threshold_db_relative_to_peak": threshold, "sound_context": summary,
    }
    task = "Return the pavilion proposal as one JSON object and no other text."
    response_format = recipe_response_format(review=review is not None)
    if planning and plan is None:
        instructions = (
            "Interpret the pavilion brief, reference images and directional sound data. "
            "Describe a coherent concept and rank its defining architectural features. "
            "Distinguish the features that must survive from details open to interpretation. "
            "Describe the proposed spatial organization, circulation and directional sound response "
            "where relevant. Call out tradeoffs with the site constraints. Keep it concise. "
            "This pass defines intent. Geometry operations will be selected in the next pass."
        )
        task = "Return a short architectural concept as one JSON object and no other text."
        response_format = concept_response_format()
    elif planning:
        design_input["design_concept"] = plan
        instructions = (
            "Turn the architectural concept into a buildable Soundform assembly. "
            "Preserve the brief's defining features and explain any simplification. "
            "For each feature specify its parts, approximate dimensions and placement, the sequence "
            "of supported operations and how their outputs connect to the next inputs. Check "
            "operation compatibility, site bounds, component connections, overlaps and openings. "
            "Give a checkable target for each feature, such as member count, opening width, "
            "coverage, or a specific connection between parts. Describe what the reviewer should "
            "verify in the built mesh or recipe and how the geometry addresses directional sound. "
            "Distinguish computed geometry from visual approximations and avoid relying on "
            "renaming a component to imply an unbuilt feature. "
            f"Operation catalog: {json.dumps(catalog)}"
        )
        task = "Return a construction plan as one JSON object and no other text."
        response_format = plan_response_format()
    elif plan is not None:
        design_input["design_plan"] = plan
        instructions += (
            " Implement the supplied plan's defining features and operation chains. "
            "If a different operation is necessary, preserve the feature and explain the approximation."
        )
    instructions += (
        " The authoritative limits are sound_context.recipe_frame.site_boundary_in_local_coordinates "
        "and sound_context.recipe_frame.max_height_m. All recipe coordinates are in that local frame. "
        "For a circle measure each final vertex from center_xy_m. For a rectangle check both X and Y "
        "intervals. The height cap applies to final Z. Account for section size, thickness, offsets, "
        "transforms, and repeated parts as well as their paths or centers. Leave clearance where feasible. "
        "The max_centered_radius_m describes clearance around the recipe origin, not the radius of "
        "the site or a target footprint size."
    )
    if repair is not None:
        design_input["repair"] = {
            "task": "Return a complete corrected recipe. Use the measured overruns to adjust the named parts, their upstream dimensions or transforms, and any related components while preserving the design intent.",
            "validation_error": repair[0],
            "previous_recipe": repair[1],
            "earlier_validation_errors": repair[2],
        }
    if review is not None:
        instructions += (
            " This is a review of a locally built proposal. Compare the labelled build preview "
            "and measurements with the brief, reference images, and the design plan's feature checks. "
            "Inspect the actual operations and their connections in the built recipe for details "
            "that the silhouettes may obscure. Verify the rationale against the actual geometry. "
            "If it claims missing features, implement them where feasible or revise the rationale "
            "to describe the actual result and its limitations. Return a complete revised "
            "graph in revision only if you can identify a concrete mismatch and improve it. "
            "Otherwise return revision: null. The preview is geometry only and the screening "
            "percentage is a line-of-sight estimate."
        )
        design_input["built_proposal"] = review[0]
    keys = ", ".join(response_format["schema"]["properties"])
    content = [{
        "type": "text",
        "text": task + f" Required keys: {keys}.\nDesign input:\n" + json.dumps(design_input),
    }]
    content.extend(_image_block(image) for image in sketches)
    if review is not None:
        content.append({"type": "text", "text": "Next image: orthographic views of the actual built proposal."})
        content.append(_image_block(review[1]))
    if review is not None or not planning:
        max_tokens = 32000
    elif plan is not None:
        max_tokens = 16000
    else:
        max_tokens = 8000
    payload = {
        "model": os.environ.get("SOUNDFORM_CLAUDE_MODEL", "claude-opus-5-5"),
        "max_tokens": max_tokens,
        "output_config": {"effort": "low"},
        "system": instructions,
        "messages": [{"role": "user", "content": content}],
    }
    request = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2026-10-21",
            "content-type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=180) as reply:
            data = json.load(reply)
    except HTTPError as exc:
        raise ValueError(_claude_error(exc, key)) from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"Could not reach Claude: {exc}") from exc
    blocks = data.get("content") or []
    if data.get("stop_reason") == "refusal":
        raise ValueError("Claude declined to generate a recipe for this request.")
    if data.get("stop_reason") == "max_tokens":
        raise ValueError("Claude's reply was cut off before the recipe finished. Try a shorter brief, then generate again.")
    parts = [block.get("text", "") for block in blocks if block.get("type") == "text"]
    try:
        return _json_object("".join(parts))
    except json.JSONDecodeError as exc:
        raise ValueError("Claude did not return a complete pavilion definition.") from exc


def validate_proposal(proposal):
    if isinstance(proposal, dict) and "recipe_version" in proposal:
        if proposal["recipe_version"] == 2:
            graph = validate_graph(proposal)
            offset = proposal.get("site_fit_offset_m")
            if offset is not None:
                if (not isinstance(offset, list) or len(offset) != 2
                        or any(type(value) not in (int, float) or not np.isfinite(value)
                               or abs(value) > _MAX_SITE_FIT_M for value in offset)
                        or np.linalg.norm(offset) > _MAX_SITE_FIT_M):
                    raise ValueError("Invalid local site-fit offset.")
                graph["site_fit_offset_m"] = offset.copy()
            return graph
        if proposal["recipe_version"] == 1:
            return validate_recipe(proposal)
        raise ValueError("Unsupported geometry recipe version.")
    if not isinstance(proposal, dict) or not isinstance(proposal.get("rationale"), str):
        raise ValueError("A pavilion proposal needs a written rationale.")
    operations = proposal.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= 6:
        raise ValueError("A pavilion proposal needs one to six geometry operations.")
    normalized = []
    for operation in operations:
        kind = operation.get("type") if isinstance(operation, dict) else None
        if not isinstance(kind, str) or kind not in OPERATIONS:
            raise ValueError("The proposal contains an unsupported geometry operation.")
        fields = OPERATIONS[kind]
        operation = {**OPERATION_DEFAULTS.get(kind, {}), **operation}
        if set(operation) != {"type", *fields}:
            raise ValueError(f"Unexpected or missing dimensions for {kind}.")
        for name, (label, low, high, _unit) in fields.items():
            value = operation[name]
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError(f"{label} must be between {low} and {high}.")
        normalized.append(operation)
    return {"rationale": proposal["rationale"][:1200], "operations": normalized}


def _direction(angle):
    angle = np.deg2rad(angle)
    return np.stack((-np.sin(angle), np.cos(angle)), axis=-1)


def build_proposal(proposal, params, directional_energy=None):
    meshes = []
    center = np.array([params.protected_center_x, params.protected_center_y])
    definition = validate_proposal(proposal)
    if "recipe_version" in definition:
        return _build_recipe_mesh(definition, params, directional_energy)
    operations = definition["operations"]
    for operation in operations:
        bearing, height = operation["bearing"], operation["height"]
        kind = operation["type"]
        if kind in OPERATION_DEFAULTS:
            radius = params.protected_radius + operation["offset"]
            thickness = operation["thickness"]
            entrances = ()
            if kind == "arc_wall":
                span = operation["sweep"]
                length = np.deg2rad(span) * radius
                width = operation["entrance_width"]
                relative = ((operation["entrance_bearing"] - bearing + 180) % 360 - 180) / span
                if width:
                    half_gap = np.arcsin(min(1.0, width / (2 * radius))) / np.deg2rad(span)
                    gap_centers = (0.5 + relative + turn * 360 / span for turn in (-1, 0, 1))
                    entrances = tuple((mid - half_gap, mid + half_gap) for mid in gap_centers
                                      if mid + half_gap > 0 and mid - half_gap < 1)
            else:
                length = operation["length"]
            steps, solid = wall_grid(length, operation["perforation"], entrances)
            if not np.any(solid):
                raise ValueError("The entrance removes the whole wall. Reduce the entrance width or extend the sweep.")
            if kind == "arc_wall":
                angles = bearing + (steps - 0.5) * span
                path = center + np.array([radius, radius + thickness])[:, None, None] * _direction(angles)
            else:
                along = (steps - 0.5)[:, None] * length * _direction(bearing + 90)
                path = center + np.array([radius, radius + thickness])[:, None, None] * _direction(bearing) + along
                angles = bearing + np.rad2deg(np.arctan2((steps - 0.5) * length, radius))
            heights = np.full(len(steps), height, dtype=float)
            if directional_energy is not None:
                bearings, energies = directional_energy
                heights += operation["sound_rise"] * np.interp(angles, bearings, energies, period=360)
            mesh = wall_mesh(path, heights[:, None] * np.array([0, 0.55, 0.82, 1]), solid)
        else:
            along = _direction(bearing) * operation["depth"] / 2
            across = _direction(bearing + 90) * operation["width"] / 2
            path = center + np.array([-along - across, -along + across, along + across, along - across])
            vertices = np.column_stack((path, [height, height, height + operation["rise"], height + operation["rise"]]))
            mesh = as_mesh(vertices, np.array([(0, 1, 2), (0, 2, 3)]))
        vertices = mesh.vertices
        _validate_site_bounds(vertices, params, label=kind.replace("_", " "))
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    combined.face_attributes["pavilion_part"] = np.concatenate([
        np.full(len(mesh.faces), f"{index + 1}. {op['type'].replace('_', ' ').title()}")
        for index, (op, mesh) in enumerate(zip(operations, meshes, strict=True))
    ])
    return combined


def _footprint_distance(triangles, point):
    """Distance from a point to each projected triangle, including vertical faces."""
    starts = triangles - point
    edges = np.roll(starts, -1, axis=1) - starts
    lengths_squared = np.sum(edges * edges, axis=2)
    fractions = np.divide(-np.sum(starts * edges, axis=2), lengths_squared,
                          out=np.zeros_like(lengths_squared), where=lengths_squared > 0)
    nearest = starts + np.clip(fractions, 0, 1)[..., None] * edges
    distance = np.linalg.norm(nearest, axis=2).min(axis=1)
    crosses = edges[..., 0] * starts[..., 1] - edges[..., 1] * starts[..., 0]
    inside = (np.all(crosses >= 0, axis=1) | np.all(crosses <= 0, axis=1))
    inside &= np.abs(crosses.sum(axis=1)) > 1e-12
    return np.where(inside, 0.0, distance)


def _preserves_occupant_clearance(mesh, params, offset):
    triangles = mesh.triangles
    z = triangles[:, :, 2]
    footprints = triangles[(z.min(axis=1) < EAR_HEIGHT_M) & (z.max(axis=1) >= 0), :, :2]
    center = np.array([params.protected_center_x, params.protected_center_y])
    current = _footprint_distance(footprints, center)
    shifted = _footprint_distance(footprints, center - offset)
    return np.all(shifted + 1e-6 >= np.minimum(current, params.protected_radius))


def _site_fit_offset(mesh, params):
    vertices = mesh.vertices
    if vertices[:, 2].max() > params.max_height + 1e-6:
        return None
    xy = vertices[:, :2]
    margin = 1e-5
    if params.site_shape == "circle":
        radii = np.linalg.norm(xy, axis=1)
        outside = radii > params.site_radius
        direction = xy[outside].mean(axis=0)
        length = np.linalg.norm(direction)
        if length < 1e-8:
            return None
        direction /= length
        along = xy @ direction
        perpendicular = xy[:, 0] * direction[1] - xy[:, 1] * direction[0]
        reach_squared = (params.site_radius - margin) ** 2 - perpendicular ** 2
        if np.any(reach_squared < 0):
            return None
        reach = np.sqrt(reach_squared)
        shift = float((along - reach).max())
        if shift > (along + reach).min():
            return None
        offset = -direction * shift
    else:
        half = np.array([params.site_width, params.site_depth]) / 2
        lower = -half + margin - xy.min(axis=0)
        upper = half - margin - xy.max(axis=0)
        if np.any(lower > upper):
            return None
        offset = np.clip(np.zeros(2), lower, upper)
    if np.linalg.norm(offset) > _MAX_SITE_FIT_M or not _preserves_occupant_clearance(mesh, params, offset):
        return None
    return offset


def _build_recipe_mesh(definition, params, directional_energy, *, fit_site=False):
    builder = build_graph if definition["recipe_version"] == 2 else build_recipe
    mesh = builder(definition, params, directional_energy)
    offset = definition.get("site_fit_offset_m")

    def check_bounds():
        _validate_site_bounds(mesh.vertices, params, faces=mesh.faces,
                              labels=mesh.face_attributes.get("pavilion_part"))

    if offset is not None:
        if not _preserves_occupant_clearance(mesh, params, offset):
            raise ValueError("The saved site adjustment would reduce occupant clearance for this geometry.")
        mesh.apply_translation((*offset, 0))
    try:
        check_bounds()
    except ValueError as original_error:
        if not fit_site or offset is not None:
            raise
        offset = _site_fit_offset(mesh, params)
        if offset is None:
            raise
        mesh.apply_translation((*offset, 0))
        try:
            check_bounds()
        except ValueError:
            raise original_error from None
        definition["site_fit_offset_m"] = offset.tolist()
    return mesh


def _record_recipe_failure(draft, error, params, directional_energy):
    """Keep the latest failed build as a local, API-free reproduction."""
    directory = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Soundform"
    temporary_path = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as file:
            temporary_path = Path(file.name)
            energy = None if directional_energy is None else [array.tolist() for array in directional_energy]
            json.dump({"recipe": draft, "error": error, "parameters": params.to_dict(),
                       "directional_energy": energy},
                      file, indent=2, allow_nan=False)
        temporary_path.replace(directory / "latest-recipe-failure.json")
    except (OSError, ValueError):
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _bounds_violations(vertices, boundary, max_height):
    violations = []

    def check(label, measured, limit, excess):
        if excess > 1e-6:
            violations.append(f"{label} {measured:.5g} m (limit {limit:.5g} m, exceeds by {excess:.3g} m)")

    if boundary["shape"] == "circle":
        distances = np.linalg.norm(vertices[:, :2] - boundary["center_xy_m"], axis=1)
        radius = boundary["radius_m"]
        index = int(distances.argmax())
        x, y = vertices[index, :2]
        check(f"radius from site center at local XY ({x:.5g}, {y:.5g})",
              distances[index], radius, distances[index] - radius)
    else:
        for index, axis in enumerate(("x", "y")):
            lower, upper = boundary[f"{axis}_m"]
            minimum, maximum = vertices[:, index].min(), vertices[:, index].max()
            check(f"local {axis.upper()} min", minimum, lower, lower - minimum)
            check(f"local {axis.upper()} max", maximum, upper, maximum - upper)
    top = vertices[:, 2].max()
    check("top Z", top, max_height, top - max_height)
    return violations


def _validate_site_bounds(vertices, params, *, faces=None, labels=None, label="Pavilion"):
    boundary = _local_site_boundary(params)
    local = vertices - (params.protected_center_x, params.protected_center_y, 0)
    violations = _bounds_violations(local, boundary, params.max_height)
    if not violations:
        return
    details = []
    if labels is not None:
        for part in np.unique(labels):
            part_vertices = local[np.unique(faces[labels == part])]
            failures = _bounds_violations(part_vertices, boundary, params.max_height)
            if failures:
                details.append(f"{part}: " + ", ".join(failures))
    if not details:
        details.append(f"{label}: " + ", ".join(violations))
    message = "Site limits exceeded. " + " | ".join(details[:4])
    if len(details) > 4:
        message += f" | {len(details) - 4} more parts exceed the limits."
    raise ValueError(message)


def _review_preview(mesh, params):
    """Rasterize three orthographic silhouettes without opening a desktop window."""
    size = 300
    image = Image.new("RGB", (3 * size, size), "#f4f3ef")
    draw = ImageDraw.Draw(image)
    vertices = mesh.vertices - (params.protected_center_x, params.protected_center_y, 0)
    scale = (size - 48) / max(float(np.abs(vertices).max()) * 2, 1)
    for index, (label, axes) in enumerate((
        ("PLAN (NORTH UP)", (0, 1)),
        ("X / Z ELEVATION", (0, 2)),
        ("Y / Z ELEVATION", (1, 2)),
    )):
        center = np.array([index * size + size / 2, size / 2 + 10])
        pixels = np.rint(vertices[:, axes] * (scale, -scale) + center).astype(int)
        for face in mesh.faces:
            draw.polygon(tuple(pixels[face].flatten()), fill="#405d65")
        draw.text((index * size + 12, 12), label, fill="#24373c")
        x, y = center
        draw.line((x - 4, y, x + 4, y), fill="#e09c41")
        draw.line((x, y - 4, x, y + 4), fill="#e09c41")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _build_online(prompt, sketches, summary, threshold, params, directional_energy, plan):
    errors = []
    repair = None
    for _ in range(3):
        draft = request_proposal(prompt, sketches, summary, threshold, plan=plan, repair=repair)
        try:
            if not isinstance(draft, dict) or draft.get("recipe_version") != 2:
                raise ValueError("Return an object with recipe_version set to 2.")
            definition = validate_proposal(draft)
            return definition, _build_recipe_mesh(definition, params, directional_energy, fit_site=True)
        except (ValueError, GeometryError) as exc:
            errors.append(str(exc))
            _record_recipe_failure(draft, errors[-1], params, directional_energy)
            repair = (errors[-1], draft, errors[:-1])
    raise ValueError(f"Claude's recipe could not be built after three attempts. {errors[-1]}")


def evaluate_proposal(field, params, threshold, proposal=None, prompt="", sketches=(), online=False):
    used, warnings = prepare_parameters(params)
    maps, selected = threshold_maps(field, threshold)
    energy_by_bearing = maps["broadband"].sum(axis=0)
    relative_energy = energy_by_bearing / max(float(energy_by_bearing.max()), 1e-30)
    directional_energy = (field.azimuth_deg + used.rotation_offset_deg, relative_energy)
    generated_online = proposal is None and online
    if generated_online:
        summary = sound_summary(field, maps, selected, used)
        concept = request_proposal(prompt, sketches, summary, threshold, planning=True)
        plan = request_proposal(prompt, sketches, summary, threshold, planning=True, plan=concept)
        proposal, mesh = _build_online(prompt, sketches, summary, threshold, used, directional_energy, plan)
        proposal = validate_proposal(proposal)
    else:
        if proposal is None:
            raise ValueError("Enter a design brief or attach a sketch and generate with Claude.")
        proposal = validate_proposal(proposal)
        mesh = build_proposal(proposal, used, directional_energy)
    response = compute_response(field, {"broadband": 1.0}, smoothing=0)
    response.energy = maps["broadband"]
    report = estimate_protection(mesh, response, used, maps)
    if generated_online:
        result = {
            "recipe": proposal,
            "size_xyz_m": mesh.extents.round(3).tolist(),
            "triangle_count": len(mesh.faces),
            "estimated_screening_percent": screening_percent(report),
            "screening_warnings": report.warnings,
        }
        try:
            review = request_proposal(prompt, sketches, summary, threshold,
                                      plan=plan, review=(result, _review_preview(mesh, used)))
            if not isinstance(review, dict) or "revision" not in review:
                raise ValueError("Missing review result.")
            if review["revision"] is not None:
                revision = review["revision"]
                if not isinstance(revision, dict) or revision.get("recipe_version") != 2:
                    raise ValueError("Review revision must be a version 2 recipe.")
                revised = validate_proposal(revision)
                revised_mesh = _build_recipe_mesh(revised, used, directional_energy, fit_site=True)
                revised_report = estimate_protection(revised_mesh, response, used, maps)
                proposal, mesh, report = revised, revised_mesh, revised_report
        except (ValueError, GeometryError, OSError):
            warnings = [*warnings, "AI review could not be applied. Showing the successfully built proposal."]
    if (offset := proposal.get("site_fit_offset_m")) is not None:
        warnings = [*warnings, f"Soundform moved the pavilion {np.linalg.norm(offset) * 100:.1f} cm in plan to fit the site."]
    reference = used.copy()
    reference.mode = "pavilion"
    try:
        reference_mesh, _, _, _ = generate_architecture(field, reference)
        baseline = screening_percent(estimate_protection(reference_mesh, response, used, maps))
    except GeometryError:
        baseline = None
    return proposal, mesh, report, baseline, warnings


def screening_percent(report):
    if any("Coverage is unavailable" in warning for warning in report.warnings):
        return None
    before = float(report.exposure_before.sum())
    return 100 * (1 - float(report.exposure_after.sum()) / before) if before > 0 else None
