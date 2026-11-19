import numpy as np
import pytest

from soundform.geometry.proposal import build_proposal
from soundform.geometry.recipe_graph import validate_graph
from soundform.models.project import default_parameters


def _node(op, **inputs):
    return {"op": op, "label": op, "inputs": inputs}


def _build(*nodes):
    return build_proposal({
        "recipe_version": 2,
        "rationale": "Geometry regression",
        "parameters": [
            {"name": "p_height", "label": "Height", "unit": "m",
             "min": 0.5, "max": 2, "step": 0.1, "value": 1},
        ],
        "nodes": list(nodes),
    }, default_parameters())


def test_angle_parameter_accepts_the_word_degrees_and_a_missing_label():
    graph = validate_graph({
        "recipe_version": 2,
        "rationale": "Quiet boards turn away from the loud bearing.",
        "parameters": [
            {"name": "p_quiet_angle", "label": "  ", "unit": "degrees",
             "min": 0, "max": 90, "step": 1, "value": 15},
        ],
        "nodes": [_node("cylinder", x=0, y=0, z=0, radius=1, height="p_quiet_angle")],
    })

    assert graph["parameters"][0]["label"] == "Quiet angle"
    assert graph["parameters"][0]["unit"] == "°"


def test_loft_caps_unequal_closed_profiles_without_degenerate_faces():
    mesh = _build(_node("loft", closed=True, cap=True, sections=[
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        [[-1, -1, "p_height"], [0, -1, "p_height"], [1, -1, "p_height"],
         [1, 1, "p_height"], [-1, 1, "p_height"], [-1, -1, "p_height"]],
    ]))

    assert mesh.is_watertight
    assert mesh.nondegenerate_faces().all()
    assert np.isclose(mesh.volume, 4)


def test_transformed_extrusion_emits_only_terminal_mesh():
    mesh = _build(
        _node("polyline", points=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], closed=True),
        _node("extrude", profile=0, x=0, y=0, z="p_height", cap=True),
        _node("move", mesh=1, x=2, y=0, z=0),
    )

    assert mesh.is_watertight
    assert np.allclose(mesh.bounds, [[2, 0, 0], [3, 1, 1]])
    assert set(mesh.face_attributes["pavilion_part"]) == {"3. move"}

def test_extrude_caps_concave_profile():
    mesh = _build(
        _node("polyline", points=[[0, 0, 0], [2, 0, 0], [2, 1, 0],
                                 [1, 1, 0], [1, 2, 0], [0, 2, 0]], closed=True),
        _node("extrude", profile=0, x=0, y=0, z="p_height", cap=True),
    )
    assert mesh.is_watertight
    assert mesh.nondegenerate_faces().all()
    assert np.isclose(mesh.volume, 3)


@pytest.mark.parametrize(("revision_height", "result_height"), [(2, 2), (0, 1)])
def test_online_generation_repairs_and_reviews_built_geometry(monkeypatch, tmp_path, revision_height, result_height):
    import base64
    import io
    import json

    from PIL import Image

    from soundform.geometry import proposal as proposals
    from soundform.tests.helpers import blob_field

    def recipe(height):
        return {
            "recipe_version": 2, "rationale": "Box",
            "parameters": [{"name": "p_height", "label": "Height", "unit": "m",
                            "min": 0.5, "max": 2, "step": 0.1, "value": height}],
            "nodes": [_node("box", x=0, y=0, z=0, width=1, depth=1, height="p_height")],
        }

    calls = []

    design_plan = {"concept": "Box", "features": [
        {"intent": "Solid block", "operations": ["box"],
         "build_approach": "Use a box with editable height", "review_check": "Check height"}
    ], "sound_response": "None requested", "approximations": ""}
    concept = {"concept": "A solid block", "defining_features": ["Solid block"],
               "spatial_organization": "Single component", "sound_response": "",
               "tradeoffs": ""}
    planning_calls = []

    def request(*args, planning=False, plan=None, repair=None, review=None):
        assert args[2]["analysis_surface"] == {"samples": []}
        if planning:
            assert not calls
            planning_calls.append(plan)
            return concept if plan is None else design_plan
        assert plan is design_plan
        calls.append((repair, review))
        if review is not None:
            measurements, image_url = review
            assert measurements["size_xyz_m"] == [1, 1, 1]
            image = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1])))
            assert image.size == (900, 300)
            return {"revision": recipe(revision_height)}
        return recipe(1 if len(calls) == 3 else 0)

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(proposals, "request_proposal", request)
    definition, mesh, _, _, warnings = proposals.evaluate_proposal(
        blob_field(), default_parameters(), -12, prompt="Box", online=True,
        surface_context={"samples": []}
    )
    failure = json.loads((tmp_path / "Soundform" / "latest-recipe-failure.json").read_text())
    assert "Invalid range" in failure["error"]
    assert failure["recipe"]["parameters"][0]["value"] == 0
    assert planning_calls == [None, concept]
    assert len(calls) == 4
    assert calls[1][0][2] == []
    assert len(calls[2][0][2]) == 1
    assert definition["parameters"][0]["value"] == result_height
    assert mesh.extents[2] == result_height
    assert any("AI review could not be applied" in warning for warning in warnings) == (revision_height == 0)

@pytest.mark.parametrize(("shape", "boundary_message"), [
    ("rectangle", "local X max 8 m (limit 7 m, exceeds by 1 m)"),
    ("circle", "radius from site center"),
])
def test_site_bounds_report_part_and_local_overruns(shape, boundary_message):
    from soundform.geometry.proposal import _validate_site_bounds

    params = default_parameters()
    params.site_shape = shape
    params.site_width = 20
    params.site_depth = 20
    params.site_radius = 10
    params.protected_center_x = 3
    params.max_height = 4
    vertices = np.array([[0, 0, 0], [11, 0, 5], [0, 1, 0],
                         [0, 2, 0], [1, 2, 0], [1, 3, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    labels = np.array(["1. Roof", "2. Screen"])

    with pytest.raises(ValueError) as error:
        _validate_site_bounds(vertices, params, faces=faces, labels=labels)

    message = str(error.value)
    assert "1. Roof" in message
    assert "2. Screen" not in message
    assert boundary_message in message
    assert "top Z 5 m (limit 4 m, exceeds by 1 m)" in message

def test_online_small_circle_overrun_fits_without_model_repair(monkeypatch):
    from soundform.geometry import proposal as proposals

    params = default_parameters()
    params.site_radius = 10
    center_x = float(np.sqrt(10.122 ** 2 - 0.1 ** 2) - 0.1)
    recipe = {
        "recipe_version": 2, "rationale": "Edge screen",
        "parameters": [{"name": "p_height", "label": "Height", "unit": "m",
                        "min": 0.5, "max": 2, "step": 0.1, "value": 1}],
        "nodes": [_node("box", x=center_x, y=0, z=0, width=0.2, depth=0.2, height="p_height")],
    }

    def request(*args, repair=None, **kwargs):
        assert repair is None
        return recipe

    monkeypatch.setattr(proposals, "request_proposal", request)
    definition, mesh = proposals._build_online("", (), {}, -12, params, None, {})

    assert 0.122 <= np.linalg.norm(definition["site_fit_offset_m"]) < 0.123
    assert np.allclose(mesh.extents, [0.2, 0.2, 1])
    rebuilt = proposals.build_proposal(definition, params)
    assert np.allclose(mesh.vertices, rebuilt.vertices)


def test_site_fit_rejects_moving_wall_into_occupant_clearance():
    import trimesh

    from soundform.geometry.proposal import _site_fit_offset

    params = default_parameters()
    params.site_radius = 10
    params.protected_radius = 5
    wall = trimesh.creation.box(extents=(5.122, 0.1, 2))
    wall.apply_translation((7.561, 0, 1))

    assert _site_fit_offset(wall, params) is None
