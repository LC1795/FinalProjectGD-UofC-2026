"""Strict OpenAI response contract for Soundform's typed geometry graph."""

from soundform.geometry.recipe_graph import NODES, PARAMETER_PATTERN


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _array(items, minimum, maximum):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def recipe_response_format(*, review=False):
    number = {"type": "number"}
    text = {"type": "string"}
    parameter_name = {"type": "string", "pattern": PARAMETER_PATTERN}
    scalar = {"$ref": "#/$defs/scalar"}
    xyz = _array(scalar, 3, 3)
    input_schema = {
        "number": scalar,
        "xyz_list": _array(xyz, 2, 24),
        "sections": _array(_array(xyz, 2, 24), 2, 16),
        "boolean": {"type": "boolean"},
    }
    node_link = {"type": "integer", "minimum": 0, "maximum": 22}
    variants = [
        _object({"op": {"type": "string", "enum": [kind]}, "label": text,
                 "inputs": _object({key: input_schema.get(value_type, node_link)
                                    for key, value_type in inputs.items()})})
        for kind, (_, inputs) in NODES.items()
    ]
    parameter = _object({
        "name": parameter_name, "label": text,
        "unit": {"type": "string", "enum": ["", "m", "°", "%"]},
        "min": number, "max": number, "value": number, "step": number,
    })
    schema = _object({
        "recipe_version": {"type": "integer", "enum": [2]},
        "rationale": text,
        "parameters": _array(parameter, 1, 16),
        "nodes": _array({"anyOf": variants}, 1, 24),
    })
    name = "soundform_geometry_graph_v2"
    if review:
        schema = _object({"revision": {"anyOf": [schema, {"type": "null"}]}})
        name = "soundform_geometry_review_v2"
    schema["$defs"] = {"scalar": {"anyOf": [number, parameter_name]}}
    return {"type": "json_schema", "name": name, "strict": True, "schema": schema}


def concept_response_format():
    text = {"type": "string"}
    return {
        "type": "json_schema",
        "name": "soundform_design_concept",
        "strict": True,
        "schema": _object({
            "concept": text,
            "defining_features": _array(text, 1, 8),
            "spatial_organization": text,
            "sound_response": text,
            "tradeoffs": text,
        }),
    }


def plan_response_format():
    text = {"type": "string"}
    feature = _object({
        "intent": text,
        "operations": _array({"type": "string", "enum": list(NODES)}, 1, 8),
        "build_approach": text,
        "geometry_targets": text,
        "review_check": text,
    })
    return {
        "type": "json_schema",
        "name": "soundform_design_plan",
        "strict": True,
        "schema": _object({
            "concept": text,
            "features": _array(feature, 1, 8),
            "sound_response": text,
            "assembly_order": text,
            "approximations": text,
        }),
    }
