"""Select and edit named portions of the generated pavilion mesh."""

import numpy as np
import trimesh

PART_ATTRIBUTE = "pavilion_part"


def part_labels(mesh):
    labels = mesh.face_attributes.get(PART_ATTRIBUTE)
    return labels if labels is not None else np.full(len(mesh.faces), "Mesh")


def selected_mesh(mesh, part):
    if mesh is None or part is None:
        return mesh
    return _extract(mesh, part_labels(mesh) == part)


def _extract(mesh, mask):
    if not mask.any():
        return None
    extracted = mesh.copy()
    extracted.update_faces(mask)
    extracted.remove_unreferenced_vertices()
    extracted.face_attributes[PART_ATTRIBUTE] = part_labels(mesh)[mask]
    return extracted


def edit_mesh(mesh, part, *, delete=False, offset=(0, 0, 0), rotation=0, scale=1):
    selected = selected_mesh(mesh, part)
    if selected is None:
        return mesh
    if not delete:
        selected = selected.copy()
        pivot = selected.bounds.mean(axis=0)
        pivot[2] = selected.bounds[0, 2]
        transform = trimesh.transformations.rotation_matrix(np.deg2rad(rotation), (0, 0, 1))
        transform[:3, :3] *= scale
        transform[:3, 3] = pivot - transform[:3, :3] @ pivot + offset
        selected.apply_transform(transform)
    if part is None:
        return None if delete else selected
    remainder = _extract(mesh, part_labels(mesh) != part)
    if delete:
        return remainder
    if remainder is None:
        return selected
    result = trimesh.util.concatenate([remainder, selected])
    result.face_attributes[PART_ATTRIBUTE] = np.concatenate([part_labels(remainder), part_labels(selected)])
    return result
