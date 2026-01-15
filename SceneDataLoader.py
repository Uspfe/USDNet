import os
import json
import numpy as np
from typing import Dict, Any, List


class SceneDataLoader:
    """
    Data loader for Articulate3D.

    This loader iterates over scenes in the dataset directory, where each scene
    consists of two JSON files:
    - `<scene_id>_parts.json`: contains part annotations, mesh face indices, and stats.
    - `<scene_id>_artic.json`: contains articulation data such as axes, origins, and motion ranges.

    Each iteration returns:
        - scene_id (str): Identifier of the scene.
        - scene_dict (dict): Dictionary mapping part IDs (int) to their articulation info:
            {
                "label": str,              # Part label string
                "axis": List[float],       # Articulation axis (3 floats)
                "range": List[float],      # Min and max range of motion
                "type": int,               # Motion type (1 = rotation, 2 = translation)
                "origin": List[float],     # Origin point of articulation (3 floats)
                "interactable": List[dict] # List of interactable parts with keys 'pid' and 'label'
            }
        - face_mask (np.ndarray): 1D integer array of length equal to total mesh faces,
          where each element is the part ID the face belongs to (0 if none).

    Methods:
    - extract_identifier(label: str) -> str
        Extracts a simplified identifier from a part label by removing the prefix before the first '.'.

    - parse_parts(annotations: List[Dict]) -> Dict[int, Dict]
        Parses part annotations into a mapping from part IDs to metadata including identifier and label.

    - find_interactables(pid: int, identifier: str, part_map: Dict[int, Dict]) -> List[Dict]
        Finds interactable sub-parts of a given part by checking identifier hierarchy.

    - load_scene(scene_id: str)
        Loads and returns the scene data for the given scene ID, combining parts and articulation info.

    Usage example:
        loader = SceneDataLoader("path/to/Articulate3D/")
        for scene_id, scene_dict, face_mask in loader:
            # process scene data

    Notes:
    - Skips scenes if the required JSON files are missing.
    - The motion types are mapped as {"rotation": 1, "translation": 2}.
    """
    def __init__(self, dataset_dir: str):
        self.motion_type = {"rotation":1, "translation":2}

        self.dataset_dir = dataset_dir
        files = os.listdir(dataset_dir)
        self.scene_ids = sorted(set(f.split('_')[0] for f in files if f.endswith('_parts.json')))
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.scene_ids):
            raise StopIteration

        scene_id = self.scene_ids[self.index]
        self.index += 1
        return self.load_scene(scene_id)

    def extract_identifier(self, label: str) -> str:
        return label.split('.', 1)[-1]  # Remove the "*."

    def parse_parts(self, annotations: List[Dict]) -> Dict[int, Dict]:
        part_map = {}
        for ann in annotations:
            identifier = self.extract_identifier(ann['label'])
            part_map[ann['partId']] = {
                'annotation': ann,
                'identifier': identifier,
                'label': ann['label'],
            }
        return part_map

    def find_interactables(self, pid: int, identifier: str, part_map: Dict[int, Dict]) -> List[Dict]:
        parent_prefix = identifier + "."
        candidates = []

        for other_pid, part in part_map.items():
            if other_pid == pid:
                continue
            uid = part['identifier']
            if uid.startswith(parent_prefix):
                suffix = uid[len(parent_prefix):]
                dot_count = suffix.count('.')
                candidates.append((dot_count, suffix, other_pid, part))

        if not candidates:
            return []

        max_depth = max(c[0] for c in candidates)
        deepest = [c for c in candidates if c[0] == max_depth]

        return [{
            "pid": other_pid,
            "label": part['label'],
        } for _, _, other_pid, part in deepest]

    def load_scene(self, scene_id: str):
        parts_path = os.path.join(self.dataset_dir, f"{scene_id}_parts.json")
        artic_path = os.path.join(self.dataset_dir, f"{scene_id}_artic.json")

        try:
            with open(parts_path, 'r', encoding='utf-8') as f:
                parts_data = json.load(f)
            with open(artic_path, 'r', encoding='utf-8') as f:
                artic_data = json.load(f)
        except FileNotFoundError:
            print(f"Missing files for scene {scene_id}, skipping.")
            return self.__next__()

        annotations = parts_data['data']['annotations']
        total_faces = parts_data['data']['stats']['totalFaces']

        part_map = self.parse_parts(annotations)

        pid_to_artic = {
            a['pid']: a
            for a in artic_data.get('data', {}).get('articulations', [])
        }

        #scene data dict
        scene_dict = {}
        face_mask = np.zeros(total_faces, dtype=int)

        face_offset = 0  # track global face index

        #map from id to triIndices face positions
        part_faces = {}
        for ann in annotations:
            part_faces[ann['partId']] = ann['triIndices']

        for pid, artic in pid_to_artic.items():
            if pid not in part_map:
                continue

            part_data = part_map[pid]
            identifier = part_data['identifier']
            interactables = self.find_interactables(pid, identifier, part_map)

            scene_dict[pid] = {
                "label": part_data['label'],
                "axis": artic['axis'],
                "range": [artic['rangeMin'], artic['rangeMax']],
                "type": self.motion_type[artic['type']],
                "origin": artic['origin'],
                "interactable": interactables
            }

            #movable part to mask
            for face_idx in part_faces.get(pid, []):
                if face_idx < total_faces:
                    face_mask[face_idx] = pid

            #inter parts to mask
            for inter in interactables:
                inter_pid = inter['pid']
                for face_idx in part_faces.get(inter_pid, []):
                    if face_idx < total_faces:
                        face_mask[face_idx] = inter_pid

        return scene_id, scene_dict, face_mask

