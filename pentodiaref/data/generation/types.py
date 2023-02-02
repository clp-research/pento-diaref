import json
import os
from typing import List, Tuple, Dict, Union
from tqdm import tqdm

from golmi.contrib.pentomino.symbolic.types import PropertyNames, Colors, Shapes, RelPositions, SymbolicPieceGroup


class Reference:

    def __init__(self, user: str, utterance_type: int, sent_type: int, utterance: str,
                 property_values: Dict[PropertyNames, Union[Colors, Shapes, RelPositions]]):
        self.property_values = property_values
        self.utterance = utterance
        self.utterance_type = utterance_type
        self.sent_type = sent_type
        self.user = user

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return f"Reference({self.user}, {self.utterance})"

    def to_json(self):
        return {
            # we actually do not need to store "instr" b.c. its derivable from "props"
            # when we use a fix template anyway (and saves lot of space)
            "user": "ia",
            "instr": self.utterance,  # the referring expression
            "type": self.utterance_type,
            "sent_type": self.sent_type,
            "props": dict([(pn.to_json(), v.to_json()) for pn, v in self.property_values.items()]),
            # Note: the preference order is fix for now
            # "props_pref": [pn.to_json() for pn in pia.preference_order],  # the preference order
        }

    @classmethod
    def __convert_json_ref_prop(cls, pn, v):
        pn = PropertyNames.from_json(pn)
        if pn == PropertyNames.COLOR:
            v = Colors.from_json(v)
        if pn == PropertyNames.SHAPE:
            v = Shapes.from_json(v)
        if pn == PropertyNames.REL_POSITION:
            v = RelPositions.from_json(v)
        return pn, v

    @classmethod
    def from_json(cls, r):
        property_values = dict([cls.__convert_json_ref_prop(pn, v) for pn, v in r["props"].items()])
        return Reference(r["user"], r["type"], r["sent_type"], r["instr"], property_values)


class Annotation:

    def __init__(self, anno_id: int, group_id: int, target_idx: int, group: SymbolicPieceGroup, refs: List[Reference],
                 bboxes: List[Tuple] = None, global_id: int = None, split_name: str = None):
        self.target_idx = target_idx
        self.group_id = group_id
        self.refs = refs
        self.group = group
        self.anno_id = anno_id  # split-id
        self.global_id = global_id
        self.bboxes = bboxes
        self.split_name = split_name

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        if self.global_id is not None:
            return f"Annotation(gid:{self.global_id},sid:{self.anno_id},iid:{self.group_id})"
        return f"Annotation(sid:{self.anno_id},iid:{self.group_id})"

    def to_json(self):
        d = {
            "id": self.anno_id,
            "group_id": self.group_id,
            "size": len(self.group),
            "pieces": self.group.to_json(),
            "target": self.target_idx,
            "refs": [ref.to_json() for ref in self.refs]
        }
        if self.bboxes is not None:
            d["bboxes"] = self.bboxes
        if self.global_id is not None:
            d["global_id"] = self.global_id
        if self.split_name is not None:
            d["split_name"] = self.split_name
        return d

    @classmethod
    def from_json(cls, json_annotation):
        refs = [Reference.from_json(r) for r in json_annotation["refs"]]
        group = SymbolicPieceGroup.from_json(json_annotation["pieces"])
        annos_id = json_annotation["id"]
        group_id = json_annotation["group_id"]
        target_idx = json_annotation["target"]
        anno = Annotation(annos_id, group_id, target_idx, group, refs)
        if "bboxes" in json_annotation:
            anno.bboxes = json_annotation["bboxes"]
        if "global_id" in json_annotation:
            anno.global_id = json_annotation["global_id"]
        if "split_name" in json_annotation:
            anno.split_name = json_annotation["split_name"]
        return anno

    @classmethod
    def load(cls, data_dir, file_name, resolve=False):
        if file_name.endswith(".json"):
            file_name = os.path.splitext(file_name)[0]  # remove extension
        file_path = os.path.join(data_dir, f"{file_name}.json")
        with open(file_path, "r") as f:
            data = json.load(f)
            if resolve:
                print("Resolve data types")
                annos = [cls.from_json(j) for j in tqdm(data)]
            else:
                annos = data
        print(f"Loaded {len(annos)} from {file_path}")
        return annos

    @staticmethod
    def store(annotations: List, file_name, data_dir):
        if not annotations:
            raise Exception("Cannot store annotations, when none given")
        if file_name.endswith(".json"):
            file_name = os.path.splitext(file_name)[0]  # remove extension
        file_path = os.path.join(data_dir, f"{file_name}.json")
        data = annotations  # these should be dicts
        if isinstance(annotations[0], Annotation):
            print("Convert annotations to json")
            data = list([a.to_json() for a in tqdm(annotations)])
        print(f"Store {len(data)} annotations to", file_path)
        with open(file_path, "w") as f:
            json.dump(data, f)
