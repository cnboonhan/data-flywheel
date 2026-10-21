"""Import the latching door (make_latch_door.py) as a BEHAVIOR asset with OmniGibson's own pipeline.

    venv-behavior/bin/python import_latch_door.py [--model flylat] [--dataset flywheel-assets] [make_latch_door.py args]

Writes $OMNIGIBSON_DATA_PATH/<dataset>/objects/door/<model>/ (usd/, urdf/, misc/metadata.json), loadable as
DatasetObject(category="door", model=<model>, dataset_name=<dataset>). Category door is BDDL's door.n.01, which is
openable; the metadata names j_leaf as the openable joint (else OmniGibson's Open state would read every joint,
the handle included) and tags the handle link. Collision is the URDF's own boxes (collision_method None), as the
importer's CoACD is not installed and the frame is not convex.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--model", default="flylat")            # BEHAVIOR model ids are 6 letters
p.add_argument("--dataset", default="flywheel-assets")
args, door_args = p.parse_known_args()

tmp = Path(tempfile.mkdtemp())
urdf = subprocess.run([sys.executable, str(HERE / "make_latch_door.py"), "--out", str(tmp), *door_args],
                      check=True, capture_output=True, text=True).stdout.strip()

import omnigibson as og  # noqa: E402

og.launch()
import carb  # noqa: E402

carb.settings.get_settings().set("/crashreporter/enabled", False)
import omnigibson.utils.asset_conversion_utils as acu  # noqa: E402

_record = acu.record_obj_metadata_from_urdf


def record_with_door_fields(urdf_path, obj_dir, joint_setting="zero", overwrite=False):
    _record(urdf_path=urdf_path, obj_dir=obj_dir, joint_setting=joint_setting, overwrite=overwrite)
    path = Path(obj_dir) / "misc" / "metadata.json"
    meta = json.loads(path.read_text())
    meta["openable_joint_ids"] = [[0, "j_leaf"]]        # the leaf's joint only: the handle doesn't open the door
    meta["link_tags"] = {"base_link": [], "leaf": [], "handle": ["handle"]}
    path.write_text(json.dumps(meta))


acu.record_obj_metadata_from_urdf = record_with_door_fields
urdf_out, usd, _ = acu.import_og_asset_from_urdf(category="door", model=args.model, dataset_name=args.dataset,
                                                 urdf_path=urdf, collision_method=None, overwrite=True)
print("usd", usd)
print("metadata", (Path(usd).parent.parent / "misc" / "metadata.json").read_text())
og.shutdown()
