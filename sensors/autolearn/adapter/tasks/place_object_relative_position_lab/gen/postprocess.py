"""Bring generator output in line with the shipped llm_task instances. Needs pxr (run in the container).

geniesim_generator.utils.usd writes payloads relative to `<src>/assets/`, a directory that only exists in
the deployed layout, so the benchmark could not resolve them; the shipped instances reference the
container's asset mount absolutely (`/geniesim_assets/objects/...`). The shipped instances also pin the
table with `physics:kinematicEnabled` so the arm cannot shove it; the generator does not.

usage: python3 postprocess.py <llm_task/<sub_task> dir>  (every numeric subdir's scene.usda is rewritten)
"""

import os
import sys

from pxr import Sdf, Usd

ASSETS_MOUNT = "/geniesim_assets"  # the GenieSimAssets bind mount inside the geniesim3 container


def rewrite(scene_path):
    stage = Usd.Stage.Open(scene_path, Usd.Stage.LoadNone)
    # GetAllChildren: with LoadNone, prims whose payload is unloaded are filtered out of GetChildren()
    for prim in stage.GetPrimAtPath("/World/Objects").GetAllChildren():
        payloads = prim.GetMetadata("payload")
        items = payloads.prependedItems if payloads else []
        if not items:
            continue
        rel = items[0].assetPath
        url = rel.split("/objects/", 1)[1] if "/objects/" in rel else None
        if url is None:
            raise SystemExit(f"{scene_path}: payload {rel} of {prim.GetPath()} is not under objects/")
        absolute = f"{ASSETS_MOUNT}/objects/{url}"
        prim.GetPayloads().SetPayloads([Sdf.Payload(absolute)])
        if "/table/" in absolute:
            entity = stage.OverridePrim(prim.GetPath().AppendChild("entity"))
            entity.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        print(f"{os.path.basename(os.path.dirname(scene_path))}/{prim.GetName()}: {absolute}")
    stage.GetRootLayer().Save()


def main(root):
    for name in sorted(n for n in os.listdir(root) if n.isdigit()):
        rewrite(os.path.join(root, name, "scene.usda"))


if __name__ == "__main__":
    main(sys.argv[1])
