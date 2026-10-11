# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""An IsaacLab-Arena environment spec of a session's generated objects, each where the session last saw it, in front of
its background (background.py's mesh room or colmap_splat's splat), with a Franka picking one and placing it at another.

    uv run sim/asset_gen/arena.py datasets/asset_gen/<scene> --pick <asset> --place <asset> [--out <spec.yaml>]

Prints the environment for sim/isaaclab_arena/envgen.sh: ARENA_ASSET_DIR (the objects), ARENA_SPLAT_SCENE (the
background) and ARENA_SPLAT_ORIGIN, where the robot stands in the scene frame: by default 0.5 m from the picked object
towards the capture point, on the floor, since the panorama saw the objects from there. Arena's frame is the scene
frame shifted to that point; positions in the spec are in it.
"""

import argparse
import json
import math
from pathlib import Path

# m from the picked object to the robot's base (untuned). On the kitchen's 1 m counter the base stood inside the counter
# and the arm knocked the cup over: counter tasks need the robot on a stand or a mobile base.
REACH = 0.5


def run(root: Path, pick: str, place: str, out: Path, background: Path | None = None):
    objects = {p.parent.name: json.loads(p.read_text()) for p in sorted((root / "objects").glob("*/pose.json"))}
    for a in (pick, place):
        if a not in objects:
            raise SystemExit(f"{a}: not among {', '.join(objects)}")
    x, y = objects[pick]["position"][:2]
    d = math.hypot(x, y)
    ox, oy = (x - REACH * x / d, y - REACH * y / d) if d > REACH else (0.0, 0.0)
    background = background or root / "background" / "scene.usda"
    lines = [f"env_name: {root.name}_generated",
             f"# Needs ARENA_SPLAT_SCENE={background}, ARENA_ASSET_DIR={root / 'objects'},",
             f"# ARENA_SPLAT_ORIGIN='{ox:.3f} {oy:.3f}' (written by sim/asset_gen/arena.py).",
             "embodiment: {id: franka, registry_name: franka_ik, params: {}}",
             "background: {id: splat, registry_name: splat_scene, params: {}}",
             "objects:"]
    lines += [f"- {{id: gen_{a}, registry_name: gen_{a}, params: {{}}}}" for a in objects]   # ids: Python names
    lines += ["object_references:", "- {id: floor, parent_id: splat, prim_path: floor, object_type: base, params: {}}",
              "relations:", "- {kind: is_anchor, subject: floor, params: {}}"]
    for a, p in objects.items():
        px, py, pz = p["position"]
        # Where the session saw it, 1 cm up so it settles onto its surface (the room collides) rather than into it.
        lines.append(f"- {{kind: at_position, subject: gen_{a}, params: {{x: {px - ox:.3f}, y: {py - oy:.3f}, "
                     f"z: {pz + 0.01:.3f}}}}}")
    lines += ["task:", "  composition: atomic",
              f"  description: Franka picks up the {objects[pick]['label']} and places it at the "
              f"{objects[place]['label']}, in a kitchen generated from one panorama.",
              "  subtasks:", "  - kind: PickAndPlaceTask", "    params:",
              f"      pick_up_object: gen_{pick}", f"      destination_location: gen_{place}",
              "      background_scene: splat"]
    out.write_text("\n".join(lines) + "\n")
    print(f"{out}\nexport ARENA_SPLAT_SCENE={background.resolve()} ARENA_ASSET_DIR={(root / 'objects').resolve()} "
          f"ARENA_SPLAT_ORIGIN='{ox:.3f} {oy:.3f}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="datasets/asset_gen/<scene> (objects/ and background/)")
    p.add_argument("--pick", required=True, help="the asset the Franka picks, as in objects/ (06_cup)")
    p.add_argument("--place", required=True, help="the asset it places it at (03_bottle)")
    p.add_argument("--background", type=Path, help="the background scene.usda (default: background.py's)")
    p.add_argument("--out", type=Path, help="spec to write (default: <root>/arena.yaml)")
    a = p.parse_args()
    run(a.root, a.pick, a.place, a.out or a.root / "arena.yaml", a.background)
