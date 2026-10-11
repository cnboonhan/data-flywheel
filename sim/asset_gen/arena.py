# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""An IsaacLab-Arena environment spec of a session's generated objects, each where the session last saw it, in front of
its background (background.py's mesh room or colmap_splat's splat), with a Franka picking one and placing it at another.

    uv run sim/asset_gen/arena.py datasets/asset_gen/<scene> --pick <asset> --place <asset> [--out <spec.yaml>]

Prints the environment for sim/isaaclab_arena/envgen.sh: ARENA_ASSET_DIR (the objects), ARENA_SPLAT_SCENE (the
background) and ARENA_SPLAT_ORIGIN, where the robot spawns: the panorama's capture point, the scene frame's origin, so
Arena's frame is the scene frame. Each object stands at its pose.json position, turned by its yaw.
"""

import argparse
import json
import math
from pathlib import Path


def run(root: Path, pick: str, place: str, out: Path, background: Path | None = None):
    objects = {p.parent.name: json.loads(p.read_text()) for p in sorted((root / "objects").glob("*/pose.json"))}
    for a in (pick, place):
        if a not in objects:
            raise SystemExit(f"{a}: not among {', '.join(objects)}")
    ox, oy = 0.0, 0.0                                                 # the capture point: the robot spawns there
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
        yaw = math.radians((p["yaw_deg"] + 180) % 360 - 180)            # asset_gen's fit: the turn about z, as Arena's
        lines.append(f"- {{kind: rotate_around_solution, subject: gen_{a}, params: {{yaw_rad: {yaw:.4f}}}}}")
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
