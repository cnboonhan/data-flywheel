# Why these numbers

Where each value in `table_task_lab_g2_op.json` and the layouts comes from. The benchmark's placement
rules, read from `geniesim_benchmark` 3.2 (`app/controllers/api_core.py:_init_robot_cfg`, `_reload_scenes`,
`benchmark/task_benchmark.py`, `plugins/tgs/layout/task_generate.py`):

- The background (`scene_usd`) is referenced at `/World` with no transform.
- The instance's `scene.usda` is referenced at `/Workspace` with no transform, so the table in
  `scene.usda` lands where the program put it: table centre at the world origin.
- The robot prim is placed at `robot_init_pose[<workspace id>]` in world coordinates. The workspace id is
  the last path element of `scene.scene_id`.
- `function_space_objects` only feeds `LayoutGenerator`, which places `objects.task_related_objects` /
  `extra_objects`. Both lists are empty here (as in every shipped table task), so these numbers never move
  anything; the shipped G1 and G2 jsons carry the same `(-4.03, -0.032, 0.806)` supermarket values under
  every background for that reason.

| Value | Chosen | Why |
|---|---|---|
| `scene_usd` | `background/laboratory/laboratory_3/background.usda` | The asset pack's only Gaussian-splat background (NuRec volume `common/laboratory_3/splat_7/splat_7.nurec`); the G1 task `table_task_g1_op.json` already uses it with the same table instances. |
| `scene_id` | `laboratory_3/workspace_00` | Only the suffix matters: it selects `robot_init_pose.workspace_00` and `function_space_objects.workspace_00`. |
| Table in the background | none, so the layout brings `benchmark_table_019` | `background.usda` holds the splat, a dome light, a 50 x 50 m ground collision plane at z = 0 and three invisible wall planes; no table geometry. `metricsAssembler` lists `/World/table`, `/World/table2` but no such prims exist. The splat volume (after its three transforms) spans x in [-11.6, 23.5], y in [-13.7, 38.0], z in [-0.68, 5.15]; the `gauss` xform is lifted 0.021 m, which reads as the splat floor being aligned to z = 0. The origin is inside the splat, and the G1 task puts the same table there. |
| Table pose | `(0, 0, 0.265)`, kinematic | `benchmark_table_019` mesh spans z in [-0.265, 0.465] about its origin (measured with pxr), so 0.265 stands it on the z = 0 ground plane with its top at 0.73 m; same as every shipped instance. `physics:kinematicEnabled` as shipped, so the arm cannot push a 0.6 x 1.0 m table. |
| `robot_init_pose.workspace_00` | `(-0.71, 0, 0)`, identity quaternion | Copied from `table_task_2_g2_op.json`: the G2 base 0.71 m behind the table centre (0.41 m from the table edge), facing +x, feet on the z = 0 ground plane. The G1 json uses (-0.8, 0, -0.01) for the taller G1. Robot-to-table geometry does not change with the background, so neither does this. |
| `function_space_objects.workspace_00.position` | `(0, 0, 0.73)` | The table-top centre in world coordinates, the only honest value (unused, see above). `size` 0 as shipped; `LayoutSolver2D` only stores it. The shipped `workspace_00_extra` / `_ground` entries describe supermarket fixtures and are dropped. |
| Object x | `[-0.24, 0.04]` | Shipped instances keep objects at x in about [-0.18, 0.15] (0.53-0.86 m from the robot base); the brief asked for [-0.35, 0.05]. [-0.24, 0.04] keeps every object on the 0.6 m-deep top (edge at -0.30) and 0.47-0.75 m from the base. |
| Object y | `[-0.26, 0.26]` | Shipped band is [-0.25, 0.23] on a 1.0 m-wide top; the G2 is asked for the right arm but the checks accept either gripper. |
| Object spacing | >= 0.20 m centre to centre | Brief asks >= 0.12 m; 0.20 leaves a 13.6 cm gripper room beside any 8 cm object. Verified by `gen/verify.py`. |
| Free target strip | 0.05-0.20 m past B along the relation axis, +-0.10 m across | What `RelativePosition` accepts: offset > 0.05 along the axis and `aligned_*` tolerance 0.1 across; 0.20 m reach so a 10 cm object fits. Verified by `gen/verify.py`. |
| Object z | bottom = 0.74 m | Table top 0.73 + 0.01 m clearance so convex hulls never start inside the top. Shipped instances start objects ~0.15 m up and let them drop, which rolls fruit. |
| Follow box | `[0.15, 0.15, 0.15]` | Shipped values are 0.10-0.25 per axis by object (0.15 for fruit); 0.15 covers every object here (tallest 11 cm) with margin for the gripper. |
| Step limits | 300 reach, 600 place | As shipped. |
| `aligned_*` tolerance | 0.1 | As shipped. |

Not verifiable without Isaac Sim: that the NuRec splat renders in this container (the G1 json is the only
evidence it loads), that the splat's floor and the z = 0 collision plane coincide (2 cm offset authored in
`background.usda`), and that the robot's camera view of the table is not occluded by splat content near
the origin.
