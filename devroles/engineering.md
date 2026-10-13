# Engineering

## Scope

Builds and runs what the flywheel stands on: the tooling that collects data and scenes, the infra that stores and serves it, and the evaluation loop that feeds results back.

## Responsibilities

![The flywheel with engineering's areas: tooling, infra, the sync specs, backup and restore, the deploy to inference compute, physical evals, the reset automation, the eval logs, the feedback loop and the upload and sync from off-robot collection and teleoperation in red; off-robot data collection, the upload of public datasets and virtual evals, shared with the Scientist, in green](../assets/architecture-engineering.svg)

| Item | Description |
|---|---|
| Off-Robot Data Collection | Build and arrange collection of data using Off-Robot methods. For example, YUBI / UMI / Egocentric / Simulation. Be familiar with hardware specs, limitations, maintenance and repair. Highlight hardware degradation that can affect data quality. Validate quality of data output (e.g. timestamping sync, calibration, joint limit readings accuracy ) |
| Robot Teleoperation | Build and arrange collection of data using On-Robot Teleoperation. Familiarity with teleoperation stack implementation for both IK and WBC. Able to tune control stack and highlight safety issues during robot deployment. Be familiar with hardware specs, limitations, maintenance and repair. Highlight hardware degradation that affects data quality. Validate quality of data output (e.g timestamping sync, calibration, joint limits reading accuracy). |
| Public Datasets | Help to pull and sync raw public datasets into data flywheel according to scientist requests |
| Virtual Scenes | Develop digital tests where applicable to unit test real world deployment of models. These should help to provide targeted signal on the model performance in real environment, and complement the Pre-Evals signals during Architecture experiments by accounting for real world constraints. For example, Pre-Eval should be a Validation Action Error (numerical), whereas a Virtual Scene would be to require the model to move into place within X seconds, to a distance of at most X m. |
| Physical Scenes | Develop and maintain the real world test environments. Actively improve the physical environments to improve the possible configuration options and signals required for deployment. (For example, mockup button lighting, output numbers). Track the physical limitations during rollout and highlight to team to implement countermeasures. (e.g time limit on lift staying open / Human traffic / Variable door heights ). Determine constraints for successful real world execution and enhance testing for this. |
| Sync Specs | Aim for Virtual Scenes to (cumulatively) provide evaluation signals on par to the real world |
| Inference Compute | Handle the administration, readiness and stability of the compute to run models and robot control stack, connectivity etc. Teleoperators should be able to collect data and upload, Evaluators should be able to launch and run tests with minimal backend / plumbing / debugging. |
| Upload and Sync | Handle the upload of raw data from On and Off Robot Sources and Public Datasets |
| Connectivity | Profile the networking limitations of the real hardware when deployed for teleoperation or physical evaluations. Optimize the connectivity deployment, and advise countermeasures in the control stack to handle network failures. Preemptively test and implement network recovery where applicable. Arrange the necessary QoS and compute reservations to minimize non-deterministic latency. |
| Data Storage | Manage the infrastructure uptime of the Data Storage Layer of Flywheel |
| Backup and Restore | Manage the backup and restore of datasets, checkpoints in case of data loss. |
| Logging | Manage the logging infrastructure and advise on monitoring for alerts and issues raised |
| Model Registry | Manage the deployment of the model registry and experiment tracking |
| Deploy (Model Registry → Inference Compute) | Handle the deployment of checkpoints and teleoperation / inference libraries to inference compute. Implement seamless deployment to multiple downstream inference compute |
| Virtual Evals | Run virtual evals on the virtual scenes and analyze the results |
| Physical Evals | Run physical evals on Physical Scenes and analyze results. |
| Reset Automation | Design ways to reduce cost / effort of running and configuring Physical Evals. For example, using a Navstack to reset the robot position and randomize position automatically; Prototype motors or a push robot to automatically close a swing door etc. |
| Eval Logs | Design infrastructure and tooling to continuously collect useful data during physical and virtual evals. Identify useful data points to feedback to scientists. Set up hardware infrastructure (multiple configurable camera views, localization) to improve the signal from evals |
| Feedback Loop | Continuously push evaluation data back to storage for processing and use in model improvements |
