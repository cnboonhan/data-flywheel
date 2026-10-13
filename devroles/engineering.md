# Engineering

## Scope

Builds and runs what the flywheel stands on: the tooling that collects data and scenes, the infra that stores and serves it, and the evaluation loop that feeds results back.

## Responsibilities

![The flywheel with engineering's areas: tooling, infra, the sync specs, backup and restore, the deploy to inference compute, physical evals, the reset automation, the eval logs, the feedback loop and the upload and sync from off-robot collection and teleoperation in red; the upload of public datasets and virtual evals, shared with the Scientist, in green](../assets/architecture-engineering.svg)

| Item | Description |
|---|---|
| Off-Robot Data Collection | Build and arrange collection of data using Off-Robot methods. For example, YUBI / UMI / Egocentric. Be familiar with hardware specs, limitations, maintenance and repair. Highlight hardware degradation that can affect data quality. Validate quality of data output (e.g. timestamping sync, calibration, joint limit readings accuracy ) |
| Robot Teleoperation | Build and arrange collection of data using On-Robot Teleoperation. Familiarity with teleoperation stack implementation for both IK and WBC. Able to tune control stack and highlight safety issues during robot deployment. Be familiar with hardware specs, limitations, maintenance and repair. Highlight hardware degradation that affects data quality. Validate quality of data output (e.g timestamping sync, calibration, joint limits reading accuracy). |
| Public Datasets | Help to pull and sync raw public datasets into data flywheel according to scientist requests |
| Virtual Scenes | Develop Simulations where applicable to unit test real world deployment of models. These should help to provide targeted signal on the model performance in real environment, and complement the Pre-Evals signals during Architecture experiments by accounting for real world constraints. For example, Pre-Eval should be a Validation Action Error (numerical), whereas a Virtual Scene would be to require the model to move into place within X seconds, to a distance of at most X m. |
| Physical Scenes | Develop and maintain the real world test environments. Actively improve the physical environments to improve the possible configuration options and signals required for deployment. (For example, mockup button lighting, output numbers). Track the physical limitations during rollout and highlight to team to implement countermeasures. (e.g time limit on lift staying open / Human traffic / Variable door heights ). Determine constraints for successful real world execution and enhance testing for this. |
| Sync Specs | Aim for Virtual Scenes to (cumulatively) provide evaluation signals on par to the real world |
| Inference Compute | |
| Upload and Sync | |
| Connectivity | |
| Data Storage | |
| Backup and Restore | |
| Logging | |
| Model Registry | |
| Deploy (Model Registry → Inference Compute) | |
| Virtual Evals | |
| Physical Evals | |
| Reset Automation | |
| Eval Logs | |
| Feedback Loop | |
