# Scientist

## Scope

Turns raw data into better models: processes it into training sets, experiments with architectures, trains, optimises and distils models, and evaluates them in simulation.

## Responsibilities

![The flywheel with the scientist's areas: data processing, architecture experiments, model training and pre-evals, optimizations, distillation and the deploy into the model registry in blue; the upload of public datasets and sim evals, shared with Engineering, in green](../assets/architecture-scientist.svg)

| Item | Example |
|---|---|
| Upload and Sync | Identify promising Public Datasets and upload them into Data Storage |
| Data Cleaning | Understand what are unhelpful features in data or misconfigurations in relation to the hardware configuration. Filter out bad data (e.g pauses, still frames) and validate data ingress assumptions against input hardware specs (e.g camera fisheye vs pinhole, resolution, joint angle ranges). |
| Data Conversion | Understand the data formats and schema needed downstream for all models we intend to train, as well as the infrastructure the training runs on. Write adapters to implement this conversion. |
| Data Mixing | Read papers and develop intuitions / rules of thumb of how to mix and compose different processed datasets and arrange the training batch schedule in order to improve performance. Provide recommendations on what datasets / tasks require more data collection |
| Architecture Experiments | Form hypothesis on what data processing, data mix strategies and architecture changes will improve model performance. |
| Model Training and Pre-Evals | |
| Inference / Model Optimizations | |
| Distillation | |
| Deploy (Train → Model Registry) | Deploy trained models onto Registry. Tag them with important information for model ranking and deployment (Eval results, training metrics, deployment flags, data mix). |
| Sim Evals | |
