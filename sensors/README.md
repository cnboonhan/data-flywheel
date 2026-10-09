# sensors

Data collection: real hardware (git submodules; see each one's upstream README) and a simulated robot behind the same ROS 2 interface.

| Path | Contents |
|---|---|
| [`yubi-hw`](https://github.com/Toyota/yubi-hw) | YUBI hardware (CAD): glove, gripper, stationary desk and portable rig |
| [`yubi-sw`](https://github.com/airoa-org/yubi-sw) | YUBI operator-side ROS 2 bringup: encoders, hand cameras, foot pedal or Quest controllers |
| [`real2sim/`](real2sim/README.md) | Isaac Sim Nova Carter warehouse + Nav2 on a known map (goals as ROS 2 poses), and map-planned image capture for splats |

Project page: [yubi.airoa.io](https://yubi.airoa.io/)
