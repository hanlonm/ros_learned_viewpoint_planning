source /opt/ros/foxy/setup.bash
colcon build
source /workspace/ros2_ws/install/setup.bash

export PYTHONPATH=/workspace/viewpoint_planning:$PYTHONPATH
export PYTHONPATH=/workspace/viewpoint_planning/viewpoint_planning:$PYTHONPATH
export PYTHONPATH=/workspace/viewpoint_learning:$PYTHONPATH