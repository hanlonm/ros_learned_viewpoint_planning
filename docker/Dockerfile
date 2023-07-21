ARG ROS_DISTRO
ARG BASE_IMAGE
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 as cuda
FROM $BASE_IMAGE

RUN /bin/bash -c "apt-get update && apt-get -y dist-upgrade"

# Add cuda&cudnn libraries
COPY --from=cuda /usr/local/cuda /usr/local/cuda
COPY --from=cuda /usr/lib/x86_64-linux-gnu/*libcudnn* /usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/targets/x86_64-linux/lib/"
ENV PATH="${PATH}:/usr/local/cuda/bin"

# https://discourse.ros.org/t/ros-gpg-key-expiration-incident/20669
# Also, the ROS ppa has to be removed while curl is being installed
RUN /bin/bash -c "mv /etc/apt/sources.list.d/ros2-latest.list /etc/ros2-latest.list; \
    apt-get update && apt-get install -y curl python3-tk; \
    mv /etc/ros2-latest.list /etc/apt/sources.list.d/ros2-latest.list; \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -; curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg"

# install python stuff
COPY ./docker/content/requirements.txt /workspace/requirements.txt
RUN apt update && apt install python3-pip -y
RUN pip install -r /workspace/requirements.txt
RUN pip install pyembree

ENV ROS2_WS /workspace/ros2_ws
RUN mkdir -p $ROS2_WS/src
WORKDIR $ROS2_WS
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; \
		  colcon build" 