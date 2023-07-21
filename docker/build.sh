#!/bin/sh

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

ROS_DISTRO="foxy"
BASE_IMAGE="osrf/ros:$ROS_DISTRO-desktop"
IMAGE_NAME="viewpoint_planning:0.0.1"

while getopts r:b:t: flag
do
    case "${flag}" in
        r) ROS_DISTRO=${OPTARG};;
        b) BASE_IMAGE=${OPTARG};;
        t) IMAGE_NAME=${OPTARG};;
        *) error "Unexpected option ${flag}" ;;
    esac
done

docker build \
    -t $IMAGE_NAME \
    --build-arg ROS_DISTRO=$ROS_DISTRO \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    -f Dockerfile ${SCRIPT_DIR}/..