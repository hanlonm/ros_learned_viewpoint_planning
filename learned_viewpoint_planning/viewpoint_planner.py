from typing import List
import rclpy
from rclpy.context import Context
from rclpy.node import Node

from viewpoint_planning.planning.viewpoint_planning import HlocViewpointPlanner

from rclpy.parameter import Parameter



class ViewpointPlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("viewpoint_planner")
        print('Hi from the viewpoint planner.')
        

def main(args=None):
    rclpy.init(args=args)
    print('Hi from learned_viewpoint_planning.')

    viewpoint_planner = ViewpointPlanningNode()

    rclpy.spin(viewpoint_planner)

    viewpoint_planner.destroy_node()
    rclpy.shutdown()

    


if __name__ == '__main__':
    main()
