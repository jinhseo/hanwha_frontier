import rospy
import numpy as np
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ColorRGBA
import math
import tf.transformations
from std_msgs.msg import MultiArrayDimension, MultiArrayLayout, Float32MultiArray

def transform_global_goal_to_local(global_goal_x, global_goal_y, odom_position_x, odom_position_y, odom_rotation_yaw):
    dx = global_goal_x - odom_position_x
    dy = global_goal_y - odom_position_y

    rospy.loginfo(f"Global goal: ({global_goal_x}, {global_goal_y})")
    rospy.loginfo(f"Robot position: ({odom_position_x}, {odom_position_y})")
    rospy.loginfo(f"Delta: ({dx}, {dy})")
    rospy.loginfo(f"Robot yaw: {odom_rotation_yaw}")

    cos_yaw = math.cos(-odom_rotation_yaw)
    sin_yaw = math.sin(-odom_rotation_yaw)

    local_goal_x = dx * cos_yaw - dy * sin_yaw
    local_goal_y = dx * sin_yaw + dy * cos_yaw

    rospy.loginfo(f"Transformed goal: ({local_goal_x:.2f}, {local_goal_y:.2f})")

    return (local_goal_x, local_goal_y)

def transform_local_to_world(local_point, odom_position_x, odom_position_y, odom_rotation_yaw):
    cos_yaw = math.cos(-odom_rotation_yaw)
    sin_yaw = math.sin(-odom_rotation_yaw)

    world_x = odom_position_x + (local_point[0] * cos_yaw - local_point[1] * sin_yaw)
    world_y = odom_position_y + (local_point[0] * sin_yaw + local_point[1] * cos_yaw)

    return (world_x, world_y)

def visualize_goal_projection(transformed_global_goal, grid_map_info, goal_projection_pub):
    if transformed_global_goal is None:
        return

    marker = Marker()
    marker.header.frame_id = "aligned_basis"
    marker.header.stamp = grid_map_info.header.stamp if grid_map_info else rospy.Time.now()
    marker.ns = "goal_projection"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.pose.position.x = transformed_global_goal[0]
    marker.pose.position.y = transformed_global_goal[1]
    marker.pose.position.z = 1.5

    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.6
    marker.scale.y = 0.6
    marker.scale.z = 0.6

    marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8)

    goal_projection_pub.publish(marker)

def visualize_frontiers(frontiers, selected_frontier, grid_map_info, frontier_viz_pub, max_frontier_id_ref):
    marker_array = MarkerArray()

    for i in range(max_frontier_id_ref[0] + 1):
        delete_marker = Marker()
        delete_marker.header.frame_id = "aligned_basis"
        delete_marker.header.stamp = grid_map_info.header.stamp if grid_map_info else rospy.Time.now()
        delete_marker.ns = "frontiers"
        delete_marker.id = i
        delete_marker.action = Marker.DELETE
        marker_array.markers.append(delete_marker)

    if frontiers:
        max_frontier_id_ref[0] = len(frontiers) - 1
    else:
        frontier_viz_pub.publish(marker_array)
        return

    for i, (x, y) in enumerate(frontiers):
        marker = Marker()
        marker.header.frame_id = "aligned_basis"
        marker.header.stamp = grid_map_info.header.stamp if grid_map_info else rospy.Time.now()
        marker.ns = "frontiers"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 1.5

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        if selected_frontier and (x, y) == selected_frontier:
            marker.scale.x = 0.8
            marker.scale.y = 0.8
            marker.scale.z = 0.8
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        else:
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=1.0)

        marker_array.markers.append(marker)

    frontier_viz_pub.publish(marker_array)

def visualize_global_goal(global_goal_x, global_goal_y, grid_map_info, global_goal_viz_pub, odom_position_x, odom_position_y, odom_rotation_yaw):
    transformed_goal = transform_global_goal_to_local(global_goal_x, global_goal_y, odom_position_x, odom_position_y, odom_rotation_yaw)

    goal_marker = Marker()
    goal_marker.header.frame_id = "world"
    goal_marker.header.stamp = grid_map_info.header.stamp if grid_map_info else rospy.Time.now()
    goal_marker.ns = "global_goal"
    goal_marker.id = 0
    goal_marker.type = Marker.SPHERE
    goal_marker.action = Marker.ADD

    goal_marker.pose.position.x = global_goal_x
    goal_marker.pose.position.y = global_goal_y
    goal_marker.pose.position.z = 2.0

    goal_marker.pose.orientation.x = 0.0
    goal_marker.pose.orientation.y = 0.0
    goal_marker.pose.orientation.z = 0.0
    goal_marker.pose.orientation.w = 1.0

    goal_marker.scale.x = 1.5
    goal_marker.scale.y = 1.5
    goal_marker.scale.z = 1.5

    goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

    marker_array = MarkerArray()
    marker_array.markers.append(goal_marker)
    global_goal_viz_pub.publish(marker_array)

def visualize_traversability_map(traversability_map, grid_map, traversability_viz_pub):
    frontier_map = GridMap()

    frontier_map.info = grid_map.info
    frontier_map.layers = ['traversability']

    inverted_map = np.copy(traversability_map)
    valid_mask = ~np.isnan(traversability_map)
    inverted_map[valid_mask] = 1.0 - traversability_map[valid_mask]

    traversability_data_msg = Float32MultiArray()

    dim1 = MultiArrayDimension()
    dim1.label = "column_index"
    dim1.size = traversability_map.shape[0]
    dim1.stride = traversability_map.size

    dim2 = MultiArrayDimension()
    dim2.label = "row_index"
    dim2.size = traversability_map.shape[1]
    dim2.stride = traversability_map.shape[1]

    traversability_data_msg.layout.dim = [dim1, dim2]
    traversability_data_msg.layout.data_offset = 0

    traversability_data_msg.data = inverted_map.flatten().tolist()

    frontier_map.data = [traversability_data_msg]

    traversability_viz_pub.publish(frontier_map)

    original_stamp = grid_map.info.header.stamp
    published_stamp = frontier_map.info.header.stamp
    time_diff = abs((original_stamp - published_stamp).to_sec()) if original_stamp != published_stamp else 0.0

    if time_diff > 0.001:
        rospy.logwarn(f"Timestamp sync issue: {time_diff:.3f}s difference")
