#!/usr/bin/env python3

import rospy
import numpy as np
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import MarkerArray, Marker

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
import math
from nav_msgs.msg import Odometry
import tf.transformations
from std_msgs.msg import Int32
from utils import (
    global_to_local,
    visualize_goal_projection,
    visualize_frontiers,
    visualize_global_goal,
    local_to_global
)

class FrontierDetector:
    def __init__(self):
        rospy.init_node('frontier_detector', anonymous=True)

        self.SEARCH_RADIUS = 9 ### 15
        self.ANGLE_RESOLUTION = 0.2
        self.STEP_RESOLUTION = 0.25
        self.COST_THRESHOLD = 0.7
        self.SAFETY_RADIUS = 3

        # --- Mode Management --- #
        self.SEARCH_MODE = 0
        self.RECOVERY_MODE = 1
        self.EXPLORATION_MODE = 2
        self.current_mode = self.SEARCH_MODE
        self.current_mode_pub = rospy.Publisher('/frontier_detector/current_mode', Int32, queue_size=1) # Added mode publisher
        self.current_mode_pub.publish(self.current_mode) # Publish initial mode

        # Dead End Detection Parameters
        self.local_goal_history = []
        self.LOCAL_GOAL_HISTORY_SIZE = 10  # Number of local goals to track
        self.LOCAL_GOAL_STAGNATION_THRESHOLD = 0.1 # meters, if local goals don't change much

        self.odom_history = []
        self.ODOM_HISTORY_SIZE = 50 # Number of odom positions to track
        self.ROBOT_STAGNATION_THRESHOLD = 0.3 # meters, if robot doesn't move much

        self.latest_local_goal = None

        self.global_goal_x = rospy.get_param('/global_goal_x', 0.0) ### ('/target_goal_x', -499.948609927)
        self.global_goal_y = rospy.get_param('/global_goal_y', 0.0) ### ('/target_goal_y', 124.862913548)

        self.odom_position_x = 0.0
        self.odom_position_y = 0.0
        self.odom_rotation_yaw = 0.0

        self.grid_map_sub = rospy.Subscriber('/GridMap_planning', GridMap, self.grid_map_callback) ### aligned_base
        self.odom_sub = rospy.Subscriber('/global/odometry', Odometry, self.odom_callback)                                ### world -> base
        #self.odom_sub = rospy.Subscriber('/odometry_gt', Odometry, self.odom_callback)
        self.global_goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.global_goal_callback)         ### aligned_base

        self.frontier_viz_pub = rospy.Publisher('/frontier_visualization', MarkerArray, queue_size=1)
        self.global_goal_viz_pub = rospy.Publisher('/global_goal_visualization', MarkerArray, queue_size=1)

        self.local_goal_pub = rospy.Publisher('/local_goal', PoseStamped, queue_size=1)
        self.goal_projection_pub = rospy.Publisher('/goal_projection_visualization', Marker, queue_size=1, latch=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.grid_map = None
        self.max_frontier_id = [-1]

        self.resolution = None
        self.width = None
        self.height = None

    def grid_map_callback(self, msg):
        try:
            self.grid_map = msg
            self.resolution = msg.info.resolution
            self.width = msg.info.length_x
            self.height = msg.info.length_y

            traversability_map = self.get_layer_data('planning_cost_map_updated')

            transformed_global_goal = global_to_local(self.tf_buffer, self.global_goal_x, self.global_goal_y, "world", "aligned_base")

            ### frontier candidates ###
            frontiers = self.find_frontiers(traversability_map, transformed_global_goal)

            if frontiers:
                local_goal = self.select_local_goal(frontiers, transformed_global_goal)

                if local_goal:
                    self.publish_local_goal(local_goal)

                    # Update local goal history
                    self.local_goal_history.append(local_goal)
                    if len(self.local_goal_history) > self.LOCAL_GOAL_HISTORY_SIZE:
                        self.local_goal_history.pop(0)

                    # Update latest local goal
                    self.latest_local_goal = local_goal

                    # Convert local_goal to global frame and set as ROS parameters
                    global_goal_from_local = local_to_global(
                        self.tf_buffer, local_goal[0], local_goal[1], "aligned_base", "world"
                    )
                    if global_goal_from_local:
                        rospy.set_param('/global_goal_x', global_goal_from_local[0])
                        rospy.set_param('/global_goal_y', global_goal_from_local[1])
                        rospy.loginfo(f"Set ROS param /global_goal_x: {global_goal_from_local[0]:.2f}, /global_goal_y: {global_goal_from_local[1]:.2f}")
                    else:
                        rospy.logwarn("Failed to convert local_goal to global frame.")

                viz_start_time = rospy.Time.now()
                rospy.loginfo(f"GridMap Update - Found {len(frontiers)} frontiers")
                visualize_frontiers(frontiers, local_goal, self.grid_map.info, self.frontier_viz_pub, self.max_frontier_id)
                visualize_global_goal(self.global_goal_x, self.global_goal_y, self.grid_map.info, self.global_goal_viz_pub, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw)
                visualize_goal_projection(transformed_global_goal, self.grid_map.info, self.goal_projection_pub)
            else:
                viz_start_time = rospy.Time.now()
                visualize_frontiers([], None, self.grid_map.info, self.frontier_viz_pub, self.max_frontier_id)
                visualize_global_goal(self.global_goal_x, self.global_goal_y, self.grid_map.info, self.global_goal_viz_pub, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw)
                # Check for no frontiers, and mode transition
                if self.current_mode == self.SEARCH_MODE and not frontiers:
                    rospy.logwarn("No frontiers found. Switching to RECOVERY_MODE.")
                    self.current_mode = self.RECOVERY_MODE
                    self.current_mode_pub.publish(self.current_mode) # Publish mode change

        except Exception as e:
            rospy.logerr(f"Error processing grid map: {e}")

    def odom_callback(self, msg):
        #self.global_goal_x = rospy.get_param('/target_goal_x', 0.0) ### ('/target_goal_x', -499.948609927)
        #self.global_goal_y = rospy.get_param('/target_goal_y', 0.0) ### ('/target_goal_y', 124.862913548)

        self.odom_position_x = msg.pose.pose.position.x
        self.odom_position_y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        self.odom_rotation_yaw = yaw

        # Update odom history for stagnation detection
        self.odom_history.append((self.odom_position_x, self.odom_position_y))
        if len(self.odom_history) > self.ODOM_HISTORY_SIZE:
            self.odom_history.pop(0)

        # Check for robot stagnation and mode transition (combined with local goal stagnation)
        if (self.current_mode == self.SEARCH_MODE and
            self._is_robot_stagnated() and
            self._is_local_goal_stagnated(self.latest_local_goal)):
            rospy.logwarn("Robot and local goal detected to be stagnated. Switching to RECOVERY_MODE.")
            self.current_mode = self.RECOVERY_MODE
            self.current_mode_pub.publish(self.current_mode) # Publish mode change

        # Log robot movement and stagnation status
        total_movement = 0.0
        if len(self.odom_history) > 1:
            for i in range(1, len(self.odom_history)):
                p1 = self.odom_history[i-1]
                p2 = self.odom_history[i]
                total_movement += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        rospy.loginfo(f"[DeadEnd Debug] Robot Total Movement ({len(self.odom_history)} points): {total_movement:.3f}m, Stagnated: {self._is_robot_stagnated()}")

        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        total_tilt = max(abs(roll_deg), abs(pitch_deg))

    def global_goal_callback(self, msg):
        target_frame = "world"
        source_frame = msg.header.frame_id

        rospy.loginfo(f"Attempting to set new goal from RViz. (Source: {source_frame})")

        try:
            pose_to_transform = PoseStamped()
            pose_to_transform.header.frame_id = source_frame
            pose_to_transform.header.stamp = rospy.Time(0)
            pose_to_transform.pose = msg.pose

            transformed_pose_stamped = self.tf_buffer.transform(
                pose_to_transform,
                target_frame,
                rospy.Duration(1.0)
            )
            self.global_goal_x = transformed_pose_stamped.pose.position.x
            self.global_goal_y = transformed_pose_stamped.pose.position.y

            rospy.set_param('/global_goal_x', self.global_goal_x)
            rospy.set_param('/global_goal_y', self.global_goal_y)

            rospy.loginfo(f"New global goal set in {target_frame}: "
                          f"({self.global_goal_x:.2f}, {self.global_goal_y:.2f})")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to transform global goal from {source_frame} to {target_frame}: {e}")
            rospy.logwarn("Global goal was NOT updated. Check TF tree (is 'world' frame available?).")

    def get_layer_data(self, layer_name):
        try:
            if layer_name not in self.grid_map.layers:
                rospy.logwarn(f"Layer {layer_name} not found in grid map")
                return None

            layer_index = self.grid_map.layers.index(layer_name)

            data_msg = self.grid_map.data[layer_index]
            raw_data = np.array(data_msg.data)
            rows = data_msg.layout.dim[0].size
            cols = data_msg.layout.dim[1].size

            data = raw_data.reshape((rows, cols))

            return data

        except Exception as e:
            rospy.logerr(f"Error getting layer data for {layer_name}: {e}")
            return None

    def find_frontiers(self, traversability_map, transformed_global_goal):
        frontiers = []
        center_x = int(self.width / (2 * self.resolution))
        center_y = int(self.height / (2 * self.resolution))

        total_nan = np.sum(np.isnan(traversability_map))
        total_safe = np.sum(traversability_map == 0.0)
        total_caution = np.sum(traversability_map == 0.5)
        total_blocked = np.sum(traversability_map == 1.0)

        rays_checked = 0
        rays_hit_nan = 0
        rays_hit_blocked = 0

        # --- Debugging Start ---
        rospy.loginfo(f"[Dense Ray Debug] Transformed Global Goal: ({transformed_global_goal[0]:.2f}, {transformed_global_goal[1]:.2f})")
        # rospy.loginfo(f"[Dense Ray Debug] Flipped Goal Vector for Angle: ({goal_vector_x:.2f}, {goal_vector_y:.2f})") # No longer applicable with direct atan2 call
        # --- Debugging End ---

        # Corrected atan2 call with x-axis flip applied to the x-component for consistency
        global_goal_angle = math.atan2(-transformed_global_goal[1], -transformed_global_goal[0])
        rospy.loginfo(f"[Dense Ray Debug] Global Goal Angle (degrees): {math.degrees(global_goal_angle):.2f}")

        # Generate angles
        angles_to_check = []

        # Denser angles around the global goal direction
        denser_angle_range = np.pi / 6.0 # +- 45 degrees around global goal (can be tuned)
        denser_angle_resolution = self.ANGLE_RESOLUTION / 6.0 # 3 times denser (can be tuned)

        # Sparse angles for other directions
        sparse_angle_resolution_factor = 3.0 # Make it 2 times sparser than normal (can be tuned)
        sparse_angle_resolution = self.ANGLE_RESOLUTION * sparse_angle_resolution_factor

        # Iterate through 360 degrees
        current_angle = 0.0
        while current_angle < 2 * np.pi:
            angle_diff = np.abs(self._normalize_angle(current_angle - global_goal_angle))

            if angle_diff <= denser_angle_range:
                angles_to_check.append(current_angle)
                current_angle += denser_angle_resolution
            else:
                # Add angle with sparser resolution
                angles_to_check.append(current_angle)
                current_angle += sparse_angle_resolution

        # Ensure all angles are unique and sorted
        all_angles = np.unique(np.array(angles_to_check))

        for angle in all_angles: # Iterate over the combined set of angles
            rays_checked += 1
            max_dist_cells = int(self.SEARCH_RADIUS / self.resolution)
            farthest_traversable_point = None

            for dist in range(1, max_dist_cells):
                x = int(center_x + dist * np.cos(angle))
                y = int(center_y + dist * np.sin(angle))

                if not (0 <= x < traversability_map.shape[1] and 0 <= y < traversability_map.shape[0]):
                    break

                current_value = traversability_map[y, x]

                if np.isnan(current_value):
                    rays_hit_nan += 1
                    break
                elif current_value >= self.COST_THRESHOLD:
                    rays_hit_blocked += 1
                    break
                elif current_value < self.COST_THRESHOLD:
                    farthest_traversable_point = (x, y)

            if farthest_traversable_point is not None:
                cell_x, cell_y = farthest_traversable_point

                #local_frontier_x = self.grid_map.info.pose.position.x - (cell_x - center_x) * self.resolution
                #local_frontier_y = self.grid_map.info.pose.position.y - (cell_y - center_y) * self.resolution

                # right-left flip
                local_frontier_x = - (cell_x - center_x) * self.resolution
                local_frontier_y = - (cell_y - center_y) * self.resolution

                dist_from_robot = np.sqrt(local_frontier_x**2 + local_frontier_y**2)

                traversability_value = traversability_map[cell_y, cell_x]

                if (dist_from_robot > self.SAFETY_RADIUS): # Removed check_min_distance call
                    frontiers.append((local_frontier_x, local_frontier_y, dist_from_robot))

        rospy.loginfo(f"Found {len(frontiers)} frontier candidates")

        if len(frontiers) == 0:
            rospy.logwarn("No frontiers found!")
            rospy.logwarn(f"Debug info:")
            rospy.logwarn(f"  Rays checked: {rays_checked}")
            rospy.logwarn(f"  Rays hit NaN: {rays_hit_nan}")
            rospy.logwarn(f"  Rays hit blocked: {rays_hit_blocked}")
            rospy.logwarn(f"  Map stats - NaN: {total_nan}, Safe: {total_safe}, Caution: {total_caution}, Blocked: {total_blocked}")

            if rays_hit_nan == 0:
                rospy.logwarn("  → Try increasing search radius or check if NaN regions exist in the map")

        return frontiers

    def _normalize_angle(self, angle):
        """Normalize angle to be within [-pi, pi)"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def select_local_goal(self, frontiers, transformed_global_goal):
        if not frontiers:
            return None

        goal_x = transformed_global_goal[0]
        goal_y = transformed_global_goal[1]

        robot_heading_vector = np.array([math.cos(self.odom_rotation_yaw), math.sin(self.odom_rotation_yaw)])

        # Calculate distances for all frontiers
        frontier_data = []
        for fx, fy, f_dist_from_robot in frontiers:
            dist_to_goal = math.sqrt((fx - goal_x)**2 + (fy - goal_y)**2)

            # robot to frontier vector
            vector_to_frontier = np.array([fx, fy])

            # Normalize vector_to_frontier (if not zero)
            norm_vector_to_frontier = np.linalg.norm(vector_to_frontier)
            if norm_vector_to_frontier > 0:
                normalized_vector_to_frontier = vector_to_frontier / norm_vector_to_frontier
            else:
                normalized_vector_to_frontier = np.array([0.0, 0.0])

            dot_product = np.dot(robot_heading_vector, normalized_vector_to_frontier)
            angle_score = np.clip(dot_product, -1.0, 1.0)

            frontier_data.append((fx, fy, f_dist_from_robot, dist_to_goal, angle_score))

        if not frontier_data:
            return None

        # Extract distances for normalization
        goal_distances = [data[3] for data in frontier_data]
        search_distances = [data[2] for data in frontier_data]
        angle_scores = [data[4] for data in frontier_data]

        # Convert to numpy arrays for vectorized operations
        goal_distances_np = np.array(goal_distances)
        search_distances_np = np.array(search_distances)
        # angle_costs_np = np.array(angle_costs) # No longer needed for min-max normalization

        min_goal_dist = np.min(goal_distances_np)
        max_goal_dist = np.max(goal_distances_np)
        # min_search_dist = np.min(search_distances_np) # No longer needed for normalization range
        # max_search_dist = np.max(search_distances_np) # No longer needed for normalization range

        # Normalize goal distances (smaller is better)
        if (max_goal_dist - min_goal_dist) > 0:
            normalized_goal_distances = (goal_distances_np - min_goal_dist) / (max_goal_dist - min_goal_dist)
        else:
            normalized_goal_distances = np.zeros_like(goal_distances_np)

        # Normalize search distances (larger is better, using fixed radius for min/max)
        # Assuming search_distances are in meters
        min_search_range = self.SAFETY_RADIUS * self.resolution # The absolute minimum traversable distance from robot
        max_search_range = self.SEARCH_RADIUS # The absolute maximum search radius

        if (max_search_range - min_search_range) > 0:
            clamped_search_distances = np.clip(search_distances_np, min_search_range, max_search_range)
            normalized_search_distances = (clamped_search_distances - min_search_range) / (max_search_range - min_search_range)
        else:
            normalized_search_distances = np.zeros_like(search_distances_np)

        weight_goal_dist = 0.3
        weight_search_dist = 0.7
        weight_angle_score = 0.0
        #print("weight_goal_dist: ", weight_goal_dist)
        #print("weight_search_dist: ", weight_search_dist)
        print("normalized_goal_distances: ", normalized_goal_distances)
        print(len(normalized_goal_distances))
        print("normalized_search_distances: ", normalized_search_distances)
        print(len(normalized_search_distances))
        print("angle_scores: ", angle_scores)
        print(len(angle_scores))
        best_score = float('inf')
        best_local_goal = None

        for i, (fx, fy, _, _, _) in enumerate(frontier_data):
            score = (weight_goal_dist * normalized_goal_distances[i]) \
                  - (weight_search_dist * normalized_search_distances[i]) \
                  - (weight_angle_score * angle_scores[i])

            if score < best_score:
                best_score = score
                best_local_goal = (fx, fy)

        return best_local_goal

    def publish_local_goal(self, world_frontier):
        if world_frontier is None:
            return

        local_goal_msg = PoseStamped()
        local_goal_msg.header.stamp = rospy.Time.now()
        local_goal_msg.header.frame_id = "aligned_base"

        local_goal_msg.pose.position.x = world_frontier[0]
        local_goal_msg.pose.position.y = world_frontier[1]
        local_goal_msg.pose.position.z = 0.0

        dx = world_frontier[0] - self.odom_position_x
        dy = world_frontier[1] - self.odom_position_y
        yaw = math.atan2(dy, dx)

        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        local_goal_msg.pose.orientation.x = quaternion[0]
        local_goal_msg.pose.orientation.y = quaternion[1]
        local_goal_msg.pose.orientation.z = quaternion[2]
        local_goal_msg.pose.orientation.w = quaternion[3]

        self.local_goal_pub.publish(local_goal_msg)

    def _is_local_goal_stagnated(self, new_local_goal):
        if len(self.local_goal_history) < self.LOCAL_GOAL_HISTORY_SIZE:
            return False

        # Calculate distance between current local goal and all historical local goals
        stagnated = True
        for old_goal in self.local_goal_history:
            dist = np.sqrt((old_goal[0] - new_local_goal[0])**2 + (old_goal[1] - new_local_goal[1])**2)
            if dist > self.LOCAL_GOAL_STAGNATION_THRESHOLD:
                stagnated = False
                break
        return stagnated

    def _is_robot_stagnated(self):
        if len(self.odom_history) < self.ODOM_HISTORY_SIZE:
            return False

        # Calculate total movement over the history
        total_movement = 0.0
        for i in range(1, len(self.odom_history)):
            p1 = self.odom_history[i-1]
            p2 = self.odom_history[i]
            total_movement += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        return total_movement < self.ROBOT_STAGNATION_THRESHOLD

if __name__ == '__main__':
    try:
        detector = FrontierDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
