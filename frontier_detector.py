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
from utils import (
    global_to_local,
    transform_local_to_world,
    visualize_goal_projection,
    visualize_frontiers,
    visualize_global_goal,
    visualize_cost_map
)

import scipy.ndimage

class FrontierDetector:
    def __init__(self):
        rospy.init_node('frontier_detector', anonymous=True)

        self.SEARCH_RADIUS = 15.0
        self.ANGLE_RESOLUTION = 0.2
        self.STEP_RESOLUTION = 0.25
        self.COST_THRESHOLD = 0.7
        self.MIN_FRONTIER_DIST = 0.0
        self.SAFETY_RADIUS = 3

        self.global_goal_x = rospy.get_param('/global_goal_x', 200.0)
        self.global_goal_y = rospy.get_param('/global_goal_y', 0.0)

        self.odom_position_x = 0.0
        self.odom_position_y = 0.0
        self.odom_rotation_yaw = 0.0

        self.grid_map_sub = rospy.Subscriber('/trip/trip_updated/terrain_local_gridmap', GridMap, self.grid_map_callback)
        self.odom_sub = rospy.Subscriber('/global/odometry', Odometry, self.odom_callback)
        self.global_goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.global_goal_callback)

        self.frontier_viz_pub = rospy.Publisher('/frontier_visualization', MarkerArray, queue_size=1)
        self.global_goal_viz_pub = rospy.Publisher('/global_goal_visualization', MarkerArray, queue_size=1)

        self.local_goal_pub = rospy.Publisher('/local_goal', PoseStamped, queue_size=1)
        self.goal_projection_pub = rospy.Publisher('/goal_projection_visualization', Marker, queue_size=1, latch=True)
        self.cost_map_viz_pub = rospy.Publisher('/frontier_map_visualization', GridMap, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.grid_map = None
        self.max_frontier_id = [-1]

        self.resolution = None
        self.width = None
        self.height = None
        self.origin_x = None
        self.origin_y = None

    def grid_map_callback(self, msg):
        try:
            self.grid_map = msg
            self.resolution = msg.info.resolution
            self.width = msg.info.length_x
            self.height = msg.info.length_y

            self.origin_x = msg.info.pose.position.x - self.width/2
            self.origin_y = msg.info.pose.position.y - self.height/2

            inclination_risk = self.get_layer_data('inclination_risk')
            collision_risk = self.get_layer_data('collision_risk')
            steepness_risk = self.get_layer_data('steepness_risk')

            if inclination_risk is not None and collision_risk is not None and steepness_risk is not None:
                traversability_map = self.compute_cost_map(inclination_risk, collision_risk, steepness_risk)

                if traversability_map is not None:
                    transformed_global_goal = global_to_local(
                        self.global_goal_x, self.global_goal_y, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw
                    )
                    ### frontier candidates ###
                    frontiers = self.find_frontiers(traversability_map)

                    if frontiers:
                        local_goal = self.select_local_goal(frontiers, transformed_global_goal)

                        if local_goal:
                            world_frontier = transform_local_to_world(
                                local_goal, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw
                            )
                            self.publish_local_goal(world_frontier)

                        rospy.loginfo(f"GridMap Update - Found {len(frontiers)} frontiers")
                        visualize_frontiers(frontiers, local_goal, self.grid_map.info, self.frontier_viz_pub, self.max_frontier_id)
                        visualize_global_goal(self.global_goal_x, self.global_goal_y, self.grid_map.info, self.global_goal_viz_pub, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw)
                        visualize_goal_projection(transformed_global_goal, self.grid_map.info, self.goal_projection_pub)
                        visualize_cost_map(traversability_map, self.grid_map.info, self.cost_map_viz_pub)
                    else:
                        visualize_frontiers([], None, self.grid_map.info, self.frontier_viz_pub, self.max_frontier_id)
                        visualize_global_goal(self.global_goal_x, self.global_goal_y, self.grid_map.info, self.global_goal_viz_pub, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw)
                else:
                    rospy.logwarn("Cost map computation failed")
            else:
                rospy.logerr("Failed to get risk layer data")

        except Exception as e:
            rospy.logerr(f"Error processing grid map: {e}")

    def odom_callback(self, msg):
        self.odom_position_x = msg.pose.pose.position.x
        self.odom_position_y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        self.odom_rotation_yaw = yaw

        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        total_tilt = max(abs(roll_deg), abs(pitch_deg))

        rospy.loginfo(f"Robot orientation: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}°")
        rospy.loginfo(f"Robot tilt: total={total_tilt:.1f}°")

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

    def compute_cost_map(self, incl, coll, steep):
        if incl is None or coll is None or steep is None:
            return None

        traversability_map = np.full_like(incl, np.nan)

        center_x = int(self.width / (2 * self.resolution))
        center_y = int(self.height / (2 * self.resolution))

        # Step 1: Compute initial risk cost
        initial_cost_map = np.full_like(incl, np.nan)
        for i in range(incl.shape[0]):
            for j in range(incl.shape[1]):
                incl_val = incl[i, j]
                coll_val = coll[i, j]
                steep_val = steep[i, j]

                if np.isnan(incl_val) or np.isnan(coll_val) or np.isnan(steep_val):
                    risk_cost = np.nan
                else:
                    # Aggressive level 1: Max
                    # Aggressive level 2: Top 2 average
                    # Aggressive level 3: Average
                    aggressive_level = 3

                    risks = sorted([incl_val, coll_val, steep_val], reverse=True)
                    if aggressive_level == 1:
                        risk_cost = risks[0]
                    elif aggressive_level == 2:
                        risk_cost = (risks[0] + risks[1]) / 2.0
                    else:
                        risk_cost = (risks[0] + risks[1] + risks[2]) / 3.0
                initial_cost_map[i, j] = risk_cost

        # Step 2: Process Cost Map
        robot_radius = self.SAFETY_RADIUS * self.resolution
        geometric_cost_threshold = self.COST_THRESHOLD
        geometric_dilation_alpha = 1.4
        geometric_gaussian_sigma = 0.5
        geometric_cost_amplification = 1.2

        # 2.1. Binarize based on threshold
        binary_image = np.zeros_like(initial_cost_map, dtype=np.uint8)
        binary_image[initial_cost_map >= geometric_cost_threshold] = 255

        # 2.2. Median Blur for denoising
        denoised_image = scipy.ndimage.median_filter(binary_image, size=3)

        # 2.3. Hole Filling
        hole_filled = scipy.ndimage.binary_fill_holes(denoised_image).astype(np.uint8) * 255

        # 2.4. Dilation
        dilation_radius_cells = int(round(robot_radius * geometric_dilation_alpha / self.resolution))
        if dilation_radius_cells > 0:
            y, x = np.ogrid[-dilation_radius_cells:dilation_radius_cells+1, -dilation_radius_cells:dilation_radius_cells+1]
            structuring_element = (x**2 + y**2 <= dilation_radius_cells**2).astype(np.uint8)
            dilated_image = scipy.ndimage.binary_dilation(hole_filled, structure=structuring_element).astype(np.uint8) * 255
        else:
            dilated_image = hole_filled.copy()

        dilated_mask = (dilated_image == 255)
        processed_cost_map = initial_cost_map.copy()
        processed_cost_map[dilated_mask] = 1.0

        # 2.5. Gaussian Blur
        gaussian_sigma_pixels = robot_radius * geometric_gaussian_sigma / self.resolution
        if gaussian_sigma_pixels < 1.0:
            gaussian_sigma_pixels = 1.0

        gaussian_image = scipy.ndimage.gaussian_filter(processed_cost_map, sigma=gaussian_sigma_pixels, mode='nearest')
        gaussian_image[dilated_mask] = 1.0

        # Step 3: Generate Final Cost Map
        final_cost_map = gaussian_image.copy()

        temp_min_cost = np.nanmax(final_cost_map)
        temp_max_cost = np.nanmin(final_cost_map)

        for i in range(final_cost_map.shape[0]):
            for j in range(final_cost_map.shape[1]):
                if not np.isnan(initial_cost_map[i, j]):
                    cost_val = final_cost_map[i, j]
                    if initial_cost_map[i, j] >= geometric_cost_threshold:
                        cost_val = 1.0
                    else:
                        cost_val *= geometric_cost_amplification

                    if cost_val > 1.0:
                        cost_val = 1.0

                    current_cell_dist_from_center = np.sqrt(((j - center_x) * self.resolution)**2 + ((i - center_y) * self.resolution)**2)
                    if current_cell_dist_from_center <= robot_radius:
                        cost_val = 0.0

                    final_cost_map[i, j] = cost_val
                    if not np.isnan(cost_val):
                        temp_min_cost = min(temp_min_cost, cost_val)
                        temp_max_cost = max(temp_max_cost, cost_val)

        min_cost = temp_min_cost
        max_cost = temp_max_cost

        # Min-Max Normalization
        cost_range = max_cost - min_cost
        if cost_range > 0 and not np.isinf(cost_range):
            normalized_cost_map = (final_cost_map - min_cost) / cost_range

            for i in range(normalized_cost_map.shape[0]):
                for j in range(normalized_cost_map.shape[1]):
                    current_cell_dist_from_center = np.sqrt(((j - center_x) * self.resolution)**2 + ((i - center_y) * self.resolution)**2)
                    if current_cell_dist_from_center <= robot_radius:
                        normalized_cost_map[i, j] = 0.0
            traversability_map = normalized_cost_map
        else:
            traversability_map = final_cost_map

        return traversability_map

    def find_frontiers(self, traversability_map):
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

        for angle in np.arange(0, 2*np.pi, self.ANGLE_RESOLUTION):
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
                elif current_value >= self.COST_THRESHOLD: # Changed from 1.0 to self.COST_THRESHOLD (e.g., 0.7)
                    rays_hit_blocked += 1
                    break
                elif current_value < self.COST_THRESHOLD: # Changed from <= 0.5 to < self.COST_THRESHOLD
                    farthest_traversable_point = (x, y)

            if farthest_traversable_point is not None:
                cell_x, cell_y = farthest_traversable_point
                world_x = self.grid_map.info.pose.position.x - (cell_x - center_x) * self.resolution
                world_y = self.grid_map.info.pose.position.y - (cell_y - center_y) * self.resolution

                dist_from_robot = np.sqrt(world_x**2 + world_y**2)

                traversability_value = traversability_map[cell_y, cell_x]

                if (dist_from_robot > self.MIN_FRONTIER_DIST and
                    self.check_min_distance(frontiers, (world_x, world_y))):
                    frontiers.append((world_x, world_y))

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

    def select_local_goal(self, frontiers, transformed_global_goal):
        if not frontiers:
            return None

        local_goal = None
        min_distance = float('inf')

        goal_x = transformed_global_goal[0]
        goal_y = transformed_global_goal[1]

        for frontier in frontiers:
            local_frontier = frontier

            dist = math.sqrt((local_frontier[0] - goal_x)**2 + (local_frontier[1] - goal_y)**2)

            if dist < min_distance:
                min_distance = dist
                local_goal = frontier

        return local_goal

    def publish_local_goal(self, world_frontier):
        if world_frontier is None:
            return

        local_goal_msg = PoseStamped()
        local_goal_msg.header.stamp = rospy.Time.now()
        local_goal_msg.header.frame_id = "aligned_basis"

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

    def check_min_distance(self, frontiers, new_point):
        for frontier in frontiers:
            dist = np.sqrt((frontier[0] - new_point[0])**2 + (frontier[1] - new_point[1])**2)
            if dist < self.MIN_FRONTIER_DIST:
                return False
        return True

if __name__ == '__main__':
    try:
        detector = FrontierDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
