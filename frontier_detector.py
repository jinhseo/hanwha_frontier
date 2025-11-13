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
    transform_global_goal_to_local,
    transform_local_to_world,
    visualize_goal_projection,
    visualize_frontiers,
    visualize_global_goal,
    visualize_traversability_map
)

class FrontierDetector:
    def __init__(self):
        rospy.init_node('frontier_detector', anonymous=True)

        self.SEARCH_RADIUS = 15.0
        self.ANGLE_RESOLUTION = 0.2
        self.STEP_RESOLUTION = 0.25
        self.COST_THRESHOLD = 0.1
        self.MIN_FRONTIER_DIST = 0.0
        self.SAFETY_RADIUS = 3

        self.global_goal_x = rospy.get_param('/global_goal_x', 200.0)
        self.global_goal_y = rospy.get_param('/global_goal_y', 0.0)

        self.odom_position_x = 0.0
        self.odom_position_y = 0.0
        self.odom_rotation_yaw = 0.0

        self.grid_map_sub = rospy.Subscriber(
            '/trip/trip_updated/terrain_local_gridmap',
            GridMap,
            self.grid_map_callback
        )

        self.odom_sub = rospy.Subscriber(
            '/global/odometry',
            Odometry,
            self.odom_callback
        )

        self.frontier_viz_pub = rospy.Publisher(
            '/frontier_visualization',
            MarkerArray,
            queue_size=1
        )

        self.traversability_viz_pub = rospy.Publisher(
            '/frontier_map',
            GridMap,
            queue_size=1
        )

        self.global_goal_viz_pub = rospy.Publisher(
            '/global_goal_visualization',
            MarkerArray,
            queue_size=1
        )

        self.local_goal_pub = rospy.Publisher(
            '/local_goal',
            PoseStamped,
            queue_size=1
        )

        self.global_goal_sub = rospy.Subscriber(
            '/move_base_simple/goal',
            PoseStamped,
            self.global_goal_callback
        )
        self.goal_projection_pub = rospy.Publisher('/goal_projection_visualization', Marker, queue_size=1, latch=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.grid_map = None
        self.last_traversability_map = None
        self.last_frontiers = None
        self.last_selected_frontier = None
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
                    self.last_traversability_map = traversability_map

                    transformed_global_goal = transform_global_goal_to_local(
                        self.global_goal_x, self.global_goal_y, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw
                    )
                    visualize_goal_projection(transformed_global_goal, self.grid_map.info, self.goal_projection_pub)
                    frontiers = self.find_frontiers(traversability_map)

                    if frontiers:
                        transformed_global_goal = transform_global_goal_to_local(
                            self.global_goal_x, self.global_goal_y, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw
                        )
                        selected_frontier = self.select_best_frontier(frontiers, transformed_global_goal)

                        if selected_frontier:
                            world_frontier = transform_local_to_world(
                                selected_frontier, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw
                            )
                            self.publish_local_goal(world_frontier)

                        rospy.loginfo(f"GridMap Update - Found {len(frontiers)} frontiers")
                        visualize_traversability_map(traversability_map, self.grid_map, self.traversability_viz_pub)
                        visualize_frontiers(frontiers, selected_frontier, self.grid_map.info, self.frontier_viz_pub, self.max_frontier_id)
                        visualize_global_goal(self.global_goal_x, self.global_goal_y, self.grid_map.info, self.global_goal_viz_pub, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw)

                        self.last_frontiers = frontiers
                        self.last_selected_frontier = selected_frontier
                    else:
                        visualize_traversability_map(traversability_map, self.grid_map, self.traversability_viz_pub)
                        visualize_frontiers([], None, self.grid_map.info, self.frontier_viz_pub, self.max_frontier_id)
                        visualize_global_goal(self.global_goal_x, self.global_goal_y, self.grid_map.info, self.global_goal_viz_pub, self.odom_position_x, self.odom_position_y, self.odom_rotation_yaw)
                        self.last_frontiers = None
                        self.last_selected_frontier = None
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

            rospy.loginfo(f"✅ [SUCCESS] New global goal set in {target_frame}: "
                          f"({self.global_goal_x:.2f}, {self.global_goal_y:.2f})")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"❌ [FAIL] Failed to transform global goal from {source_frame} to {target_frame}: {e}")
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

        collision_threshold = 0.8
        steepness_threshold = 0.7
        inclination_threshold = 0.7

        traversability_map = np.full_like(incl, np.nan)

        for i in range(incl.shape[0]):
            for j in range(incl.shape[1]):
                incl_val = incl[i, j]
                coll_val = coll[i, j]
                steep_val = steep[i, j]

                if np.isnan(incl_val) or np.isnan(coll_val) or np.isnan(steep_val):
                    traversability_map[i, j] = np.nan
                else:
                    if coll_val > collision_threshold:
                        traversability_map[i, j] = 1.0
                    elif steep_val > steepness_threshold or incl_val > inclination_threshold:
                        traversability_map[i, j] = 1.0
                    elif (coll_val > collision_threshold * 0.5 or
                          steep_val > steepness_threshold * 0.5 or
                          incl_val > inclination_threshold * 0.5):
                        traversability_map[i, j] = 0.5
                    else:
                        traversability_map[i, j] = 0.0

        safe_cells = np.sum(traversability_map == 0.0)
        caution_cells = np.sum(traversability_map == 0.5)
        blocked_cells = np.sum(traversability_map == 1.0)
        unknown_cells = np.sum(np.isnan(traversability_map))

        rospy.loginfo(f"Traversability Map - Safe: {safe_cells}, Caution: {caution_cells}, "
                     f"Blocked: {blocked_cells}, Unknown: {unknown_cells}")
        rospy.loginfo(f"Fixed thresholds: collision={collision_threshold:.2f}, "
                     f"steepness={steepness_threshold:.2f}, inclination={inclination_threshold:.2f}")

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
                elif current_value >= 1.0:
                    rays_hit_blocked += 1
                    break
                elif current_value <= 0.5:
                    if self.is_safe_area(traversability_map, x, y):
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

    def select_best_frontier(self, frontiers, transformed_global_goal):
        if not frontiers:
            return None

        best_frontier = None
        min_distance = float('inf')

        goal_x = transformed_global_goal[0]
        goal_y = transformed_global_goal[1]

        for frontier in frontiers:
            local_frontier = frontier

            dist = math.sqrt((local_frontier[0] - goal_x)**2 + (local_frontier[1] - goal_y)**2)

            if dist < min_distance:
                min_distance = dist
                best_frontier = frontier

        return best_frontier

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

    def is_safe_area(self, traversability_map, x, y):
        height, width = traversability_map.shape

        safe_cells = 0
        total_cells = 0

        for dx in range(-self.SAFETY_RADIUS, self.SAFETY_RADIUS + 1):
            for dy in range(-self.SAFETY_RADIUS, self.SAFETY_RADIUS + 1):
                check_x = x + dx
                check_y = y + dy

                if 0 <= check_x < width and 0 <= check_y < height:
                    total_cells += 1
                    cell_value = traversability_map[check_y, check_x]

                    if not np.isnan(cell_value) and cell_value <= 0.5:
                        safe_cells += 1

        if total_cells > 0:
            safety_ratio = safe_cells / total_cells
            return safety_ratio >= 0.7

        return False

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
