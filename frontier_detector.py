#!/usr/bin/env python3

import rospy
import numpy as np
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
import math
from nav_msgs.msg import Odometry

class FrontierDetector:
    def __init__(self):
        rospy.init_node('frontier_detector', anonymous=True)

        self.SEARCH_RADIUS = 15.0
        self.ANGLE_RESOLUTION = 0.2
        self.STEP_RESOLUTION = 0.25
        self.COST_THRESHOLD = 0.1
        self.MIN_FRONTIER_DIST = 0.0
        self.SAFETY_RADIUS = 3

        ### intersection 1: (70, 230) - 150s
        ### steep : (150, 150) - 610s
        self.global_goal_x = rospy.get_param('/global_goal_x', 200.0)
        self.global_goal_y = rospy.get_param('/global_goal_y', 0.0)

        self.odom_position_x = 0.0
        self.odom_position_y = 0.0
        self.odom_rotation_yaw = 0.0
        self.odom_roll = 0.0
        self.odom_pitch = 0.0

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

        self.robot_path_pub = rospy.Publisher(
            '/robot_path',
            MarkerArray,
            queue_size=1
        )

        self.global_goal_direction_pub = rospy.Publisher(
            '/global_goal_direction',
            MarkerArray,
            queue_size=1
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.grid_map = None
        self.last_traversability_map = None
        self.last_frontiers = None
        self.last_selected_frontier = None
        self.max_frontier_id = -1  # 이전에 사용된 최대 frontier ID 추적

        # 로봇 경로 저장
        self.robot_path = []  # (x, y) 튜플들의 리스트
        self.path_timer = rospy.Timer(rospy.Duration(1.0), self.save_robot_position)

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

            rospy.loginfo(f"Original GridMap Info:")
            rospy.loginfo(f"  Resolution: {self.resolution}")
            rospy.loginfo(f"  Width: {self.width}, Height: {self.height}")
            rospy.loginfo(f"  Pose: ({msg.info.pose.position.x:.2f}, {msg.info.pose.position.y:.2f})")
            rospy.loginfo(f"  Orientation: ({msg.info.pose.orientation.x:.3f}, {msg.info.pose.orientation.y:.3f}, {msg.info.pose.orientation.z:.3f}, {msg.info.pose.orientation.w:.3f})")
            rospy.loginfo(f"  Frame ID: {msg.info.header.frame_id}")
            rospy.loginfo(f"  Calculated Origin: ({self.origin_x:.2f}, {self.origin_y:.2f})")

            inclination_risk = self.get_layer_data('inclination_risk')
            collision_risk = self.get_layer_data('collision_risk')
            steepness_risk = self.get_layer_data('steepness_risk')

            if inclination_risk is not None and collision_risk is not None and steepness_risk is not None:
                traversability_map = self.compute_cost_map(inclination_risk, collision_risk, steepness_risk)

                if traversability_map is not None:
                    self.last_traversability_map = traversability_map

                    frontiers = self.find_frontiers(traversability_map)

                    if frontiers:
                        transformed_global_goal = self.transform_global_goal_to_local()
                        selected_frontier = self.select_best_frontier(frontiers, transformed_global_goal)

                        if selected_frontier:
                            world_frontier = self.transform_local_to_world(selected_frontier)
                            self.publish_local_goal(world_frontier)

                        # visualization
                        rospy.loginfo(f"GridMap Update - Found {len(frontiers)} frontiers")
                        self.visualize_traversability_map(traversability_map)
                        self.visualize_frontiers(frontiers, selected_frontier)
                        self.visualize_global_goal()
                        self.visualize_global_goal_direction()

                        # 저장 (다음 시각화를 위해)
                        self.last_frontiers = frontiers
                        self.last_selected_frontier = selected_frontier
                    else:
                        # Frontier가 없어도 맵은 시각화하고, 이전 frontier들도 삭제
                        self.visualize_traversability_map(traversability_map)
                        self.visualize_frontiers([], None)  # 빈 리스트로 호출하여 이전 마커들 삭제
                        self.visualize_global_goal()
                        self.visualize_global_goal_direction()
                        self.last_frontiers = None
                        self.last_selected_frontier = None
                else:
                    rospy.logwarn("Cost map computation failed")
            else:
                rospy.logerr("Failed to get risk layer data")

        except Exception as e:
            rospy.logerr(f"Error processing grid map: {e}")

    def odom_callback(self, msg):
        prev_yaw = self.odom_rotation_yaw

        self.odom_position_x = msg.pose.pose.position.x
        self.odom_position_y = msg.pose.pose.position.y

        import tf.transformations
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        self.odom_rotation_yaw = yaw
        self.odom_roll = roll
        self.odom_pitch = pitch
        
        # 로봇 기울기 로그 출력
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        total_tilt = max(abs(roll_deg), abs(pitch_deg))
        
        rospy.loginfo(f"Robot orientation: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}°")
        rospy.loginfo(f"Robot tilt: total={total_tilt:.1f}°")

    def save_robot_position(self, event):
        """1초마다 로봇의 현재 위치를 저장"""
        if self.odom_position_x is not None and self.odom_position_y is not None:
            current_position = (self.odom_position_x, self.odom_position_y)

            # 이전 위치와 너무 가까우면 저장하지 않음 (최소 0.5m 간격)
            if not self.robot_path or self.calculate_distance(self.robot_path[-1], current_position) > 0.5:
                self.robot_path.append(current_position)

                # 경로가 너무 길어지면 오래된 부분 제거 (최대 1000개 포인트)
                if len(self.robot_path) > 1000:
                    self.robot_path = self.robot_path[-1000:]

                # 경로 시각화 업데이트
                self.visualize_robot_path()

    def calculate_distance(self, pos1, pos2):
        """두 위치 간의 거리 계산"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def visualize_robot_path(self):
        """저장된 로봇 경로를 시각화"""
        if len(self.robot_path) < 2:
            return

        marker_array = MarkerArray()

        # 기존 경로 마커들 삭제
        for i in range(len(self.robot_path)):
            delete_marker = Marker()
            delete_marker.header.frame_id = "aligned_basis"
            delete_marker.header.stamp = rospy.Time.now()
            delete_marker.ns = "robot_path"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # 경로를 선으로 연결하는 마커 생성
        line_marker = Marker()
        line_marker.header.frame_id = "aligned_basis"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "robot_path"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD

        # 경로 포인트들을 Point 리스트로 변환
        for x, y in self.robot_path:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1  # 지면에서 약간 위에 표시
            line_marker.points.append(point)

        # 선 스타일 설정
        line_marker.scale.x = 0.1  # 선 두께
        line_marker.color = ColorRGBA()
        line_marker.color.r = 0.0  # 파란색
        line_marker.color.g = 0.0
        line_marker.color.b = 1.0
        line_marker.color.a = 0.8  # 약간 투명

        marker_array.markers.append(line_marker)

        # 현재 위치를 특별한 마커로 표시
        if self.robot_path:
            current_marker = Marker()
            current_marker.header.frame_id = "aligned_basis"
            current_marker.header.stamp = rospy.Time.now()
            current_marker.ns = "robot_path"
            current_marker.id = 1
            current_marker.type = Marker.SPHERE
            current_marker.action = Marker.ADD

            current_x, current_y = self.robot_path[-1]
            current_marker.pose.position.x = current_x
            current_marker.pose.position.y = current_y
            current_marker.pose.position.z = 0.2

            current_marker.scale.x = 0.3
            current_marker.scale.y = 0.3
            current_marker.scale.z = 0.3

            current_marker.color = ColorRGBA()
            current_marker.color.r = 1.0  # 빨간색 (현재 위치)
            current_marker.color.g = 0.0
            current_marker.color.b = 0.0
            current_marker.color.a = 1.0

            marker_array.markers.append(current_marker)

        self.robot_path_pub.publish(marker_array)

    def visualize_global_goal_direction(self):
        """로봇을 중심으로 글로벌 골 방향을 가리키는 화살표 시각화"""
        marker_array = MarkerArray()
        
        # 글로벌 골을 로컬 좌표계로 변환
        transformed_goal = self.transform_global_goal_to_local()
        
        # 화살표 길이 (로봇에서 글로벌 골까지의 거리, 최대 5m)
        goal_distance = math.sqrt(transformed_goal[0]**2 + transformed_goal[1]**2)
        arrow_length = min(goal_distance, 5.0)  # 최대 5m로 제한
        
        # 화살표 마커 생성
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "aligned_basis"
        arrow_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
        arrow_marker.ns = "global_goal_direction"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # 화살표 시작점 (로봇 위치)
        start_point = Point()
        start_point.x = 0.0  # 로봇 중심
        start_point.y = 0.0
        start_point.z = 0.5  # 지면에서 0.5m 위
        
        # 화살표 끝점 (글로벌 골 방향)
        end_point = Point()
        if goal_distance > 0:
            # 정규화된 방향 벡터에 화살표 길이 적용
            end_point.x = (transformed_goal[0] / goal_distance) * arrow_length
            end_point.y = (transformed_goal[1] / goal_distance) * arrow_length
        else:
            end_point.x = 0.0
            end_point.y = 0.0
        end_point.z = 0.5
        
        arrow_marker.points = [start_point, end_point]
        
        # 화살표 스타일 설정
        arrow_marker.scale.x = 0.2  # 화살표 몸통 두께 (0.3 → 0.2)
        arrow_marker.scale.y = 0.3  # 화살표 머리 두께 (0.5 → 0.3)
        arrow_marker.scale.z = 0.0  # 2D 화살표
        
        # 밝은 노란색으로 표시
        arrow_marker.color = ColorRGBA()
        arrow_marker.color.r = 1.0  # 노란색
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 0.8  # 약간 투명
        
        marker_array.markers.append(arrow_marker)
        
        # 글로벌 골 방향 각도 표시를 위한 텍스트 마커
        text_marker = Marker()
        text_marker.header.frame_id = "aligned_basis"
        text_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
        text_marker.ns = "global_goal_direction"
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # 텍스트 위치 (화살표 중간 지점)
        text_marker.pose.position.x = end_point.x / 2
        text_marker.pose.position.y = end_point.y / 2
        text_marker.pose.position.z = 1.0  # 화살표 위에 표시
        
        # 각도 계산 및 표시
        goal_angle = math.atan2(transformed_goal[1], transformed_goal[0])
        text_marker.text = f"Goal: {math.degrees(goal_angle):.0f}°"
        
        # 텍스트 스타일
        text_marker.scale.z = 0.2  # 텍스트 크기 (0.3 → 0.2)
        text_marker.color = ColorRGBA()
        text_marker.color.r = 1.0  # 흰색
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        
        marker_array.markers.append(text_marker)
        
        self.global_goal_direction_pub.publish(marker_array)

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

            valid_count = np.sum(~np.isnan(data))
            total_count = data.size

            return data

        except Exception as e:
            rospy.logerr(f"Error getting layer data for {layer_name}: {e}")
            return None

    def compute_cost_map(self, incl, coll, steep):
        """Frontier 탐색에 특화된 traversability map 생성 (적응적 임계값 사용)"""
        if incl is None or coll is None or steep is None:
            return None

        # 로봇의 기울기에 따른 적응적 임계값 계산
        collision_threshold, steepness_threshold, inclination_threshold = self.get_adaptive_thresholds()

        # Frontier 탐색을 위한 traversability map 생성
        traversability_map = np.full_like(incl, np.nan)

        # 각 셀별로 주행 가능성 판단
        for i in range(incl.shape[0]):
            for j in range(incl.shape[1]):
                incl_val = incl[i, j]
                coll_val = coll[i, j]
                steep_val = steep[i, j]

                # 하나라도 NaN이면 미탐사 영역 (NaN 유지)
                if np.isnan(incl_val) or np.isnan(coll_val) or np.isnan(steep_val):
                    traversability_map[i, j] = np.nan  # 미탐사 영역
                else:
                    # 적응적 임계값을 사용한 traversability 계산
                    # 1. 충돌 위험이 높으면 주행 불가
                    if coll_val > collision_threshold:
                        traversability_map[i, j] = 1.0  # 주행 불가
                    # 2. 경사가 너무 심하면 주행 불가
                    elif steep_val > steepness_threshold or incl_val > inclination_threshold:
                        traversability_map[i, j] = 1.0  # 주행 불가
                    # 3. 약간의 위험이 있지만 주행 가능 (임계값의 절반 사용)
                    elif (coll_val > collision_threshold * 0.5 or 
                          steep_val > steepness_threshold * 0.5 or 
                          incl_val > inclination_threshold * 0.5):
                        traversability_map[i, j] = 0.5  # 조심스럽게 주행 가능
                    # 4. 안전한 영역
                    else:
                        traversability_map[i, j] = 0.0  # 안전하게 주행 가능

        # 통계 출력
        safe_cells = np.sum(traversability_map == 0.0)
        caution_cells = np.sum(traversability_map == 0.5)
        blocked_cells = np.sum(traversability_map == 1.0)
        unknown_cells = np.sum(np.isnan(traversability_map))

        rospy.loginfo(f"Traversability Map - Safe: {safe_cells}, Caution: {caution_cells}, "
                     f"Blocked: {blocked_cells}, Unknown: {unknown_cells}")

        return traversability_map

    def get_adaptive_thresholds(self):
        """로봇의 기울기에 따라 적응적 임계값 계산"""
        # 로봇의 현재 기울기 (라디안)
        roll_deg = math.degrees(abs(self.odom_roll))
        pitch_deg = math.degrees(abs(self.odom_pitch))
        
        # 전체 기울기 (최대값 사용)
        total_tilt = max(roll_deg, pitch_deg)
        
        # 기본 임계값들
        base_collision_threshold = 0.8
        base_steepness_threshold = 0.7
        base_inclination_threshold = 0.7
        
        # 기울기에 따른 임계값 조정 (갈 수 있는 영역 확대)
        if total_tilt < 3.0:  # 평지 (3도 미만)
            # 기본 임계값 (평지에서는 정상적인 탐색)
            collision_threshold = base_collision_threshold
            steepness_threshold = base_steepness_threshold
            inclination_threshold = base_inclination_threshold
        elif total_tilt < 8.0:  # 약간 기울어진 지형 (3-8도)
            # 기울어진 상태: 갈 수 있는 영역 확대 (임계값 낮춤)
            collision_threshold = base_collision_threshold - 0.1
            steepness_threshold = base_steepness_threshold - 0.2  # 경사 임계값 낮춤
            inclination_threshold = base_inclination_threshold - 0.2  # 경사도 임계값 낮춤
        elif total_tilt < 15.0:  # 기울어진 지형 (8-15도)
            # 기울어진 상태: 갈 수 있는 영역 더 확대
            collision_threshold = base_collision_threshold - 0.2
            steepness_threshold = base_steepness_threshold - 0.4  # 경사 임계값 더 낮춤
            inclination_threshold = base_inclination_threshold - 0.4  # 경사도 임계값 더 낮춤
        elif total_tilt < 25.0:  # 매우 기울어진 지형 (15-25도)
            # 기울어진 상태: 갈 수 있는 영역 매우 확대
            collision_threshold = base_collision_threshold - 0.3
            steepness_threshold = base_steepness_threshold - 0.6  # 경사 임계값 매우 낮춤
            inclination_threshold = base_inclination_threshold - 0.6  # 경사도 임계값 매우 낮춤
        else:  # 극한 지형 (25도 이상)
            # 기울어진 상태: 갈 수 있는 영역 극한 확대
            collision_threshold = base_collision_threshold - 0.4
            steepness_threshold = base_steepness_threshold - 0.8  # 경사 임계값 극한 낮춤
            inclination_threshold = base_inclination_threshold - 0.8  # 경사도 임계값 극한 낮춤
        
        # 임계값 범위 제한 (0.05 ~ 1.0) - 더 관대한 범위
        collision_threshold = max(0.05, min(1.0, collision_threshold))
        steepness_threshold = max(0.05, min(1.0, steepness_threshold))
        inclination_threshold = max(0.05, min(1.0, inclination_threshold))
        
        rospy.loginfo(f"Robot tilt: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, total={total_tilt:.1f}°")
        rospy.loginfo(f"Adaptive thresholds: collision={collision_threshold:.2f}, "
                     f"steepness={steepness_threshold:.2f}, inclination={inclination_threshold:.2f}")
        
        return collision_threshold, steepness_threshold, inclination_threshold

    def find_frontiers(self, traversability_map):
        """Traversability map에서 프런티어 포인트 탐색"""
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
                # 원본 GridMap의 pose 기준으로 좌표 계산
                # GridMap의 pose는 맵의 중심점이므로, 셀 좌표를 world 좌표로 변환
                # 좌우, 위아래 모두 반전을 위해 x축과 y축 모두 뒤집음
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

    def transform_global_goal_to_local(self):
        dx = self.global_goal_x - self.odom_position_x
        dy = self.global_goal_y - self.odom_position_y
        
        rospy.loginfo(f"Global goal: ({self.global_goal_x}, {self.global_goal_y})")
        rospy.loginfo(f"Robot position: ({self.odom_position_x}, {self.odom_position_y})")
        rospy.loginfo(f"Delta: ({dx}, {dy})")
        rospy.loginfo(f"Robot yaw: {self.odom_rotation_yaw}")
        
        cos_yaw = math.cos(-self.odom_rotation_yaw)
        sin_yaw = math.sin(-self.odom_rotation_yaw)
        
        local_goal_x = dx * cos_yaw - dy * sin_yaw
        local_goal_y = dx * sin_yaw + dy * cos_yaw
        
        rospy.loginfo(f"Transformed goal: ({local_goal_x:.2f}, {local_goal_y:.2f})")
        
        return (local_goal_x, local_goal_y)

    def select_best_frontier(self, frontiers, transformed_global_goal):
        if not frontiers:
            return None

        best_frontier = None
        min_distance = float('inf')

        for frontier in frontiers:
            local_frontier = self.transform_world_to_local(frontier)

            dist = math.sqrt((local_frontier[0] - transformed_global_goal[0])**2 + (local_frontier[1] - transformed_global_goal[1])**2)

            if dist < min_distance:
                min_distance = dist
                best_frontier = frontier

        return best_frontier

    def transform_world_to_local(self, world_point):
        dx = world_point[0] - self.odom_position_x
        dy = world_point[1] - self.odom_position_y
        cos_yaw = math.cos(-self.odom_rotation_yaw)
        sin_yaw = math.sin(-self.odom_rotation_yaw)

        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        return (local_x, local_y)

    def transform_local_to_world(self, local_point):
        cos_yaw = math.cos(-self.odom_rotation_yaw)
        sin_yaw = math.sin(-self.odom_rotation_yaw)

        world_x = self.odom_position_x + (local_point[0] * cos_yaw - local_point[1] * sin_yaw)
        world_y = self.odom_position_y + (local_point[0] * sin_yaw + local_point[1] * cos_yaw)

        return (world_x, world_y)

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

        import tf.transformations
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        local_goal_msg.pose.orientation.x = quaternion[0]
        local_goal_msg.pose.orientation.y = quaternion[1]
        local_goal_msg.pose.orientation.z = quaternion[2]
        local_goal_msg.pose.orientation.w = quaternion[3]

        #rospy.set_param('/global_goal_x', world_frontier[0])
        #rospy.set_param('/global_goal_y', world_frontier[1])
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

    def visualize_frontiers(self, frontiers, selected_frontier=None):
        """프런티어 포인트들을 MarkerArray로 시각화 (선택된 frontier는 다른 색으로)"""
        marker_array = MarkerArray()

        # 이전에 사용된 모든 마커들을 먼저 삭제 (잔상 방지)
        for i in range(self.max_frontier_id + 1):
            delete_marker = Marker()
            delete_marker.header.frame_id = "aligned_basis"
            delete_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
            delete_marker.ns = "frontiers"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # 현재 최대 ID 업데이트
        if frontiers:
            self.max_frontier_id = len(frontiers) - 1
        else:
            # frontier가 없으면 이전 마커들만 삭제하고 종료
            self.frontier_viz_pub.publish(marker_array)
            return

        for i, (x, y) in enumerate(frontiers):
            marker = Marker()
            marker.header.frame_id = "aligned_basis"  # 또는 적절한 frame_id
            # 원본 GridMap과 동일한 타임스탬프 사용 (동기화)
            marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 1.5

            # Orientation 명시적으로 설정 (quaternion 초기화)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # 선택된 frontier는 더 크게 표시
            if selected_frontier and (x, y) == selected_frontier:
                marker.scale.x = 0.8  # 0.5에서 0.8로 증가
                marker.scale.y = 0.8
                marker.scale.z = 0.8
                # 선택된 frontier는 밝은 녹색으로 표시
                marker.color = ColorRGBA()
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                marker.scale.x = 0.5  # 0.3에서 0.5로 증가
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                # 일반 frontier는 더 잘 보이는 주황색으로 표시
                marker.color = ColorRGBA()
                marker.color.r = 1.0  # 밝은 주황색
                marker.color.g = 0.65
                marker.color.b = 0.0
                marker.color.a = 1.0  # 완전 불투명으로 변경

            # lifetime 설정 없음 - 지속적으로 표시

            selected_frontiers = self.transform_local_to_world(selected_frontier)

            rospy.set_param('/global_goal_x', selected_frontier[0])
            rospy.set_param('/global_goal_y', selected_frontier[1])


            marker_array.markers.append(marker)

        self.frontier_viz_pub.publish(marker_array)

    def visualize_global_goal(self):
        """글로벌 골을 시각화 (프론티어보다 크게)"""
        marker_array = MarkerArray()

        # Global goal을 로컬 좌표계로 변환
        transformed_goal = self.transform_global_goal_to_local()

        # 글로벌 골 마커 생성
        goal_marker = Marker()
        goal_marker.header.frame_id = "aligned_basis"
        goal_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
        goal_marker.ns = "global_goal"
        goal_marker.id = 0
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD

        goal_marker.pose.position.x = transformed_goal[0]
        goal_marker.pose.position.y = transformed_goal[1]
        goal_marker.pose.position.z = 2.0  # 더 높게 표시

        # Orientation 명시적으로 설정 (quaternion 초기화)
        goal_marker.pose.orientation.x = 0.0
        goal_marker.pose.orientation.y = 0.0
        goal_marker.pose.orientation.z = 0.0
        goal_marker.pose.orientation.w = 1.0

        # 프론티어보다 크게 설정
        goal_marker.scale.x = 1.5  # 1.0에서 1.5로 증가 (프론티어는 0.5-0.8)
        goal_marker.scale.y = 1.5
        goal_marker.scale.z = 1.5

        # 밝은 빨간색으로 표시
        goal_marker.color = ColorRGBA()
        goal_marker.color.r = 1.0  # 빨간색
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 1.0

        marker_array.markers.append(goal_marker)
        self.global_goal_viz_pub.publish(marker_array)

    def visualize_traversability_map(self, traversability_map):
        """Traversability map을 GridMap으로 시각화"""
        from std_msgs.msg import MultiArrayDimension, MultiArrayLayout, Float32MultiArray

        frontier_map = GridMap()

        frontier_map.info = self.grid_map.info
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

        self.traversability_viz_pub.publish(frontier_map)

        original_stamp = self.grid_map.info.header.stamp
        published_stamp = frontier_map.info.header.stamp
        time_diff = abs((original_stamp - published_stamp).to_sec()) if original_stamp != published_stamp else 0.0

        if time_diff > 0.001:
            rospy.logwarn(f"Timestamp sync issue: {time_diff:.3f}s difference")

if __name__ == '__main__':
    try:
        detector = FrontierDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
