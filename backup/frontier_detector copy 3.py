#!/usr/bin/env python3

import rospy
import numpy as np
from grid_map_msgs.msg import GridMap
import tf
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
import math
from nav_msgs.msg import Odometry
from dynamic_reconfigure.server import Server
from planning_module.cfg import HyperparametersConfig
import tf.transformations as tft

class FrontierDetector:
    def __init__(self):
        rospy.init_node('frontier_detector', anonymous=True)

        # 기본값은 ROS param 또는 하드코딩 → 곧 DR이 덮어씀
        self.SEARCH_RADIUS = rospy.get_param("~SEARCH_RADIUS", 15.0)
        self.ANGLE_RESOLUTION = rospy.get_param("~ANGLE_RESOLUTION", 0.2)
        self.STEP_RESOLUTION = rospy.get_param("~STEP_RESOLUTION", 0.25)
        self.COST_THRESHOLD = rospy.get_param("~COST_THRESHOLD", 0.1)
        self.MIN_FRONTIER_DIST = rospy.get_param("~MIN_FRONTIER_DIST", 0.0)
        self.SAFETY_RADIUS = rospy.get_param("~SAFETY_RADIUS", 3)

        self.ADAPT_ENABLE = rospy.get_param("~ADAPT_ENABLE", True)
        self.BASE_COLLISION = rospy.get_param("~BASE_COLLISION", 0.8)
        self.BASE_STEEPNESS = rospy.get_param("~BASE_STEEPNESS", 0.7)
        self.BASE_INCLINATION = rospy.get_param("~BASE_INCLINATION", 0.7)

        self.HEADING_BIAS = 0.6
        self.GOAL_BIAS = 0.6
        self.BIAS_SPREAD = math.radians(30.0)
        self.BIAS_MULT = 2
        self.MIN_ANG_SEP = math.radians(8.0)
        self.NMS_RADIUS = 1.0
        self.SECTOR_K = 4
        self.USE_BOUNDARY_FRONTIER = True

        self.ARROW_MAX_LEN = rospy.get_param("~ARROW_MAX_LEN", 5.0)

        # 경로 저장 주기: Timer 재생성할 수 있게 보관
        self.PATH_UPDATE_HZ = rospy.get_param("~PATH_UPDATE_HZ", 1.0)
        self.path_timer = rospy.Timer(rospy.Duration(1.0/self.PATH_UPDATE_HZ), self.save_robot_position)

        # self.SEARCH_RADIUS = 15.0
        # self.ANGLE_RESOLUTION = 0.2
        # self.STEP_RESOLUTION = 0.25
        # self.COST_THRESHOLD = 0.1
        # self.MIN_FRONTIER_DIST = 0.0
        # self.SAFETY_RADIUS = 3

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

        # self.frontier_viz_pub = rospy.Publisher(
        #     '/frontier_visualization',
        #     MarkerArray,
        #     queue_size=1
        # )

        # self.traversability_viz_pub = rospy.Publisher(
        #     '/frontier_map',
        #     GridMap,
        #     queue_size=1
        # )

        # self.global_goal_viz_pub = rospy.Publisher(
        #     '/global_goal_visualization',
        #     MarkerArray,
        #     queue_size=1
        # )

        # self.local_goal_pub = rospy.Publisher(
        #     '/local_goal',
        #     PoseStamped,
        #     queue_size=1
        # )

        # self.robot_path_pub = rospy.Publisher(
        #     '/robot_path',
        #     MarkerArray,
        #     queue_size=1
        # )

        # self.global_goal_direction_pub = rospy.Publisher(
        #     '/global_goal_direction',
        #     MarkerArray,
        #     queue_size=1
        # )

        self.frontier_viz_pub = rospy.Publisher('/frontier_visualization', MarkerArray, queue_size=1, latch=True)
        self.traversability_viz_pub = rospy.Publisher('/frontier_map', GridMap, queue_size=1, latch=True)
        self.global_goal_viz_pub = rospy.Publisher('/global_goal_visualization', MarkerArray, queue_size=1, latch=True)
        self.local_goal_pub = rospy.Publisher('/local_goal', PoseStamped, queue_size=1, latch=True)
        self.robot_path_pub = rospy.Publisher('/robot_path', MarkerArray, queue_size=1, latch=True)
        self.global_goal_direction_pub = rospy.Publisher('/global_goal_direction', MarkerArray, queue_size=1, latch=True)


        self.global_goal_sub = rospy.Subscriber(
            '/move_base_simple/goal',
            PoseStamped,
            self.global_goal_callback
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
        # self.path_timer = rospy.Timer(rospy.Duration(1.0), self.save_robot_position)

        self.resolution = None
        self.width = None
        self.height = None
        self.origin_x = None
        self.origin_y = None
        
        self.dr_srv = Server(HyperparametersConfig, self._reconfigure_cb)


    def _reconfigure_cb(self, config, level):
        # 프런티어/탐색
        self.SEARCH_RADIUS     = float(config.get('SEARCH_RADIUS',     self.SEARCH_RADIUS))
        self.ANGLE_RESOLUTION  = float(config.get('ANGLE_RESOLUTION',  self.ANGLE_RESOLUTION))
        self.STEP_RESOLUTION   = float(config.get('STEP_RESOLUTION',   self.STEP_RESOLUTION))
        self.COST_THRESHOLD    = float(config.get('COST_THRESHOLD',    self.COST_THRESHOLD))
        self.MIN_FRONTIER_DIST = float(config.get('MIN_FRONTIER_DIST', self.MIN_FRONTIER_DIST))
        self.SAFETY_RADIUS     = int(  config.get('SAFETY_RADIUS',     self.SAFETY_RADIUS))

        # 임계값/적응
        self.ADAPT_ENABLE      = bool(  config.get('ADAPT_ENABLE',      self.ADAPT_ENABLE))
        self.BASE_COLLISION    = float( config.get('BASE_COLLISION',    self.BASE_COLLISION))
        self.BASE_STEEPNESS    = float( config.get('BASE_STEEPNESS',    self.BASE_STEEPNESS))
        self.BASE_INCLINATION  = float( config.get('BASE_INCLINATION',  self.BASE_INCLINATION))

        self.HEADING_BIAS = float(config.get('HEADING_BIAS', getattr(self,'HEADING_BIAS',0.6)))
        self.GOAL_BIAS    = float(config.get('GOAL_BIAS',    getattr(self,'GOAL_BIAS',0.6)))
        self.BIAS_SPREAD  = math.radians(float(config.get('BIAS_SPREAD_DEG', getattr(self,'BIAS_SPREAD',30.0))))
        self.BIAS_MULT    = int(config.get('BIAS_MULT', getattr(self,'BIAS_MULT',2)))
        self.MIN_ANG_SEP  = math.radians(float(config.get('MIN_ANG_SEP_DEG', getattr(self,'MIN_ANG_SEP',8.0))))
        self.NMS_RADIUS   = float(config.get('NMS_RADIUS', getattr(self,'NMS_RADIUS',1.0)))
        self.SECTOR_K     = int(config.get('SECTOR_K', getattr(self,'SECTOR_K',4)))
        self.USE_BOUNDARY_FRONTIER = bool(config.get('USE_BOUNDARY_FRONTIER', getattr(self,'USE_BOUNDARY_FRONTIER', True)))

        self.USE_RING_NMS     = bool(  config.get('USE_RING_NMS',     getattr(self,'USE_RING_NMS', True)))
        self.MIN_ARC_SEP_M    = float( config.get('MIN_ARC_SEP_M',    getattr(self,'MIN_ARC_SEP_M', 1.0)))
        self.FRONTIER_INSET_M = float( config.get('FRONTIER_INSET_M', getattr(self,'FRONTIER_INSET_M', 0.3)))

        # 시각화/타이머
        new_arrow = float(config.get('ARROW_MAX_LEN', self.ARROW_MAX_LEN))
        self.ARROW_MAX_LEN = new_arrow

        new_hz = float(config.get('PATH_UPDATE_HZ', self.PATH_UPDATE_HZ))
        if abs(self.PATH_UPDATE_HZ - new_hz) > 1e-6:
            self.PATH_UPDATE_HZ = new_hz
            try:
                self.path_timer.shutdown()
            except Exception:
                pass
            self.path_timer = rospy.Timer(rospy.Duration(1.0/self.PATH_UPDATE_HZ), self.save_robot_position)

        rospy.loginfo("[DR] Updated params: "
                    f"R={self.SEARCH_RADIUS:.2f}, dθ={self.ANGLE_RESOLUTION:.2f}, "
                    f"dr={self.STEP_RESOLUTION:.2f}, minΔ={self.MIN_FRONTIER_DIST:.2f}, "
                    f"safety={self.SAFETY_RADIUS}, arrowMax={self.ARROW_MAX_LEN:.1f}, "
                    f"adapt={self.ADAPT_ENABLE}")
        # return config
        try:
            if self.grid_map is not None:
                incl = self.get_layer_data('inclination_risk')
                coll = self.get_layer_data('collision_risk')
                steep = self.get_layer_data('steepness_risk')

                if incl is not None and coll is not None and steep is not None:
                    traversability_map = self.compute_cost_map(incl, coll, steep)
                    if traversability_map is not None:
                        self.last_traversability_map = traversability_map

                        frontiers = self.find_frontiers(traversability_map)
                        transformed_global_goal = self.transform_global_goal_to_local()
                        selected_frontier = self.select_best_frontier(frontiers, transformed_global_goal) if frontiers else None

                        if selected_frontier:
                            # ★ 여기: 변환 제거
                            self.publish_local_goal(selected_frontier)

                        self.visualize_traversability_map(traversability_map)
                        self.visualize_frontiers(frontiers if frontiers else [], selected_frontier)
                        self.visualize_global_goal()
                        self.visualize_global_goal_direction()
                        self.last_frontiers = frontiers
                        self.last_selected_frontier = selected_frontier
            else:
                rospy.logdebug("No grid_map yet; will reflect on next update.")
        except Exception as e:
            rospy.logwarn(f"[DR] immediate refresh failed: {e}")

        return config

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

    # def global_goal_callback(self, msg):
    #     """RViz '2D Nav Goal' 클릭 시 호출되어 global_goal 업데이트"""
    #     # msg는 PoseStamped 타입입니다.
    #     # RViz에서 설정한 Fixed Frame이 'aligned_basis'라고 가정합니다.
    #     self.global_goal_x = msg.pose.position.x
    #     self.global_goal_y = msg.pose.position.y

    #     # 파라미터 서버에도 업데이트 (다른 노드와 공유 시 유용)
    #     rospy.set_param('/global_goal_x', self.global_goal_x)
    #     rospy.set_param('/global_goal_y', self.global_goal_y)

    #     rospy.loginfo(f"New global goal set from RViz: ({self.global_goal_x:.2f}, {self.global_goal_y:.2f})")
    def global_goal_callback(self, msg):
        """RViz '2D Nav Goal' 클릭 시 호출되어 global_goal 업데이트"""
        
        target_frame = "world"
        source_frame = msg.header.frame_id
        
        rospy.loginfo(f"Attempting to set new goal from RViz. (Source: {source_frame})")

        try:
            # ★★★★★★★★★★★★★★★★★★★★★★★
            # 해결책: 타임스탬프를 rospy.Time(0)으로 설정하여
            # "가장 최신의" TF transform을 사용하도록 강제합니다.
            # rosbag 재생 시 발생하는 타임스탬프 불일치 문제 해결
            # ★★★★★★★★★★★★★★★★★★★★★★★
            pose_to_transform = PoseStamped()
            pose_to_transform.header.frame_id = source_frame
            pose_to_transform.header.stamp = rospy.Time(0) # <--- 이것이 핵심입니다!
            pose_to_transform.pose = msg.pose # 원본 msg의 pose 데이터 사용
            
            # 원본 msg 대신, 타임스탬프가 0인 복사본(pose_to_transform)을 변환합니다.
            transformed_pose_stamped = self.tf_buffer.transform(
                pose_to_transform, 
                target_frame, 
                rospy.Duration(1.0)
            )
            # ★★★★★★★★★★★★★★★★★★★★★★★

            # 변환된 좌표를 global_goal로 저장
            self.global_goal_x = transformed_pose_stamped.pose.position.x
            self.global_goal_y = transformed_pose_stamped.pose.position.y

            # 파라미터 서버에도 업데이트
            rospy.set_param('/global_goal_x', self.global_goal_x)
            rospy.set_param('/global_goal_y', self.global_goal_y)

            rospy.loginfo(f"✅ [SUCCESS] New global goal set in {target_frame}: "
                          f"({self.global_goal_x:.2f}, {self.global_goal_y:.2f})")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"❌ [FAIL] Failed to transform global goal from {source_frame} to {target_frame}: {e}")
            rospy.logwarn("Global goal was NOT updated. Check TF tree (is 'world' frame available?).")

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
            delete_marker.header.frame_id = "world"
            delete_marker.header.stamp = rospy.Time.now()
            delete_marker.ns = "robot_path"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # 경로를 선으로 연결하는 마커 생성
        line_marker = Marker()
        line_marker.header.frame_id = "world"
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
            current_marker.header.frame_id = "world"
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

    # def visualize_global_goal_direction(self):
    #     """로봇을 중심으로 글로벌 골 방향을 가리키는 화살표 시각화"""
    #     marker_array = MarkerArray()
        
    #     # 글로벌 골을 로컬 좌표계로 변환
    #     transformed_goal = self.transform_global_goal_to_local()
        
    #     # 화살표 길이 (로봇에서 글로벌 골까지의 거리, 최대 5m)
    #     goal_distance = math.sqrt(transformed_goal[0]**2 + transformed_goal[1]**2)
    #     arrow_length = min(goal_distance, 5.0)  # 최대 5m로 제한
        
    #     # 화살표 마커 생성
    #     arrow_marker = Marker()
    #     arrow_marker.header.frame_id = "aligned_basis"
    #     arrow_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
    #     arrow_marker.ns = "global_goal_direction"
    #     arrow_marker.id = 0
    #     arrow_marker.type = Marker.ARROW
    #     arrow_marker.action = Marker.ADD
        
    #     # 화살표 시작점 (로봇 위치)
    #     start_point = Point()
    #     start_point.x = 0.0  # 로봇 중심
    #     start_point.y = 0.0
    #     start_point.z = 0.5  # 지면에서 0.5m 위
        
    #     # 화살표 끝점 (글로벌 골 방향)
    #     end_point = Point()
    #     if goal_distance > 0:
    #         # 정규화된 방향 벡터에 화살표 길이 적용
    #         end_point.x = (transformed_goal[0] / goal_distance) * arrow_length
    #         end_point.y = (transformed_goal[1] / goal_distance) * arrow_length
    #     else:
    #         end_point.x = 0.0
    #         end_point.y = 0.0
    #     end_point.z = 0.5
        
    #     arrow_marker.points = [start_point, end_point]
        
    #     # 화살표 스타일 설정
    #     arrow_marker.scale.x = 0.2  # 화살표 몸통 두께 (0.3 → 0.2)
    #     arrow_marker.scale.y = 0.3  # 화살표 머리 두께 (0.5 → 0.3)
    #     arrow_marker.scale.z = 0.0  # 2D 화살표
        
    #     # 밝은 노란색으로 표시
    #     arrow_marker.color = ColorRGBA()
    #     arrow_marker.color.r = 1.0  # 노란색
    #     arrow_marker.color.g = 1.0
    #     arrow_marker.color.b = 0.0
    #     arrow_marker.color.a = 0.8  # 약간 투명
        
    #     marker_array.markers.append(arrow_marker)
        
    #     # 글로벌 골 방향 각도 표시를 위한 텍스트 마커
    #     text_marker = Marker()
    #     text_marker.header.frame_id = "aligned_basis"
    #     text_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
    #     text_marker.ns = "global_goal_direction"
    #     text_marker.id = 1
    #     text_marker.type = Marker.TEXT_VIEW_FACING
    #     text_marker.action = Marker.ADD
        
    #     # 텍스트 위치 (화살표 중간 지점)
    #     text_marker.pose.position.x = end_point.x / 2
    #     text_marker.pose.position.y = end_point.y / 2
    #     text_marker.pose.position.z = 1.0  # 화살표 위에 표시
        
    #     # 각도 계산 및 표시
    #     goal_angle = math.atan2(transformed_goal[1], transformed_goal[0])
    #     text_marker.text = f"Goal: {math.degrees(goal_angle):.0f}°"
        
    #     # 텍스트 스타일
    #     text_marker.scale.z = 0.2  # 텍스트 크기 (0.3 → 0.2)
    #     text_marker.color = ColorRGBA()
    #     text_marker.color.r = 1.0  # 흰색
    #     text_marker.color.g = 1.0
    #     text_marker.color.b = 1.0
    #     text_marker.color.a = 1.0
        
    #     marker_array.markers.append(text_marker)
        
    #     self.global_goal_direction_pub.publish(marker_array)

    def visualize_global_goal_direction(self):
        """로봇을 중심으로 글로벌 골 방향을 가리키는 화살표 시각화"""
        marker_array = MarkerArray()
        
        # 1. 글로벌 골까지의 방향 벡터 (aligned_basis 프레임 기준)
        dx = self.global_goal_x - self.odom_position_x
        dy = self.global_goal_y - self.odom_position_y
        
        goal_distance = math.sqrt(dx**2 + dy**2)
        # arrow_length = min(goal_distance, 5.0)  # 최대 5m로 제한
        arrow_length = min(goal_distance, self.ARROW_MAX_LEN)

        
        # 2. 화살표 마커 생성
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "world"
        arrow_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
        arrow_marker.ns = "global_goal_direction"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # 3. 화살표 시작점 (로봇의 현재 위치)
        start_point = Point()
        start_point.x = self.odom_position_x
        start_point.y = self.odom_position_y
        start_point.z = 0.5  # 지면에서 0.5m 위
        
        # 4. 화살표 끝점 (글로벌 골 방향)
        end_point = Point()
        if goal_distance > 0:
            # 정규화된 방향 벡터에 화살표 길이 적용
            end_point.x = self.odom_position_x + (dx / goal_distance) * arrow_length
            end_point.y = self.odom_position_y + (dy / goal_distance) * arrow_length
        else:
            end_point.x = self.odom_position_x
            end_point.y = self.odom_position_y
        end_point.z = 0.5
        
        arrow_marker.points = [start_point, end_point]
        
        # 화살표 스타일 설정
        arrow_marker.scale.x = 0.2
        arrow_marker.scale.y = 0.3
        arrow_marker.scale.z = 0.0
        
        arrow_marker.color = ColorRGBA()
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 0.8
        
        marker_array.markers.append(arrow_marker)
        
        # 5. 글로벌 골 방향 각도 표시를 위한 텍스트 마커
        text_marker = Marker()
        text_marker.header.frame_id = "world"
        text_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
        text_marker.ns = "global_goal_direction"
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # 텍스트 위치 (화살표 중간 지점)
        text_marker.pose.position.x = (start_point.x + end_point.x) / 2
        text_marker.pose.position.y = (start_point.y + end_point.y) / 2
        text_marker.pose.position.z = 1.0  # 화살표 위에 표시
        
        # 각도 계산 (로봇의 yaw를 기준으로 한 상대 각도)
        # 로봇 프레임 기준의 골 각도
        local_goal_angle = math.atan2(dy, dx) - self.odom_rotation_yaw
        # -pi ~ pi 범위로 정규화
        local_goal_angle = (local_goal_angle + math.pi) % (2 * math.pi) - math.pi 
        
        text_marker.text = f"Goal: {math.degrees(local_goal_angle):.0f}°"
        
        text_marker.scale.z = 0.2
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        
        marker_array.markers.append(text_marker)
        
        self.global_goal_direction_pub.publish(marker_array)
        
    def get_layer_data(self, layer_name):
        try:
            if layer_name not in self.grid_map.layers:
                rospy.logwarn(f"Layer {layer_name} not found in grid map"); return None
            data_msg = self.grid_map.data[self.grid_map.layers.index(layer_name)]
            raw = np.array(data_msg.data)
            dim0 = data_msg.layout.dim[0]  # usually columns
            dim1 = data_msg.layout.dim[1]  # usually rows

            # ✅ GridMap: dim0=columns, dim1=rows → NumPy wants (rows, cols)
            cols = int(dim0.size)
            rows = int(dim1.size)
            data = raw.reshape((rows, cols))

            rospy.loginfo(f"[{layer_name}] shape={data.shape} (rows={rows}, cols={cols}), nan={np.isnan(data).sum()}/{data.size}")
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
        # base_collision_threshold = 0.8
        # base_steepness_threshold = 0.7
        # base_inclination_threshold = 0.7
        base_collision_threshold   = self.BASE_COLLISION
        base_steepness_threshold   = self.BASE_STEEPNESS
        base_inclination_threshold = self.BASE_INCLINATION
        if not self.ADAPT_ENABLE:
            rospy.loginfo("Adaptive thresholds: DISABLED")
            return (base_collision_threshold, base_steepness_threshold, base_inclination_threshold)

        
        # 기울기에 따른 임계값 조정 (갈 수 있는 영역 확대)
        if total_tilt < 3.0:  # 평지 (3도 미만)
            # 기본 임계값 (평지에서는 정상적인 탐색)
            collision_threshold = base_collision_threshold
            steepness_threshold = base_steepness_threshold
            inclination_threshold = base_inclination_threshold
        elif total_tilt < 8.0:  # 약간 기울어진 지형 (3-8도)
            # 기울어진 상태: 갈 수 있는 영역 확대 (임계값 낮춤)
            # collision_threshold = base_collision_threshold - 0.1
            # steepness_threshold = base_steepness_threshold - 0.2  # 경사 임계값 낮춤
            # inclination_threshold = base_inclination_threshold - 0.2  # 경사도 임계값 낮춤
            collision_threshold = base_collision_threshold + 0.1 # 더 관대하게
            steepness_threshold = base_steepness_threshold + 0.1 # 더 관대하게
            inclination_threshold = base_inclination_threshold + 0.1 # 더 관대하게
        elif total_tilt < 15.0:  # 기울어진 지형 (8-15도)
            # 기울어진 상태: 갈 수 있는 영역 더 확대
            # collision_threshold = base_collision_threshold - 0.2
            # steepness_threshold = base_steepness_threshold - 0.4  # 경사 임계값 더 낮춤
            # inclination_threshold = base_inclination_threshold - 0.4  # 경사도 임계값 더 낮춤
            collision_threshold = base_collision_threshold + 0.2
            steepness_threshold = base_steepness_threshold + 0.2
            inclination_threshold = base_inclination_threshold + 0.2
        elif total_tilt < 25.0:  # 매우 기울어진 지형 (15-25도)
            # 기울어진 상태: 갈 수 있는 영역 매우 확대
            # collision_threshold = base_collision_threshold - 0.3
            # steepness_threshold = base_steepness_threshold - 0.6  # 경사 임계값 매우 낮춤
            # inclination_threshold = base_inclination_threshold - 0.6  # 경사도 임계값 매우 낮춤
            collision_threshold = base_collision_threshold + 0.3
            steepness_threshold = base_steepness_threshold + 0.3
            inclination_threshold = base_inclination_threshold + 0.3
        else:  # 극한 지형 (25도 이상)
            # 기울어진 상태: 갈 수 있는 영역 극한 확대
            # collision_threshold = base_collision_threshold - 0.4
            # steepness_threshold = base_steepness_threshold - 0.8  # 경사 임계값 극한 낮춤
            # inclination_threshold = base_inclination_threshold - 0.8  # 경사도 임계값 극한 낮춤
            collision_threshold = base_collision_threshold + 0.4
            steepness_threshold = base_steepness_threshold + 0.4
            inclination_threshold = base_inclination_threshold + 0.4
        
        # 임계값 범위 제한 (0.05 ~ 1.0) - 더 관대한 범위
        collision_threshold = max(0.05, min(1.0, collision_threshold))
        steepness_threshold = max(0.05, min(1.0, steepness_threshold))
        inclination_threshold = max(0.05, min(1.0, inclination_threshold))
        
        rospy.loginfo(f"Robot tilt: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, total={total_tilt:.1f}°")
        rospy.loginfo(f"Adaptive thresholds: collision={collision_threshold:.2f}, "
                     f"steepness={steepness_threshold:.2f}, inclination={inclination_threshold:.2f}")
        
        return collision_threshold, steepness_threshold, inclination_threshold

    # def make_biased_angles(self):
    #     base = np.arange(0, 2*np.pi, self.ANGLE_RESOLUTION)

    #     # 진행방향(=0)과 골 방향
    #     yaw = 0.0  # 로컬 프레임 기준
    #     dx = self.global_goal_x - self.odom_position_x
    #     dy = self.global_goal_y - self.odom_position_y
    #     goal_ang = math.atan2(dy, dx) - self.odom_rotation_yaw
    #     goal_ang = (goal_ang + math.pi) % (2*math.pi) - math.pi

    #     extra = []
    #     def oversample(center, mult, spread):
    #         if mult <= 0: return
    #         # spread 범위에 균일 분포로 mult*2 개 추가
    #         offs = np.linspace(-spread, spread, 2*mult+1)
    #         for o in offs:
    #             extra.append((center + o + 2*np.pi) % (2*np.pi))

    #     oversample(yaw,      int(round(self.BIAS_MULT * self.HEADING_BIAS)), self.BIAS_SPREAD)
    #     oversample(goal_ang, int(round(self.BIAS_MULT * self.GOAL_BIAS)),    self.BIAS_SPREAD)

    #     angles = np.concatenate([base, np.array(extra, dtype=float)])
    #     angles = np.mod(angles, 2*np.pi)
    #     # 중복/가까운 각도 정리
    #     angles = np.unique(np.round(angles, 4))
    #     return angles
    def make_biased_angles(self):
        base = np.arange(0, 2*np.pi, self.ANGLE_RESOLUTION)

        # ✅ 진행방향: 실제 로봇 yaw
        yaw = (self.odom_rotation_yaw + 2*math.pi) % (2*math.pi)

        # ✅ 골 방향: 맵(aligned_basis)에서의 절대 각도
        dx = self.global_goal_x - self.odom_position_x
        dy = self.global_goal_y - self.odom_position_y
        goal_ang = (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)

        extra = []
        def oversample(center, mult, spread):
            if mult <= 0: return
            offs = np.linspace(-spread, spread, 2*mult+1)
            for o in offs:
                extra.append((center + o) % (2*math.pi))

        oversample(yaw,      int(round(self.BIAS_MULT * self.HEADING_BIAS)), self.BIAS_SPREAD)
        oversample(goal_ang, int(round(self.BIAS_MULT * self.GOAL_BIAS)),    self.BIAS_SPREAD)

        angles = np.concatenate([base, np.array(extra, dtype=float)])
        angles = np.mod(angles, 2*np.pi)
        angles = np.unique(np.round(angles, 4))
        return angles


    # def thin_frontiers(self, pts_world):
    #     if not pts_world: return pts_world

    #     # 각 후보 점의 각도/거리 계산
    #     def polar(p):
    #         dx, dy = p[0]-self.odom_position_x, p[1]-self.odom_position_y
    #         ang = math.atan2(dy, dx); ang = (ang + 2*math.pi)%(2*math.pi)
    #         dist = math.hypot(dx, dy)
    #         return ang, dist

    #     # 섹터(예: 15°)로 나눠 섹터당 최대 K개만 보유
    #     sector_size = math.radians(15.0)
    #     buckets = {}
    #     for p in pts_world:
    #         ang,_ = polar(p)
    #         s = int(ang/sector_size)
    #         buckets.setdefault(s, []).append(p)

    #     kept = []
    #     for s, plist in buckets.items():
    #         # 각도/거리 기준 정렬(예: 로봇/골 정렬 점수 우선)
    #         def score(p):
    #             ang, dist = polar(p)
    #             # 진행방향/골 방향 정렬 보너스
    #             yaw = (0.0 + 2*np.pi)%(2*np.pi)
    #             dx,dy = self.global_goal_x-self.odom_position_x, self.global_goal_y-self.odom_position_y
    #             goal_ang = (math.atan2(dy,dx)-self.odom_rotation_yaw + 2*np.pi)%(2*np.pi)
    #             def align(a,b): 
    #                 d = abs(((a-b)+math.pi)%(2*math.pi)-math.pi)
    #                 return math.cos(d)  # 1(정렬)~ -1(반대)
    #             return 1.0*align(ang,yaw) + 1.0*align(ang,goal_ang) - 0.05*dist
    #         plist.sort(key=score, reverse=True)

    #         # 공간 NMS + 각도 최소분리
    #         picked = []
    #         for p in plist:
    #             pang, _ = polar(p)
    #             ok = True
    #             for q in picked:
    #                 qang,_ = polar(q)
    #                 if math.hypot(p[0]-q[0], p[1]-q[1]) < self.NMS_RADIUS: ok=False; break
    #                 d_ang = abs(((pang-qang)+math.pi)%(2*math.pi)-math.pi)
    #                 if d_ang < self.MIN_ANG_SEP: ok=False; break
    #             if ok:
    #                 picked.append(p)
    #             if len(picked) >= self.SECTOR_K: break
    #         kept.extend(picked)

    #     return kept

    def thin_frontiers(self, pts_world):
        if not pts_world: return pts_world

        def polar(p):
            dx, dy = p[0]-self.odom_position_x, p[1]-self.odom_position_y
            ang = math.atan2(dy, dx) % (2*math.pi)
            dist = math.hypot(dx, dy)
            return ang, dist

        # ✅ 섹터 크기를 파라미터화하고 싶으면 cfg로 빼도 좋음(기본 15°)
        sector_size = math.radians(15.0)

        buckets = {}
        for p in pts_world:
            ang,_ = polar(p)
            s = int(ang / sector_size)
            buckets.setdefault(s, []).append(p)

        kept = []
        # ✅ 실제 로봇 yaw 사용
        yaw = (self.odom_rotation_yaw + 2*math.pi) % (2*math.pi)

        # ✅ goal 각도(절대각)
        gdx, gdy = self.global_goal_x - self.odom_position_x, self.global_goal_y - self.odom_position_y
        goal_ang = (math.atan2(gdy, gdx) + 2*math.pi) % (2*math.pi)

        def align(a,b):
            d = abs(((a-b)+math.pi)%(2*math.pi) - math.pi)
            return math.cos(d)  # 1(정렬) ~ -1(반대)

        for s, plist in buckets.items():
            # 진행/골 정렬 보너스, 거리 패널티
            def score(p):
                ang, dist = polar(p)
                return 1.0*align(ang, yaw) + 1.0*align(ang, goal_ang) - 0.05*dist

            plist.sort(key=score, reverse=True)

            picked = []
            for p in plist:
                pang, _ = polar(p)
                ok = True
                for q in picked:
                    # ✅ 공간 NMS
                    if math.hypot(p[0]-q[0], p[1]-q[1]) < self.NMS_RADIUS:
                        ok = False; break
                    # ✅ 각도 최소 분리
                    qang,_ = polar(q)
                    d_ang = abs(((pang-qang)+math.pi)%(2*math.pi) - math.pi)
                    if d_ang < self.MIN_ANG_SEP:
                        ok = False; break
                if ok:
                    picked.append(p)
                if len(picked) >= self.SECTOR_K:  # ✅ 섹터당 최대 K
                    break

            kept.extend(picked)

        return kept


    # def find_frontiers(self, traversability_map):
        # """Traversability map에서 프런티어 포인트 탐색"""
        # frontiers = []
        # center_x = int(self.width / (2 * self.resolution))
        # center_y = int(self.height / (2 * self.resolution))


        # total_nan = np.sum(np.isnan(traversability_map))
        # total_safe = np.sum(traversability_map == 0.0)
        # total_caution = np.sum(traversability_map == 0.5)
        # total_blocked = np.sum(traversability_map == 1.0)

        # rays_checked = 0
        # rays_hit_nan = 0
        # rays_hit_blocked = 0

        # for angle in np.arange(0, 2*np.pi, self.ANGLE_RESOLUTION):
        #     rays_checked += 1
        #     max_dist_cells = int(self.SEARCH_RADIUS / self.resolution)
        #     farthest_traversable_point = None

        #     for dist in range(1, max_dist_cells):
        #         x = int(center_x + dist * np.cos(angle))
        #         y = int(center_y + dist * np.sin(angle))

        #         if not (0 <= x < traversability_map.shape[1] and 0 <= y < traversability_map.shape[0]):
        #             break

        #         current_value = traversability_map[y, x]

        #         if np.isnan(current_value):
        #             rays_hit_nan += 1
        #             break
        #         elif current_value >= 1.0:
        #             rays_hit_blocked += 1
        #             break
        #         elif current_value <= 0.5:
        #             if self.is_safe_area(traversability_map, x, y):
        #                 farthest_traversable_point = (x, y)

        #     if farthest_traversable_point is not None:
        #         cell_x, cell_y = farthest_traversable_point
        #         # 원본 GridMap의 pose 기준으로 좌표 계산
        #         # GridMap의 pose는 맵의 중심점이므로, 셀 좌표를 world 좌표로 변환
        #         # 좌우, 위아래 모두 반전을 위해 x축과 y축 모두 뒤집음
        #         world_x = self.grid_map.info.pose.position.x - (cell_x - center_x) * self.resolution
        #         world_y = self.grid_map.info.pose.position.y - (cell_y - center_y) * self.resolution

        #         dist_from_robot = np.sqrt(world_x**2 + world_y**2)

        #         traversability_value = traversability_map[cell_y, cell_x]

        #         if (dist_from_robot > self.MIN_FRONTIER_DIST and
        #             self.check_min_distance(frontiers, (world_x, world_y))):
        #             frontiers.append((world_x, world_y))


        # rospy.loginfo(f"Found {len(frontiers)} frontier candidates")

        # if len(frontiers) == 0:
        #     rospy.logwarn("No frontiers found!")
        #     rospy.logwarn(f"Debug info:")
        #     rospy.logwarn(f"  Rays checked: {rays_checked}")
        #     rospy.logwarn(f"  Rays hit NaN: {rays_hit_nan}")
        #     rospy.logwarn(f"  Rays hit blocked: {rays_hit_blocked}")
        #     rospy.logwarn(f"  Map stats - NaN: {total_nan}, Safe: {total_safe}, Caution: {total_caution}, Blocked: {total_blocked}")

        #     if rays_hit_nan == 0:
        #         rospy.logwarn("  → Try increasing search radius or check if NaN regions exist in the map")

        # return frontiers
    def _map_center_yaw(self):
        """GridMap info.pose에서 (cx, cy, yaw) 안전하게 추출. 쿼터니언이 비정상이면 yaw=0으로."""
        pose = self.grid_map.info.pose
        cx, cy = pose.position.x, pose.position.y
        q = pose.orientation
        qv = [q.x, q.y, q.z, q.w]

        try:
            # 쿼터니언 노름이 너무 작으면(=0,0,0,0 같은) 회전 없음으로 간주
            if abs(q.x) + abs(q.y) + abs(q.z) + abs(q.w) < 1e-9:
                yaw = 0.0
                rospy.logwarn("GridMap pose quaternion is zero; assuming yaw=0.")
            else:
                (_, _, yaw) = tft.euler_from_quaternion(qv)
        except Exception as e:
            yaw = 0.0
            rospy.logwarn(f"euler_from_quaternion failed: {e}; fallback yaw=0.")
        return cx, cy, yaw

    def _cell_to_world_no_rot(self, ix:int, iy:int, cols:int, rows:int):
        """원래 find_frontiers 공식: 맵 yaw=0 가정, GridMap.pose가 맵 중심."""
        cx = self.span_center_x()
        cy = self.span_center_y()
        wx = cx - (ix - int(cols/2)) * self.resolution
        wy = cy - (iy - int(rows/2)) * self.resolution
        return wx, wy

    def _world_to_cell_no_rot(self, xw:float, yw:float, cols:int, rows:int):
        """원래 find_frontiers 역변환(중심 기준, x는 -, y도 -)"""
        cx = self.span_center_x()
        cy = self.span_center_y()
        ix = int(round( int(cols/2)  - (xw - cx) / self.resolution ))
        iy = int(round( int(rows/2)  - (yw - cy) / self.resolution ))
        return ix, iy

    def span_center_x(self):  # world center X of grid_map
        return self.grid_map.info.pose.position.x or 0.0
    def span_center_y(self):
        return self.grid_map.info.pose.position.y or 0.0
        
    def find_frontiers(self, traversability_map):
        """원래 좌표계(yaw=0 가정) + 로봇 위치는 맵 중앙과 동일하다고 가정된 버전.
        (terrain_local_gridmap 이면 보통 맵 중심이 로봇 위치입니다)"""
        frontiers = []

        rows, cols = traversability_map.shape
        cx = self.grid_map.info.pose.position.x
        cy = self.grid_map.info.pose.position.y
        center_x = int(self.width  / (2.0 * self.resolution))   # 원래 코드와 동일하게
        center_y = int(self.height / (2.0 * self.resolution))

        # 레이 스캔 파라미터
        angles = self.make_biased_angles()   # world==map 축 가정 (yaw=0)
        max_dist_cells = max(1, int(self.SEARCH_RADIUS / self.resolution))
        rays_checked = rays_hit_nan = rays_hit_blocked = 0

        rospy.loginfo(f"[shape] rows={rows}, cols={cols}, center=({center_x},{center_y}), res={self.resolution:.3f}")

        for ang in angles:
            rays_checked += 1
            far_ix = far_iy = None

            for dist in range(1, max_dist_cells+1):
                ix = int(round(center_x + dist * math.cos(ang)))
                iy = int(round(center_y + math.sin(ang) * dist))

                if not (0 <= ix < cols and 0 <= iy < rows):
                    break

                v = float(traversability_map[iy, ix])
                if np.isnan(v):
                    rays_hit_nan += 1
                    break
                if v >= 1.0:
                    rays_hit_blocked += 1
                    break
                if v <= 0.5 and self.is_safe_area(traversability_map, ix, iy):
                    far_ix, far_iy = ix, iy

            if far_ix is not None:
                wx, wy = self._cell_to_world_no_rot(far_ix, far_iy, cols, rows)

                # 인셋(옵션)
                if self.FRONTIER_INSET_M > 1e-6:
                    vx, vy = (wx - self.odom_position_x), (wy - self.odom_position_y)
                    d = math.hypot(vx, vy)
                    if d > 1e-6:
                        s = max(0.0, (d - self.FRONTIER_INSET_M) / d)
                        wx = self.odom_position_x + vx * s
                        wy = self.odom_position_y + vy * s

                if math.hypot(wx - self.odom_position_x, wy - self.odom_position_y) > self.MIN_FRONTIER_DIST and \
                self.check_min_distance(frontiers, (wx, wy)):
                    frontiers.append((wx, wy))

        rospy.loginfo(f"Found {len(frontiers)} frontier candidates "
                    f"(NaN={int(np.isnan(traversability_map).sum())}, "
                    f"safe={int((traversability_map==0.0).sum())}, "
                    f"blocked={int((traversability_map==1.0).sum())}, "
                    f"rays={rays_checked}, hitNaN={rays_hit_nan}, hitBlk={rays_hit_blocked})")

        if not frontiers:
            # 디버깅 팁 로그
            rospy.logwarn("No frontiers found. Try: smaller SAFETY_RADIUS (e.g.,2), set USE_BOUNDARY_FRONTIER=False, "
                        "reduce ANGLE_RESOLUTION to 0.2, verify GridMap length_x/length_y ≈ (cols*res)/(rows*res).")
            return []

        # 억제 단계 (섹터+NMS+링)
        rospy.loginfo(f"[frontier] before thin: {len(frontiers)}")
        frontiers = self.thin_frontiers(frontiers)
        rospy.loginfo(f"[frontier] after  thin: {len(frontiers)}")
        if getattr(self, "USE_RING_NMS", True):
            frontiers = self.ring_nms(frontiers)
        rospy.loginfo(f"[frontier] after  ring_nms: {len(frontiers)}")

        return frontiers



    def ring_nms(self, pts_world):
        if not pts_world: return pts_world
        # 로봇 기준 극좌표로 변환
        polars = []
        for (x,y) in pts_world:
            dx, dy = x - self.odom_position_x, y - self.odom_position_y
            r = math.hypot(dx, dy)
            theta = (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
            polars.append([r, theta, (x,y)])
        # 반경으로 정렬 후, 유사 반경끼리 묶기
        polars.sort(key=lambda t: t[0])

        kept = []
        i = 0
        RAD_EPS = self.resolution * 1.5  # 동일 ring 판단 허용 오차
        while i < len(polars):
            # 한 ring(반경 대역) 묶기
            r0 = polars[i][0]
            ring = []
            while i < len(polars) and abs(polars[i][0] - r0) <= RAD_EPS:
                ring.append(polars[i]); i += 1
            # 각도 기준 정렬
            ring.sort(key=lambda t: t[1])
            # 아크 길이 기반 억제
            acc = []
            for r, theta, pt in ring:
                ok = True
                for rr, tt, _ in acc:
                    # 같은 ring이므로 아크 길이 = r * 최소각도차
                    dtheta = abs(((theta - tt) + math.pi) % (2*math.pi) - math.pi)
                    arc = r * dtheta
                    if arc < self.MIN_ARC_SEP_M:
                        ok = False
                        break
                if ok:
                    acc.append([r, theta, pt])
            kept.extend([pt for _,_,pt in acc])
        return kept


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

            # rospy.set_param('/global_goal_x', selected_frontier[0])
            # rospy.set_param('/global_goal_y', selected_frontier[1])


            marker_array.markers.append(marker)

        self.frontier_viz_pub.publish(marker_array)

    def visualize_global_goal(self):
        """글로벌 골을 시각화 (프론티어보다 크게)"""
        marker_array = MarkerArray()

        # Global goal을 로컬 좌표계로 변환
        transformed_goal = self.transform_global_goal_to_local()

        # 글로벌 골 마커 생성
        goal_marker = Marker()
        goal_marker.header.frame_id = "world"
        goal_marker.header.stamp = self.grid_map.info.header.stamp if self.grid_map else rospy.Time.now()
        goal_marker.ns = "global_goal"
        goal_marker.id = 0
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD

        # goal_marker.pose.position.x = transformed_goal[0]
        # goal_marker.pose.position.y = transformed_goal[1]
        goal_marker.pose.position.x = self.global_goal_x 
        goal_marker.pose.position.y = self.global_goal_y
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