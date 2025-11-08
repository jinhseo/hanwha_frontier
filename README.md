# hanwha_frontier

### CM 1109
- frontier_detector.py : goal 좌표계 수정
  - /global_goal_visualization (빨간색 구): 글로벌 골의 월드좌표계 기준 위치
  - /goal_projection_visualization (청록색 구) : 글로벌 골의 aligned basis 기준 위치 
    -> 이 위치 기준으로 best frontier 찾음
- frontier_detector_finetuning.py : finetuning 중 + goal 방향 및 odom 방향에 frontier 후보 더 뽑히도록 구현
  - in_the_planning_module_folder에 있는 cfg 폴더 & CMakeList, package.xml를 planning_module 폴더에 복사
  - rosrun planning_module frontier_detector_finetuning.py
  - rosrun rqt_reconfigure rqt_reconfigure -> fontier_detector 더블클릭 -> 스크롤로 조절