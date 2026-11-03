#!/usr/bin/env python3
"""Run an isolated Gazebo integration check and save both camera views."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import String
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from kinova_marker_pick.vision import Detector
from ament_index_python.packages import get_package_share_directory


def main():
    share = Path(get_package_share_directory('kinova_marker_pick'))
    defaults = yaml.safe_load((share/'config/observation_poses.yaml').read_text())['marker_pick']['ros__parameters']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-source', choices=['markers', 'fixture'], default='markers')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--output', type=Path, default=Path('/tmp/kinova_sim_validation'))
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--observation-offset', type=float, default=defaults['observation_offset_m'])
    parser.add_argument('--observation-angle', type=float, default=defaults['observation_angle_rad'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('ROS_DOMAIN_ID', '92')
    os.environ.setdefault('GZ_PARTITION', 'kinova_validation')
    os.environ.setdefault('ROS_LOG_DIR', str(args.output.resolve() / 'ros_logs'))
    status = {'stage':'waiting', 'detail':''}
    observed = {camera:{'image_frame':None, 'info_frame':None, 'marker_ids':[], 'images':0}
                for camera in ('overhead', 'eih')}
    latest_images = {}
    latest_joints = {}
    stage_joints = {}
    eih_positions = []
    ibvs_samples = []
    overlay = {'images':0, 'frame':None}
    bridge, detector = CvBridge(), Detector()
    with (args.output / 'launch.log').open('w') as log:
        process = subprocess.Popen([
            'ros2', 'launch', 'kinova_marker_pick', 'cable_insertion.launch.py',
            f'gui:={str(args.gui).lower()}', 'auto_execute:=true',
            f'target_source:={args.target_source}',
            f'observation_offset_m:={args.observation_offset}',
            f'observation_angle_rad:={args.observation_angle}',
        ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rclpy.init()
        node = rclpy.create_node('simulation_validator')
        buffer = Buffer()
        listener = TransformListener(buffer, node)

        def on_image(msg, camera):
            image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            latest_images[camera] = image
            observation = observed[camera]
            observation['images'] += 1
            if observation['image_frame'] is None:
                observation['image_frame'] = msg.header.frame_id
                cv2.imwrite(str(args.output / f'{camera}_initial.png'), image)
            if observation['images'] % 5 == 1:
                ids = detector.detect(image)[1]
                if ids is not None:
                    found = set(observation['marker_ids']) | set(map(int, ids.flatten()))
                    observation['marker_ids'] = sorted(found)
                    if camera == 'eih' and 8 in ids:
                        cv2.imwrite(str(args.output / 'eih_port.png'), image)
                if camera == 'eih':
                    try:
                        tf = buffer.lookup_transform('world', msg.header.frame_id, Time.from_msg(msg.header.stamp))
                        p = tf.transform.translation
                        eih_positions.append([p.x, p.y, p.z])
                    except Exception:
                        pass

        def on_info(msg, camera):
            observed[camera]['info_frame'] = msg.header.frame_id

        def on_status(msg):
            status.update(json.loads(msg.data))
            stage_joints[status['stage']] = latest_joints.copy()
            print(status['stage'] + ': ' + status['detail'], flush=True)

        def on_joints(msg):
            latest_joints.update(dict(zip(msg.name, msg.position)))

        def on_ibvs(msg):
            sample = json.loads(msg.data)
            ibvs_samples.append(sample)
            print(f"{sample['phase']}: {sample['error_px']:.3f} px, iteration {sample['iteration']}", flush=True)
            if 'eih' in latest_images:
                annotated = latest_images['eih'].copy()
                for point in sample['desired_corners_px']:
                    cv2.circle(annotated, tuple(np.rint(point).astype(int)), 5, (0, 200, 255), 2)
                cv2.imwrite(str(args.output/f"{sample['phase']}.png"), annotated)

        def on_overlay(msg):
            overlay['images'] += 1
            overlay['frame'] = msg.header.frame_id
            stage = status['stage']
            if stage.startswith(('pose1_ibvs', 'pose2_ibvs')):
                name = 'pose1_overlay.png' if stage.startswith('pose1') else 'pose2_overlay.png'
                cv2.imwrite(str(args.output/name), bridge.imgmsg_to_cv2(msg, 'bgr8'))

        for camera, prefix in [('overhead','/camera'), ('eih','/eih')]:
            node.create_subscription(Image, prefix+'/image',
                lambda msg,camera=camera:on_image(msg,camera), qos_profile_sensor_data)
            node.create_subscription(CameraInfo, prefix+'/camera_info',
                lambda msg,camera=camera:on_info(msg,camera), qos_profile_sensor_data)
        node.create_subscription(String, '/marker_pick/status', on_status,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        node.create_subscription(JointState, '/joint_states', on_joints, qos_profile_sensor_data)
        node.create_subscription(String, '/marker_pick/ibvs', on_ibvs, 100)
        node.create_subscription(Image, '/marker_pick/eih_debug_image', on_overlay, 1)
        started = time.monotonic()
        try:
            while process.poll() is None and time.monotonic()-started < args.timeout:
                rclpy.spin_once(node, timeout_sec=0.2)
                if status['stage'] in ('complete', 'aborted'):
                    break
        except KeyboardInterrupt:
            status.update(stage='interrupted', detail='Validation interrupted')
        finally:
            for camera, image in latest_images.items():
                cv2.imwrite(str(args.output / f'{camera}_final.png'), image)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        excursion = (float(np.max(np.linalg.norm(np.array(eih_positions)-eih_positions[0],axis=1)))
                     if eih_positions else 0.0)
        cameras_valid = all(o['image_frame'] is not None and o['image_frame'] == o['info_frame']
                            for o in observed.values())
        success = (status['stage'] == 'complete' and cameras_valid and excursion > .05
                   and overlay['images'] > 0 and overlay['frame'] == observed['eih']['image_frame']
                   and (args.target_source != 'markers' or 8 in observed['eih']['marker_ids']))
        if args.target_source == 'markers':
            success = success and all(status.get(f'pose{i}_ibvs_converged') is True
                and status.get(f'pose{i}_ibvs_final_error_px', float('inf')) <= .9 for i in (1, 2))
        (args.output/'ibvs.json').write_text(json.dumps(ibvs_samples, indent=2)+'\n')
        report = {'success':success, 'target_source':args.target_source,
                  'observation_offset_m':args.observation_offset, 'observation_angle_rad':args.observation_angle,
                  'elapsed_s':round(time.monotonic()-started, 2), **status,
                  'cameras':observed, 'wrist_camera_excursion_m':excursion,
                  'stage_joint_states':stage_joints}
        report['alignment_overlay'] = overlay
        (args.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
