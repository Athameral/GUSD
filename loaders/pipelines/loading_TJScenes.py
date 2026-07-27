from loaders.utils import compose_ego2img


import mmcv
import numpy as np
from mmcv.runner import get_dist_info
from mmdet3d.datasets.builder import PIPELINES


import os.path as osp


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsTJScenes:
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 train_interval=[4, 8],
                 test_interval=6,
                 force_offline=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.force_offline = force_offline

        self.train_interval = train_interval
        self.test_interval = test_interval

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def _get_cam_types(self, results):
        sweeps = self._get_sweeps(results)
        prev_sweeps = sweeps.get('prev', [])
        if len(prev_sweeps) > 0 and len(prev_sweeps[0].keys()) > 0:
            return list(prev_sweeps[0].keys())

        if 'cam_types' in results and len(results['cam_types']) > 0:
            return results['cam_types']

        if 'filename' in results and len(results['filename']) > 0:
            names = [osp.basename(osp.dirname(p)) for p in results['filename'][:6]]
            if len(set(names)) == 6:
                return names

        return [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

    def _resolve_img_path(self, results, path):
        if osp.isabs(path):
            return path

        data_root = results.get('data_root', None)
        if data_root is not None:
            return osp.normpath(osp.join(data_root, path))

        return path

    def _find_sensor_key(self, sweep, sensor):
        if sensor in sweep:
            return sensor

        sensor_lower = str(sensor).lower()
        for key in sweep.keys():
            if str(key).lower() == sensor_lower:
                return key

        raise KeyError('Cannot find sensor {} in sweep keys {}.'.format(sensor, list(sweep.keys())))

    def _get_sweeps(self, results):
        if 'cam_sweeps' in results:
            return results['cam_sweeps']
        if 'sweeps' in results:
            return results['sweeps']
        return {'prev': [], 'next': []}

    def _compose_ego2img_from_sweep(self, results, sweep_cam):
        if 'sensor2global_translation' in sweep_cam and 'sensor2global_rotation' in sweep_cam:
            return compose_ego2img(
                results['ego2global_translation'],
                results['ego2global_rotation'],
                sweep_cam['sensor2global_translation'],
                sweep_cam['sensor2global_rotation'].T,
                sweep_cam['cam_intrinsic'],
            )

        if 'sensor2lidar_rotation' in sweep_cam and 'sensor2lidar_translation' in sweep_cam:
            sensor2lidar = np.eye(4, dtype=np.float32)
            sensor2lidar[:3, :3] = np.asarray(sweep_cam['sensor2lidar_rotation'], dtype=np.float32)
            sensor2lidar[:3, 3] = np.asarray(sweep_cam['sensor2lidar_translation'], dtype=np.float32)
            lidar2cam = np.linalg.inv(sensor2lidar)

            intrinsic = np.asarray(sweep_cam['cam_intrinsic'], dtype=np.float32)
            viewpad = np.eye(4, dtype=np.float32)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            return viewpad @ lidar2cam

        raise KeyError('Sweep camera dict must contain sensor2global_* or sensor2lidar_* fields.')

    def _append_identity_copy(self, results, cam_types):
        for _ in range(self.sweeps_num):
            for j in range(len(cam_types)):
                results['img'].append(results['img'][j])
                results['img_timestamp'].append(results['img_timestamp'][j])
                results['filename'].append(results['filename'][j])
                results['ego2img'].append(np.copy(results['ego2img'][j]))

    def _choices(self, num_prev):
        if self.test_mode:
            interval = self.test_interval
            return [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

        if num_prev <= self.sweeps_num:
            pad_len = self.sweeps_num - num_prev
            return list(range(num_prev)) + [num_prev - 1] * pad_len

        max_interval = num_prev // self.sweeps_num
        max_interval = min(max_interval, self.train_interval[1])
        min_interval = min(max_interval, self.train_interval[0])
        interval = np.random.randint(min_interval, max_interval + 1)
        return [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

    def load_offline(self, results):
        cam_types = self._get_cam_types(results)
        sweeps = self._get_sweeps(results)
        prev_sweeps = sweeps.get('prev', [])

        if len(prev_sweeps) == 0:
            self._append_identity_copy(results, cam_types)
            return results

        for idx in sorted(self._choices(len(prev_sweeps))):
            sweep_idx = min(idx, len(prev_sweeps) - 1)
            sweep = prev_sweeps[sweep_idx]

            if len(sweep.keys()) < len(cam_types) and sweep_idx > 0:
                sweep = prev_sweeps[sweep_idx - 1]

            for sensor in cam_types:
                sensor_key = self._find_sensor_key(sweep, sensor)
                sweep_cam = sweep[sensor_key]
                img_path = self._resolve_img_path(results, sweep_cam['data_path'])
                results['img'].append(mmcv.imread(img_path, self.color_type))
                results['img_timestamp'].append(sweep_cam['timestamp'] / 1e6)
                results['filename'].append(img_path)
                results['ego2img'].append(self._compose_ego2img_from_sweep(results, sweep_cam))

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval % 6 == 0

        cam_types = self._get_cam_types(results)
        sweeps = self._get_sweeps(results)
        prev_sweeps = sweeps.get('prev', [])

        if len(prev_sweeps) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
            return results

        interval = self.test_interval
        choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

        for idx in sorted(choices):
            sweep_idx = min(idx, len(prev_sweeps) - 1)
            sweep = prev_sweeps[sweep_idx]

            if len(sweep.keys()) < len(cam_types) and sweep_idx > 0:
                sweep = prev_sweeps[sweep_idx - 1]

            for sensor in cam_types:
                sensor_key = self._find_sensor_key(sweep, sensor)
                sweep_cam = sweep[sensor_key]
                img_path = self._resolve_img_path(results, sweep_cam['data_path'])
                results['img_timestamp'].append(sweep_cam['timestamp'] / 1e6)
                results['filename'].append(img_path)
                results['ego2img'].append(self._compose_ego2img_from_sweep(results, sweep_cam))

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode and (not self.force_offline):
            return self.load_online(results)
        return self.load_offline(results)


@PIPELINES.register_module()
class LoadOcc3DFromFileTJScenes:

    def __init__(self, occ_root, ignore_class_names=[]):
        self.occ_root = occ_root
        self.ignore_class_names = ignore_class_names
        self.occ_class_names = [
            'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation', 'free'
        ]

    def __call__(self, results):
        scene_name, sample_token = results.get('scene_name', None), results.get('sample_token', None)
        occ_file = None

        # Prefer per-sample annotation path when available (e.g. TJScenes).
        if 'occ_gt_path' in results and results['occ_gt_path'] is not None:
            if osp.isabs(results['occ_gt_path']):
                occ_file = results['occ_gt_path']
            elif osp.exists(results['occ_gt_path']):
                occ_file = results['occ_gt_path']
            else:
                occ_file = osp.join(self.occ_root, results['occ_gt_path'])

        if occ_file is None:
            occ_file = osp.join(self.occ_root, scene_name, sample_token, 'labels.npz')

        # load lidar and camera visible label
        occ_labels = np.load(occ_file)

        if "semantics" in occ_labels.keys():
            semantics = occ_labels['semantics']
        else:
            semantics = occ_labels[list(occ_labels.keys())[0]]  # [200, 200, 16]
        
        semantics[semantics > 17] = 0  # map > 17 to unknowns

        mask_lidar = occ_labels.get('mask_lidar', np.ones_like(semantics, dtype=np.bool_))  # [200, 200, 16]
        mask_camera = occ_labels.get('mask_camera', np.ones_like(semantics, dtype=np.bool_))  # [200, 200, 16]
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        for class_id in range(len(self.occ_class_names) - 1):
            mask = semantics == class_id
            if mask.sum() == 0:
                continue
            if self.occ_class_names[class_id] in self.ignore_class_names:
                semantics[mask] = len(self.occ_class_names) - 1
        results['voxel_semantics'] = semantics
        return results