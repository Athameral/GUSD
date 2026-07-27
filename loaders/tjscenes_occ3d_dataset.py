import os
import mmcv
import numpy as np
import os.path as osp
from typing import Sequence
from tqdm import tqdm
import logging
import queue
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from models.utils import sparse2dense
from .old_metrics import Metric_mIoU_Occ3D


@DATASETS.register_module()
class TJScenesOcc3DDataset(NuScenesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.data_infos = self.load_annotations(self.ann_file)
        self._upgrade_paths_inplace()

    def _to_abs_path(self, path):
        if path is None:
            return None
        path = str(path)
        if osp.isabs(path):
            return path
        return osp.abspath(osp.join(self.data_root, path))

    def _upgrade_paths_inplace(self):
        for info in self.data_infos:
            if info.get('occ_gt_path', None) is not None:
                info['occ_gt_path'] = self._to_abs_path(info['occ_gt_path'])

            if info.get('lidar_path', None) is not None:
                info['lidar_path'] = self._to_abs_path(info['lidar_path'])

            cams = info.get('cams', {})
            for _, cam_info in cams.items():
                if cam_info.get('data_path', None) is not None:
                    cam_info['data_path'] = self._to_abs_path(cam_info['data_path'])

    def _to_rotation_matrix(self, rotation):
        rotation = np.asarray(rotation)
        if rotation.shape == (3, 3):
            return rotation.astype(np.float32)
        if rotation.shape == (4,):
            return Quaternion(rotation.tolist()).rotation_matrix.astype(np.float32)
        raise ValueError('Unsupported rotation format, expected (3, 3) matrix or (4,) quaternion.')

    def _is_same_scene(self, idx_a, idx_b):
        if idx_b < 0 or idx_b >= len(self.data_infos):
            return False

        info_a = self.data_infos[idx_a]
        info_b = self.data_infos[idx_b]

        if 'tj_scene_name' in info_a and 'tj_scene_name' in info_b:
            return info_a['tj_scene_name'] == info_b['tj_scene_name']
        if 'scene_name' in info_a and 'scene_name' in info_b:
            return info_a['scene_name'] == info_b['scene_name']

        token_a = str(info_a.get('token', ''))
        token_b = str(info_b.get('token', ''))
        return token_a.split('/')[0] == token_b.split('/')[0]

    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index - 1
        while len(all_sweeps_prev) < into_past and self._is_same_scene(index, curr_index):
            all_sweeps_prev.append(self.data_infos[curr_index]['cams'])
            curr_index -= 1

        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future and self._is_same_scene(index, curr_index):
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index += 1

        return all_sweeps_prev, all_sweeps_next

    def _compute_ego2img(self, cam_info):
        sensor2lidar = np.eye(4, dtype=np.float32)
        sensor2lidar[:3, :3] = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float32)
        sensor2lidar[:3, 3] = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float32)
        lidar2cam = np.linalg.inv(sensor2lidar)

        intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        return viewpad @ lidar2cam

    def _resolve_occ_path(self, info):
        occ_path = info.get('occ_gt_path')
        if occ_path is None:
            return None
        return self._to_abs_path(occ_path)

    def _ordered_cam_items(self, cams):
        preferred = [
            'cam_front', 'cam_front_right', 'cam_front_left',
            'cam_back', 'cam_back_left', 'cam_back_right'
        ]
        rank = {name: idx for idx, name in enumerate(preferred)}
        ordered_keys = sorted(cams.keys(), key=lambda k: (rank.get(str(k).lower(), 999), str(k)))
        return [(k, cams[k]) for k in ordered_keys]

    def get_data_info(self, index):
        info = self.data_infos[index]

        ego2global_translation = np.asarray(info['ego2global_translation'], dtype=np.float32)
        ego2global_rotation_mat = self._to_rotation_matrix(info['ego2global_rotation'])
        lidar2ego_translation = np.asarray(info['lidar2ego_translation'], dtype=np.float32)
        lidar2ego_rotation_mat = self._to_rotation_matrix(info['lidar2ego_rotation'])

        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = lidar2ego_rotation_mat
        lidar2ego[:3, 3] = lidar2ego_translation
        ego2lidar = np.linalg.inv(lidar2ego)

        scene_name = info.get('scene_name', info.get('tj_scene_name', 'unknown_scene'))
        input_dict = dict(
            sample_token=info['token'],
            scene_name=scene_name,
            data_root=self.data_root,
            timestamp=info['timestamp'] / 1e6,
            ego2lidar=ego2lidar,
            ego2obj=ego2lidar,
            ego2occ=np.eye(4, dtype=np.float32),
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
            occ_gt_path=self._resolve_occ_path(info),
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            ego2img = []

            for _, cam_info in self._ordered_cam_items(info['cams']):
                img_paths.append(cam_info['data_path'])
                img_timestamps.append(cam_info['timestamp'] / 1e6)
                ego2img.append(self._compute_ego2img(cam_info))

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)
            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                ego2img=ego2img,
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
            ))

        if self.modality['use_lidar']:
            input_dict.update(dict(
                pts_filename=info.get('lidar_path', None),
                lidar_sweeps={'prev': info.get('lidar_sweeps', []), 'next': []},
            ))

        # if not self.test_mode:
        #     annos = self.get_ann_info(index)
        #     input_dict['ann_info'] = annos

        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        return self.eval_miou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs)

    def eval_miou(self, occ_results: Sequence[dict], runner=None, show_dir=None, offset=0, **eval_kwargs):
        # print('\nStarting Evaluation...')
        logging.info(f'Starting eval_miou (offset={offset})...')
        metric = Metric_mIoU_Occ3D(use_image_mask=True, device="cuda")

        if isinstance(occ_results, queue.Queue):
            logging.info('Using streaming eval...')
            logging.info('Collecting results from the queue...')
            occ_results = iter(occ_results.get, None)

        num_skipped = 0
        pbar = tqdm(enumerate(occ_results), total=len(self.data_infos), desc='eval_miou', position=0, leave=True, ncols=80)
        for i, result_dict in pbar:
            gt_idx = i + offset

            # skip if gt index is out of bounds
            if gt_idx < 0 or gt_idx >= len(self.data_infos):
                num_skipped += 1
                continue

            # skip if prediction and gt are from different scenes
            if offset != 0 and not self._is_same_scene(i, gt_idx):
                num_skipped += 1
                continue

            info = self.data_infos[gt_idx]
            occ_file = self._resolve_occ_path(info)
            if occ_file is None or (not osp.exists(occ_file)):
                raise FileNotFoundError('Cannot find occ gt file for index {} (pred idx={}, offset={}): {}'.format(gt_idx, i, offset, occ_file))

            occ_infos = np.load(occ_file)
            if "semantics" in occ_infos.keys():
                occ_labels = occ_infos['semantics']
            else:
                occ_labels = occ_infos[list(occ_infos.keys())[0]]
            occ_labels[occ_labels > 17] = 0
            mask_lidar = occ_infos['mask_lidar'].astype(np.bool_) if 'mask_lidar' in occ_infos else np.ones_like(occ_labels, dtype=np.bool_)
            mask_camera = occ_infos['mask_camera'].astype(np.bool_) if 'mask_camera' in occ_infos else np.ones_like(occ_labels, dtype=np.bool_)

            occ_pred = result_dict['dense_occ']

            metric.add_batch(occ_pred, occ_labels, mask_lidar, mask_camera)
            pbar.set_postfix(mIoU=metric.compute(), skip=num_skipped)

        if num_skipped > 0:
            logging.info(f'eval_miou: skipped {num_skipped} samples (offset={offset})')
        metric.log(metric.compute())
        return {'mIoU': metric.count_miou()}

    def eval_riou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        # Reserved for future use in TJScenes. Keep this method for API compatibility.
        return {}

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        print('\nFinished.')