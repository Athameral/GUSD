import os
import mmcv
import numpy as np
import torch
import pickle
import os.path as osp
from tqdm import tqdm
from typing import Union, Sequence
import queue
import logging
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from models.utils import sparse2dense
from .old_metrics import Metric_mIoU_Occ3D
from .utils import compose_ego2img


@DATASETS.register_module()
class NuScenesOcc3DDataset(NuScenesDataset):    
    def __init__(self, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.data_infos = self.load_annotations(self.ann_file)
    
    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def collect_lidar_sweeps(self, index, into_past=20, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        last_timestamp = self.data_infos[index]['timestamp']
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps'][::-1]
            if curr_sweeps[0]['timestamp'] == last_timestamp:
                curr_sweeps = curr_sweeps[1:]
            all_sweeps_next.extend(curr_sweeps)
            curr_index = curr_index + 1
            last_timestamp = all_sweeps_next[-1]['timestamp']

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix
        ego2lidar = transform_matrix(
            lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)

        input_dict = dict(
            sample_token=info['token'],
            scene_name=info['scene_name'],
            timestamp=info['timestamp'] / 1e6,
            ego2lidar=ego2lidar,
            ego2obj=ego2lidar,
            ego2occ=np.eye(4),
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        if self.modality['use_lidar']:
            lidar_sweeps_prev, lidar_sweeps_next = self.collect_lidar_sweeps(index)
            input_dict.update(dict(
                pts_filename=info['lidar_path'],
                lidar_sweeps={'prev': lidar_sweeps_prev, 'next': lidar_sweeps_next},
            ))

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            ego2img = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)
                ego2img.append(
                    compose_ego2img(
                        ego2global_translation,
                        ego2global_rotation_mat,
                        cam_info['sensor2global_translation'],
                        cam_info['sensor2global_rotation'].T,
                        cam_info['cam_intrinsic']
                    )
                )

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                ego2img=ego2img,
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        results_dict = {}
        results_dict.update(
            self.eval_miou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        # results_dict.update(
        #     self.eval_riou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        return results_dict
        # results_dict = {}
        # if isinstance(occ_results, queue.Queue):
        #     logging.info('Using streaming eval...')
        #     logging.info('Collecting results from the queue...')
        #     occ_results = iter(occ_results.get, None)

        # q_miou = queue.Queue(maxsize=100)
        # q_riou = queue.Queue(maxsize=100)

        # def bridge():
        #     """从原始数据源读取，扇出到两个队列"""
        #     if isinstance(occ_results, queue.Queue):
        #         source = iter(occ_results.get, None)
        #     else:
        #         source = iter(occ_results)
        #     for item in source:
        #         q_miou.put(item)
        #         q_riou.put(item)
        #     # 发送终止哨兵
        #     q_miou.put(None)
        #     q_riou.put(None)

        # def eval_miou_thread():
        #     results_dict.update(
        #         self.eval_miou(q_miou, runner=runner, show_dir=show_dir, **eval_kwargs))
        # def eval_riou_thread():
        #     results_dict.update(
        #         self.eval_riou(q_riou, runner=runner, show_dir=show_dir, **eval_kwargs))

        # t_bridge = Thread(target=bridge, daemon=True)
        # t_miou = Thread(target=eval_miou_thread)
        # t_riou = Thread(target=eval_riou_thread)

        # t_bridge.start()
        # t_miou.start()
        # t_riou.start()

        # t_bridge.join()
        # t_miou.join()
        # t_riou.join()
        # return results_dict

    def eval_miou(self, occ_results: Union[Sequence[dict], queue.Queue], runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        occ_preds = []
        lidar_origins = []

        logging.info('Starting Evaluation mIoU...')
        metric = Metric_mIoU_Occ3D(use_image_mask=True, device="cuda")

        if isinstance(occ_results, queue.Queue):
            logging.info('Using streaming eval...')
            logging.info('Collecting results from the queue...')
            occ_results = iter(occ_results.get, None)

        pbar = tqdm(enumerate(occ_results), total=len(self.data_infos), desc='eval_miou', position=0, leave=True, ncols=80)
        for i, result_dict in pbar:
            info = self.get_data_info(i)
            token = info['sample_token']
            scene_name = info['scene_name']
            occ_root = 'data/nuscenes/gts/'
            occ_file = osp.join(occ_root, scene_name, token, 'labels.npz')
            occ_infos = np.load(occ_file)

            occ_labels = occ_infos['semantics']
            mask_lidar = occ_infos['mask_lidar'].astype(np.bool_)
            mask_camera = occ_infos['mask_camera'].astype(np.bool_)

            occ_pred = result_dict['dense_occ']
            
            metric.add_batch(occ_pred, occ_labels, mask_lidar, mask_camera)
            pbar.set_postfix(mIoU=metric.compute())
        metric.log(metric.compute())
        return {'mIoU': metric.count_miou()}
    
    # def eval_riou(self, occ_results: Union[Sequence[dict], queue.Queue], runner=None, show_dir=None, **eval_kwargs):
    #     logging.info('Starting Evaluation RayIoU...')

    #     # from .ray_metrics import main as calc_rayiou
    #     from .ego_pose_dataset import EgoPoseDataset

    #     if isinstance(occ_results, queue.Queue):
    #         logging.info('Using streaming eval...')
    #         logging.info('Collecting results from the queue...')
    #         occ_results = iter(occ_results.get, None)

    #     ego_pose_dataset = EgoPoseDataset(self.data_infos)

    #     def sample_generator():
    #         for i, result_dict in enumerate(occ_results):
    #             info = self.data_infos[i]
    #             token = info['token']
    #             scene_name = info['scene_name']
    #             occ_root = 'data/nuscenes/gts/'
    #             occ_file = osp.join(occ_root, scene_name, token, 'labels.npz')
    #             occ_infos = np.load(occ_file)
    #             gt_semantics = occ_infos['semantics']

    #             _, output_origin = ego_pose_dataset[i]          # [T, 3]
    #             output_origin = output_origin.unsqueeze(0)      # [1, T, 3]

    #             dense_sem_pred = result_dict['dense_occ']

    #             yield dense_sem_pred, gt_semantics, output_origin

    #     return calc_rayiou(
    #         sample_generator(),
    #         total=len(self.data_infos),
    #         pc_range=[-40, -40, -1.0, 40, 40, 5.4],
    #         voxel_size=0.4,
    #         grid_shape=[1, 16, 200, 200],
    #     )

    def format_results(self, occ_results,submission_prefix,**kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.astype(np.uint8))
        print('\nFinished.')