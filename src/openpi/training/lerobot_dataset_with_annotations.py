import json
import numpy as np
import os
import torch


def _get_default_lerobot_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


def _extract_joint(state):
    return np.concatenate([
        state[..., 0:8],   # left joint + gripper
        state[..., 17:25], # right joint + gripper
    ])



class LeRobotDatasetWithAnnotations(torch.utils.data.Dataset):
    def __init__(
        self,
        lerobot_dataset
    ):
        self.lerobot_dataset = lerobot_dataset
        self.repo_id = lerobot_dataset.repo_id
    
    def __len__(self):
        return len(self.lerobot_dataset)
    
    def __getitem__(self, idx):
        item = self.lerobot_dataset[idx]
        episode_index = item['episode_index']
        frame_index = item['frame_index']
        prompt = self._parse_annotation(item, episode_index, frame_index)
        item['prompt'] = prompt
        item['observation.state'] = _extract_joint(item['observation.state'])
        item['action'] = _extract_joint(item['action'])
        return item
    
    def _parse_annotation(self, item, episode_index, frame_index):
        annotation_path = os.path.join(_get_default_lerobot_root(), self.repo_id, 'annotations', f'episode_{episode_index:06d}.json')
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        annotation = annotations[frame_index]
        prompt = 'scene: {}<wrap>task: {}<wrap>subtask: {}<wrap>movement: {} {}.'.format(
            annotation['scene_description'].split(',')[0],
            item['task'],
            annotation['subtask'],
            annotation['movement_summary_left'],
            annotation['movement_summary_right'],
        )
        return (
            prompt.replace('. ', '')
            .replace('.', '')
            .replace('a ', '')
            .replace('the ', '')
            .replace('is ', '')
            .replace('are ', '')
            .replace('<wrap>', '. ')
        )