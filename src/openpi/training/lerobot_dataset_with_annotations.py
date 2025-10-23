"""
This module defines the LeRobotDatasetWithAnnotations class, which extends the LeRobot dataset
by incorporating textual annotations for training language-conditioned policies.

Example:
```python
from openpi.training.lerobot_dataset import LeRobotDataset
from openpi.training.lerobot_dataset_with_annotations import LeRobotDatasetWithAnnotations

base_dataset = LeRobotDataset(repo_id='lerobot/repo', split='train')
annotated_dataset = LeRobotDatasetWithAnnotations(base_dataset, use_annotation=True)
for data in annotated_dataset:
    print(data['prompt'])  # Annotated prompt for the task
``` 
"""

import json
import numpy as np
import os
import re
import torch
from typing import List, Tuple, Optional

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# Template for constructing prompts from annotations
_PROMPT_TEMPLATE = 'scene: {scene_description}. task: {task}, {subtask}. movement: {movement_summary_left} {movement_summary_right}. '

# Mapping of special symbols to unique tokens
_SPECIAL_SYMBOLS = {
    ': ': '<colon>',
    '. ': '<full_stop>',
    ', ': '<comma>',
}

# List of common stop words to be removed from prompts
_STOP_WORDS = [
    'a', 'the', 'is', 'are',
]


def _get_default_lerobot_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


def _extract_indices(state, indices):
    states = []
    for start, end in indices:
        states.append(state[..., start:end])
    return np.concatenate(states, axis=-1)


class LeRobotDatasetWithAnnotations(torch.utils.data.Dataset):
    """
    LeRobot dataset with annotations for training language-conditioned policies.
    Extends a base LeRobot dataset by incorporating textual annotations into the data samples.

    Params:
    - lerobot_dataset: Base LeRobot dataset instance.
    - use_annotation: Whether to use textual annotations for prompts. If False, uses the original task description.
    - use_indices: Optional list of (start, end) tuples to extract specific indices from state and action vectors.
    """
    
    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        use_annotation: bool = False,
        use_indices: Optional[List[Tuple[int, int]]] = None,
    ):
        self.lerobot_dataset = lerobot_dataset
        self.use_annotation = use_annotation
        self.use_indices = use_indices
    
    def __len__(self):
        return len(self.lerobot_dataset)
    
    def __getitem__(self, idx):
        item = self.lerobot_dataset[idx]
        episode_index = item['episode_index']
        frame_index = item['frame_index']

        item['prompt'] = self._parse_annotation(item, episode_index, frame_index)

        if self.use_indices is not None:
            item['observation.state'] = _extract_indices(item['observation.state'], self.use_indices)
            item['action'] = _extract_indices(item['action'], self.use_indices)

        return item
    
    def _parse_annotation(self, item, episode_index, frame_index):
        if not self.use_annotation:
            return item['task']
        
        annotation_path = os.path.join(
            _get_default_lerobot_root(), 
            self.lerobot_dataset.repo_id, 
            'annotations', 
            f'episode_{episode_index:06d}.json')
        
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        annotation = annotations[frame_index]

        return self._make_prompt(item, annotation)
    
    def _replace_special_symbols(self, text):
        for symbol, replacement in _SPECIAL_SYMBOLS.items():
            text = text.replace(symbol, replacement)
        return text
    
    def _remove_symbols(self, text):
        for symbol in [',', '.', ':', ';', '!', '?']:
            text = text.replace(symbol, ' ')
        return text
    
    def _remove_stop_words(self, text):
        if not text.startswith(' '):
            text = ' ' + text
        if not text.endswith(' '):
            text = text + ' '
        for stop_word in _STOP_WORDS:
            text = text.replace(f' {stop_word} ', ' ')
        return text.strip()

    def _recover_special_symbols(self, text):
        for symbol, replacement in _SPECIAL_SYMBOLS.items():
            text = text.replace(replacement, symbol)
        return text
    
    def _clear_text(self, text):
        for symbol in [',', '.', ':', ';', '!', '?']:
            text = text.replace(f' {symbol} ', symbol + ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')
        return text.strip()

    def _make_prompt(self, item, annotation):
        prompt = _PROMPT_TEMPLATE
        prompt = self._replace_special_symbols(prompt).lower()

        keys = re.findall(r'\{(.*?)\}', prompt)
        for key in keys:
            if key in annotation:
                value = annotation[key]
            else:
                value = item[key]
            value = self._remove_symbols(value).lower()
            if not value.startswith(' '):
                value = ' ' + value
            if not value.endswith(' '):
                value = value + ' '
            prompt = prompt.replace(f'{{{key}}}', value)

        prompt = self._remove_stop_words(prompt)
        prompt = self._recover_special_symbols(prompt)
        prompt = self._clear_text(prompt)
        return prompt