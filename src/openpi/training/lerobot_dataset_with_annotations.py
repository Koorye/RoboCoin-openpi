import json
import os


def get_default_lerobot_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


class LeRobotDatasetWithAnnotations:
    def __init__(
        self,
        lerobot_dataset
    ):
        self.lerobot_dataset = lerobot_dataset
    
    def __getitem__(self, idx):
        item = self.lerobot_dataset[idx]
        episode_index = item['episode_index']
        frame_index = item['frame_index']
        prompt = self._parse_annotation(episode_index, frame_index)
        item['task'] = prompt
        return item
    
    def _parse_annotation(self, item, episode_index, frame_index):
        annotation_path = os.path.join(get_default_lerobot_root(), self.repo_id, 'annotations', f'episode_{episode_index:06d}.json')
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        annotation = annotations[frame_index]
        prompt = 'task: {}\nscene: {}\nsubtask: {}\nmovement left: {} right: {}'.format(
            item['task'],
            annotation['scene'],
            annotation['subtask'],
            annotation['movement_left'],
            annotation['movement_right']
        )
        return (
            prompt.replace('.', '')
            .replace('a ', '')
            .replace('the ', '')
            .replace('is ', '')
            .replace('are ', '')
        )