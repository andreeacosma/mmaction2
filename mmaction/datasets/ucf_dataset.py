import copy
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core.evaluation.ava_utils import ava_eval, read_labelmap, results2csv
from ..utils import get_root_logger
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class UCFDataset(BaseDataset):
    """Rawframe dataset for action recognition.
      The dataset loads raw frames and apply specified transforms to return a
      dict containing the frame tensors and other information.
      The ann_file is a text file with multiple lines, and each line indicates
      the directory to frames of a video, total frames of the video and
      the label of a video, which are split with a whitespace.
      Example of a annotation file:
      .. code-block:: txt
          some/directory-1 163 1
          some/directory-2 122 1
          some/directory-3 258 2
          some/directory-4 234 2
          some/directory-5 295 3
          some/directory-6 121 3
      Example of a multi-class annotation file:
      .. code-block:: txt
          some/directory-1 163 1 3 5
          some/directory-2 122 1 2
          some/directory-3 258 2
          some/directory-4 234 2 4 6 8
          some/directory-5 295 3
          some/directory-6 121 3
      Example of a with_offset annotation file (clips from long videos), each
      line indicates the directory to frames of a video, the index of the start
      frame, total frames of the video clip and the label of a video clip, which
      are split with a whitespace.
      .. code-block:: txt
          some/directory-1 12 163 3
          some/directory-2 213 122 4
          some/directory-3 100 258 5
          some/directory-4 98 234 2
          some/directory-5 0 295 3
          some/directory-6 50 121 3
      Args:
          ann_file (str): Path to the annotation file.
          pipeline (list[dict | callable]): A sequence of data transforms.
          data_prefix (str | None): Path to a directory where videos are held.
              Default: None.
          test_mode (bool): Store True when building test or validation dataset.
              Default: False.
          filename_tmpl (str): Template for each filename.
              Default: 'img_{:05}.jpg'.
          with_offset (bool): Determines whether the offset information is in
              ann_file. Default: False.
          multi_class (bool): Determines whether it is a multi-class
              recognition dataset. Default: False.
          num_classes (int | None): Number of classes in the dataset.
              Default: None.
          modality (str): Modality of data. Support 'RGB', 'Flow'.
              Default: 'RGB'.
          sample_by_class (bool): Sampling by class, should be set `True` when
              performing inter-class data balancing. Only compatible with
              `multi_class == False`. Only applies for training. Default: False.
          power (float | None): We support sampling data with the probability
              proportional to the power of its label frequency (freq ^ power)
              when sampling data. `power == 1` indicates uniformly sampling all
              data; `power == 0` indicates uniformly sampling all classes.
              Default: None.
      """
    _FPS = 30

    def __init__(self,
                 ann_file,
                 exclude_file,
                 pipeline,
                 label_file=None,
                 filename_tmpl='{:05}.jpg',
                 proposal_file=None,
                 person_det_score_thr=0.9,
                 num_classes=81,
                 custom_classes=None,
                 data_prefix=None,
                 test_mode=False,
                 modality='RGB',
                 num_max_proposals=1000,
                 timestamp_start=900,
                 timestamp_end=1800,
                 ):
        # since it inherits from `BaseDataset`, some arguments
        # should be assigned before performing `load_annotations()`
        self.custom_classes = custom_classes

        if custom_classes is not None:
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            _, class_whitelist = read_labelmap(open(label_file))
            assert set(custom_classes).issubset(class_whitelist)

            self.custom_classes = tuple([0] + custom_classes)

        self.exclude_file = exclude_file
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr

        self.num_classes = num_classes
        self.filename_tmpl = filename_tmpl
        self.num_max_proposals = num_max_proposals
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.logger = get_root_logger()

        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            modality=modality,
            num_classes=num_classes)

        if self.proposal_file is not None:
            #self.proposals = mmcv.load(self.proposal_file)
            import pandas as pd
            self.proposals = pd.read_pickle(self.proposal_file)
            self.proposals = self.proposals['gttubes']
        else:
            self.proposals = None


        if not test_mode:
            valid_indexes = self.filter_exclude_file()
            self.logger.info(
                f'{len(valid_indexes)} out of {len(self.video_infos)} '
                f'frames are valid.')
            self.video_infos = self.video_infos = [
                self.video_infos[i] for i in valid_indexes
            ]

    def parse_img_record(self, img_records):
        bboxes, labels = [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)
            selected_records = list(
                filter(
                    lambda x: np.array_equal(x['entity_box'], img_record[
                        'entity_box']), img_records))
            num_selected_records = len(selected_records)
            img_records = list(
                filter(
                    lambda x: not np.array_equal(x['entity_box'], img_record[
                        'entity_box']), img_records))
            assert len(img_records) + num_selected_records == num_img_records

            bboxes.append(img_record['entity_box'])
            valid_labels = np.array([
                selected_record['label']
                for selected_record in selected_records
            ])

            # The format can be directly used by BCELossWithLogits
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1.

            labels.append(label)


        bboxes = np.stack(bboxes)
        labels = np.stack(labels)

        return bboxes, labels

    def filter_exclude_file(self):
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.video_infos)))
        else:
            exclude_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, video_info in enumerate(self.video_infos):
                valid_indexes.append(i)
                for video_id, timestamp in exclude_video_infos:
                    if (video_info['video_id'] == video_id
                            and video_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break
        return valid_indexes

    def load_annotations(self):
        #video_name timestamp x1 y1 x2 y2 label
        video_infos = []
        records_dict_by_img = defaultdict(list)
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(',')

                label = int(line_split[6])
                if self.custom_classes is not None:
                    if label not in self.custom_classes:
                        continue
                    label = self.custom_classes.index(label)

                video_id = line_split[0]
                timestamp = int(line_split[1])

                #for proposal file AVA: 'video_id,timestamp'
                #for ucf [video_id][label][0][timestamp]
                img_key = f'{video_id},{timestamp:04d},{label}'

                entity_box = np.array(list(map(float, line_split[2:6])))
                #entity_id = int(line_split[7])
                shot_info = (0, (self.timestamp_end - self.timestamp_start) *
                             self._FPS)

                video_info = dict(
                    video_id=video_id,
                    timestamp=timestamp,
                    entity_box=entity_box,
                    label=label,
                    #entity_id=entity_id,
                    shot_info=shot_info)
                records_dict_by_img[img_key].append(video_info)

        #total_frames = self.proposals['nframes'][video_id]
        #print("In ucf_dataset.py self.proposals: {}".format(self.proposals))

        for img_key in records_dict_by_img:
            video_id, timestamp, label = img_key.split(',')
            bboxes, labels = self.parse_img_record(
                records_dict_by_img[img_key])
            ann = dict(
                gt_bboxes=bboxes, gt_labels=labels)
            frame_dir = video_id
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                img_key=img_key,
                shot_info=shot_info,
                fps=self._FPS,
                ann=ann,
                total_frames=150
                )
            video_infos.append(video_info)



        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']

        video_id, timestamp, label = img_key.split(',')

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[video_id][int(label)-1][0][int(timestamp)]

                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']


        #print("Video_id: {}, timestamp: {}\n".format(video_id, timestamp))
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']

        video_id, timestamp, label = img_key.split(',')

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                #proposals = self.proposals[img_key]
                proposals = self.proposals[video_id][int(label) - 1][0][int(timestamp)]

                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        # Follow the mmdet variable naming style.
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']

        return self.pipeline(results)

    def dump_results(self, results, out):
        assert out.endswith('csv')
        results2csv(self, results, out, self.custom_classes)

    def evaluate(self,
                 results,
                 metrics=('mAP', ),
                 metric_options=None,
                 logger=None):
        # need to create a temp result file
        assert len(metrics) == 1 and metrics[0] == 'mAP', (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            'See https://github.com/open-mmlab/mmaction2/pull/567 '
            'for more info.')
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = f'AVA_{time_now}_result.csv'
        results2csv(self, results, temp_file, self.custom_classes)

        ret = {}
        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            eval_result = ava_eval(
                temp_file,
                metric,
                self.label_file,
                self.ann_file,
                self.exclude_file,
                custom_classes=self.custom_classes)
            log_msg = []
            for k, v in eval_result.items():
                log_msg.append(f'\n{k}\t{v: .4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            ret.update(eval_result)

        os.remove(temp_file)

        return ret
