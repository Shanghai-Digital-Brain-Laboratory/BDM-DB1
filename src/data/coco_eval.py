# Copyright 2022 Digital Brain Laboratory
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Part of the code here has been copied from:
  https://github.com/OFA-Sys/OFA/blob/main/run_scripts/caption/coco_eval.py
with some modifications.
"""

import json
import sys
import os.path as op

from pycocotools.coco import COCO
from mycocoevalcap.eval import COCOEvalCap

from src.data.vqa_dataset import VQA
from src.evaluation.vqaEval import VQAEval

_eval_coco = None
_eval_cap = None

_eval_vqa = None
_eval_vqa_eval = None


def create_coco_caption_evaluator(label_file):
    global _eval_coco, _eval_cap
    _eval_coco = COCO(label_file)


def create_coco_vqa_evaluator(vqa):
    global _eval_vqa, _eval_vqa_eval
    _eval_vqa = vqa


def get_img_path(
    ques_id,
) -> str:
    global _eval_vqa, _eval_vqa_eval
    return _eval_vqa.get_img_path(ques_id)


def evaluate_on_coco_caption(res_file, skip_metrics=None, outfile=None):
    """
    res_file: txt file, each row is [image_key, json format list of captions].
             Each caption is a dict, with fields "caption", "conf".
    label_file: JSON file of ground truth captions in COCO format.
    """
    global _eval_coco, _eval_cap

    # if _eval_coco is None:
    # _eval_coco = COCO(label_file)
    coco = _eval_coco
    cocoRes = coco.loadRes(res_file)

    if _eval_cap is None:
        _eval_cap = COCOEvalCap(coco, cocoRes, skip_metrics)
        cocoEval = _eval_cap
    else:
        cocoEval = _eval_cap
        cocoEval.reset(coco, cocoRes)
        cocoEval.reset_skip_metrics(skip_metrics)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params["image_id"] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, "w") as fp:
            json.dump(result, fp, indent=4)
    return result


def evaluate_on_vqa_v2(res_file, outfile=None):
    global _eval_vqa, _eval_vqa_eval
    vqa = _eval_vqa
    vqaRes = vqa.loadRes(res_file)

    # create vqaEval object by taking vqa and vqaRes
    if _eval_vqa_eval is None:
        # n is precision of accuracy (number of places after decimal), default is 3
        vqaEval = VQAEval(vqa, vqaRes, n=3)
        _eval_vqa_eval = vqaEval
    else:
        vqaEval = _eval_vqa_eval
        vqaEval.reset(vqa, vqaRes)

    quesIds = [ann["question_id"] for ann in vqaRes.dataset["annotations"]]

    # evaluate results
    vqaEval.evaluate(quesIds)
    result = vqaEval.accuracy
    if not outfile:
        print(result)
    else:
        with open(outfile, "w") as fp:
            json.dump(result, fp, indent=4)

    # overall, perQuestionType, perAnswerType
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate_on_coco_caption(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        evaluate_on_coco_caption(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise NotImplementedError
