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

"""Dataset for VQA"""
import os.path
from typing import Any, Callable, Optional, Tuple, List

from PIL import Image

from torchvision.datasets.vision import VisionDataset
import json
import datetime
import copy
import torch
import random
import numpy as np
import torch.nn.functional as F


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
           Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        
        if not annotation_file == None and not question_file == None:
            print('loading VQA annotations and questions into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            questions = json.load(open(question_file, 'r'))
            self.subtype = dataset["data_subtype"]
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def get_img_path(self, ques_id, ) -> str:
        data_subtype = self.questions["data_subtype"]
        img_id = self.loadImgs(ques_id)[0]
        return "{}/COCO_{}_{:0>12d}.jpg".format(data_subtype, data_subtype, img_id)

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}

        imgToQA = {ann['image_id']: [] for ann in self.questions['questions']}
        qa =  {ann['question_id']:       [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']:       [] for ann in self.questions['questions']}
        for ann in self.dataset['annotations']:
            qa[ann['question_id']] = ann
        for ques in self.questions['questions']:
            imgToQA[ques['image_id']] += [ques]
            qqa[ques['question_id']] = ques
        
        print('index created!')
         # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

        # self.imgs=imgs

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.datset['info'].items():
            print('%s: %s'%(key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                quesTypes (str array)   : get question ids for given question types
                ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds 	  = imgIds    if type(imgIds)    == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],[])
            else:
                 anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['question_id'] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds   = quesIds   if type(quesIds)   == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.questions['questions']
        else:
            if not len(quesIds) == 0:
                anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa],[])
            else:
                anns = self.questions['questions']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['image_id'] for ann in anns]
        return ids

    def loadQues(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if _isArrayLike(ids):
            return [self.qqa[id] for id in ids]
        elif type(ids) == int:
            return [self.qqa[ids]]

    def loadAns(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if _isArrayLike(ids):
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.qqa[id]["image_id"] for id in ids]
        elif type(ids) == int:
            return [self.qqa[ids]["image_id"]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann['question_id']
            print("Question: %s" %(self.qqa[quesId]['question']))
            for ans in ann['answers']:
                print("Answer %d: %s" %(ans['answer_id'], ans['answer']))
        
    def loadRes(self, resFile, quesFile=None):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        if quesFile is None:
            res.questions = copy.deepcopy(self.questions)
        else:
            res.questions = json.load(open(quesFile))

        res.dataset['info'] = copy.deepcopy(self.questions['info'])
        res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
        res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
        res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
        res.dataset['license'] = copy.deepcopy(self.questions['license'])

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        if type(resFile) == str:
            with open(resFile) as f:
                anns = json.load(f)
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        # anns    = json.load(open(resFile))
        assert type(anns) == list, 'results is not an array of objects'
        annsQuesIds = [ann['question_id'] for ann in anns]
        # assert set(annsQuesIds) == set(self.getQuesIds()), \
        assert set(annsQuesIds) == (set(annsQuesIds) & set(self.getQuesIds())), \
        'Results do not correspond to current VQA set. There is atleast one question id that does not belong to the question ids in the annotation file.'
        for ann in anns:
            quesId 			     = ann['question_id']
            if res.dataset['task_type'] == 'Multiple Choice':
                assert ann['answer'] in self.qqa[quesId]['multiple_choices'], 'predicted answer is not one of the multiple choices'
            qaAnn                = self.qa[quesId]
            ann['image_id']      = qaAnn['image_id'] 
            ann['question_type'] = qaAnn['question_type']
            ann['answer_type']   = qaAnn['answer_type']
        print('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


class CocoVQA(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        quesFile: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        seq_length:int =None, 
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        self.vqa = VQA(annFile, quesFile)
        self.ids = list(sorted(self.vqa.qqa.keys()))

        self.prompt_items = self.vqa.questions["prompt_items"]
        self.seq_length = seq_length - len(self.prompt_items[0])


    def get_img_path(self, ques_id, ) -> str:
        data_subtype = self.vqa.questions["data_subtype"]
        img_id = self.vqa.loadImgs(ques_id)[0]
        return "{}/COCO_{}_{:0>12d}.jpg".format(data_subtype, data_subtype, img_id)

    def _load_image(self, id: int) -> Image.Image:
        # img_id = self.vqa.loadImgs(id)[0]
        # data_subtype = self.vqa.questions["data_subtype"]
        # path = "{}/COCO_{}_{:0>12d}.jpg".format(
        #         data_subtype, data_subtype, img_id)
        path = self.get_img_path(id)
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        assert index < len(self), f"error {index} {len(self)}"
        id = self.ids[index]
        image = self._load_image(id)
        ques, target = self._load_QA(id)
        ques = ques[0]
        target = target[0]
        prompt = self.prompt_items[0]

        # remove eod
        ques = self.prompt_items[1][:-1] + ques[:-1] + self.prompt_items[2][:-1]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        prompt = torch.IntTensor(prompt).squeeze()
        ques = torch.IntTensor(ques).squeeze()
        target = torch.IntTensor(target).squeeze()

        # padding zero
        if target.shape[-1] >= self.seq_length:
            target = target[..., :self.seq_length]
        else:
            target = F.pad(target, (0, self.seq_length - target.shape[-1] - ques.shape[-1]), "constant", 0)
        img_id = self.vqa.loadImgs(id)[0]

        return {"prompt": prompt, "img": image, "ques": ques, "ans": target, 
                "img_id": img_id,
                "ques_id": id, "ques_len": len(ques)}

    def __len__(self) -> int:
        return len(self.ids)

    def _load_QA(self, id: int) -> List[str]:
        q_ret = self.vqa.loadQues(id)
        a_ret = self.vqa.loadAns(id)
        # print(f"q {q_ret[0].keys()}, a {a_ret[0].keys()}")
        return [ann["question"][0] for ann in q_ret], \
                [ann["multiple_choice_answer"][0] for ann in a_ret]
        