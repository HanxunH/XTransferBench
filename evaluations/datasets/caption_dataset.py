import json
import os
from PIL import Image
from torch.utils.data import Dataset
import re


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        if dataset_name == 'coco':
            annotations = json.load(open(annotations_path))
            images_captions = {}
            not_in_counter, in_counter = 0, 0
            for item in annotations["images"]:
                filename = os.path.join(self.image_train_dir_path, item['file_name']) if self.is_train else os.path.join(self.image_val_dir_path, item['file_name'])
                if not os.path.exists(filename):
                    not_in_counter += 1
                    continue
                else:
                    images_captions[item['id']] = {
                        'file_name': item['file_name']
                    }
                    in_counter+=1
            print(annotations_path, not_in_counter, in_counter, flush=True)
            for item in annotations["annotations"]:
                if item['image_id'] in images_captions:
                    images_captions[item['image_id']]['caption'] = item['caption']
                    images_captions[item['image_id']]['image_id'] = item['image_id']
    
            self.annotations = [images_captions[key] for key in images_captions]
            print(len(self.annotations), flush=True)
        elif dataset_name == 'flickr':
            self.annotations = json.load(open(annotations_path))['annotations']                
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["file_name"]
                )
                if self.is_train
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["file_name"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, 'flickr30k-images', self.annotations[idx]["image_id"]+'.jpg'
                )
            )
        image.load()
        caption = self.annotations[idx]["caption"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["image_id"]
        }



class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        return results
    

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


def pre_captions(captions, max_words):
    caption_list = []
    for caption in captions:
        caption_list.append(pre_caption(caption, max_words))
    return caption_list


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=77):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        if type(self.ann[0]['caption']) == list:
            txt_id = 0
            for img_id, ann in enumerate(self.ann):
                self.image.append(ann['image'])
                self.img2txt[img_id] = []
                for i, caption in enumerate(ann['caption']):
                    self.text.append(pre_caption(caption, self.max_words))
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1
        # elif 'image_id' not in self.ann[0].keys():
        #     image_set = set()
        #     img_id = -1
        #     for txt_id, ann in enumerate(self.ann):
        #         self.text.append(pre_caption(ann['caption'], self.max_words))
        #         if ann['image'] not in image_set:
        #             img_id += 1
        #             image_set.add(ann['image'])
        #             self.image.append(ann['image'])
        #             self.img2txt[img_id] = []
        #         self.img2txt[img_id].append(txt_id)
        #         self.txt2img[txt_id] = img_id
        else:
            for txt_id, ann in enumerate(self.ann):
                self.text.append(pre_caption(ann['caption'], self.max_words))
                img_id = ann['image_id']
                if img_id not in self.img2txt.keys():
                    self.img2txt[img_id] = []
                    self.image.append(ann['image'])
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                                    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        # image = self.transform(image)  
        return image, self.text[index], index