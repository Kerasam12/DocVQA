import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class SP_VQADataset(Dataset):
    def __init__(self, annotations_dir, ocr_dir, images_dir, transform):
        # Initialize the ColorizationDataset class with the specified root directory and transformation
        self.annotations_dir = annotations_dir
        self.ocr_dir = ocr_dir
        self.images_dir = images_dir
        self.transform = transform
        #self.transform = transform
        # Get a list of image files in the root directory
        self.ocr_files = [f for f in os.listdir(ocr_dir) if os.path.isfile(os.path.join(ocr_dir, f))]
        self.image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    
    def __len__(self):
        # Return the length of the dataset (number of image files)
        
        with open(self.annotations_dir,'r') as annotations:
            
            ann = json.load(annotations)
            
            return len(ann['data'])

    def __getitem__(self, idx):
        # Get the image at the specified index
        with open(self.annotations_dir) as ann:
            annotations = json.load(ann)
            annotations_data = annotations['data'][idx]
            
            image_name = annotations_data['image'][10:]#Pick the image directory and eliminate the directory associated and only keep the image name 
            image_path = os.path.join(self.images_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            ocr_name = image_name[:-3] + 'json'#Erase the 'png' part and replace it with 'json'
            ocr_path = os.path.join(self.ocr_dir, ocr_name)
            ocr_route = open(ocr_path)
            ocr = json.load(ocr_route)
             
            
            #ocr_list = ocr['recognitionResults'][0]['lines']
            #data = ocr['recognitionResults'][0]['lines']
            question, questionId = self.get_questions(annotations_data)
            context,context_bbox,context_txt = self.process_ocr(ocr)
            answer, start_answ_idx, end_answ_idx = self.get_start_end_answer_idx(context, annotations_data)
        print(len(question), len(context_txt), context_bbox.shape, image.size(), len(answer))   
        return question, context_txt#, context_bbox, image, answer
    
    def process_ocr(self, ocr):
        context = [txt['text'] for txt in ocr['recognitionResults'][0]['lines']]#get all the text in the image by sentences recognized by the OCR
        context_bbox = []
        context_txt = []
        padding_bbox = torch.tensor([[0,0], [0,0], [0,0], [0,0]])
        max_bb = 250
        
        
        
        data = ocr['recognitionResults'][0]['lines']
        #for data in ocr:
        for d in data:
            
            context_bbox.append(d['boundingBox'])#get all the bounding boxes of the text in the image by words 

            context_txt.append(d['text'])#get all the text in the image by words 
            
        pad_len = max_bb - len(context_bbox) 
        expanded_bbox = padding_bbox.unsqueeze(0).expand(pad_len, -1, -1)
        
             
        context_bbox = torch.tensor(context_bbox).reshape((len(context_bbox),4,2))
        
        # Concatenate along the first dimension
        context_bbox = torch.cat([context_bbox, expanded_bbox], dim=0)   
        
        return context,context_bbox,context_txt

    def get_questions(self, annotations_data):
        question = annotations_data['question']
        questionId = annotations_data['questionId']
        return question, questionId
    
    def get_start_end_answer_idx(self, context, annotations_data):
        answers = annotations_data['answers']
        context_joined = "".join(context)
        answer_positions = []
        for answer in answers:
            start_idx = context_joined.find(answer)

            if start_idx != -1:
                end_idx = start_idx + len(answer)
                answer_positions.append([start_idx, end_idx])

        if len(answer_positions) > 0:
            start_idx, end_idx = random.choice(answer_positions)  # If both answers are in the context. Choose one randomly.
            answer = context_joined[start_idx: end_idx]
        else:
            start_idx, end_idx = 0, 0  # If the indices are out of the sequence length they are ignored. Therefore, we set them as a very big number.

        return answer, start_idx, end_idx