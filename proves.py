from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer, T5Model
from DataLoaderVQA import SP_VQADataset

annotations_dir = '/home/jsamper/Desktop/DocVQA/Data/Annotations/train_v1.0_withQT.json'
ocr_dir = '/home/jsamper/Desktop/DocVQA/Data/OCR'
images_dir = '/home/jsamper/Desktop/DocVQA/Data/Images'
# Create an instance of the custom dataset
new_width = 1500
new_height = 2000 
reshape_transform = transforms.Compose([
    transforms.Resize((new_width, new_height)),  # Specify the new dimensions
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])

MAX_LEN_STR = 512
EOS_CHAR = '<eos>'
TOKENIZER = T5Tokenizer.from_pretrained("t5-small")
MAX_LEN_BBOX = 290
MAX_LEN_QUESTION = 80
MAX_LEN_ANSWER = 50

train_dataset = SP_VQADataset(annotations_dir, ocr_dir, images_dir, transform = reshape_transform,max_len_answer = MAX_LEN_ANSWER, max_len_question = MAX_LEN_QUESTION, max_len_bbox = MAX_LEN_BBOX, max_len_str=MAX_LEN_STR, tokenizer = TOKENIZER)

batch_size = 32
# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



for data in train_loader:#, context_txt, context_bbox, image, answer
    print(data['question'], data['context'],data['context_bbox'], data['image'], data['answer'])