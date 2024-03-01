import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer, T5Model
from DataLoaderVQA import SP_VQADataset
from VQAModel import ModelVT5
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration

import wandb

# Initialize WandB
wandb.init(project='SP-DocVQA', name='Basic-DocVQA-OnlyTXT')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


kwargs_dir = 'args.json'
kwargs = {
    "MAX_LEN_STR" :512,
    "TOKENIZER": T5Tokenizer.from_pretrained('t5-small'),
    "MAX_LEN_BBOX":310,
    "MAX_LEN_QUESTION":80,
    "MAX_LEN_ANSWER":50
}

phisical_dev = "PC"
path_strt = "paths.json"

annotations_dir = '/home/jsamper/Desktop/DocVQA/Data/Annotations/train_v1.0_withQT.json'
ocr_dir = '/home/jsamper/Desktop/DocVQA/Data/OCR'
images_dir = '/home/jsamper/Desktop/DocVQA/Data/Images'
# Create an instance of the custom dataset
new_width = 1400
new_height = 1980 
reshape_transform = transforms.Compose([
    transforms.Resize((new_width, new_height)),  # Specify the new dimensions
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    
]) 
#
train_dataset = SP_VQADataset(annotations_dir, ocr_dir, images_dir, transform = reshape_transform,**kwargs)#max_len_answer = MAX_LEN_ANSWER, max_len_question = MAX_LEN_QUESTION, max_len_bbox = MAX_LEN_BBOX, max_len_str=MAX_LEN_STR, tokenizer = TOKENIZER)

batch_size = 8
# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



'''for data in train_loader:#, context_txt, context_bbox, image, answer
    print(data['question'], data['context'],data['context_bbox'], data['image'], data['answer'])'''
model = ModelVT5().to(device)
tokenizer =  T5Tokenizer.from_pretrained('t5-small')
#model_gen = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

learning_rate = 0.00002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 1000

wandb.config.learning_rate = learning_rate
wandb.config.batch_size = batch_size
wandb.config.epochs = epochs
#wandb.watch(model, criterion, log='all')


for step in range(epochs):
    print('epoch:', step)
    tot_loss = 0
    model.train()
    for data in train_loader:
        question = data['question'].to(device)
        context = data['context'].to(device)#ocr tokenized ids text
        context_bbox = data['context_bbox'].to(device)
        image = data['image'].to(device)
        answer = data['answer'].to(device)

        output = model.forward(image, context, context_bbox)
        output = output.to(device)
        loss = model.model(input_ids=output, labels=answer).loss
        
        wandb.log({'loss': loss})
        print(loss)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(tot_loss)
        #out_words = tokenizer.batch_decode(output)
        #print(out_words[0])