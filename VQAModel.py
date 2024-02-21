import torch 
import torch.nn as nn
from utils import VisualEmbeddings, SpatialEmbeddings
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration

class ModelVT5(nn.Module):
    def __init__(self):
        super(ModelVT5, self).__init__()
        self.visual_embedding = VisualEmbeddings()
        self.spatial_embedding = SpatialEmbeddings()
        self.semantic_embedding = T5EncoderModel.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.decoder_start_token_id = self.tokenizer.pad_token_id
            
    def encoder(self,image, ocr, ocr_bounding_box):
        input_ids = self.tokenizer(ocr) 
        visual_embed, visual_emb_mask = self.visual_embedding(image)
        spatial_embed = self.spatial_embedding(ocr_bounding_box)
        semantic_embed = self.semantic_embedding(input_ids)
        
        
        input_embeds = torch.add(semantic_embed, spatial_embed)
        input_embeds = torch.cat([input_embeds, visual_embed], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)
        encoder_embed = []
        return encoder_embed
    
    def decoder(self, encoder_embed):
        outputs = self.model.generate(inputs_embeds= encoder_embed,decoder_start_token_id=self.decoder_start_token_id)
        return outputs
        
    def forward(self,x):
        encoder_embed = self.encoder(x)
        output = self.decoder(encoder_embed)
        return output