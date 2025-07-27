import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, RobertaModel

class HybridQGQAModel(nn.Module):
    def __init__(self, encoder_model='t5-base', qa_encoder_model='roberta-base', hidden_size=768):
        super(HybridQGQAModel, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(encoder_model)
        self.encoder = T5EncoderModel.from_pretrained(encoder_model)
        self.qa_encoder = RobertaModel.from_pretrained(qa_encoder_model)
        self.qa_span_predictor = nn.Linear(hidden_size, 2)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, 
            nhead=8,
            batch_first=True
        )
        self.qg_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.qg_output = nn.Linear(hidden_size, self.tokenizer.vocab_size)
        
        self.pos_embedding = nn.Embedding(512, hidden_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, qa_input_ids, qa_attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        batch_size, seq_len, hidden_dim = decoder_input_ids.shape[0], decoder_input_ids.shape[1], encoder_outputs.shape[-1]
        decoder_embeddings = torch.zeros(batch_size, seq_len, hidden_dim).to(encoder_outputs.device)
        
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(encoder_outputs.device)
        decoder_embeddings += self.pos_embedding(pos_ids)
        
        memory_key_padding_mask = ~attention_mask.bool()
        decoded = self.qg_decoder(
            decoder_embeddings, 
            encoder_outputs,
            memory_key_padding_mask=memory_key_padding_mask
        )
        qg_logits = self.qg_output(decoded)
        
        qa_outputs = self.qa_encoder(input_ids=qa_input_ids, attention_mask=qa_attention_mask)
        span_logits = self.qa_span_predictor(qa_outputs.last_hidden_state)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return qg_logits, start_logits, end_logits