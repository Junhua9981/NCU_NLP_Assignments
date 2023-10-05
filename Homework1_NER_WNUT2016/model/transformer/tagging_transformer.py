import lightning.pytorch as pl
import torch
import torch.nn as nn
from model.transformer.encoder import Encoder
from model.embedding_model.word_embeddings import vocab_size, word_embedding

def exact_match_f1_score(true_labels, predicted_labels):
    # 將真實值和預測值轉換為PyTorch張量
    true_labels = torch.Tensor(true_labels)
    predicted_labels = torch.Tensor(predicted_labels)

    # 計算精確匹配的正確預測數量
    correct_matches = torch.sum(true_labels == predicted_labels)

    # 計算真實值中的正樣本數量和預測值中的正樣本數量
    true_positives = torch.sum(true_labels)
    predicted_positives = torch.sum(predicted_labels)

    # 計算F1-score
    precision = correct_matches / predicted_positives
    recall = correct_matches / true_positives
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score.item()  # 將F1-score轉換為Python數字


class TaggingTransformer(pl.LightningModule):
    def __init__(self, num_layers=4, d_model=128, num_heads=8, dff=512, input_vocab_size=vocab_size,
               label_size=22, pe_input=vocab_size, word_emb=word_embedding, rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                            dff=dff, input_vocab_size=input_vocab_size,
                            maximum_position_encoding=pe_input, word_emb=word_emb)

        self.final_layer = nn.Linear(d_model, label_size)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index= 0)

    def forward(self, inp, enc_padding_mask):

        enc_output = self.encoder(inp, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, label_size)

        return final_output
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        enc_padding_mask = self.create_padding_mask(inputs)
        output = self(inputs, enc_padding_mask).permute(0,2,1)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        enc_padding_mask = self.create_padding_mask(inputs)
        output = self(inputs, enc_padding_mask).permute(0,2,1)
        loss = exact_match_f1_score(target, output)
        self.log("val_extract_f1", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    # def test_step(self, batch, batch_idx):
    #     inputs = batch
    #     enc_padding_mask = self.create_padding_mask(inputs)
    #     output = self(inputs).permute(0,2,1)
    #     with open("work_dirs"+"/"+"test_predictions.txt", "w") as f:
    #         for i in range(len(output)):
    #             for j in range(len(output[i])):
    #                 f.write(str(output[i][j].item())+"\n")
    #             f.write("\n")
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch
        enc_padding_mask = self.create_padding_mask(inputs)
        predictions = self(inputs, enc_padding_mask)
        _, predicted_id = torch.max(predictions, -1)

        predicted_id *= inputs.bool().long()

        result += predicted_id.view(-1).cpu().numpy().tolist()
        self.log("test_predictions", result, on_step=True, on_epoch=True, prog_bar=True, logger=True)
  
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer