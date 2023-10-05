from model.transformer.tagging_transformer import TaggingTransformer
from model.embedding_model.word_embeddings import generate_word_embedding
import lightning.pytorch as pl
from data.wnut_loader import WNUTDataModule

# word_embedding, vocab = generate_word_embedding([], 128, "./word_dirs", "fast_text")
# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8

# input_vocab_size = len(vocab)
# label_size = 22
# dropout_rate = 0.1
# model = TaggingTransformer(num_layers, d_model, num_heads, dff, input_vocab_size, label_size, input_vocab_size, word_embedding, dropout_rate)

# trainer = Trainer()
# trainer.fit(model, train_dataloader, val_dataloader)


from argparse import ArgumentParser


def main(hparams):
    model = TaggingTransformer()
    trainer = pl.Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
    trainer.fit(model, datamodule=WNUTDataModule())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)