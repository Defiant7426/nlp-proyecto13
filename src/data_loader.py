from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len_article, max_len_highlight, bos_token_id, eos_token_id):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len_article = max_len_article - 2  # Reservar espacio para BOS y EOS
        self.max_len_highlight = max_len_highlight - 2
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        article_text = self.dataframe.iloc[idx]['article']
        highlight_text = self.dataframe.iloc[idx]['highlights']

        # Tokenizar y truncar artículo
        self.tokenizer.enable_truncation(max_length=self.max_len_article)
        encoded_article = self.tokenizer.encode(article_text)
        article_token_ids = encoded_article.ids

        # Tokenizar y truncar resumen
        self.tokenizer.enable_truncation(max_length=self.max_len_highlight)
        encoded_highlight = self.tokenizer.encode(highlight_text)
        highlight_token_ids = encoded_highlight.ids

        # Añadir tokens BOS/EOS y convertir a tensor
        article_tensor = torch.cat(
            (torch.tensor([self.bos_token_id]),
             torch.tensor(article_token_ids, dtype=torch.long),
             torch.tensor([self.eos_token_id]))
        )

        highlight_tensor = torch.cat(
            (torch.tensor([self.bos_token_id]),
             torch.tensor(highlight_token_ids, dtype=torch.long),
             torch.tensor([self.eos_token_id]))
        )

        return article_tensor, highlight_tensor

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=1)  # 1 = PAD_IDX
    tgt_batch_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=1)

    return src_batch_padded, tgt_batch_padded