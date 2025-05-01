# En este archivo vamos vamos a definir el Encoder, Decoder y Seq2Seq

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,input_dim, emb_dim, hidden_dim, n_layers, dropout, pad_idx):
        """
            Inicializador del Encoder.
            Args: 
                input_dim(int): tamano del vocabulario de entrada (fuente o sorce o src).
                emb_dim (int): Dimension de los embeddings.
                hidden_dim(int): dimension de la capa oculta del LSTM.
                n_layers (int): Numero de capas del LSTM
                dropout (float): Probabilidad de dropout
                pad_idx (idx): Indice del token de padding en el vocabulario
        """
        super().__init__() # Configuraciones internas de nn.Module
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)
    