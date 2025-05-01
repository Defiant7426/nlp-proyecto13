# En este archivo vamos vamos a definir el Encoder, Decoder y Seq2Seq

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,input_dim, emb_dim, hidden_dim, n_layers, dropout, pad_idx):
        """
            Constructor del Encoder.
            Args: 
                input_dim(int): tamanio del vocabulario de entrada (fuente o sorce o src).
                emb_dim (int): Dimension de los embeddings.
                hidden_dim(int): dimension de la capa oculta del LSTM.
                n_layers (int): Numero de capas del LSTM
                dropout (float): Probabilidad de dropout
                pad_idx (idx): Indice del token de padding en el vocabulario
        """
        super().__init__() # Configuraciones internas de nn.Module en el Encoder
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        """
            Procesa la secuencia fuente.
            Arg:
                src (Tensor): Secuencia de tokens de entrada [batch_size, src_len]
            Return:

        """
        embedded = self.dropout(self.embedding(src))

        pass

class Decoder(nn.Module):
    def __init__(self,output_dim, emb_dim, hidden_dim, n_layers, dropout, pad_idx):
        """
            Inicializador del Decoder.
            Args:
                output_dim(int): tamanio del vocabulario de salida.
                emb_dim(int): dimension de los embeddings.
                hidden_dim(int): dimension de la capa oculta del LSTM
                n_layers(int): Numero de capas del LSTM
                dropout (float): Probabilidad de dropout
                pad_idx: Indice del token de padding en el vocabulario
        """
        super.__init__() # Configuraciones internas del Module.nn en el Decoder

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        pass

    def forward(self, input, hidden, cell):
        """
            Procesa un paso de la decodificación:
            Arg:
                input(Tensor): Token de entrada actual [batch size]
                hidden(Tensor): Estado oculto del paso anterior [n_layers, batch_size, hidden_dim].
                cell(Tensor): Estado de la celda en el paso anterior  [n_layers, batch_size, hidden_dim].
            Return:
        """

        input = input.unsqueeze(1) # input = [batch size, 1]
        embedded = self.dropout(self.embedding(input)) # embedded = [batch size, 1, emb dim]

        prediction = None

        pass

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
            Args:
                encoder(Encoder): instancia del encoder
                decoder(Decoder): instancia del decoder
                device(torch.device): cpu o cuda 
        """
        super().__init__() # Configuraciones internas de nn.Modules en Seq2Seq

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        """
            Procesa el par de secuencias fuente y objetivo.
            Args:
                src(Tensor): secuencia fuente [batch_size, src_len].
                trg(Tensor): secuencia target [batch_size, trg_len].
                teacher_forcing_ratio (float): Probabilidad de usar teacher forcing.
            
            Return:
                output(Tensor): predicciones del decoder.
        """
        outputs = None

        return outputs