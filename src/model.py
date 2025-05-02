# En este archivo vamos vamos a definir el Encoder, Decoder y Seq2Seq

import torch
import torch.nn as nn
import random

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

        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers,
                           dropout=dropout if n_layers>1 else 0,
                           bidirectional=True, batch_first=True)
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        """
            Procesa la secuencia fuente.
            Arg:
                src (Tensor): Secuencia de tokens de entrada [batch_size, src_len]
            Return:

        """
        embedded = self.dropout(self.embedding(src))

        outputs, (hidden, cell) = self.rnn(embedded)

        hidden = hidden.permute(1, 0, 2)
        
        hidden = hidden.reshape(hidden.size(0), self.n_layers, 2 * self.hidden_dim)

        hidden = hidden.permute(1, 0, 2)

        cell = cell.permute(1, 0, 2)
        cell = cell.reshape(cell.size(0), self.n_layers, 2 * self.hidden_dim)
        cell = cell.permute(1, 0, 2)

        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))

        return outputs, hidden, cell

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

        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

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

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell

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

        assert encoder.hidden_dim == decoder.hidden_dim, \
            "Las dimensiones ocultas del encoder y decoder deben de ser iguales"
        assert encoder.n_layers == decoder.n_layers, \
            "El encoder y decoder deben de tener el mismo numero de capas"

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
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len): # Predecimos a partir del segundo token

            output, hidden, cell = self.decoder(input, hidden, cell)

            # Guardamos las predicciones en el tensor de salida
            outputs[:, t, :] = output 

            # Decidir si usar teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Obtener el token predicho con mayor probabilidad
            top1 = output.argmax(1) 

            # Si es teacher forcing, usar el token real como siguiente input
            # Si no, usar el token predicho
            input = trg[:, t] if teacher_force else top1


        return outputs