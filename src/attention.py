"""

    ¿Que es la atencion de Bandanau?

        Como habiamos visto antes, un modelo de seq2seq se encuentra compuesto por un encoder que procesa
        la secuencia de entrada lo comprime y lo convierte en un vector de contexto y esto es procesado por 
        un decoder que genera la salida. Pero el decoder solo "ve" por unica vez lo que genero el ultimo paso del 
        encoder por lo que el decoder debe de "recordar" toda la secuencia. La ATENCION viene a resolver 
        esto haciendo que el decoder pueda "mirar" todo los pasos generados por el encoder. 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,enc_hid_dim, dec_hid_dim):
        """
            Constructor de la capa de atencion:
            Arg:  
                enc_hid_dim (int): Dimension oculta del encoder (BiLSTM).
                dec_hid_dim (int): Dimension oculta del decoder (LSTM).
        """
        super().__init__()

        # Capa lineal para transformar el estado oculto del encoder
        self.attn_W_enc = nn.Linear(enc_hid_dim * 2, dec_hid_dim, bias=False)
        # Capa lineal para transformar el estado oculto del decoder
        self.attn_W_dec = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        # Capa lineal para calcular el score final
        self.attn_v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
            Calcula los pesos de atencion y el vector de contexto
            Arg:
                decoder_hidden (Tensor): Estado oculto del decoder del paso anterior.
                encoder_outputs (Tensor): Salidas de todos los pasos de tiempo del encoder.
            
            Returns:
            context_vector (Tensor): Vector de contexto calculado.
            attention_weights (Tensor): Pesos de atencion calculados.
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        decoder_hidden_repeated = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn_W_enc(encoder_outputs) + self.attn_W_dec(decoder_hidden_repeated))

        attention_scores = self.attn_v(energy).squeeze(2)

        attention_weights = F.softmax(attention_scores, dim=1)

        attention_weights_unsqueezed = attention_weights.unsqueeze(1)

        context_vector = torch.bmm(attention_weights_unsqueezed, encoder_outputs)

        context_vector = context_vector.squeeze(1)

        return context_vector, attention_weights


