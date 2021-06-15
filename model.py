import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nout, nhid, nlayers, task, dropout=0.1):
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, nout)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = nout
        self.sigmoid = torch.nn.Sigmoid()
        self.task = task

    def forward(self, input, masks):
        hidden = self.init_hidden(input.size(0))
        lengths = masks.size(1) - masks.sum(dim=1)
        input = torch.swapaxes(input, 0, 1)
        emb = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, enforce_sorted=False)
        output, hidden = self.rnn(emb, hidden)
        padded_out, lens = torch.nn.utils.rnn.pad_packed_sequence(output)
        padded_out = torch.swapaxes(padded_out, 0, 1)
        if self.task == "sequence_classification":
            return self.sigmoid(self.decoder(padded_out[torch.arange(padded_out.size(0)), lens - 1]))
        elif self.task == "sequence_learning":
            return self.decoder(padded_out[torch.arange(padded_out.size(0)), lens - 1])
        else:
            print("unknown task!")
            exit()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def count_parameters(self):
        r"""count the total number of parameters of the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ninp, nhead, nhid, nlayers, task, dropout=0.1):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.embed = nn.Linear(ninp, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid * 2, dropout)
        encoder_norm = nn.LayerNorm(nhid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, encoder_norm)
        self.ninp = ninp
        self.decoder = nn.Linear(nhid, 1)
        self.task = task
        self.sigmoid = nn.Sigmoid()
        self._reset_parameters()

    def forward(self, src, masks):

        src = self.embed(src)
        src = self.pos_encoder(torch.transpose(src, 0, 1))
        output = self.transformer_encoder(src, src_key_padding_mask=masks)
        output = self.decoder(output[0])

        if self.task == "sequence_classification":
            return self.sigmoid(output)
        elif self.task == "sequence_learning":
            return output
        else:
            print("unknown task!")
            exit()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def count_parameters(self):
        r"""count the total number of parameters of the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)