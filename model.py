import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout_rate=0.1):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, title, text):
        # Encoder
        embedded_title = self.dropout(self.embedding(title))
        encoder_outputs, (hidden, cell) = self.encoder(embedded_title)

        # Decoder (input previous token and hidden state)
        embedded_text = self.embedding(text)
        decoder_outputs, _ = self.decoder(embedded_text, (hidden, cell))

        # Output layer
        outputs = self.fc(decoder_outputs)
        return outputs
