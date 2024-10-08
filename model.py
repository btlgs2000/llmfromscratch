import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Un'attenzione multi-testa. """
    def __init__(self, input_dim, output_dim, num_heads, qkv_bias, max_seq_len, dropout):
        """Inizializza l'attenzione multi-testa.

        Args:
            input_dim (int): la dimensione di input.
            output_dim (int): la dimensione di output.
            num_heads (int): numero di teste per l'attenzione.
            qkv_bias (bool): se True, aggiunge un termine bias alle matrici di proiezione di queries, keys e values.
            max_seq_len (itn): lunghezza massima delle sequenze.
            dropout (float): probabilità di dropout sulla matrice delle probabilità dell'attenzione.
        """
        super().__init__()

        # la dimensione di output deve essere divisibile per il numero di teste
        assert output_dim % num_heads == 0
        
        # dimensione di output per ogni testa
        self.head_dim = output_dim // num_heads

        self.num_heads = num_heads

        # Matrici di proiezione per queries, keys e values.
        # Calcolano le proiezioni per tutte le teste in un'unica operazione.
        # Il tensore di output ha dimensione (batch_size, seq_len, output_dim),
        # ma può essere visto come un tensore di dimensione (batch_size, seq_len, num_heads, head_dim)
        self.query_proj = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.key_proj = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.value_proj = nn.Linear(input_dim, output_dim, bias=qkv_bias)

        # Matrice di proiezione per l'output dell'attenzione
        self.output_proj = nn.Linear(output_dim, output_dim, bias=qkv_bias)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # Maschera per la causalità. Valgono True le coppie che non devono essere considerate.
        # Nella forward questi indici verranno settati a -inf in attention_scores
        self.register_buffer('mask', torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # queries, keys, values: (batch_size, num_heads, seq_len, head_dim)
        queries = self.query_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # attention_scores: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(queries, keys.transpose(2, 3))

        # normalizza gli attention scores relativamente alla dimensione di ogni testa
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        # aggiunge la causalità all'attenzione
        attention_scores.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)

        # calcola una distribuzione di probabilità per ogni query trmite la funzione softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # applica dropout
        attention_probs = self.dropout(attention_probs)

        # calcola l'output dell'attenzione come la somma pesata dei valori
        # attention_output: (batch_size, seq_len, output_dim)
        attention_output = torch.matmul(attention_probs, values).transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.output_proj(attention_output)
    
class TransformerBlock(nn.Module):
    """Un transformer block.

    E' il modulo base di un transformer. Un transformer completo è composto da più blocchi.

    """

    def __init__(self, model_dim, num_heads, qkv_bias, max_seq_len, attention_dropout, block_dropout, ff_hidden_dim):
        """Inizializza il transformer block.

        Args:
            model_dim (int): dimensione caratteristica del modello, ossia la dimensione degli embedding.
            num_heads (int): numero di teste per l'attenzione.
            qkv_bias (bool): se True, aggiunge un termine bias alle matrici di proiezione di queries, keys e values.
            max_seq_len (int): lunghezza massima delle sequenze.
            attention_dropout (float): probabilità di dropout nella matrice di attenzione.
            block_dropout (float): probabilità di dropout dopo l'attenzione e dopo il feedforward.
            ff_hidden_dim (int): dimensione dello strato nascosto del feedforward.
        """
        super().__init__()

        # layer normalization layers
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        # dropout layers
        self.dropout1 = nn.Dropout(block_dropout)
        self.dropout2 = nn.Dropout(block_dropout)

        # multi-head attention
        self.mha = MultiHeadAttention(
            model_dim, 
            model_dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            max_seq_len=max_seq_len, 
            dropout=attention_dropout)
        
        # feedforward
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_hidden_dim, model_dim))

    def forward(self, x):
        # x: (batch_size, seq_len, model_dim)

        #############################
        # attention
        #############################

        # salva l'input per la connessione residuale
        res = x
        # layer normalization
        x = self.ln1(x)
        # multi-head attention
        x = self.mha(x)
        # dropout
        x = self.dropout1(x)
        # connessione residuale
        x += res

        #############################
        # feedforward
        #############################

        # salva l'input per la connessione residuale
        res = x
        # layer normalization
        x = self.ln2(x)
        # feedforward
        x = self.ff(x)
        # dropout
        x = self.dropout2(x)
        # connessione residuale
        x += res

        return x
    
class TransformerDecoder(nn.Module):
    """Un decoder transformer.
    """

    def __init__(self, cfg):
        """Inizializza il decoder.

        Args:
            cfg (Dict): dizionario di configurazione.
        """
        super().__init__(cfg)

        # token embedding
        self.token_embedding = nn.Embedding(cfg['vocab_size'], cfg['model_dim'])
        # positional embedding
        self.positional_embedding = nn.Embedding(cfg['max_seq_len'], cfg['model_dim'])
        # transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                cfg['model_dim'], 
                cfg['num_heads'], 
                cfg['qkv_bias'], 
                cfg['max_seq_len'], 
                cfg['attention_dropout'], 
                cfg['block_dropout'], 
                cfg['ff_hidden_dim'] 
            ) for _ in range(cfg['num_blocks'])
        ])
        # layer normalization
        self.ln = nn.LayerNorm(cfg['model_dim'])
        # output layer
        self.output_layer = nn.Linear(cfg['model_dim'], cfg['vocab_size'])

    def forward(self, x):
        # x: (batch_size, seq_len)

        # token embedding: (batch_size, seq_len, model_dim)
        token_embedding = self.token_embedding(x)
        batch_size, seq_len, _ = token_embedding.size()

        # positional embedding: (1, seq_len, model_dim)
        positional_embedding = self.positional_embedding(torch.arange(seq_len, device=x.device)).unsqueeze(0)

        # somma gli embedding
        x = token_embedding + positional_embedding

        # applica i transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # layer normalization
        x = self.ln(x)

        # output layer
        x = self.output_layer(x)

        return x