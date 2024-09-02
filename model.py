import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, qkv_bias, max_seq_len, dropout):
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