import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len=512, dropout_prob=0.1):
        super(TransformerEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_ids):
        embeddings = self.token_embeddings(input_ids)
        embeddings = self.position_encoding(embeddings)
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout_prob)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        original_input = hidden_states
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        output = self.output_projection(context_layer)
        output = self.output_dropout(output)
        output = self.output_layer_norm(output + original_input)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states):
        residual = hidden_states
        
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.layer_norm(hidden_states + residual)
        
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, dropout_prob=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout_prob)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        output = self.feed_forward(attention_output)
        return output

class CustomTransformerEncoder(nn.Module):
    def __init__(self, 
                vocab_size,
                hidden_size=256,
                num_hidden_layers=6,
                num_attention_heads=4,
                intermediate_size=512,
                dropout_prob=0.1,
                max_seq_len=512):
        super(CustomTransformerEncoder, self).__init__()
        
        self.embeddings = TransformerEmbedding(
            vocab_size=vocab_size, 
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            dropout_prob=dropout_prob
        )
        
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                dropout_prob=dropout_prob
            ) for _ in range(num_hidden_layers)
        ])
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
        
    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids)
        
        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        pooled_output = hidden_states[:, 0]
        pooled_output = self.pooler(pooled_output)
        pooled_output = self.pooler_activation(pooled_output)
        
        return {
            'last_hidden_state': hidden_states,
            'pooled_output': pooled_output
        }

class TransformerForClassification(nn.Module):
    def __init__(self, vocab_size, num_labels=2, **kwargs):
        super(TransformerForClassification, self).__init__()
        self.transformer = CustomTransformerEncoder(vocab_size=vocab_size, **kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout_prob', 0.1))
        hidden_size = kwargs.get('hidden_size', 256)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask)
        pooled_output = outputs['pooled_output']
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs['last_hidden_state']
        }
