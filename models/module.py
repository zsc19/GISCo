import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import graph_adj

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
class QKVAttention(nn.Module):
    
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-1, -2)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class EnSelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(EnSelfAttention, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x


class Attention(nn.Module):
    def __init__(self, dropout_rate):
        super(Attention, self).__init__()
    
    def forward(self, input_query, input_key, input_value):
        score_tensor = F.softmax(torch.matmul(
            input_query,
            input_key.transpose(-2, -1)
        ), dim=-1)
        forced_tensor = torch.matmul(score_tensor, input_value)
        return forced_tensor 

class GraphConvolution(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hw_linear = nn.Linear(input_dim, output_dim)

    def forward(self, input_feature, adjacency):
        ah = torch.matmul(adjacency, input_feature)
        output = self.hw_linear(ah)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.input_dim) + '->' + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    def __init__(self, args, input_dim, output_dim, dropout_rate):
        super(GcnNet, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.conv1 = GraphConvolution(self.args, self.input_dim, self.output_dim)

    def forward(self, node_features, adjacency):
        node_features = self.conv1(node_features, adjacency)
        node_features = F.dropout(node_features, self.dropout_rate, training=self.training)
        output = node_features
        return output


class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(BiLSTMEncoder, self).__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        dropout_text = self.__dropout_layer(embedded_text)
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)
        return padded_hiddens


class TaskSharedEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__args = args

        self.__attention = EnSelfAttention(
            self.__args.word_embedding_dim,
            self.__args.self_attention_hidden_dim,
            self.__args.self_attention_output_dim,
            self.__args.dropout_rate
        )

        self.__encoder = BiLSTMEncoder(
            self.__args.word_embedding_dim,
            self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )

    def forward(self, word_tensor, seq_lens):
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        return hiddens


class IntraUtteranceAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(IntraUtteranceAttention, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__w1_layer = nn.Linear(self.__input_dim, self.__hidden_dim, bias=False)
        self.__w2_layer = nn.Linear(self.__hidden_dim, self.__output_dim, bias=False)

    def forward(self, input_x, seq_lens):
        input_x = self.__dropout_layer(input_x)
        o_w1 = self.__w1_layer(input_x)
        o_w1 = F.tanh(o_w1) 
        o_w2 = self.__w2_layer(o_w1)
        value = F.softmax(o_w2, dim=-1)
        attention_x = torch.matmul(value.transpose(-1, -2), input_x)

        return attention_x


class SlotDecoderBlock(nn.Module):
    def __init__(self, args, hidden_size):
        super(SlotDecoderBlock, self).__init__()
        self.__args = args

        self.dense_in = nn.Linear(hidden_size * 6, hidden_size)
        self.act_fun = nn.ReLU()
        self.dense_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.__args.dropout_rate)

    def forward(self, intent_tensor, slot_tensor):
        cat_tensor = torch.cat([intent_tensor, slot_tensor], dim=2)
        batch_size, max_length, hidden_size = cat_tensor.size()
        h_pad = torch.zeros(batch_size, 1, hidden_size)
        if self.__args.gpu and torch.cuda.is_available():
            h_pad = h_pad.cuda()
        h_left = torch.cat([h_pad, cat_tensor[:, :max_length - 1, :]], dim=1)
        h_right = torch.cat([cat_tensor[:, 1:, :], h_pad], dim=1)
        cat_tensor = torch.cat([cat_tensor, h_left, h_right], dim=2)

        cat_tensor = self.dense_in(cat_tensor)
        cat_tensor = self.act_fun(cat_tensor)
        cat_tensor =self.dense_out(cat_tensor)
        cat_tensor = self.dropout(cat_tensor)
        slot_tensor_new = cat_tensor + slot_tensor
        return slot_tensor_new

class RAN_layer(nn.Module):
    def __init__(self, args):
        super(RAN_layer, self).__init__()
        self.__args = args
        self.__attention_1 = Attention(self.__args.dropout_rate)
        self.__attention_2 = Attention(self.__args.dropout_rate)
        self.__norm_1 = nn.LayerNorm(384) # LayerNorm(d_model=384)
        self.__norm_2 = nn.LayerNorm(384) # LayerNorm(d_model=384)
        self.__norm_3 = nn.LayerNorm(384) # LayerNorm(d_model=384)
        self.__r_ffn = nn.Sequential(
            nn.Linear(384,384),
            nn.Dropout(self.__args.slot_decoder_dropout_rate),
            nn.LeakyReLU(args.alpha),
            nn.Linear(384,384)
        )

    def forward(self, S, R, I):
        S_ = self.__attention_1(S, R, I)
        S = self.__norm_1(S + S_)
        I_ = self.__attention_2(I, R, S)
        I = self.__norm_2(I + I_)
        R = S + I
        R_ = self.__r_ffn(R)
        R = self.__norm_3(R + R_)
        return S, R, I

class ResultAttentionNetwork(nn.Module):
    def __init__(self, args):
        super(ResultAttentionNetwork, self).__init__()
        self.layers = nn.ModuleList([RAN_layer(args) for _ in range(3)])

    def forward(self, S, R, I):
        for layer in self.layers:
            S, R, I = layer(S, R, I)

        return S, R, I
class ModelManager(nn.Module):
    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # word embedding
        self.__embedding = nn.Embedding(self.__num_word, self.__args.word_embedding_dim)
        # task-shared: self-attentive encoder
        self.__text_encoder = TaskSharedEncoder(args)
        # task-specific encoder
        self.__text_lstm_intent = BiLSTMEncoder(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                                 self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                                 self.__args.dropout_rate)
        self.__text_lstm_slot = BiLSTMEncoder(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                                 self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                                 self.__args.dropout_rate) 

        # # intra-corpus label embedding is updated during training
        # # intent label embedding
        # self.__intent_embedding = nn.Parameter(torch.FloatTensor(self.__num_intent, self.__args.intent_embedding_dim))
        # nn.init.normal_(self.__intent_embedding.data)
        # # slot label embedding
        # self.__slot_embedding = nn.Parameter(torch.FloatTensor(self.__num_slot, self.__args.slot_embedding_dim))
        # nn.init.normal_(self.__slot_embedding.data)
        #
        # # intra-utterance attention for intent and slot label representation
        # self.__intent_attention = IntraUtteranceAttention(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
        #                                        self.__args.self_attention_hidden_dim,
        #                                        self.__num_intent,
        #                                        self.__args.dropout_rate)
        # self.__slot_attention = IntraUtteranceAttention(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
        #                                        self.__args.self_attention_hidden_dim,
        #                                        self.__num_slot,
        #                                        self.__args.dropout_rate)
        #
        # # intent-slot co-occurrence gcn
        # self.graph_Adj = graph_adj.get_graph_adj(self.__args)
        # if torch.cuda.is_available():
        #     self.graph_Adj = self.graph_Adj.cuda()
        # self.graph_Adj.requires_grad = False
        # self.__graph_gcn = GcnNet(self.__args,
        #                           self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
        #                           self.__args.gcn_output_dim,
        #                           self.__args.gcn_dropout_rate)
        #
        # # intent and slot embedding adaptively fuse with w1 and w2
        # self.__intent_weight1 = nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, 1)
        # self.__intent_weight2 = nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, 1)
        # self.__slot_weight1 = nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, 1)
        # self.__slot_weight2 = nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, 1)
        #
        # # extract intent and slot information from utterance using in slot decoder
        # self.__intent_text_qkv = Attention(self.__args.dropout_rate)
        # self.__slot_text_qkv = Attention(self.__args.dropout_rate)
        # # information fuse block used in slot decoder
        # self.__fuse_block = SlotDecoderBlock(self.__args, self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim)

        # intent decoder mlp改
        self.__intent_decoder = nn.Sequential(
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, self.__num_intent)
        )
        
        # slot decoder mlp
        self.__slot_decoder = nn.Sequential(
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim),
            nn.Dropout(self.__args.slot_decoder_dropout_rate),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, self.__num_slot)
        )
        """Encoder"""
        self.__intent = nn.Linear(self.__num_intent, 384, bias=False)
        self.__slot = nn.Linear(self.__num_slot, 384, bias=False)
        """RAN"""
        self.__self_attention = EnSelfAttention(
            384,#self.__args.encoder_hidden_dim,
            self.__args.self_attention_hidden_dim,
            384,#self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )
        self.__norm_1 = nn.LayerNorm(384) # LayerNorm(d_model=384)   #LayerNormalization(d_model)
        self.__RAN = ResultAttentionNetwork(args)
        self.__norm_11 = nn.LayerNorm(384) # LayerNorm(d_model=384)

        self.__attention = Attention(self.__args.dropout_rate)
        # Define FFN
        self.__r_ffn = nn.Sequential(
            nn.Linear(384,384),
            nn.Dropout(self.__args.slot_decoder_dropout_rate),
            nn.LeakyReLU(args.alpha),
            nn.Linear(384,384)
        )
        self.__h_ffn = nn.Sequential(
            nn.Linear(384,384),
            nn.Dropout(self.__args.slot_decoder_dropout_rate),
            nn.LeakyReLU(args.alpha),
            nn.Linear(384,384)
        )
    def forward(self, text, seq_lens, n_predicts=None):
        word_tensor = self.__embedding(text)

        # utterance encoder
        # task-shared
        text_encoder = self.__text_encoder(word_tensor, seq_lens)
        # task-specific
        text_hiddens_intent = self.__text_lstm_intent(text_encoder, seq_lens)
        text_hiddens_intent = F.dropout(text_hiddens_intent, p=self.__args.dropout_rate, training=self.training)
        text_hiddens_slot = self.__text_lstm_slot(text_encoder, seq_lens)
        text_hiddens_slot = F.dropout(text_hiddens_slot, p=self.__args.dropout_rate, training=self.training)

        seq_lens_tensor = torch.tensor(seq_lens)
        if self.__args.gpu:
            seq_lens_tensor = seq_lens_tensor.cuda()

        # # intent and slot label embedder
        # # intra-utterance
        # intent_attention_out = self.__intent_attention(text_hiddens_intent, seq_lens)
        # intent_attention_out = F.dropout(intent_attention_out, p=self.__args.dropout_rate, training=self.training)
        # slot_attention_out = self.__slot_attention(text_hiddens_slot, seq_lens)
        # slot_attention_out = F.dropout(slot_attention_out, p=self.__args.dropout_rate, training=self.training)
        # # intra-corpus
        # intent_embedding = torch.matmul(torch.matmul(self.__intent_embedding.unsqueeze(0).repeat(len(seq_lens), 1, 1), text_hiddens_intent.transpose(-1, -2)), text_hiddens_intent)
        # slot_embedding = torch.matmul(torch.matmul(self.__slot_embedding.unsqueeze(0).repeat(len(seq_lens), 1, 1), text_hiddens_slot.transpose(-1, -2)), text_hiddens_slot)
        # # adaptive fusion
        # intent_weight1 = torch.sigmoid(self.__intent_weight1(intent_embedding))
        # intent_weight2 = torch.sigmoid(self.__intent_weight2(intent_attention_out))
        # intent_weight1 = intent_weight1 / (intent_weight1 + intent_weight2)
        # intent_weight2 = 1 - intent_weight1
        # slot_weight1 = torch.sigmoid(self.__slot_weight1(slot_embedding))
        # slot_weight2 = torch.sigmoid(self.__slot_weight2(slot_attention_out))
        # slot_weight1 = slot_weight1 / (slot_weight1 + slot_weight2)
        # slot_weight2 = 1 - slot_weight1
        # intent_attention_out = intent_weight1 * intent_embedding + intent_weight2 * intent_attention_out
        # intent_attention_out = F.dropout(intent_attention_out, p=self.__args.dropout_rate, training=self.training)
        # slot_attention_out = slot_weight1 * slot_embedding + slot_weight2 * slot_attention_out
        # slot_attention_out = F.dropout(slot_attention_out, p=self.__args.dropout_rate, training=self.training)
        #
        # # intent-slot co-occurrence gcn
        # graph_H = torch.cat([intent_attention_out, slot_attention_out], dim=1)
        # graph_Adj = self.graph_Adj.unsqueeze(0).repeat(len(seq_lens), 1, 1)
        # graph_H = self.__graph_gcn(graph_H, graph_Adj)
        # # updated label representation through gcn
        # intent_label_H = graph_H[:, 0:self.__num_intent, :]
        # slot_label_H = graph_H[:, self.__num_intent:, :]
        #
        # # similarity of intent and utterance
        # text_fuse_intent_pred = torch.matmul(text_hiddens_intent, intent_label_H.transpose(-1, -2))
        #
        # # information enhance and fuse block in slot decoder
        # text_fuse_intent = self.__intent_text_qkv(text_hiddens_intent, intent_label_H, intent_label_H)
        # text_fuse_intent = text_fuse_intent + text_hiddens_intent
        # text_fuse_slot = self.__slot_text_qkv(text_hiddens_slot, slot_label_H, slot_label_H)
        # text_fuse_slot = text_fuse_slot + text_hiddens_slot
        # text_fuse_slot = self.__fuse_block(text_fuse_intent, text_fuse_slot)
        """"""
        #intent decoder mlp
        pred_intent = self.__intent_decoder(text_hiddens_intent)    # torch.Size([16, 30, 18])
        
        #slot decoder mlp
        pred_slot = self.__slot_decoder(text_hiddens_slot)  # torch.Size([16, 30, 117])
        """Encoder"""
        I = F.softmax(pred_intent, dim=-1)
        I = self.__intent(I)
        S = F.softmax(pred_slot, dim=-1)
        S = self.__slot(S)
        """SR"""
        H = text_encoder
        """RAN"""
        R = S + I   # torch.Size([16, 36, 256])
        # R = torch.cat([pred_intent, pred_slot], dim=-1) # torch.Size([16, 30, 135])??
        R_ = self.__self_attention(R, seq_lens)
        # R_att = F.normalize(R+R_, dim=-1)
        R = self.__norm_1(R+R_)
        """3 layers"""
        S, R, I = self.__RAN(S, R, I)
        # print(R.shape)
        """Decoder"""
        H_ = torch.mean(H, dim=1) # [b, 384]
        H_ = self.__h_ffn(H_)
        H_ = H_.unsqueeze(1).repeat(1,H.shape[1],1)
        # H_rs = F.normalize(H + R + H_, dim=-1)
        H_rs = self.__norm_11(H + R + H_)
        """"""
        pred_slot = H_rs + S
        pred_intent = H_rs + I
        # intent decoder mlp
        pred_intent = self.__intent_decoder(pred_intent)  # torch.Size([16, 30, 18])

        # slot decoder mlp
        pred_slot = self.__slot_decoder(pred_slot)  # torch.Size([16, 30, 117])
        """"""
        pred_slot_list = []
        for i in range(0, len(seq_lens)):
            pred_slot_list.append(pred_slot[i, 0:seq_lens[i], :])
        pred_slot = torch.cat(pred_slot_list, dim=0)
        pred_slot = F.log_softmax(pred_slot, dim=1)
        # print(pred_slot.shape)
        if n_predicts is None:
            return pred_slot, pred_intent
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)

            intent_index_sum = torch.cat(
                [
                    torch.sum(torch.sigmoid(pred_intent[i, 0:seq_lens[i], :]) > self.__args.threshold, dim=0).unsqueeze(
                        0)
                    for i in range(len(seq_lens))
                ],
                dim=0
            )
            intent_index = (intent_index_sum > (seq_lens_tensor // 2).unsqueeze(1)).nonzero()

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot embedding:		    	{};'.format(self.__args.slot_embedding_dim))
        print('\toutput dimension of gcn graph:             {};'.format(self.__args.gcn_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')
