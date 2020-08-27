import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config
from config import USE_CUDA, DEVICE
import os
import numpy as np
def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class embedding(nn.Module):
    def __init__(self,vocab_size,embedding_dim,pretrained_weight):
        super(embedding, self).__init__()
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        pretrained_weight = np.array(pretrained_weight)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
    def forward(self,x):
        embedded = self.word_embeds(input_x)
        return embedded


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # 双层lstm
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    # seq_lens: 1D tensor 应该降序排列
    def forward(self, input_emb, seq_lens):
        embedded = input_emb

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True) #长度要排好序
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(packed)  # hidden is tuple([2, batch, hid_dim], [2, batch, hid_dim])

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # [batch, max(seq_lens), 2*hid_dim]
        encoder_outputs = encoder_outputs.contiguous()                      # [batch, max(seq_lens), 2*hid_dim]

        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  #ouput转成二维[batch,神经元含有h和c]
        encoder_feature = self.W_h(encoder_feature)       # [batch*max(seq_lens), 2*hid_dim]

        return encoder_outputs, encoder_feature, hidden   # [B, max(seq_lens), 2*hid_dim], [B*max(seq_lens), 2*hid_dim], tuple([2, batch, hid_dim], [2, batch, hid_dim])
        #[batch,time,units]   [batch*time,2*units]  
class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden    # h, c dim = [2, batch, hidden_dim]
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)  # [batch, hidden_dim*2]
        # hidden_reduced_h = F.relu(self.reduce_h(h_in))                         # [batch, hidden_dim]
        hidden_reduced_h = F.tanh(self.reduce_h(h_in))

        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        # hidden_reduced_c = F.relu(self.reduce_c(c_in))
        hidden_reduced_c = F.tanh(self.reduce_c(c_in))


        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = [1, batch, hidden_dim]

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())
        # batch，时间步，神经元
        # t_k为时间步
        # n为神经元数量

        # s_t_hat 为en或者de隐藏
        dec_fea = self.decode_proj(s_t_hat)               # B x 2*hid_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hid_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)   # B * t_k x 2*hid_dim

        att_features = encoder_feature + dec_fea_expanded #  [B * t_k , 2*hidden_dim]
        # encoder不变，有多时间特性     dec_fea是不断循环的deco的隐藏，单时间

        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)         # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)   # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)   # B * t_k x 2*hidden_dim   
        scores = self.v(e)             # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k   Pvocab*en_pad_mask(就是Pgen)
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)           # B x 1 x t_k
        c_t = torch.bmm(attn_dist,  )  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)    # B x 2*hidden_dim
        # c_t就是context vector
        attn_dist = attn_dist.view(-1, t_k)          # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1_embd, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            # St'其实是de_h_n
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next
        # print('y_t_1是什么:',y_t_1)
        
        # print('字典输出y_t_1_embd',y_t_1_embd) #[2,128]
        # print('c_t_1:',c_t_1) # [2,512]
        x_cat=torch.cat([c_t_1, y_t_1_embd], 1)

        # print('cat之后的x:',x_cat.shape)
        x = self.x_context(x_cat)
        self.lstm.flatten_parameters()
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)
        # print('att的输出:',c_t, attn_dist, coverage_next)
        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim
        # print('output是什么:',output)
        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
                # print('vocab_dist_是什么:',vocab_dist_.shape)# [2,50004]
                # print('extra_zeros是什么:',extra_zeros.shape)# [2,4]
            # print('attn_dist_是什么',attn_dist_.shape) # [2,96]
            # print('enc_batch_extend_vocab是什么',enc_batch_extend_vocab.shape) #[2,96]
            # print('vocab_dist_是什么:',vocab_dist_.shape)  #[2,50004]
            # print('扩展词典：',enc_batch_extend_vocab)#这个越位了，超过50004
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

from gensim.models.word2vec import LineSentence, Word2Vec

class Model(object):#只是构建组件
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        wv_model = Word2Vec.load('./wv_model')
        embedding_matrix = wv_model.wv.vectors

        word_emb=embedding(len(embedding_matrix),config.emb_dim,embedding_matrix)

        # shared the embedding between encoder and decoder
        # decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()
            word_emb=word_emb.eval()
        if USE_CUDA:
            encoder = encoder.to(DEVICE)
            decoder = decoder.to(DEVICE)
            reduce_state = reduce_state.to(DEVICE)
            word_emb=word_emb.to(DEVICE)
        #if NUM_CUDA > 1:
        #    encoder = nn.DataParallel(encoder)
        #    decoder = nn.DataParallel(decoder)
        #    reduce_state = nn.DataParallel(reduce_state)
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state
        self.word_emb=word_emb

        # if model_file_path is not None:
        #     state = torch.load(model_file_path, map_location= lambda storage, location: storage)
        #     self.encoder.load_state_dict(state['encoder_state_dict'])
        #     self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
        #     self.reduce_state.load_state_dict(state['reduce_state_dict'])
            
    def save_model(self,epo,loss, iter_step,optimizer):
    # """保存模型"""
        state = {
            # 'epo':epo
            'iter': iter_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'reduce_state_dict': self.reduce_state.state_dict(),
            'optimizer': optimizer.state_dict(),
            'current_loss': loss
        }
        weight_ROOT = "./weight/pointer"
        model_save_path = os.path.join(weight_ROOT, 'model_{:.4f}_{}_{}.ckpt'.format(loss,epo,iter_step ))
        torch.save(state, model_save_path)


def run_model(model,input_data,output_data):
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab,\
    extra_zeros, c_t_1, coverage=input_data
    # print('input拿到的词典:',enc_batch_extend_vocab)
    dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch=output_data
    # target_batch正常
    # print('max_dec_len是cuda吗',max_dec_len) # 24

    enc_batch_emb=model.word_emb(enc_batch)
    encoder_outputs, encoder_feature, encoder_hidden = model.encoder(enc_batch_emb, enc_lens)
    s_t_1 = model.reduce_state(encoder_hidden)
    step_losses = []    
    # print('target是什么',target)
    # max_dec_len
    for di in range(min(max_dec_len, config.max_dec_steps)): #不是，是遍历0到目标长度
        y_t_1 = dec_batch[:, di] # 
        y_t_1_embd=model.word_emb(y_t_1)

        final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage =\
            model.decoder(y_t_1_embd, s_t_1,encoder_outputs, \
            encoder_feature, enc_padding_mask, c_t_1,extra_zeros, \
            enc_batch_extend_vocab, coverage, di)
        
        target = target_batch[:, di]  # 摘要的下一个单词的编码 
        gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()   # 取出目标单词的概率gold_probs
        g_=gold_probs + config.eps
        step_loss = -torch.log(g_)  # 最大化gold_probs，也就是最小化step_loss（添加负号）
        
        if config.is_coverage:
            step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
            coverage = next_coverage
        step_mask = dec_padding_mask[:, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

    sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    batch_avg_loss = sum_losses/dec_lens_var
    loss = torch.mean(batch_avg_loss)
    return loss
