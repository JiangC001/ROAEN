'''
Description:
Author: J Chen
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#pass
def init_weights(m):

    if type(m)==torch.nn.modules.linear.Linear:
        try:
            torch.nn.init.uniform_(m.weight,a=-0.1,b=0.1)#weight
            torch.nn.init.uniform_(m.bias,a=-0.1,b=0.1)#bias
        except Exception:
            pass

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):  # features
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):  # x
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Myloss(torch.nn.Module):
    def __init__(self,opt):
        super(Myloss, self).__init__()
        self.opt = opt

    def forward(self,inputs,polarity,standar_score):
        standar_score.requires_grad = False
        inputs_h_extend = torch.unsqueeze(inputs, dim=1).repeat(1, self.opt.polarities_dim, 1)

        loss_all = inputs_h_extend-standar_score.to(self.opt.device)
        loss_all_sqr = torch.bmm(loss_all,torch.permute(loss_all,dims=(0,2,1)).contiguous()) #(batch,polarities_dim,polarities_dim）
        loss_all_sqr_eye = torch.diagonal(loss_all_sqr,dim1=-1,dim2=-2) # (batch,polarities_dim)
        mask_min = torch.zeros(inputs_h_extend.size(0), self.opt.polarities_dim, requires_grad=False).to(self.opt.device)
        mask_max = torch.ones(inputs_h_extend.size(0), self.opt.polarities_dim, requires_grad=False).to(self.opt.device)
        for i, j in enumerate(polarity.cpu().numpy()):
            mask_min[i][int(j)] = 1
            mask_max[i][int(j)] = 0
        loss_min = torch.sum(torch.sum(loss_all_sqr_eye * mask_min,dim=-1),dim=-1)
        loss_max = torch.sum(torch.sum(loss_all_sqr_eye * mask_max,dim=-1),dim=-1)
        cos0_1 = cos_(standar_score[0],standar_score[1]) #cos(0,1)
        cos0_2 = cos_(standar_score[0],standar_score[2]) #cos(0,2)
        cos1_2 = cos_(standar_score[1],standar_score[2]) #cos(1,2)
        loss = loss_min + cos0_1*cos0_1 + cos0_2*cos0_2 + cos1_2*cos1_2
        return loss

def cos_(input1,input2):
    return torch.sum(input1*input2,dim=-1)/torch.sqrt(torch.sum(input1*input1,dim=-1))*torch.sqrt(torch.sum(input2*input2,dim=-1))

def d_(input1,input2):
    return torch.sum(input1-input2,dim=-1)

class OurModelBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        # self.ave_pool = AvePool(bert, opt)
        self.ave_pool = AvePool(bert, opt)
        self.batch_pool = BatchPoolLoss(opt)
        self.fnn1 = nn.Linear(opt.attention_dim * 2, opt.polarities_dim)  # FFN1
        self.fnn2 = nn.Linear(opt.attention_dim * 2, opt.polarities_dim)  # FFN2

    def forward(self, inputs):

        outputs1, outputs2, porality = self.ave_pool(inputs)
        standar_score = self.batch_pool(outputs2[1],porality)
        logits_bgcn_aspect = self.fnn1(outputs1[0])
        logits_gate_aspect = self.fnn2(outputs1[1])

        # (batch,3)、list、score
        return [logits_bgcn_aspect, logits_gate_aspect, outputs2[1], porality, standar_score], None
########################################################################################################################
########################################################################################################################

class BatchPoolLoss(torch.nn.Module):
    def __init__(self,opt):
        super(BatchPoolLoss, self).__init__()
        self.opt = opt

    def forward(self,inputs,porality):

        sta_score_init = torch.zeros(3, inputs[1].size(-1)).to(self.opt.device)
        for i, j in enumerate(porality.cpu().numpy()):
            sta_score_init[int(j)] = sta_score_init[int(j)] + inputs[i]

        '''(polariti_dim,attention_dim*2)'''
        num_0_1_2 = torch.unique(porality, return_counts=True)
        num_0_1_2_init = torch.ones(3).to(self.opt.device) #
        for i, j in zip(num_0_1_2[0].cpu().numpy(), num_0_1_2[1].cpu().numpy()):
            num_0_1_2_init[int(i)] = j

        num_0_1_2_init = torch.unsqueeze(num_0_1_2_init, dim=-1)
        sta_score = sta_score_init / num_0_1_2_init
        return sta_score

########################################################################################################################
# pooling
class AvePool(torch.nn.Module):
    def __init__(self, bert, opt):
        super(AvePool, self).__init__()
        self.opt = opt
        self.bertstruture = BertStructure(bert, opt)

    def forward(self, inputs):
        bgcn_result, attention_result, gate_result, porality = self.bertstruture(inputs)
        ave_bgcn_result_aspect = torch.sum(bgcn_result[1], dim=1) / bgcn_result[2]

        ave_gate_result_aspect = torch.sum(gate_result[1], dim=1) / gate_result[2]

        ave_bgcn_result_context = torch.sum(bgcn_result[0], dim=1) / torch.unsqueeze(
            torch.Tensor([self.opt.max_length]), dim=0).repeat(bgcn_result[0].size(0),1).to(self.opt.device)
        ave_gate_result_context = torch.sum(gate_result[0], dim=1) / torch.unsqueeze(
            torch.Tensor([self.opt.max_length]), dim=0).repeat(attention_result[0].size(0),1).to(self.opt.device)
        #ave aspect:gcn、gate|ave context：gan、gate
        return [ave_bgcn_result_aspect, ave_gate_result_aspect],[ave_bgcn_result_context, ave_gate_result_context], porality
########################################################################################################################
########################################################################################################################

class BertStructure(torch.nn.Module):
    def __init__(self, bert, opt):
        super(BertStructure, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layer_norm = LayerNorm(opt.bert_dim)
        self.bgcn = BGCN(opt)
        self.attentionmodule1 = AttentionModule(opt, opt.c_c_Attention_num)
        self.attentionmodule2 = AttentionModule(opt, opt.c_a_Attention_num)

    def forward(self, inputs):
        adj_f, adj_b, adj_f_aspect, adj_b_aspect, text_bert_indices, bert_segments_ids, attention_mask, \
        text_len, post_1, asp_start, asp_end, src_mask, aspect_mask, polarity = inputs
        lin_shi = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output, pooled_output = lin_shi[0], lin_shi[1]
        sequence_output = self.layer_norm(sequence_output)
        aspect_output = sequence_output * aspect_mask.unsqueeze(-1).repeat(1, 1,sequence_output.size(-1))

        aspect_mask1 = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bgcn_dim)
        aspect_mask2 = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bgcn_dim * 2)
        aspect_mask3 = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.attention_dim)  # attention add
        aspect_mask4 = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.attention_dim * 2)

        context_output_bgcn_cat = self.bgcn(sequence_output, adj_b, adj_b_aspect)

        aspect_output_bgcn_cat = context_output_bgcn_cat * aspect_mask2
        # c_c_attention
        context_output_att1, context_output_att2, heads_context_scores = self.attentionmodule1(opt=self.opt,
                                                                                               sequence1=sequence_output,
                                                                                               sequence2=sequence_output,
                                                                                               aspect_mask=None)  # context_h、context_h、score

        context_output_att3, aspect_output_att, heads_aspect_scores = self.attentionmodule2(opt=self.opt,
                                                                                            sequence1=sequence_output,
                                                                                            sequence2=aspect_output,
                                                                                            aspect_mask=aspect_mask3)  # context_h、aspect_h、score


        context_output_att_cat = torch.cat([context_output_att1, aspect_output_att], dim=-1)  # x
        aspect_output_att_cat = context_output_att_cat * aspect_mask4

        '''x = context_output_att_cat  
           y = context_output_bgcn_cat 
           z = xyy/|y||y| = y|x||y|cosθ/|y||y| = |x|cosθy/|y| = P_y
           x_ = (g)x+(1-g)w
        '''
        x_y_y = torch.mul(torch.mul(context_output_att_cat, context_output_bgcn_cat), context_output_bgcn_cat)  # xyy

        y_L = torch.zeros(1,context_output_bgcn_cat.size(2)).to(self.opt.device)
        y_L[-1] = 0.5
        y_L = y_L.repeat(context_output_bgcn_cat.size(1),1)
        y_mo2 = torch.unsqueeze(torch.sum(torch.mul(context_output_bgcn_cat+y_L, context_output_bgcn_cat)+y_L, dim=-1),
                                dim=-1)

        ortho_output = x_y_y - (x_y_y / y_mo2)

        o_x_similar = torch.bmm(context_output_att_cat, torch.permute(ortho_output, dims=(0, 2, 1)).contiguous())
        o_x_similar = torch.unsqueeze(F.sigmoid(torch.diagonal(o_x_similar, dim1=-1, dim2=-2)), dim=-1)
        # σ*x+(1-σ)*z
        all_output = o_x_similar * context_output_att_cat + (1 - o_x_similar) * ortho_output
        all_aspect_output = all_output * aspect_mask4

        aspect_len = torch.unsqueeze((asp_end - asp_start + 1), dim=-1)

        # bgcn、attention、ortho（context、aspect）、porality
        return [context_output_bgcn_cat, aspect_output_bgcn_cat, aspect_len], \
               [context_output_att_cat, aspect_output_att_cat, aspect_len], \
               [all_output, all_aspect_output, aspect_len],\
               polarity
########################################################################################################################

########################################################################################################################

class AttentionModule(torch.nn.Module):
    def __init__(self, opt, layer_num):
        super(AttentionModule, self).__init__()
        self.opt = opt
        self.attention = Attention(opt, layer_num)

    def forward(self, opt, sequence1, sequence2, aspect_mask=None):
        # MultiHeadSelfAttention
        sequence_list1 = [sequence1]
        sequence_list2 = [sequence2]  # context、aspect
        score_list = []
        for i in range(opt.c_c_Attention_num):
            # context、context、score
            c_c_attention1, c_c_attention2, c_c_score = self.attention(sequence1=sequence_list1[-1],
                                                                       sequence2=sequence_list2[-1],
                                                                       head=opt.c_c_heads,
                                                                       len_=len(sequence_list1),
                                                                       mask=aspect_mask)
            sequence_list1.append(c_c_attention1)
            sequence_list2.append(c_c_attention1)
            score_list.append(c_c_score)

        return sequence_list1[-1], sequence_list2[-1], score_list[-1]

class Attention(torch.nn.Module):
    def __init__(self, opt, layer_num):
        super(Attention, self).__init__()
        self.opt = opt
        self.dropout = torch.nn.Dropout(p=opt.gcn_dropout)
        self.w_b_q = torch.nn.Linear(opt.bert_dim, opt.attention_dim,bias=False)
        self.w_b_k = torch.nn.Linear(opt.bert_dim, opt.attention_dim,bias=False)
        self.w_b_v = torch.nn.Linear(opt.bert_dim, opt.attention_dim,bias=False)

        self.w_b_q1 = [torch.nn.Linear(opt.attention_dim, opt.attention_dim, device=opt.device) for _ in range(layer_num - 1)]
        self.w_b_k1 = [torch.nn.Linear(opt.attention_dim, opt.attention_dim, device=opt.device) for _ in range(layer_num - 1)]
        self.w_b_v1 = [torch.nn.Linear(opt.attention_dim, opt.attention_dim, device=opt.device) for _ in range(layer_num - 1)]

    def forward(self, sequence1, sequence2, head, len_, mask=None):
        '''dropout,qkv,tanh(qk),softmax'''

        # print('s1',sequence1)
        # print('s2',sequence2)

        sequence1 = self.dropout(sequence1)
        sequence2 = self.dropout(sequence2)

        if len_ > 1:
            '''Q K=V'''

            querys = self.w_b_q1[len_-2](sequence1)
            keys = self.w_b_k1[len_-2](sequence2)
            values = self.w_b_v1[len_-2](sequence2)

        else:
            '''Q K=V'''

            querys = self.w_b_q(sequence1)
            keys = self.w_b_k(sequence2)
            values = self.w_b_v(sequence2)

        # querys = self.W_query(sequence1)  # [N, T_q, num_units]
        # keys = self.W_key(sequence2)  # [N, T_k, num_units]
        # values = self.W_value(sequence2)

        split_size = self.opt.attention_dim // head
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        # tanh
        scores = F.tanh(scores)
        scores = scores / (self.opt.bert_dim ** 0.5)
        scores = F.softmax(scores, dim=3)  # softmax
        ## out = score * V
        context_out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        context_out = torch.cat(torch.split(context_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        if mask is not None:
            aspect_out = context_out * mask
            return context_out, aspect_out, scores  # aspect_h

        return context_out, context_out, scores
########################################################################################################################

########################################################################################################################

class BGCN(torch.nn.Module):
    def __init__(self, opt):
        super(BGCN, self).__init__()

        self.opt = opt

        self.w_b = torch.nn.Linear(opt.bert_dim, opt.bgcn_dim)
        self.gcn1 = GCN(opt, opt.BGCNlayer_num)
        self.gcn2 = GCN(opt, opt.AGCNlayer_num)

    def forward(self, input_, adj_b, adj_b_aspect):

        input_ = self.w_b(input_)

        gcn_b_output_list = [input_]
        for i in range(self.opt.BGCNlayer_num):
            gcn_b = self.gcn1(gcn_b_output_list[-1], adj_b, len(gcn_b_output_list))
            gcn_b_output_list.append(gcn_b)

        gcn_b_aspect_output_list = [input_]
        for i in range(self.opt.AGCNlayer_num):
            gcn_b_aspect = self.gcn2(gcn_b_aspect_output_list[-1], adj_b_aspect,
                                     len(gcn_b_aspect_output_list))
            gcn_b_aspect_output_list.append(gcn_b_aspect)

        context_output_bgcn_cat = torch.cat([gcn_b_output_list[-1], gcn_b_aspect_output_list[-1]], dim=-1)
        return context_output_bgcn_cat

class GCN(torch.nn.Module):
    def __init__(self, opt, layer_num):
        super(GCN, self).__init__()
        self.opt = opt

        self.dropout = torch.nn.Dropout(p=opt.gcn_dropout)

        self.w_b = [torch.nn.Linear(opt.bgcn_dim, opt.bgcn_dim, bias=False, device=opt.device) for _ in range(layer_num)]

        self.bais = [torch.nn.Linear(opt.max_length, 1, bias=False, device=opt.device) for _ in range(layer_num)]

    def forward(self, input_, adj, len_):

        # adj（batch，max_len，max_len）

        input_ = self.dropout(input_)
        w_i = self.w_b[len_-1](input_)
        adj_sum = torch.unsqueeze(torch.sum(adj, dim=-1) + 1, dim=-1)  # (batch,max_len)

        adj_w_i = torch.bmm(adj, w_i) / adj_sum  # sum(adj*w*x)  (bacth,max_len,dim)
        '''b'''
        eye = torch.eye(self.opt.max_length).to(self.opt.device)
        bais = self.bais[len_-1](eye)

        output_gcn = F.relu(adj_w_i+bais)
        return output_gcn  # GCN
########################################################################################################################
