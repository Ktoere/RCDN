# @Time    : 2025/1/2 18:05
# @Author  : zeng
# @File    : RCDN-CFGKT.py
# @Software: PyCharm

import torch.nn as nn
from torch import nn
# from EmbeddingM import EmbeddingModule
# from models.BCDN_M.BCDN_C import BCDN_CC
# from models.BCDN_M.BCDN_B import BCDN_BC
import torch
import torch.nn.functional as F

# from models.EmbeddingM import EmbeddingModule
# from CFGKT_adator import CFGKT




class BCDN_cfgkt(nn.Module):
    def __init__(self, n_at, n_it,exercise_size,concept_size, input_dim,dropout,hidden_dim,col_student_num,seq_max_length,n_blocks,kq_same,
                 memory_size, final_fc_dim, n_heads,d_ff,alpha=0.4):
        super(BCDN_cfgkt, self).__init__()
        self.exercise_size = exercise_size
        self.concept_num = concept_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.col_student_num = col_student_num
        self.seq_max_length = seq_max_length
        self.alpha = alpha

        self.input_embedding = EmbeddingModule(self.exercise_size, self.concept_num, self.input_dim)
        # 将融入答案后的表示与不融入答案表示的维度处理成一样

        # 协同一致性去噪子模块
        self.CC_collaration = BCDN_CC(self.input_embedding,self.input_dim, self.col_student_num, self.hidden_dim)
        # 演化一致性去噪模块
        self.BC_collaration = BCDN_BC(self.input_dim, self.seq_max_length, 4)

        self.cfpkt = CFGKT(
            n_concepts=concept_size, n_pid=exercise_size, d_model = input_dim, n_blocks=n_blocks, kq_same=kq_same,
            dropout=dropout, model_type='CFGKT', memory_size=memory_size, final_fc_dim=final_fc_dim, n_heads=n_heads,
            d_ff=d_ff, time=n_at, interval=n_it
        )
        # self.B_Rnn = nn.RNN(self.input_dim * 4, self.hidden_dim, num_layers=1, batch_first=True)
        # self.linear_out = nn.Linear(self.hidden_dim, self.concept_num)

    def forward(self, concept_seq, response_seq, exercise_seq, student_query_matrix, query_correct_sequence_ids,taken_time_seq,interval_time_seq):
        # seq = seq + self.concept_num * r
        # seq[seq < 0] = self.concept_num * 2
        exercise_concept_embedded = self.input_embedding(exercise_seq, concept_seq)
        exercice_concept_response_seq = self.input_embedding.exercise_concept_res_embedded(exercise_seq, concept_seq,
                                                                                           response_seq)

        # 协同一致性系数，获得去噪后的表示
        seq_emb_collaration = self.CC_collaration(student_query_matrix, query_correct_sequence_ids,
                                                  exercice_concept_response_seq, exercise_concept_embedded,
                                                  exercise_seq,concept_seq)
        denoising_seq_emb_collaration = torch.mul(seq_emb_collaration, exercice_concept_response_seq)

        # 演化一致性系数，获得去噪后的表示
        seq_emb_evolutionary = self.BC_collaration(exercice_concept_response_seq)
        denoising_seq_emb_evolutionary = torch.mul(seq_emb_evolutionary, exercice_concept_response_seq)

        # 采用对比学习
        #  rnn_out_collaration, _ = self.C_Rnn(denoising_seq_emb_collaration)
        #  rnn_out_evolutionary, _ = self.B_Rnn(denoising_seq_emb_evolutionary)
        info_loss = self.contrastive_loss(denoising_seq_emb_collaration, denoising_seq_emb_evolutionary)

        # denoising_seq_emb = (denoising_seq_emb_collaration + denoising_seq_emb_evolutionary) / 2
        denoising_seq_emb = self.alpha * denoising_seq_emb_collaration + (1 - self.alpha) * denoising_seq_emb_evolutionary
        cfgkt_out = self.cfpkt(exercise_concept_embedded,denoising_seq_emb, taken_time_seq,interval_time_seq)

        # res = self.linear_out(rnn_out)
        # rnn_res = torch.sigmoid(res)
        # res = self.dropout_layer(res)
        return cfgkt_out, info_loss

    def contrastive_loss(self, rnn_out_collaration, rnn_out_evolutionary, temperature=0.3, num_negatives=50):
        """
        Contrastive Loss function to maximize the similarity of positive pairs and minimize it for negative pairs.
        :param rnn_out_collaration: Embedding tensor of shape [batch_size, seq_length, hidden_dim]
        :param rnn_out_evolutionary: Embedding tensor of shape [batch_size, seq_length, hidden_dim]
        :param temperature: Temperature scaling parameter for sharpening the similarity distribution
        :param num_negatives: Number of negative samples to consider for each positive pair

        :return:omputed contrastive loss
        """
        batch_size, seq_length, hidden_dim = rnn_out_collaration.shape
        # Normalize the embeddings to ensure they are on the unit hypersphere
        rnn_out_collaration = F.normalize(rnn_out_collaration, dim=-1)
        rnn_out_evolutionary = F.normalize(rnn_out_evolutionary, dim=-1)

        # 计算相似性
        rnn_out_collaration_flat = rnn_out_collaration.reshape(batch_size * seq_length,
                                                               hidden_dim)  # 展开为 [batch_size*seq_length, hidden_dim]
        rnn_out_evolutionary_flat = rnn_out_evolutionary.reshape(batch_size * seq_length,
                                                                 hidden_dim)  # 展开为 [batch_size*seq_length, hidden_dim]

        # 计算余弦相似性
        sim_matrix = torch.mm(rnn_out_collaration_flat,
                              rnn_out_evolutionary_flat.T) / temperature  # [batch_size*seq_length, batch_size*seq_length]

        # 计算正样本对的相似度（对角线上的值）
        positive_sim = torch.diag(sim_matrix)

        # 计算负样本对的相似度（除了对角线上的值）
        neg_mask = ~torch.eye(batch_size * seq_length, device=sim_matrix.device).bool()
        negative_sim = sim_matrix[neg_mask].view(batch_size * seq_length, -1)

        # 计算InfoNCE Loss
        loss = -torch.mean(positive_sim - torch.log(torch.sum(torch.exp(negative_sim), dim=1)))

        return loss



