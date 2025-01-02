# @Time    : 2024/9/11 2:27
# @Author  : zeng
# @File    : BCDN_cfgkt.py
# @Software: PyCharm


import torch.nn as nn
from torch import nn
from models.EmbeddingM import EmbeddingModule
from models.BCDN_M.BCDN_C import BCDN_CC
from models.BCDN_M.BCDN_B import BCDN_BC
import torch
import torch.nn.functional as F

from models.EmbeddingM import EmbeddingModule
from CFGKT_adator import CFGKT
class BCDN_cfgkt(nn.Module):
    def __init__(self, n_at, n_it,exercise_size,concept_size, input_dim,dropout,hidden_dim,col_student_num,seq_max_length,n_blocks,kq_same,
                 memory_size, final_fc_dim, n_heads,d_ff,alpha=0.7):
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
        self.CC_collaration = BCDN_CC(self.input_embedding,self.input_dim, self.col_student_num, self.hidden_dim,dropout)
        # 演化一致性去噪模块
        self.BC_collaration = BCDN_BC(n_it,self.input_dim, self.seq_max_length, 8,dropout)

        self.cfpkt = CFGKT(
            n_concepts=concept_size, n_pid=exercise_size, d_model = input_dim, n_blocks=n_blocks, kq_same=kq_same,
            dropout=dropout, model_type='CFGKT', memory_size=memory_size, final_fc_dim=final_fc_dim, n_heads=n_heads,
            d_ff=d_ff, time=n_at, interval=n_it
        )
        # self.B_Rnn = nn.RNN(self.input_dim * 4, self.hidden_dim, num_layers=1, batch_first=True)




    def forward(self, concept_seq, response_seq, exercise_seq, student_query_matrix, query_correct_sequence_ids,taken_time_seq,interval_time_seq):
        # seq = seq + self.concept_num * r
        # seq[seq < 0] = self.concept_num * 2
        # exercise_concept_embedded = self.input_embedding(exercise_seq, concept_seq)
        # exercice_concept_response_seq = self.input_embedding.exercise_concept_res_embedded(exercise_seq, concept_seq,
        #                                                                                    response_seq)

        exercise_concept_embedded, exercice_concept_response_seq, pid_embed_data = self.input_embedding(exercise_seq, concept_seq,response_seq)

        # # 协同一致性系数，获得去噪后的表示
        seq_emb_collaration = self.CC_collaration(student_query_matrix, query_correct_sequence_ids,
                                                  exercice_concept_response_seq, exercise_concept_embedded,
                                                  exercise_seq,concept_seq)
        denoising_seq_emb_collaration = torch.mul(seq_emb_collaration, exercice_concept_response_seq)

        # 演化一致性系数，获得去噪后的表示
        # seq_emb_evolutionary = self.BC_collaration(exercice_concept_response_seq,interval_time_seq)
        # denoising_seq_emb_evolutionary = torch.mul(seq_emb_evolutionary, exercice_concept_response_seq)

        # 采用对比学习
        #  rnn_out_collaration, _ = self.C_Rnn(denoising_seq_emb_collaration)
        #  rnn_out_evolutionary, _ = self.B_Rnn(denoising_seq_emb_evolutionary)
        # info_loss = self.contrastive_loss(denoising_seq_emb_collaration, denoising_seq_emb_evolutionary)
        # info_loss = self.contrastive_losssss(denoising_seq_emb_collaration, denoising_seq_emb_evolutionary,exercise_seq)


        # denoising_seq_emb = (denoising_seq_emb_collaration + denoising_seq_emb_evolutionary) / 2
        # denoising_seq_emb = self.alpha * denoising_seq_emb_collaration + (1 - self.alpha) * denoising_seq_emb_evolutionary




        # cfgkt_out = self.cfpkt(exercise_concept_embedded,denoising_seq_emb, taken_time_seq,interval_time_seq)


        cfgkt_out = self.cfpkt(exercise_concept_embedded,denoising_seq_emb_collaration, taken_time_seq,interval_time_seq)
        info_loss = 0

        # res = self.linear_out(rnn_out)
        # rnn_res = torch.sigmoid(res)
        # res = self.dropout_layer(res)
        return cfgkt_out,info_loss

    def contrastive_loss(self, rnn_out_collaration, rnn_out_evolutionary, temperature=0.2, num_negatives=50):
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


    def contrastive_losssss(self, rnn_out_collaration, rnn_out_evolutionary, seq, temperature=0.2,
                            num_negatives=50):
        """
        Contrastive Loss function to maximize the similarity of positive pairs and minimize it for negative pairs.
        :param rnn_out_collaration: Embedding tensor of shape [batch_size, seq_length, hidden_dim]
        :param rnn_out_evolutionary: Embedding tensor of shape [batch_size, seq_length, hidden_dim]
        :param seq_lengths: List or tensor indicating the length of each sequence in the batch
        :param temperature: Temperature scaling parameter for sharpening the similarity distribution
        :param num_negatives: Number of negative samples to consider for each positive pair

        :return: Computed contrastive loss
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

        # 创建负样本掩码
        seq_flat = seq.reshape(-1)  # 将seq展开为一维
        indices = torch.arange(batch_size * seq_length).unsqueeze(1)
        same_seq_mask = seq_flat[indices] == seq_flat.unsqueeze(0)

        # 将相同序列的元素设置为0，排除它们作为负样本
        mask = torch.ones_like(sim_matrix, device=sim_matrix.device)  # 创建一个全1的掩码
        mask[same_seq_mask] = 0

        # 为每个正样本选择num_negatives个负样本
        negative_indices = torch.randint(0, batch_size * seq_length, (batch_size * seq_length, num_negatives),
                                         device=sim_matrix.device)
        negative_indices = negative_indices[torch.randperm(batch_size * seq_length), :]
        negative_indices = negative_indices[:, :num_negatives]

        # 确保不选择相同序列的样本
        negative_indices = negative_indices * (1 - mask[:, negative_indices].any(dim=1).long())

        # 计算负样本的相似度
        negative_sim = sim_matrix[torch.arange(batch_size * seq_length).unsqueeze(1), negative_indices]

        # 计算InfoNCE Loss
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim],
                           dim=1)  # [batch_size*seq_length, 1 + num_negatives]
        labels = torch.zeros(batch_size * seq_length, dtype=torch.long, device=sim_matrix.device)  # 正样本标签
        loss = F.cross_entropy(logits / temperature, labels)

        return loss


