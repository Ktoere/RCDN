# -- coding: utf-8 --
# @Time : 2024/10/12
# @File : EmbeddingM.py
# @Software: PyCharm
from torch import nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingModule(nn.Module):
    def __init__(self, Exercise_size, Concept_size, embedding_dim):
        super(EmbeddingModule, self).__init__()
        self.exercie_embed = nn.Embedding(Exercise_size + 1,  embedding_dim)
        self.concept_embed = nn.Embedding(Concept_size + 1, embedding_dim)
        self.difficult_param = nn.Embedding(Exercise_size + 1, 1)
        self.a_embed = nn.Embedding(2, embedding_dim)

        # self.qa_embed_diff = nn.Embedding(2 * Concept_size + 1, embedding_dim)
        # # n_question+1 ,d_model
        # self.q_embed = nn.Embedding(Concept_size + 1, embedding_dim)
        # self.qa_embed = nn.Embedding(2, embedding_dim)
        self.liea1 = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, exercise_seq, concept_seq,response_seq):

        exercise_embed = self.exercie_embed(exercise_seq)
        concept_embed = self.concept_embed(concept_seq)
        pid_embed_data = self.difficult_param(exercise_seq)
        # q_embed_data = exercise_embed + pid_embed_data * concept_embed
        q_embed_data = concept_embed + pid_embed_data * exercise_embed
        anser_embed_data = self.a_embed(response_seq)
        qa_embed_data = self.liea1(torch.cat([q_embed_data, anser_embed_data], dim=-1))
        return q_embed_data, qa_embed_data,pid_embed_data

