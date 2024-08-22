import math
import time

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F


class ResidualModule(nn.Module):
    def __init__(self, look_back, horizon, layer_size, res_layer):
        super(ResidualModule, self).__init__()
        kernel_size = [7, 2, 5, 2, 3, 2]
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=(kernel_size[0],)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size[1], stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=(kernel_size[2],)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size[3], stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=(kernel_size[4],)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size[5], stride=1)
        )
        self.relu = nn.LeakyReLU()
        self.linear1 = nn.Linear(look_back - sum(kernel_size) + len(kernel_size), layer_size)
        self.linear2 = nn.Linear(layer_size, look_back + horizon)

        self.res_layer = res_layer
        self.look_back = look_back

    def forward(self, x):
        res_x = x.unsqueeze(1)
        forecast = 0

        for _ in range(self.res_layer):
            out = self.conv1(res_x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.linear1(out)
            out = self.relu(out)
            out = self.linear2(out)

            x_back, x_forecast = out[:, :, :self.look_back], out[:, :, self.look_back:]
            res_x = res_x - x_back
            forecast = forecast + x_forecast
        return res_x.squeeze(1), forecast.squeeze(1)


class PredictModule(nn.Module):
    def __init__(self, d_model, look_back, horizon, dropout, pre_layer, thetas_dim):
        super(PredictModule, self).__init__()
        self.pe = PositionalEncoding(d_model, dropout)
        self.multi_att = MultiHeadAttention(d_model, thetas_dim)
        # normalize the input
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model * look_back, 512)
        self.linear2 = nn.Linear(512, look_back + horizon)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.look_back = look_back
        self.pre_layer = pre_layer


    def forward(self, x):
        res_x = x
        forecast = 0

        for _ in range(self.pre_layer):
            q = self.pe(res_x)
            k = self.pe(res_x)
            v = self.pe(res_x)
            out = self.multi_att(q, k, v)
            out = q + out
            out = self.norm1(out)

            # flatten, the shape: [batch_size, input_window * d_model]
            out = out.view(out.size()[0], -1)
            # feedback through two FC
            out = self.linear1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.linear2(out)

            z_back, z_forecast = out[:, :self.look_back], out[:, self.look_back:]
            res_x = res_x - z_back
            forecast = forecast + z_forecast

        return res_x, forecast


class TrendModule(nn.Module):
    def __init__(self, units, thetas_dim, device, look_back, horizon, tre_layer):
        super(TrendModule, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = look_back
        self.forecast_length = horizon
        self.device = device
        self.tre_layer = tre_layer
        # 4 Linear with relu active function
        self.fc1 = nn.Linear(self.backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.hidden_fc = []
        # get multiple different feature figure
        for i in range(self.thetas_dim):
            fc_feature = nn.Linear(units, thetas_dim)
            self.hidden_fc.append(fc_feature)
        # backcast_linspace[-bl, 0], forecast_linspace[0, fl]
        self.linspace = np.linspace(-self.backcast_length, self.forecast_length, self.backcast_length + self.forecast_length)


    def forward(self, x, mode='train'):
        res_x = x
        forecast = 0

        for j in range(self.tre_layer):
            multi_out = 0
            out = F.relu(self.fc1(res_x))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = F.relu(self.fc4(out))

            for i in range(self.thetas_dim):
                feature_out = F.relu(self.hidden_fc[i].to(self.device)(out))
                current_out = self.trend_model(feature_out, self.linspace, self.device)
                multi_out = multi_out + current_out

            z_back, z_forecast = multi_out[:, :self.backcast_length], multi_out[:, self.backcast_length:]
            res_x = res_x - z_back
            forecast = forecast + z_forecast
        return res_x, forecast

    # linspace ** thetas_dim as trend polynomial coefficient
    def trend_model(self, thetas, t, device):
        p = thetas.size()[-1]
        assert p <= 4, 'thetas_dim is too big.'
        # T shape: [thetas_dim, t_len]
        T = torch.tensor(np.array([t ** i for i in range(p)])).float()
        return thetas.mm(T.to(device))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.embed_dim = d_model

        # linear transformations for query, key, and value
        self.q_linear = nn.Linear(self.embed_dim, n_heads * self.embed_dim)
        self.k_linear = nn.Linear(self.embed_dim, n_heads * self.embed_dim)
        self.v_linear = nn.Linear(self.embed_dim, n_heads * self.embed_dim)

        # linear transformation for concatenated outputs
        self.fc = nn.Linear(self.embed_dim * self.n_heads, self.embed_dim)

    def forward(self, input_q, input_k, input_v, mask=None):
        """
                X_embedding: [batch_size,sequence_length,embedding dimension]
                input_Q: [batch_size, len_q(=len_k=input_window), d_model]
                input_K: [batch_size, len_k, d_model]
                input_V: [batch_size, len_v, d_model]
                attn_mask: [batch_size, seq_len, seq_len]
        """
        batch_size = input_q.shape[0]
        q = self.q_linear(input_q).view(batch_size, -1, self.n_heads, self.embed_dim).transpose(1, 2)
        k = self.k_linear(input_k).view(batch_size, -1, self.n_heads, self.embed_dim).transpose(1, 2)
        v = self.v_linear(input_v).view(batch_size, -1, self.n_heads, self.embed_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim) ** 0.5
        # if mask is not None:
        #     mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        #     attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)

        # concatenate and fully connect outputs
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim * self.n_heads)
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # the shape of pe: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # the shape of x: [batch_size, input_window, 1]
        x = x.unsqueeze(-1)
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        # the shape of x: [batch_size, input_window, d_model]
        return self.dropout(x)


class IP2SDCD(nn.Module):
    def __init__(self, look_back, horizon, dropout, pre_layer, layer_size, res_layer, layer_num, d_model, units, thetas_dim, device, tre_layer):
        """
        look_back: input_window
        horizon: output_window
        dropout: the position_encoding dropout in RM
        pre_layer: the num of Predict Module (seasonal)
        d_model: the dim of position encoding layer and self_attention layer
        layer_size: the kernel_size of FC in Residual Module
        res_layer: the num of Residual Module
        layer_num: the num of stack
        units: the kernel_size of FC in TrendModule
        thetas_dim: the dim of thetas in TrendModule
        device: the running device
        """
        super(IP2SDCD, self).__init__()
        self.TM = TrendModule(units, thetas_dim, device, look_back, horizon, tre_layer)
        self.PM = PredictModule(d_model, look_back, horizon, dropout, pre_layer, thetas_dim)
        self.RM = ResidualModule(look_back, horizon, layer_size, res_layer)
        self.layer_num = layer_num
        self.time = 0

    def forward(self, x, mode="train"):
        res_x = x
        forecast = 0
        trend_forecast = 0
        prediction_forecast = 0
        residual_forecast = 0

        for i in range(self.layer_num):
            res_x, t_forecast = self.TM(res_x, mode=mode)
            res_x, p_forecast = self.PM(res_x)
            res_x, r_forecast = self.RM(res_x)
            # dual residual connection in stack
            forecast = forecast + t_forecast + p_forecast + r_forecast
            trend_forecast = trend_forecast + t_forecast
            prediction_forecast = prediction_forecast + p_forecast
            residual_forecast = residual_forecast + r_forecast

        return forecast, trend_forecast, prediction_forecast, residual_forecast
