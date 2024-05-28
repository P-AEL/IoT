import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import pandas as pd


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    

# @torch.jit.script
# def train(model, train_dataloader, optimizer: torch.optim.Optimizer=torch.optim.AdamW, loss_fn: nn.Module=nn.MSELoss(), accumulation_steps: int=1, evaluation_steps: int=1000):
    
#     tr_loss = 0
#     for step, batch in enumerate(train_dataloader):
#         model.train()
#         optimizer.zero_grad(set_to_none=True)
#         x, y = batch
#         y_pred = model(x)
#         loss = loss_fn(y_pred, y)
#         loss.backward()
#         if (step+1) % accumulation_steps == 0:
#             optimizer.step()
#             model.zero_grad(set_to_none=True)
#             tr_loss += loss.item()
#             if (step+1) % evaluation_steps == 0:
#                 print(f"Step {step+1}, Loss: {tr_loss/evaluation_steps}")
#                 tr_loss = 0

# def evaluate_model():
#     pass


class PositionalEncoding(torch.nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len,device):
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
        pos = torch.arange(0, max_len,device=device)
        pos = pos.float().unsqueeze(dim=1)
        # # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        return self.encoding
       
class ScaledDotProduct(torch.nn.Module):
    """
    scaled dot product attention class
    """
    def __init__(self):
        """
        constructor of scaled dot product attention class
        """
        super(ScaledDotProduct, self).__init__()
        
    def forward(self, Q, K, V, mask=None):
        """
        forward pass of scaled dot product attention
        :param Q: query tensor
        :param K: key tensor
        :param V: value tensor
        :param mask: mask tensor
        :return: output tensor
        """
        d_k = K.size(-1)
        # get dimension of key
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)
        # compute attention score
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        #     # apply mask to score
        attention = torch.nn.functional.softmax(scores, dim=-1)
        # apply softmax to score
        output = attention @ V
        # compute output tensor
        return output, attention
    
class MultiHeadAttention(torch.nn.Module):
    """
    multihead attention class
    """
    def __init__(self, d_model, num_heads):
        """
        constructor of multihead attention class

        :param d_model: dimension of model
        :param num_heads: number of head in multihead attention
        """
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        # get dimension of key

        self.W_Q = torch.nn.Linear(d_model, d_model)
        self.W_K = torch.nn.Linear(d_model, d_model)
        self.W_V = torch.nn.Linear(d_model, d_model)
        # linear transformation for query, key, value

        self.scaled_dot_product = ScaledDotProduct()
        # scaled dot product attention

        self.linear = torch.nn.Linear(d_model, d_model)
        # linear transformation for output

    def forward(self, Q, K, V, mask=None):
        """
        forward pass of multihead attention

        :param Q: query tensor
        :param K: key tensor
        :param V: value tensor
        :param mask: mask tensor
        :return: output tensor
        """
        batch_size = Q.size(0)
        # get batch size

        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # linear transformation and split into multihead

        if mask is not None:
            mask = mask.unsqueeze(1)
            # unsqueeze mask

        output, attention = self.scaled_dot_product(Q, K, V, mask)
        # scaled dot product attention

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # concatenate multihead attention

        return self.linear(output), attention
    
class Decoder(torch.nn.Module):
    """
    decoder layer class
    """
    def __init__(self, input,d_model,max_len,num_heads,d_ff,device):
        """
        constructor of decoder layer

        :param d_model: dimension of model
        :param num_heads: number of head in multihead attention
        :param d_ff: dimension of feed forward layer
        :param dropout: dropout rate
        """
        super(Decoder, self).__init__()

        self.embed = torch.nn.Linear(input, d_model).to(device)
        self.positonal_encoding = PositionalEncoding(d_model, max_len=max_len,device=device).to(device)
        self.norm = torch.nn.LayerNorm(d_model).to(device)
        self.attn = MultiHeadAttention(d_model, num_heads).to(device)
        self.norm_2 = torch.nn.LayerNorm(d_model).to(device)
        self.linear1 = torch.nn.Linear(d_model, d_model).to(device)
        self.Relu = torch.nn.SELU().to(device)
        self.attn2 = MultiHeadAttention(d_model, num_heads).to(device)
        self.norm_3 = torch.nn.LayerNorm(d_model).to(device)
        self.linear2 = torch.nn.Linear(d_model, d_model).to(device)
        self.attn3 = MultiHeadAttention(d_model, num_heads).to(device)
        self.ff = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_ff),
            torch.nn.SELU(),
            torch.nn.Linear(d_ff, 1)
        ).to(device)

    def forward(self, x):
        """
        forward pass of decoder layer

        :param x: input tensor (query)
        :param memory: input tensor (key, value)
        :param src_mask: source mask
        :param tgt_mask: target mask
        :return: output tensor
        """
        x = self.embed(x)
        x = x + self.positonal_encoding(x)
        x_norm = self.norm(x)
        x_att, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + x_att
        x = self.norm_2(x)
        x_att2, _ = self.attn2(x, x, x)
        x_att2 = self.linear1(x_att2)
        x_att2 = self.Relu(x_att2)
        x = x + x_att2
        x = self.norm_3(x)
        x_att3, _ = self.attn3(x, x, x)
        x_att3 = self.linear2(x_att3)
        x_att3 = self.Relu(x_att3)
        x = x + x_att3
        x = self.ff(x) #shape am ende noch (500,1), soll des so ?
        return x[:, -1].squeeze()
    
def evaluate_model(model, columns, window_size, device_id):
    """
    args:   model: torch.nn.Module
            columns: on which columns the model was trained
            window_size: lookback window size
            device_id: Room

    returns: dataframe
    """
    if model == "lstm":
        #model = LSTM(input_size=len(columns), hidden_size=128, num_layers=2, output_size=1)
        print("not implemented")
    elif model == "rnn":
        #model = RNN(input_size=len(columns), hidden_size=128, num_layers=2, output_size=1)
        print("not implemented")
    elif model == "transformer":
        device = torch.device('cpu')
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Decoder(input=len(columns),d_model=128,max_len=window_size,num_heads=4,d_ff=120,device=device)
        model.load_state_dict(torch.load('Decoder1.pth', map_location=device)) # map_device weil cpu only
        model.float()
    df = pd.read_csv('aggregated_hourly.csv')
    df = df[df["device_id"] == device_id]
    df = df[columns+["date_time"]]
    min_date = df['date_time'].min()
    max_date = df['date_time'].max()
    hourly_range = pd.date_range(start=min_date, end=max_date, freq='h')
    #missing_hours = hourly_range[~hourly_range.isin(df['date_time'])]
    missing_hours = hourly_range[~hourly_range.isin(df['date_time'].astype(hourly_range.dtype))]
    df.date_time = pd.to_datetime(df.date_time)
    for i in missing_hours:
        df_temp = df.loc[df["date_time"] < i][columns].copy()
        df_temp = df_temp.astype(float)
        df_temp_values = df_temp.values
        if len(df_temp_values) <= window_size:
            X = torch.stack([torch.cat((torch.zeros(window_size-len(df_temp_values), df_temp_values.shape[1]), torch.from_numpy(df_temp_values)), dim=0)]).float()
        else:
            X = torch.stack([torch.from_numpy(df_temp_values[-window_size:])]).float()
        model.eval()
        y_pred = model(X)
        new_row = df_temp.iloc[-1].copy()
        new_row.at['date_time'] = i
        index_of_tmp = columns.index("tmp")
        #new_row[index_of_tmp] = y_pred.item()
        new_row.iloc[index_of_tmp] = y_pred.item()
        new_row_series = pd.Series(new_row, index=df.columns)
        df = pd.concat([df, new_row_series.to_frame().T], ignore_index=True)
    return df


def evaluate_model_new(model, df, columns, window_size):
    """
    args:   model: torch.nn.Module
            df: preprocessed dataframe
            columns: on which columns the model was trained
            window_size: lookback window size
            device_id: Room

    returns: dataframe
    """
    device = torch.device('cpu')

    if model == "transformer":
        model = Decoder(input=len(columns),d_model=128,max_len=window_size,num_heads=4,d_ff=120,device=device)
        model.load_state_dict(torch.load('Decoder1.pth', map_location=device))
        model.float()
    else:
        print("not implemented")
        return

    hourly_range = pd.date_range(start=df['date_time'].min(), end=df['date_time'].max(), freq='h')
    missing_hours = hourly_range[~hourly_range.isin(df['date_time'].astype(hourly_range.dtype))]

    for i in missing_hours:
        df_temp = df.loc[df["date_time"] < i][columns].astype(float)
        X = torch.stack([torch.cat((torch.zeros(max(0, window_size-len(df_temp)), len(columns)), torch.from_numpy(df_temp.values[-window_size:])), dim=0)]).float()
        new_row = df_temp.iloc[-1].copy()
        new_row.at['date_time'] = i
        new_row.iloc[columns.index("tmp")] = model(X).item()
        df = pd.concat([df, pd.DataFrame(new_row).T], ignore_index= True)

    return df