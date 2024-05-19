import torch, torch.nn as nn, torch.optim as optim


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

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