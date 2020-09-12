import pdb

import numpy as np
import torch.nn as nn

class CombiNet(nn.Module):
	def __init__(self, in_dim = 2560, hidden_units = 512, out_dim = 2560):
		super().__init__()
		self.fc1 = nn.Linear(in_dim, 2*hidden_units)
		# self.bn1 = nn.BatchNorm1d(hidden_units)
		self.fc2 = nn.Linear(2*hidden_units, 2*hidden_units)
		# self.bn2 = nn.BatchNorm1d(2*hidden_units)
		self.fc3 = nn.Linear(2*hidden_units, out_dim)
		self.relu = nn.ReLU()
		self.apply(weight_init)
	def forward(self, x):
		# out = nn.functional.normalize(x)
		skip = x
		out = self.fc1(x)
		# out = self.bn1(out)
		out = self.relu(out)
		out = self.fc2(out)
		# out = self.bn2(out)
		out = self.relu(out)
		out = self.fc3(out)
		# out = nn.functional.normalize(out)
		out += skip
		return out

class CombiLSTM(nn.Module):
	def __init__(self, in_dim = 2560, hidden_units = 512, out_dim = 2560):
		super().__init__()
		self.in_linear1 = nn.Linear(in_dim, hidden_units)
		# self.bn1 = nn.BatchNorm1d(hidden_units)
		self.in_linear2 = nn.Linear(hidden_units, hidden_units)
		self.rnn = nn.LSTM(input_size = hidden_units, hidden_size = hidden_units, dropout = 0)
		self.out_linear1 = nn.Linear(hidden_units, hidden_units)
		# self.bn2 = nn.BatchNorm1d(hidden_units)
		self.out_linear2 = nn.Linear(hidden_units, out_dim)
		self.relu = nn.ReLU()
		self.apply(weight_init)

	def forward(self, x, hidden = None):
		out = nn.functional.normalize(x)
		skip = out
		out = self.in_linear1(out)
		# out = self.bn1(out)
		out = self.relu(out)
		out = self.in_linear2(out)
		out = out.unsqueeze(1) #Adding batch dimension
		if hidden is None:
			out, hidden = self.rnn(out)
		else:
			out, hidden = self.rnn(out, hidden)

		out = out.squeeze(1) #removing batch dimension
		out = self.out_linear1(out)
		# out = self.bn2(out)
		out = self.relu(out)
		out = self.out_linear2(out)
		out = nn.functional.normalize(out)
		out += skip
		return out, hidden

def weight_init(m):
	if type(m)==nn.Linear:
		nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
	elif type(m)==nn.LSTM:
		nn.init.xavier_normal_(m.weight_ih_l0)
		nn.init.xavier_normal_(m.weight_hh_l0)
