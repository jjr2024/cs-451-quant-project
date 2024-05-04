from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, window):
        self.data1, self.data2 = data
        self.window = window

    def __getitem__(self, index):
        x = self.data1[index:index+self.window]
        y = self.data2[index:index+self.window]
        return x, y

    def __len__(self):
        return len(self.data1) - self.window