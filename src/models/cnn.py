import torch
import torch.nn.functional as F

# DCNN model
# Giduthuri Sateesh Babu, 2016
class DCNN1d(torch.nn.Module):
    # TODO 1: Conv1d 相当于用 (feature_size, d) 的核沿着 seq 去卷，但是这种做法会相当于把 feature 也卷了，
    #   有点不 make sense；而且feature 的顺序一换，用卷积核卷出来的结果也会有差异。
    # TODO 1: 是否正确的做法是用 Conv2d, 在每个 feature 上沿着 seq 去卷，即 kernel size 为 (1,d), d 为沿着 seq
    #   卷的长度
    # TODO 2: 卷积本质上就是在抽取一定范围的特征，在一定范围内用核去相乘，然后再加，卷积核和数据都应该有正有负，卷积才能更好的发挥作用。
    #   实验一下不同 normalization 不同的范围对结果的影响。
    # TODO 3：看一下卷积是否抽取到了一些 sensor 波动比较大的特征，从而对结果做出一定的解释性。 
    # num_output_features = int((num_input_features - kernel_size + 2 * padding) / stride) + 1
    
    def __init__(self, feature_size, window_size, hidden1, hidden2, kernel_size, stride):
        super().__init__()
        # batch_size x seq_len x features: (256 x 15 x 30) -- > 256 x out_channels1 x (30-4+1)
        self.conv1 = torch.nn.Conv1d(in_channels=feature_size, 
                                     out_channels=hidden1, 
                                     kernel_size=kernel_size, 
                                     stride=stride)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.num_output_features1 = int((window_size - kernel_size) / stride + 1)
        
        # self.conv2 = torch.nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, 
        #                              kernel_size=kernel_size, stride=stride)
        # self.relu2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.MaxPool1d(kernel_size=2)
        # self.num_output_features2 = int((int(self.num_output_features1/2) - kernel_size) / stride + 1)
        
        self.fc1 = torch.nn.Linear(in_features=int(self.num_output_features1/2) * hidden1, out_features=hidden2)
        self.fc2 = torch.nn.Linear(in_features=int(hidden2), out_features=1)

    def forward(self, x):  # x: batch_size x seq_len x features
        x = x.permute(0,2,1) # --> batch_size x features x seq_len, convoluted on the last dimension(seq len)
        # print(f"input size: {x.size()}")  # 256*15*30
        x = self.conv1(x)  
        # print(f"size after the first convolution layer: {x.size()}")
        x = self.relu1(x) 
        x = self.pool1(x) 
        # print(f"size after the first pooling layer: {x.size()}")

        # x = self.conv2(x)
        # # print(f"size after the second convolution layer: {x.size()}")
        # x = self.relu2(x)
        # x = self.pool2(x) 
        # # print(f"size after the second pooling layer: {x.size()}")
        x = torch.flatten(x, 1) 
        # print(f"size after flatting: {x.size()}")
        output = self.fc1(x)
        output = F.relu(output)
        output = self.fc2(output)
        output = output.squeeze(1) # (B, 1) -> (B)
        return output



