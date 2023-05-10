import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        if num_layers > 1:
            self.rnn = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, 
                                      dropout = 0.5, batch_first = True)
        else:
            self.rnn = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
       

    
    def forward(self, features, captions):
   
        captions = captions[:, :-1] 
        embeddings = self.word_embeddings(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 
                                                        dim=1)
        lstm_out, _ = self.rnn(embeddings)
        outputs = self.linear(lstm_out)
        return outputs


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        
        states = (torch.randn(1, 1, 512).to(inputs.device),
              torch.randn(1, 1, 512).to(inputs.device))
        
        while (len(output) < max_len):
            output_rnn,_ = self.rnn(inputs, states)
            outputs = self.linear(output_rnn.squeeze(dim = 1))
   
            _, max_pred_index = torch.max(outputs, dim = 1)
            output.append(max_pred_index.cpu().numpy()[0].item())
            if (max_pred_index == 1):
                break
            inputs = self.word_embeddings(max_pred_index)   
            inputs = inputs.unsqueeze(1)
        return output