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
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = 20
    
    def forward(self, features, captions):
        ### features is pytorch of size torch.Size([batch_size, 256])
        ### captions is pytorch of size torch.Size([batch_size, 11])
        ### images are of shape and size torch.Size([batch_size, 3, 224, 224])
        
        """Decode image feature vectors and generates captions."""
        # print("embed size is", embed_size, "\n hidden size is :", hidden_size,"\n vocab size is :",vocab_size, "\n features size is :",
             # features.shape,"\n caption size is :", captions.shape)
        captions1 = captions[:,:-1]
        embeddings = self.embed(captions1)
        #print("embeddings with captions shape is", embeddings.shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #print("embeddings with features shape is", embeddings.shape)
        #print("length shape is", torch.ones(()).new_full((captions.shape[1],1),captions.shape[2]).shape)
        #packed = pack_padded_sequence(embeddings, torch.ones(()).new_full((10,1),256), batch_first=True) 
        hiddens, out2 = self.lstm(embeddings)
        #print('output from lstm has the shape',hiddens.shape,'and',[x.shape for x in out2])
        scores = self.linear(hiddens)
        #print("output from the linear layer has the shape",scores.shape)
        #print('the target output has the shape',[batch_size, captions.shape[1], vocab_size])
        outputs = scores
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids=[]
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # Get the index (in the vocabulary) of the most likely integer that
            # represents a word
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids