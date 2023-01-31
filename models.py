import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv

class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BiLSTM(nn.Module):
    def __init__(self, num_words, embedding_dim = 100, dropout = 0.25):
        super(BiLSTM, self).__init__()
        """
        Given embedding_matrix: numpy array with vector for all words
        return prediction ( in torch tensor format)
        """
        self.embedding = nn.Embedding(
                                      num_embeddings=num_words+1,
                                      embedding_dim=embedding_dim)
        # LSTM with hidden_size = 128
        self.lstm = nn.LSTM(
                            embedding_dim, 
                            128,
                            bidirectional=True,
                            batch_first=True,
                             )
        # Input(512) because we use bi-directional LSTM ==> hidden_size*2 + maxpooling **2  = 128*4 = 512, 
        #will be explained more on forward method
        self.out = nn.Linear(512, 1)
    def forward(self, x):
        # pass input (tokens) through embedding layer
        x = self.embedding(x)
        # fit embedding to LSTM
        self.lstm.flatten_parameters()
        hidden, _ = self.lstm(x)
        # apply mean and max pooling on lstm output
        avg_pool= torch.mean(hidden, 1)
        max_pool, index_max_pool = torch.max(hidden, 1)
        # concat avg_pool and max_pool ( so we have 256 size, also because this is bidirectional ==> 256*2 = 512)
        out = torch.cat((avg_pool, max_pool), 1)
        # fit out to self.out to conduct dimensionality reduction from 512 to 1
        out = self.out(out)
        # return output
        return out

def setup_model(model_architecture, num_classes = None, tokenizer = None, embedding_dim = None):

        available_models = {
            "CNNMNIST": CNNMNIST,
            "BiLSTM": BiLSTM,
            "ResNet18" : tv.models.resnet18,
            "VGG16" : tv.models.vgg16,
            "DN121": tv.models.densenet121,
            "SHUFFLENET":tv.models.shufflenet_v2_x1_0
        }
        print('--> Creating {} model.....'.format(model_architecture))
        # variables in pre-trained ImageNet models are model-specific.
        if "ResNet18" in model_architecture:
            model = available_models[model_architecture]()
            n_features = model.fc.in_features
            model.fc = nn.Linear(n_features, num_classes)
        elif "VGG16" in model_architecture:
            model = available_models[model_architecture]()
            n_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(n_features, num_classes)
        elif "SHUFFLENET" in model_architecture: 
            model = available_models[model_architecture]()
            model.fc = nn.Linear(1024, num_classes)
        elif 'BiLSTM' in model_architecture:
             model = available_models[model_architecture](num_words =  len(tokenizer.word_index), embedding_dim = embedding_dim)
        else:
            model = available_models[model_architecture]()

        if model is None:
            print("Incorrect model architecture specified or architecture not available.")
            raise ValueError(model_architecture)
        print('--> Model has been created!')
        return model