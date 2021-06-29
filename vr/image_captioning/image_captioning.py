# %% [code]
####################################################################################################
RESIZE = 356
CROP = 299
####################################################################################################

# %% [code]
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

# %% [code]
import os 
import pandas as pd 
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import sys
import numpy as np
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
spacy_eng = spacy.load("en")

def getNumberOfParameter(model):
    print('Number of trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Total params: ', sum(p.numel() for p in model.parameters()))

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, choose_file, isTrain, transform=None, freq_threshold=2):
        self.root_dir = root_dir
        
        self.df = pd.read_csv(captions_file, sep='\t', names=['image', 'caption'])
        self.df['image'] = self.df["image"].str.split('#', 1, expand=True)[0]
        
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df["caption"].tolist())
        
        #self.df = self.df.groupby('image').first().reset_index()
        
        self.choose_df = pd.read_csv(choose_file, names=['image'])
        if isTrain:
            validation_df = pd.read_csv('../input/flickr8k/Flickr8k_text-20210419T153635Z-001/Flickr8k_text/Flickr_8k.valImages.txt',
                                    names=['image'])
            self.choose_df = pd.concat([self.choose_df, validation_df]).reset_index()
        
        self.choose_df = self.choose_df[:300]
            
        self.df = self.df.loc[self.df['image'].isin(self.choose_df['image'].values)].reset_index(drop=True)
        
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.new('RGB', (RESIZE,RESIZE))
        try:
            img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        except:
            print(img_id)

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        
        return imgs, targets


def get_loader(
    root_folder,
    annotation_file,
    choose_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    isTrain=True
):
    dataset = FlickrDataset(root_folder, annotation_file, choose_file, transform=transform, isTrain=isTrain)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset
    
    

# %% [code]
# Implement CNN and RNN
import torch.nn as nn
import statistics
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, train_CNN, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.encoded_image_size = encoded_image_size
        
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        # Fine tune
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Only train last 2 layers of resnet if at all required
        if train_CNN:
            for c in list(self.resnet.children())[-2:]:
                for p in c.parameters():
                    p.requires_grad = trainCNN
        

    def forward(self, images):
        features = self.resnet(images) 
        features = self.adaptive_pool(features) # batch, 512, encoded_image_size, encoded_image_size
        features = features.permute(0, 2, 3, 1) # batch, encoded_image_size, encoded_image_size, 512
        return features
    

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout, teacher):
        """
        decoder_dim is hidden_size for lstm cell
        """
        super(DecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.teacher = teacher
        
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions):
        """
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Embedding
        if teacher:
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        else:
            preds = torch.ones((batch_size, 1), dtype=torch.int64).to(device)
                
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        decode_length = encoded_captions.size(1)-1
        
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, decode_length, vocab_size).to(device)
        alphas = torch.zeros(batch_size, decode_length, num_pixels).to(device)
        
        for t in range(decode_length):
            if teacher:
                inputEmbeddings = embeddings[:, t, :]
            else:
                inputEmbeddings = self.embedding(preds)[:, 0, :]
            
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h, c = self.decode_step(torch.cat([inputEmbeddings, attention_weighted_encoding], dim=1), (h, c))  #(batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            preds = preds.argmax(1).unsqueeze(1)
            alphas[:, t, :] = alpha

        return predictions, alphas
    


class CNNtoRNN(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=512, dropout=0.5, train_CNN=False, teacher=True):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(train_CNN=train_CNN)
        self.decoderRNN = DecoderRNN(attention_dim, embed_dim, decoder_dim, vocab_size,
                                     encoder_dim=encoder_dim, dropout=dropout, teacher=teacher)

    def forward(self, images, captions):
        encoder_out = self.encoderCNN(images)
        outputs, alphas = self.decoderRNN(encoder_out, captions)
        return outputs, alphas

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = [1]
    
        with torch.no_grad():
            encoder_out = self.encoderCNN(image)
            
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)
            vocab_size = self.decoderRNN.vocab_size
            
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)
            
            # initially start with sos as a predicted word
            predicted = torch.tensor([vocabulary.stoi["<SOS>"]]).to(device)
            h, c = self.decoderRNN.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
            
            for t in range(max_length):
                embeddings = self.decoderRNN.embedding(predicted)  # (1, embed_dim)
                
                attention_weighted_encoding, alpha = self.decoderRNN.attention(encoder_out, h)
                gate = self.decoderRNN.sigmoid(self.decoderRNN.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                
                h, c = self.decoderRNN.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))  #(batch_size_t, decoder_dim)
                preds = self.decoderRNN.fc(self.decoderRNN.dropout(h))  # (batch_size_t, vocab_size)
                    
                predicted = preds.argmax(1)
                result_caption.append(predicted.item())
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
            
            return [vocabulary.itos[idx] for idx in result_caption]
            
            
            

# %% [code]
# Train the model
batch_size=8
transform = transforms.Compose(
    [
        transforms.Resize((RESIZE, RESIZE)),
        transforms.RandomCrop((CROP, CROP)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_loader, dataset = get_loader(
    '../input/flickr8k/Flicker8k_Images-20210419T121716Z-001/Flicker8k_Images',
    '../input/flickr8k/Flickr8k_text-20210419T153635Z-001/Flickr8k_text/Flickr8k.token.txt',
    '../input/flickr8k/Flickr8k_text-20210419T153635Z-001/Flickr8k_text/Flickr_8k.trainImages.txt',
    transform=transform,
    num_workers=8,
    batch_size=batch_size,
    shuffle=True,
    isTrain=True
)

# Hyperparameters
attention_dim = 700
embed_dim = 700
decoder_dim = 700
dropout = 0.5
vocab_size = len(dataset.vocab)
learning_rate = 1e-03
num_epochs = 10
load_model = True
save_model = True
train_CNN = False
alpha_c = 1
teacher = False

def train():

    # for tensorboard
    #writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(attention_dim, embed_dim, decoder_dim, vocab_size,
                     train_CNN=train_CNN, dropout=dropout, teacher=teacher).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
        #step = load_checkpoint(torch.load("../input/flickr8k/my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        
        loss = 400
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)
            torch.save(model.state_dict(), 'puremodel.pth.tar')

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs, alphas = model(imgs, captions.permute(1, 0))
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.permute(1, 0)[:, 1:].reshape(-1))

            #writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            
            # Add doubly stochastic attention regularization
            #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            
            optimizer.step()
        
        print('Epoch {} completed with loss {}'.format(epoch+1, loss))


train()

# %% [code]
test_loader, test_dataset = get_loader(
    '../input/flickr8k/Flicker8k_Images-20210419T121716Z-001/Flicker8k_Images',
    '../input/flickr8k/Flickr8k_text-20210419T153635Z-001/Flickr8k_text/Flickr8k.token.txt',
    '../input/flickr8k/Flickr8k_text-20210419T153635Z-001/Flickr8k_text/Flickr_8k.testImages.txt',
    transform=transform,
    num_workers=8,
    shuffle=False,
    isTrain=False
)

model = CNNtoRNN(attention_dim, embed_dim, decoder_dim, vocab_size, train_CNN=train_CNN, dropout=dropout).to(device)
model.load_state_dict(torch.load("./my_checkpoint.pth.tar")['state_dict'])
#model.load_state_dict(torch.load("../input/flickr8k/my_checkpoint.pth.tar")['state_dict'])
model.eval()

predicted_captions = []
i = 0
for idx, (imgs, captions) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
    for k in range(imgs.shape[0]):
        img = imgs[k].unsqueeze(0)
        real_caption = [dataset.vocab.itos[j.item()] for j in captions[:, k]]
        predicted_captions.append([model.caption_image(img.to(device), dataset.vocab), real_caption])
        i += 1


# %% [code]
i = 0
references_corpus = []
candidate_corpus = []
for e in predicted_captions:
    if i % 5 == 0:
        if i < 21:
            print('Image name: {}'.format(test_dataset.df['image'][i]))
            print('Real caption: ', [ e for e in predicted_captions[i][1] if e != '<PAD>'])
            print('Predicted caption: ', [ e for e in predicted_captions[i][0] if e != '<PAD>'])
            print('\n')
        references_corpus.append([[ e for e in predicted_captions[i][1] if e != '<PAD>']])
        candidate_corpus.append([ e for e in predicted_captions[i][0] if e != '<PAD>'])
        
    else:
        references_corpus[i//5].append([ e for e in predicted_captions[i][1] if e != '<PAD>'])
        
    i+=1

from torchtext.data.metrics import bleu_score
print(bleu_score(candidate_corpus, references_corpus))


# %% [code]
getNumberOfParameter(model)

# %% [code]
import matplotlib.pyplot as plt

base_path = '../input/flickr8k/subjective_img/subjective_img/'
def showAndCaptionImage(img, model):
    
    img = Image.open(base_path + img).convert("RGB")
    plt.imshow(img)
    plt.show()
    img = transform(img)
    caption = model.caption_image(img.unsqueeze(0).to(device), dataset.vocab)[1:-1]
    captionStr = ""
    for e in caption:
        captionStr += e + " "
    print(captionStr)

subjective_images = ['sample1.jpg','sample2.jpg','sample3.jpg','sample4.jpg','sample5.jpg']
for image in subjective_images:
    showAndCaptionImage(image, model)

# %% [markdown]
# 
