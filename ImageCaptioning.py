
# coding: utf-8

# In[1]:


from collections import Counter, defaultdict
from gensim.models import Word2Vec
from IPython import display
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms

import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F



# In[2]:


# Define a global transformer to appropriately scale images and subsequently convert them to a Tensor.
img_size = 224
loader = transforms.Compose([
  transforms.Scale(img_size),
  transforms.CenterCrop(img_size),
  transforms.ToTensor(),
]) 
def load_image(filename, volatile=False):
    """
    Simple function to load and preprocess the image.

    1. Open the image.
    2. Scale/crop it and convert it to a float tensor.
    3. Convert it to a variable (all inputs to PyTorch models must be variables).
    4. Add another dimension to the start of the Tensor (b/c VGG expects a batch).
    5. Move the variable onto the GPU.
    """
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor, volatile=volatile).unsqueeze(0)
    return image_var.cuda()
   

load_image('data/train2014/COCO_train2014_000000000009.jpg')


# In[3]:


# Load annotations file for the training images.
mscoco_train = json.load(open('data/annotations/train_captions.json'))
train_ids = [entry['id'] for entry in mscoco_train['images']]
train_id_to_file = {entry['id']: 'data/train2014/' + entry['file_name'] for entry in mscoco_train['images']}

# Extract out the captions for the training images
train_id_set = set(train_ids)
train_id_to_captions = defaultdict(list)
for entry in mscoco_train['annotations']:
    if entry['image_id'] in train_id_set:
        train_id_to_captions[entry['image_id']].append(entry['caption'])

# Load annotations file for the validation images.
mscoco_val = json.load(open('data/annotations/val_captions.json'))
val_ids = [entry['id'] for entry in mscoco_val['images']]
val_id_to_file = {entry['id']: 'data/val2014/' + entry['file_name'] for entry in mscoco_val['images']}

# Extract out the captions for the validation images
val_id_set = set(val_ids)
val_id_to_captions = defaultdict(list)
for entry in mscoco_val['annotations']:
    if entry['image_id'] in val_id_set:
        val_id_to_captions[entry['image_id']].append(entry['caption'])

# Load annotations file for the testing images
mscoco_test = json.load(open('data/annotations/test_captions.json'))
test_ids = [entry['id'] for entry in mscoco_test['images']]
test_id_to_file = {entry['id']: 'data/val2014/' + entry['file_name'] for entry in mscoco_test['images']}


# # Preprocessing
# 
# We do the same preprocessing done in assignment 3. 

# In[4]:


sentences = [sentence for caption_set in train_id_to_captions.values() for sentence in caption_set]

# Lower-case the sentence, tokenize them and add <SOS> and <EOS> tokens
sentences = [["<SOS>"] + word_tokenize(sentence.lower()) + ["<EOS>"] for sentence in sentences]

# Create the vocabulary. Note that we add an <UNK> token to represent words not in our vocabulary.
vocabularySize = 1000
word_counts = Counter([word for sentence in sentences for word in sentence])
vocabulary = ["<UNK>"] + [e[0] for e in word_counts.most_common(vocabularySize-1)]
word2index = {word:index for index,word in enumerate(vocabulary)}
one_hot_embeddings = np.eye(vocabularySize)

# Build the word2vec embeddings
wordEncodingSize = 300
filtered_sentences = [[word for word in sentence if word in word2index] for sentence in sentences]
w2v = Word2Vec(filtered_sentences, min_count=0, size=wordEncodingSize)
w2v_embeddings = np.concatenate((np.zeros((1, wordEncodingSize)), w2v.wv.syn0))

# Define the max sequence length to be the longest sentence in the training data. 
maxSequenceLength = max([len(sentence) for sentence in sentences])

def preprocess_numberize(sentence):
    """
    Given a sentence, in the form of a string, this function will preprocess it
    into list of numbers (denoting the index into the vocabulary).
    """
    tokenized = word_tokenize(sentence.lower())
        
    # Add the <SOS>/<EOS> tokens and numberize (all unknown words are represented as <UNK>).
    tokenized = ["<SOS>"] + tokenized + ["<EOS>"]
    numberized = [word2index.get(word, 0) for word in tokenized]
    
    return numberized

def preprocess_one_hot(sentence):
    """
    Given a sentence, in the form of a string, this function will preprocess it
    into a numpy array of one-hot vectors.
    """
    numberized = preprocess_numberize(sentence)
    
    # Represent each word as it's one-hot embedding
    one_hot_embedded = one_hot_embeddings[numberized]
    
    return one_hot_embedded

def preprocess_word2vec(sentence):
    """
    Given a sentence, in the form of a string, this function will preprocess it
    into a numpy array of word2vec embeddings.
    """
    numberized = preprocess_numberize(sentence)
    
    # Represent each word as it's one-hot embedding
    w2v_embedded = w2v_embeddings[numberized]
    
    return w2v_embedded

def compute_bleu(reference_sentences, predicted_sentence):
    """
    Given a list of reference sentences, and a predicted sentence, compute the BLEU similary between them.
    """
    reference_tokenized = [word_tokenize(ref_sent.lower()) for ref_sent in reference_sentences]
    predicted_tokenized = word_tokenize(predicted_sentence.lower())
    return sentence_bleu(reference_tokenized, predicted_tokenized)


# # 1. Setup Image Encoder
# 
# We load in the pre-trained VGG-16 model, and remove the final layer, as done in assignment 2.

# In[5]:



encoder = models.vgg16(pretrained=True)
encoder.eval()

encoder.classifier = nn.Sequential(*list(encoder.classifier.children())[:-2])

for param in encoder.parameters():   
    param.requires_grad = False

encoder = encoder.cuda()


# # 2. Setup a Language Decoder
# 
# We're going to reuse our decoder from Assignment 3.

# In[6]:


class DecoderLSTM(nn.Module):
    def __init__(self, encoderOutput, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoderOutput = encoderOutput

        self.linear1 = nn.Linear(encoderOutput, hidden_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers = 1)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden, firstRun = True):
        
        if firstRun:
            hidden = self.linear1(hidden)
            hidden = (hidden, hidden)
            
        
        output = F.relu(input)
        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.linear(output[0]))
        return output, hidden
    
   
    
    def initHidden(self):
  
        hidden = Variable(torch.zeros(1,1, self.hidden_size))
       
        return hidden.cuda()
       


# # 3. Train encoder-decoder
# 
# 

# In[7]:


def FullModelTrain(target_variable, image, decoder, decoder_optimizer, encoder, criterion, embeddings=one_hot_embeddings): 
    
    softmax = nn.Softmax()
    encoder_output = encoder(image).unsqueeze(0)
    
    for i in range(5):
        
        target_variable2 = preprocess_one_hot(target_variable[i])

        decoder_optimizer.zero_grad()
        target_length = target_variable2.shape[0]
        
        loss = 0
            
        decoder_input = Variable(torch.FloatTensor(target_variable2[0]).unsqueeze(0).unsqueeze(0))
        decoder_input = decoder_input.cuda()
        
        dhidden = (decoder.initHidden() , decoder.initHidden())
        #dhidden = dhidden.cuda()
        
     
   
        for di in range(1, target_length):
        
            if di == 1:
                decoder_output, dhidden = decoder(decoder_input, encoder_output, firstRun = True)
            
            else:
                decoder_output, dhidden = decoder(decoder_input, dhidden, firstRun = False)
                
            
            output = softmax(decoder_output)
            topv, topi = output.data.topk(1)
            topi = topi[0]
            
    
            decoder_input = Variable(torch.FloatTensor(target_variable2[di]).unsqueeze(0).unsqueeze(0))
            decoder_input = decoder_input.cuda() 
            ti = int(np.argmax(target_variable2[di]))
            t = Variable(torch.LongTensor(1))
            t.data[0] = ti
            t = t.cuda()
            
            loss += criterion(decoder_output, t)
       
            if topi[0] == word2index["<EOS>"]: #EOS index
                break
                
        torch.nn.utils.clip_grad_norm(decoder.parameters(), 10.0)
        loss.backward()
        decoder_optimizer.step()
    
    return loss.data[0] / target_length


# In[8]:


decoder = DecoderLSTM(4096, 512, 1000)
optimizerD = torch.optim.Adam(decoder.parameters(), 0.001)
criterion = nn.CrossEntropyLoss().cuda()
decoder = decoder.cuda()


# In[14]:


print_every=1000
iter = 0
print_loss_total = 0  

#for h in range (2):
for image_id in train_ids:
    img = load_image(train_id_to_file[image_id])
    captionList = train_id_to_captions[image_id]
    loss = FullModelTrain(captionList, img, decoder, optimizerD, encoder, criterion)
    
    print_loss_total += loss
    if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        #print('%d %.4f' % (iter, print_loss_avg))
    
    iter = iter +1
    

print("DONE")


# In[10]:


torch.save(decoder, 'decoderAss4.pt')
#decoder = torch.load('decoderAss4.pt')


# # 4. MAP and Sampling Inference
# 

# In[11]:


def inference(decoder, encoder, image, embeddings=one_hot_embeddings, max_length=maxSequenceLength):
    
    encoder_output = encoder(image).unsqueeze(0)
    
    SOSToken = one_hot_embeddings[word2index.get("<SOS>", 0)]

    softmax = nn.Softmax()
       
    decoder_input = Variable(torch.FloatTensor(SOSToken).unsqueeze(0).unsqueeze(0))
    decoder_input = decoder_input.cuda() 
    decoded_words = []

    dhidden = (decoder.initHidden() , decoder.initHidden())
    #dhidden = dhidden.cuda()
   
    for di in range(1, max_length):
        
        if di == 1:
            decoder_output, dhidden = decoder(decoder_input, encoder_output, firstRun = True)
            
        else:
            decoder_output, dhidden = decoder(decoder_input, dhidden, firstRun = False)
                
        output = softmax(decoder_output)
        topv, topi = output.data.topk(1)
        topi = topi[0]            
           
        if topi[0] == word2index["<EOS>"]: #EOS index
            break
        else:    
            decoded_words.append(vocabulary[topi[0]])
        
        newWord = one_hot_embeddings[topi[0]]
        decoder_input = Variable(torch.FloatTensor(newWord).unsqueeze(0).unsqueeze(0))
        decoder_input = decoder_input.cuda() 

     
    return ' '.join(decoded_words)
        


# In[12]:


def Sample_inference(decoder, encoder, image, embeddings=one_hot_embeddings, max_length=maxSequenceLength):
    
    encoder_output = encoder(image).unsqueeze(0)
    
    SOSToken = one_hot_embeddings[word2index.get("<SOS>", 0)]

    softmax = nn.Softmax()
    decoded_sentences = []
    
    for i in range (5):
        
       
        decoder_input = Variable(torch.FloatTensor(SOSToken).unsqueeze(0).unsqueeze(0))
        decoder_input = decoder_input.cuda() 
        decoded_words = []

        dhidden = (decoder.initHidden() , decoder.initHidden())
        #dhidden = dhidden.cuda()
   
        for di in range(1, max_length):
        
            if di == 1:
                decoder_output,  dhidden = decoder(decoder_input, encoder_output, firstRun = True)
            
            else:
                decoder_output, dhidden = decoder(decoder_input, dhidden, firstRun = False)
                
            output = softmax(decoder_output)
            preds = output.data.cpu().numpy()
            
            ind = np.random.choice(np.arange(1000), 1, p = preds[0])    
           
            if ind[0] == word2index["<EOS>"]: #EOS index
                break
            else:    
                decoded_words.append(vocabulary[ind[0]])
        
            newWord = one_hot_embeddings[ind[0]]
            decoder_input = Variable(torch.FloatTensor(newWord).unsqueeze(0).unsqueeze(0))
            decoder_input = decoder_input.cuda() 

     
        sen =  ' '.join(decoded_words)
        decoded_sentences.append(sen)
        
    return decoded_sentences


# In[15]:


img = load_image(train_id_to_file[57870])
print("Ground truth captions: ")
for k in train_id_to_captions[57870]:
    print(k)
print("\n predicted caption: ")
print(inference(decoder, encoder, img))
print("\n predicted sampled captions: ")
print(Sample_inference(decoder, encoder, img))
print("-------------------")

img = load_image(train_id_to_file[222016])
print("Ground truth captions: ")
for k in train_id_to_captions[222016]:
    print(k)
print("\n predicted caption: ")
print(inference(decoder, encoder, img))
print("\n predicted sampled captions: ")
print(Sample_inference(decoder, encoder, img))
print("-------------------")


img = load_image(train_id_to_file[69675])
print("Ground truth captions: ")
for k in train_id_to_captions[69675]:
    print(k)
print("\n predicted caption: ")
print(inference(decoder, encoder, img))
print("\n predicted sampled captions: ")
print(Sample_inference(decoder, encoder, img))


# # 5. Evaluate performance
# 
# For validation images compute the average BLEU score.

# In[17]:


sumOfBleu = 0
i = 1

for image_id in val_ids[:100]:
    img = load_image(val_id_to_file[image_id])
    captionList = val_id_to_captions[image_id]
    predicted_sentence = inference(decoder, encoder, img)
    
    for j in range (5):
        #print("ground truth: ", captionList[j])
        #print("predicted: ", predicted_sentence)
        #print("BLEU score: ", compute_bleu(predicted_sentence, captionList[j] ))
        sumOfBleu = sumOfBleu + compute_bleu(predicted_sentence, captionList[j] )
        i = i+1


avgBleu = sumOfBleu/(i)    
print("Average BLEU Score: " ,avgBleu)
        

