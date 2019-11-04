import torch
import torch.utils.data as data
from data_loader import DataSet
import argparse
from baselines.enc_dec import EncDec
import transformers
from tqdm import trange
import time
import matplotlib.pyplot as plt
use_cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=16,help='Enter the batch size')
parser.add_argument('--mode',default='train',help='Enter the mode - train/eval')
parser.add_argument('--order',default=1,type=int,help='Enter the dataset order - 1/2/3/4')
parser.add_argument('--epochs',default=4,type=int)
args = parser.parse_args()
LEARNING_RATE = 3e-5

MODEL_NAME = 'enc_dec'

def train(order,model):
    """
    """
    if use_cuda:
        model.cuda()
    # time at the start of training    
    start = time.time()

    train_data = DataSet(order,split='train')
    train_sampler = data.SequentialSampler(train_data)
    train_dataloader = data.DataLoader(train_data,sampler=train_sampler,batch_size=args.batch_size)
    param_optimizer = list(model.classifier.named_parameters())
    # parameters that need not be decayed
    no_decay = ['bias', 'gamma', 'beta']
    # Grouping the parameters based on whether each parameter undergoes decay or not.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0} ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters,lr=LEARNING_RATE)
    scheduler = transformers.WarmupLinearSchedule(optimizer,warmup_steps=100,t_total=1000)
    # Store our loss and accuracy for plotting
    train_loss_set = []
    # trange is a tqdm wrapper around the normal python range
    for epoch in trange(args.epochs,desc="Epoch"):
         # Training begins

         # Set our model to training mode (as opposed to evaluation mode)
        model.classifier.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
         # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Unpacking the batch items
            content,attn_masks,labels = batch
            print("Epoch ",epoch+1,"step ",step+1)
            # Place the batch items on the appropriate device: cuda if avaliable
            if use_cuda:
                content = content.cuda()
                attn_masks = attn_masks.cuda()
                labels = labels.cuda()
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss,logits = model.classify(content.squeeze(1),attn_masks.squeeze(1),labels.squeeze(1))
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += content.size(0)
            nb_tr_steps += 1

        now = time.time()
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        print("Time taken till now: {} hours".format((now-start)/3600))
        torch.save(model.classifier.state_dict(),'../model_checkpoints/'+MODEL_NAME+'/classifier_order_'+str(order)+'epoch_'+str(epoch)+'.pth')
    save_trainloss(train_loss_set)    

# Function to calculate the accuracy of our predictions vs labels
def num_correct(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) 

def test(order,model):
    """
    evaluate the model for accuracy
    """

    if use_cuda:
        model.cuda()

    test_data = DataSet(order,split='test')
    test_sampler = data.SequentialSampler(test_data)
    test_dataloader = data.DataLoader(test_data,sampler=test_sampler,batch_size=args.batch_size)

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    print("Validation step started...")
    for step,batch in enumerate(test_dataloader):
        
        print("Step",step)
        content,attn_masks,labels = batch

        if use_cuda:
            content.cuda()
            attn_masks.cuda()
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model.infer(content.squeeze(1),attn_masks.squeeze(1))

        logits = logits.detach().cpu().numpy()
        # Dropping the 1 dim to match the logits' shape
        # shape : (batch_size,num_labels)
        labels = labels.squeeze(1)
        tmp_correct = num_correct(logits, labels)
        total_correct += tmp_correct
        total_samples += len(labels)

    print("Validation Accuracy: {}".format(total_correct/total_samples))


def save_trainloss(train_loss_set):

    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.savefig('../loss_images/'+MODEL_NAME+'/train_loss.png')


if __name__ == '__main__':

    model = EncDec()

    if args.mode == 'train':
        train(args.order,model)
       
    if args.mode == 'test':
        model_state = torch.load('../model_checkpoints/enc_dec/enc_dec_classifier_1epoch_3.pth')
        model = EncDec(mode='test',model_state=model_state)
        test(args.order,model)
