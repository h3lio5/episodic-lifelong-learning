import torch
import torch.utils.data as data
from data_loader import DataSet
import argparse
from baselines.MbPA import ReplayMemory, MbPA
import transformers
from tqdm import trange, tqdm
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
use_cuda = True if torch.cuda.is_available() else False
# Use cudnn backends instead of vanilla backends when the input sizes
# are similar so as to enable cudnn which will try to find optimal set
# of algorithms to use for the hardware leading to faster runtime.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='Enter the batch size')
parser.add_argument('--mode', default='train',
                    help='Enter the mode - train/eval')
parser.add_argument('--order', default=1, type=int,
                    help='Enter the dataset order - 1/2/3/4')
parser.add_argument('--epochs', default=2, type=int)
args = parser.parse_args()
LEARNING_RATE = 3e-5

MODEL_NAME = 'MbPA'


def train(order, model, memory):
    """
    """
    workers = 0
    if use_cuda:
        model.cuda()
        # Number of workers should be 4*num_gpu_available
        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
        workers = 4
    # time at the start of training
    start = time.time()

    train_data = DataSet(order, split='train')
    train_sampler = data.SequentialSampler(train_data)
    train_dataloader = data.DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size, num_workers=workers)
    param_optimizer = list(model.classifier.named_parameters())
    # parameters that need not be decayed
    no_decay = ['bias', 'gamma', 'beta']
    # Grouping the parameters based on whether each parameter undergoes decay or not.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]
    optimizer = transformers.AdamW(
        optimizer_grouped_parameters, lr=LEARNING_RATE)
    scheduler = transformers.WarmupLinearSchedule(
        optimizer, warmup_steps=100, t_total=1000)
    # Store our loss and accuracy for plotting
    train_loss_set = []
    # trange is a tqdm wrapper around the normal python range
    for epoch in trange(args.epochs, desc="Epoch"):
         # Training begins
        print("Training begins")
        # Set our model to training mode (as opposed to evaluation mode)
        model.classifier.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps, num_curr_exs = 0, 0, 0
        # Train the data for one epoch
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Release file descriptors which function as shared
            # memory handles otherwise it will hit the limit when
            # there are too many batches at dataloader
            batch_cp = copy.deepcopy(batch)
            del batch
            # Unpacking the batch items
            content, attn_masks, labels = batch_cp
            content = content.squeeze(1)
            attn_masks = attn_masks.squeeze(1)
            labels = labels.squeeze(1)
            # number of examples in the current batch
            num_curr_exs = content.size(0)
            # Place the batch items on the appropriate device: cuda if avaliable
            if use_cuda:
                content = content.cuda()
                attn_masks = attn_masks.cuda()
                labels = labels.cuda()
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, _ = model.classify(content, attn_masks, labels)
            train_loss_set.append(loss.item())
            # Get the key representation of documents
            keys = model.get_keys(content, attn_masks)
            # Push the examples into the replay memory
            memory.push(keys.cpu().numpy(), (content.cpu().numpy(),
                                             attn_masks.cpu().numpy(), labels.cpu().numpy()))
            # delete the batch data to freeup gpu memory
            del keys
            del content
            del attn_masks
            del labels
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += num_curr_exs
            nb_tr_steps += 1

        now = time.time()
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        print("Time taken till now: {} hours".format((now-start)/3600))
        model_dict = model.save_state()
        save_checkpoint(model_dict, order, epoch+1, memory=memory.memory)

    save_trainloss(train_loss_set, order)


def save_checkpoint(model_dict, order, epoch, memory=None, base_loc='../model_checkpoints/'):
    """
    Function to save a model checkpoint to the specified location
    """
    checkpoints_dir = base_loc + MODEL_NAME
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_file = 'classifier_order_' + \
        str(order) + '_epoch_'+str(epoch)+'.pth'
    torch.save(model_dict, os.path.join(checkpoints_dir, checkpoints_file))
    if memory is not None:
        with open(checkpoints_dir+'/order_'+str(order)+'_epoch_'+str(epoch)+'.pkl', 'wb') as f:
            pickle.dump(memory, f)


def calc_correct(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def test(order, model, memory):
    """
    evaluate the model for accuracy
    """
    # time at the start of validation
    start = time.time()
    if use_cuda:
        model.cuda()

    test_data = DataSet(order, split='test')
    test_dataloader = data.DataLoader(
        test_data, shuffle=True, batch_size=args.batch_size)

    # Tracking variables
    total_correct, tmp_correct, t_steps = 0, 0, 0

    print("Validation step started...")
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch_cp = copy.deepcopy(batch)
        del batch
        contents, attn_masks, labels = batch_cp
        print("before batch put on cuda ", torch.cuda.memory_allocated())
        if use_cuda:
            contents = contents.squeeze(1).cuda()
            attn_masks = attn_masks.squeeze(1).cuda()
        print("after batch put on cuda ", torch.cuda.memory_allocated())
        keys = model.get_keys(contents, attn_masks)
        print("after getting key and batch on cuda ",
              torch.cuda.memory_allocated())
        # contents = contents.cpu()
        # attn_masks = attn_masks.cpu()
        retrieved_batches = memory.get_neighbours(keys.cpu().numpy())
        del keys
        ans_logits = []
        # Iterate over the test batch to calculate label for each document(i.e,content)
        # and store them in a list for comparision later
        for content, attn_mask, (rt_contents, rt_attn_masks, rt_labels) in zip(contents, attn_masks, retrieved_batches):
            if use_cuda:
                # content = content.cuda()
                # attn_mask = attn_mask.cuda()
                rt_contents = rt_contents.cuda()
                rt_attn_masks = rt_attn_masks.cuda()
                rt_labels = rt_labels.cuda()
            print("after rt_batch and doc,attn_mask on cuda ",
                  torch.cuda.memory_allocated())
            logits = model.infer(content, attn_mask,
                                 rt_contents, rt_attn_masks, rt_labels)

            ans_logits.append(logits.cpu().numpy())
        # Dropping the 1 dim to match the logits' shape
        # shape : (batch_size,num_labels)
        labels = labels.squeeze(1).numpy()
        tmp_correct = calc_correct(np.asarray(ans_logits), labels)
        del labels
        total_correct += tmp_correct
        t_steps += len(labels.flatten())
    end = time.time()
    print("Time taken for validation {} minutes".format((end-start)/60))
    print("Validation Accuracy: {}".format(total_correct/t_steps))


def save_trainloss(train_loss_set, order, base_loc='../loss_images/'):
    """
    Function to save the image of training loss v/s iterations graph
    """
    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    image_dir = base_loc + MODEL_NAME
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    plt.savefig(train_loss_set, image_dir+'/order_' +
                str(order)+'_train_loss.png')


if __name__ == '__main__':

    if args.mode == 'train':
        model = MbPA()
        memory = ReplayMemory()
        train(args.order, model, memory)

    if args.mode == 'test':
        model_state = torch.load(
            '../model_checkpoints/MbPA/classifier_order_1_epoch_2.pth')
        model = MbPA(model_state=model_state)
        buffer = {}
        with open('../model_checkpoints/MbPA/order_1_epoch_2.pkl', 'rb') as f:
            buffer = pickle.load(f)
        memory = ReplayMemory(buffer=buffer)
        test(args.order, model, memory)
