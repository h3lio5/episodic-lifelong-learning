import torch
import torch.utils.data as data
from data_loader import DataSet
import argparse
from models.MbPAplusplus import ReplayMemory, MbPAplusplus
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
                    help='Enter the mode - train/test')
parser.add_argument('--order', default=1, type=int,
                    help='Enter the dataset order - 1/2/3/4')
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--model_path', type=str,
                    help='Enter the path to the model weights')
parser.add_argument('--memory_path', type=str,
                    help='Enter the path to the replay memory')

args = parser.parse_args()
LEARNING_RATE = 3e-5
MODEL_NAME = 'MbPA++'
# Due to memory restraint, we sample only 64 examples from
# stored memory after every 6400(1% replay rate) new examples seen
# as opposed to 100 suggested in the paper. The sampling is done after
# performing 200 steps(6400/32).
REPLAY_FREQ = 201


def train(order, model, memory):
    """
    Train function
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
            # Perform sparse experience replay after every REPLAY_FREQ steps
            if (step+1) % REPLAY_FREQ == 0:
                # sample 64 examples from memory
                content, attn_masks, labels = memory.sample(sample_size=64)
                if use_cuda:
                    content = content.cuda()
                    attn_masks = attn_masks.cuda()
                    labels = labels.cuda()
                 # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, logits = model.classify(content, attn_masks, labels)
                train_loss_set.append(loss.item())
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += content.size(0)
                nb_tr_steps += 1

                del content
                del attn_masks
                del labels
                del loss
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


def save_checkpoint(model_dict, order, epoch, memory=None):
    """
    Function to save a model checkpoint to the specified location
    """
    base_loc = './model_checkpoints'
    if not os.path.exists(base_loc):
        os.mkdir('model_checkpoints')

    checkpoints_dir = base_loc + '/' + MODEL_NAME
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
        test_data, shuffle=True, batch_size=64, num_workers=4)

    # Tracking variables
    total_correct, tmp_correct, t_steps = 0, 0, 0

    print("Validation step started...")
    for batch in tqdm(test_dataloader):
        batch_cp = copy.deepcopy(batch)
        del batch
        contents, attn_masks, labels = batch_cp
        if use_cuda:
            contents = contents.squeeze(1).cuda()
            attn_masks = attn_masks.squeeze(1).cuda()
        keys = model.get_keys(contents, attn_masks)
        retrieved_batches = memory.get_neighbours(keys.cpu().numpy())
        del keys
        ans_logits = []
        # Iterate over the test batch to calculate label for each document(i.e,content)
        # and store them in a list for comparision later
        for content, attn_mask, (rt_contents, rt_attn_masks, rt_labels) in tqdm(zip(contents, attn_masks, retrieved_batches), total=len(contents)):
            if use_cuda:
                rt_contents = rt_contents.cuda()
                rt_attn_masks = rt_attn_masks.cuda()
                rt_labels = rt_labels.cuda()

            logits = model.infer(content, attn_mask,
                                 rt_contents, rt_attn_masks, rt_labels)

            ans_logits.append(logits.cpu().numpy())
        # Dropping the 1 dim to match the logits' shape
        # shape : (batch_size,num_labels)
        labels = labels.squeeze(1).numpy()
        tmp_correct = calc_correct(np.asarray(ans_logits), labels)
        # del labels
        total_correct += tmp_correct
        t_steps += len(labels.flatten())
    end = time.time()
    print("Time taken for validation {} minutes".format((end-start)/60))
    print("Validation Accuracy: {}".format(total_correct/t_steps))


def save_trainloss(train_loss_set, order):
    """
    Function to save the image of training loss v/s iterations graph
    """
    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    base_loc = './loss_images'
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)

    image_dir = base_loc + '/' + MODEL_NAME
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    plt.savefig(train_loss_set, image_dir+'/order_' +
                str(order)+'_train_loss.png')


if __name__ == '__main__':

    if args.mode == 'train':
        model = MbPAplusplus()
        memory = ReplayMemory()
        train(args.order, model, memory)

    if args.mode == 'test':
        model_state = torch.load(
            args.model_path)
        model = MbPAplusplus(model_state=model_state)
        buffer = {}
        with open(args.memory_path, 'rb') as f:
            buffer = pickle.load(f)
        memory = ReplayMemory(buffer=buffer)
        test(args.order, model, memory)
