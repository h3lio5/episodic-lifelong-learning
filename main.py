import torch
import torch.utils.data as data
from data_loader import DataSet
import argparse
from baselines.replay import ReplayMemory, ReplayModel
import transformers
from tqdm import trange
import time
import matplotlib.pyplot as plt
import numpy as np
use_cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='Enter the batch size')
parser.add_argument('--mode', default='train',
                    help='Enter the mode - train/eval')
parser.add_argument('--order', default=1, type=int,
                    help='Enter the dataset order - 1/2/3/4')
parser.add_argument('--epochs', default=4, type=int)
args = parser.parse_args()
LEARNING_RATE = 3e-5

MODEL_NAME = 'REPLAY'
# replay frequency = 1% of total number of steps per epoch
# i.e., REPLAY_FREQ = 1/100(total_examples/batch_size) = 1/100(575000/32) ~ 180
REPLAY_FREQ = 180


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
        nb_tr_examples, nb_tr_steps = 0, 0
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):

            # Perform sparse experience replay after every REPLAY_FREQ steps
            if step % REPLAY_FREQ == 0:
                # sample 100 examples from memory
                content, attn_masks, labels = memory.sample(sample_size=100)
            else:
                # Unpacking the batch items
                content, attn_masks, labels = batch

            # Push the examples into the replay memory
            memory.push(batch)
            print("Epoch ", epoch+1, "step ", step+1)
            # Place the batch items on the appropriate device: cuda if avaliable
            if use_cuda:
                content = content.cuda()
                attn_masks = attn_masks.cuda()
                labels = labels.cuda()
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, logits = model.classify(content.squeeze(
                1), attn_masks.squeeze(1), labels.squeeze(1))
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
        model_dict = model.save_state()
        torch.save(model_dict, '../model_checkpoints/' +
                   MODEL_NAME+'/classifier_order_'+str(order)+'_epoch_'+str(epoch+1)+'.pth')
        np.save('../model_checkpoints/' + MODEL_NAME + '/epoch_' + str(epoch+1),
                np.asarray(memory.memory))
    save_trainloss(train_loss_set)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def test(order, model):
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
    for step, batch in enumerate(test_dataloader):

        print("Step", step)
        content, attn_masks, labels = batch

        if use_cuda:
            content = content.cuda()
            attn_masks = attn_masks.cuda()
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model.infer(content.squeeze(1), attn_masks.squeeze(1))

        logits = logits.detach().cpu().numpy()
        # Dropping the 1 dim to match the logits' shape
        # shape : (batch_size,num_labels)
        labels = labels.squeeze(1).numpy()
        tmp_correct = flat_accuracy(logits, labels)
        total_correct += tmp_correct
        t_steps += 1
    end = time.time()
    print("Time taken for validation {} minutes".format((end-start)/60))
    print("Validation Accuracy: {}".format(total_correct/t_steps))


def save_trainloss(train_loss_set):

    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.savefig('../loss_images/'+MODEL_NAME+'/train_loss.png')


if __name__ == '__main__':

    if args.mode == 'train':
        model = ReplayModel()
        memory = ReplayMemory()
        train(args.order, model, memory)

    if args.mode == 'test':
        model_state = torch.load(
            '../model_checkpoints/enc_dec/classifier_order_1_epoch_4.pth')
        model = EncDec(mode='test', model_state=model_state)
        test(args.order, model)
