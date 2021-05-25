from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch
import time
import datetime
import random
import numpy as np
from lstm import Model
from sklearn.metrics import accuracy_score
import os

class BertDetector():

    def get_device(self):
        if torch.cuda.is_available():
            print('The following GPU will be used:', torch.cuda.get_device_name(0))
            return torch.device("cuda")
        else:
            print('Using CPU.')
            return torch.device("cpu")


    def __init__(self, max_length, lr, eps, batch_size, name='Bert Detector'):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.lr = lr
        self.eps = eps
        self.name = name

        self.device = self.get_device()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
        )
        self.model.cuda()

    def tokenize(self, x):
      input_ids = []
      attention_masks = []
      i = 0
      size = len(x)
      five_percent = size*0.05
      print('Starting to tokenize data\n')
      for tweet in x:
          proccessed_percante = i/size*100
          if i%five_percent == 0:
            print(f'Preprocessed {proccessed_percante}% of accounts')
          i+=1
          encoded_dict = self.tokenizer.encode_plus(
                              tweet,
                              add_special_tokens = True,
                              max_length = self.max_length,
                              pad_to_max_length = True,
                              return_attention_mask = True,
                              return_tensors = 'pt',
                        )

          input_ids.append(encoded_dict['input_ids'])
          attention_masks.append(encoded_dict['attention_mask'])

      print('Done Tokenizing\n')
      return input_ids, attention_masks

    def get_dataset_loader(self, input_ids, attention_mask, y, sample_randomly=False):
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        y = torch.tensor(y)

        dataset = TensorDataset(input_ids, attention_mask, y)

        if sample_randomly:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        return DataLoader(
            dataset,
            sampler = sampler,
            batch_size = self.batch_size
        )

    def tokenize_and_get_dataset_loader(self, x, y, sample_randomly=False):
        input_ids, attention_masks = self.tokenize(x)
        return self.get_dataset_loader(input_ids, attention_masks, y, sample_randomly)

    def get_time(self, t):
        t_rounded = int(round((t)))
        return str(datetime.timedelta(seconds=t_rounded))
    import numpy as np

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def get_prediction_with_dloader(self, dataset_loader):
        predictions = []
        probabilities = []
        accuracies = 0.0
        total_loss = 0.0
        loss_steps = 0
        self.model.eval()
        with torch.no_grad():
            for batch in dataset_loader:
                b_input_ids = batch[0].to(self.device)
                b_input_masks = batch[1].to(self.device)
                b_y = batch[2].to(self.device)

                output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks, labels=b_y, output_hidden_states=True)
                loss = output.loss
                logits = output.logits.detach().cpu().numpy()
                print(output.hidden_states )
                label_ids = b_y.to('cpu').numpy()

                accuracies += self.flat_accuracy(logits, label_ids)
                pred = np.argmax(logits, axis=1).flatten()
                predictions.extend(pred)

                probability = torch.nn.functional.softmax(output.logits, 1)
                probabilities.extend(probability.detach().cpu().numpy())

                total_loss += loss.item()
                loss_steps +=1

        return accuracies/len(dataset_loader),  total_loss/loss_steps, probabilities, predictions, output.hidden_states[12][0]

    def get_prediction(self, x, y):
        dataset_loader = self.tokenize_and_get_dataset_loader(x, y)
        return self.get_prediction_with_dloader(dataset_loader)

    def train(self, x_train, y_train, x_val, y_val, epochs=4):
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        train_dataloader = self.tokenize_and_get_dataset_loader(x_train, y_train, sample_randomly=False)
        val_dataloader = self.tokenize_and_get_dataset_loader(x_val, y_val)

        optimizer = AdamW(self.model.parameters(),
                  lr = self.lr,
                  eps = self.eps)

        training_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = training_steps)
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(0, epochs):
            print('======== Training Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            self.model.train()
            train_loss, train_steps, train_accuracy = 0, 0, 0
            start_time = time.time()
            predictions = []
            for step, batch in enumerate(train_dataloader):

                if step % 40 == 0 and not step == 0:
                    elapsed = self.get_time(time.time() - start_time)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_masks = batch[1].to(self.device)
                b_y = batch[2].to(self.device)

                self.model.zero_grad()

                output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks, labels=b_y)
                loss = output.loss
                logits = output.logits

                logits = logits.detach().cpu().numpy()
                label_ids = b_y.to('cpu').numpy()
                train_accuracy += self.flat_accuracy(logits, label_ids)

                #predictions.extend(np.argmax(logits, axis=1).flatten())

                train_loss += loss.item()
                train_steps += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            train_loss = train_loss/train_steps
            avg_train_accuracy = train_accuracy/len(train_dataloader)

            print('Starting to validate the model')
            start_val_time = time.time()
            val_acc, val_loss, probabilites, predictions = self.get_prediction_with_dloader(val_dataloader)
            val_acc_based_on_pred = accuracy_score(y_val, predictions)
            elapsed = self.get_time(time.time() - start_val_time)
            print(f'Validation took {elapsed}')
            print(f'Training loss: {train_loss} Training Accuracy {avg_train_accuracy} Validation Accuracy: {val_acc_based_on_pred}, Validation loss: {val_loss}')
            train_losses.append(train_loss)
            train_accuracies.append(avg_train_accuracy)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)

        Model.plot_accuracies(val_accuracies, train_accuracies, self.name)
        Model.plot_losses(val_losses, train_losses, self.name)


    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        print("Saving model to %s" % path)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)

        self.model.cuda()