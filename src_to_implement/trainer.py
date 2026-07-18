import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        
        
        self.all_labels = None
        self.all_outputs = None
        self.all_predictions = None
        
        
        
        
        self.true_crack = None
        self.pred_crack = None
        
        self.true_inactive = None
        self.pred_inactive = None
        
        
        self.best_val_fit = -1
        self.curr_f1 = 10
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    # def restore_checkpoint(self, epoch_n):
    #     ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
    #     self._model.load_state_dict(ckp['state_dict'])
        
    # def save_onnx(self, fn):
    #     m = self._model.cpu()
    #     m.eval()
    #     x = t.randn(1, 3, 300, 300, requires_grad=True)
    #     y = self._model(x)
    #     t.onnx.export(m,                 # model being run
    #           x,                         # model input (or a tuple for multiple inputs)
    #           fn,                        # where to save the model (can be a file or file-like object)
    #           export_params=True,        # store the trained parameter weights inside the model file
    #           opset_version=10,          # the ONNX version to export the model to
    #           do_constant_folding=True,  # whether to execute constant folding for optimization
    #           input_names = ['input'],   # the model's input names
    #           output_names = ['output'], # the model's output names
    #           dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
    #                         'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        
        labels = y
        inputs = x
        
        ### Zero Parameter gradients
        self._optim.zero_grad()
        
        ### Forward + Backward + optimize
        outputs = self._model(inputs)
        
        ### Calculate the loss
        loss = self._crit(outputs,labels)
        
        ### Compute gradient by backward propagation
        loss.backward()
        
        ### Update the weights
        self._optim.step()
        
        return loss
        
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO

        labels = y
        inputs = x

        ### Forward + backward
        outputs = self._model(inputs)
        
        loss = self._crit(outputs,labels)
        
        ### keine backward
        return loss, outputs
        
        
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self._model.train(True)
        avg_loss = []
        for idx, (inputs_batch, label_batch) in enumerate(self._train_dl):

            ### transfer
            inputs_batch = inputs_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            ### iterate through 
            loss = self.train_step(inputs_batch,label_batch)
            avg_loss.append(loss.item())

        avg_loss = np.array(avg_loss)
        avg_loss = np.mean(avg_loss)
        return avg_loss

        # for sample in training:
        #     do training step

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval()
        
        
        curr_labels = []
        curr_outputs = []
        
        val_loss_sum = 0.0
        num_samples = 0
        
        with t.no_grad():
            for i, (input_batch, label_batch) in enumerate(self._val_test_dl):
                input_batch = input_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                loss, output = self.val_test_step(input_batch,label_batch)
                
                
                ### Size of current batch
                batch_size = input_batch.size(0)
                
                ### letztes Batch kann kleiner sein
                val_loss_sum += loss.item() * batch_size
                num_samples += batch_size
                
                curr_labels.append(label_batch.cpu())
                curr_outputs.append(output.cpu())
                             
                             
        self.all_labels = t.cat(curr_labels, dim=0)     
        self.all_outputs = t.cat(curr_outputs, dim=0)     
        
        ## Binarisieren fuer OneHot
        self.all_predictions = (self.all_outputs >= 0.5).int()
        
        
        self.all_labels = self.all_labels.numpy()
        self.all_predictions = self.all_predictions.numpy()
        
        ### f1 pro Klasse
        true_crack = self.all_labels[:,0]
        pred_crack = self.all_predictions[:,0]
        
        true_inactive = self.all_labels[:,1]
        pred_inactive = self.all_predictions[:,1]
        
        
        
        
        self.curr_f1 = f1_score(self.all_labels,self.all_predictions, average="macro")
        

      #  losses = np.array(losses)
       # avg_loss = np.mean(losses)   
        print("====== both ==========")
        print("macro f1: ",self.curr_f1)
        print("samples f1:", f1_score(self.all_labels,self.all_predictions, average = "samples"))
        print("========================")
        print("======= crack ==========")
        print("macro f1: ",f1_score(true_crack,pred_crack, average="macro"))
      #  print("samples f1: ",f1_score(true_crack,pred_crack, average = "samples"))
        print("========================")
        print("======== inactive=======")
        print("macro f1: ", f1_score(true_inactive,pred_inactive, average="macro"))
       # print("samples f1: ",f1_score(true_inactive,pred_inactive, average = "samples"))
        print("========================")
        print("========================")
  
        
        
        avg_loss = val_loss_sum/num_samples
                        
        return avg_loss
                                   
                                   
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_loss = []
        val_loss = []
        ### Counter for epocjh
        epoch = 0
        
        #TODO
        
        while True:
            if epoch < epochs:
                print("Epoch: ",epoch)
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
                avg_train_loss = self.train_epoch()
                avg_val_loss = self.val_test()
                self.save_checkpoint(epoch=epoch)
                
                if self.best_val_fit > self.curr_f1:
                    self.best_val_fit = self.curr_f1
                    t.save(
                        self._model.state_dict(),
                        "/content/drive/MyDrive/local_best_model.pt"
                        )
                if epoch > self._early_stopping_patience and np.max(val_loss[-2:-1])>avg_val_loss:
                    break
                train_loss.append(avg_train_loss)
                val_loss.append(avg_val_loss)
                epoch+=1
            
        return train_loss, val_loss
                    
        
        
        
