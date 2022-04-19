import torch
import segmentation_models_pytorch as smp
import os, json
from plot import plot
def modeltrain(
            model,
            trainloader,
            validloader,
            optimizer,
            criterion,
            metrics,
            epochs,
            save_path,
            device,
            earlystop=5,
            ):
    save_model_path = os.path.join(save_path, 'model_weight.pth')
    save_path_history = os.path.join(save_path, 'history.json')
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )
    history = {
        'trainloss' : [],
        'trainfscore' : [],
        'validloss' : [],
        'validfscore' : [],
    }
    state = {
        'epoch' : 0,
        'state_dict' : model.state_dict(),
        'trainloss' : 0,
        'trainfscore' : 0,
        'validloss' : 0,
        'validfscore' : 0,
    }
    valid_loss_min = 1e10
    trigger = 0

    for epoch in range(epochs):
        print(f'running epoch: {epoch+1}')
        train_logs = train_epoch.run(trainloader)
        valid_logs = valid_epoch.run(validloader)

        # save record
        trainloss = train_logs['dice_loss']
        trainfscore = train_logs['fscore']
        validloss = valid_logs['dice_loss']
        validfscore = valid_logs['fscore']
        history['trainloss'].append(trainloss)
        history['trainfscore'].append(trainfscore)
        
        history['validloss'].append(validloss)
        history['validfscore'].append(validfscore)
        
        # save model if validation loss has decreased
        if validloss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.4f} --> {validloss:.4f}).  Saving model ...\n')
            state['epoch'] = epoch
            state['state_dict'] = model
            state['trainloss'] = trainloss
            state['trainfscore'] = trainfscore
            state['validloss'] = validloss
            state['validfscore'] = validfscore

            os.makedirs(save_path, exist_ok=True)
            torch.save(state, save_model_path)
            valid_loss_min = validloss
            trigger = 0
        # if model dont improve for 5 times, interupt.
        else:
            trigger += 1
            print(f'Validation loss increased ({valid_loss_min:.4f} --> {validloss:.4f}). Trigger {trigger}/{earlystop}\n')
            if trigger == earlystop:
                break

    bestepoch = state['epoch']
    validloss = state['validloss']
    validfscore = state['validfscore']
    print(f'Best model on epoch : {bestepoch}/{epoch}')
    print(f'validation loss: {validloss:.4f}\t\t validation acc : {validfscore:.4f}')


    # save result
    with open(save_path_history, 'w') as f:
        json.dump(history, f)
    
    plot('Training Dice Loss', os.path.join(save_path,'Diceloss.png'), 
        history['trainloss'], history['validloss']
        )

    plot('Training F-score', os.path.join(save_path,'fscore.png'), 
        history['trainfscore'], history['validfscore']
        )

    return history


