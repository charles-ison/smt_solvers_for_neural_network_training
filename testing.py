import torch

def test(model, all_data, labels, criterion): 
    running_loss = 0.0
    num_correct = 0
    
    for (data, label) in zip(all_data, labels): 
        output = model(data)
        prediction = torch.argmax(output.data)

        loss = criterion(output, label)
        num_correct += (label == prediction)
        
        running_loss += loss.item()

    loss = running_loss/len(labels)
    accuracy = num_correct/len(labels)
    
    print("testing accuracy: " + str(accuracy.item()) + " and loss: " + str(loss))
