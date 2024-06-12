import torch

def train_backpropagation(model, all_data, labels, criterion, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
    
        for (data, label) in zip(all_data, labels): 
            optimizer.zero_grad()
        
            output = model(data)
            prediction = torch.argmax(output.data)

            loss = criterion(output, label)
            num_correct += (label == prediction)
        
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss = running_loss/len(labels)
        accuracy = num_correct/len(labels)
        print("epoch: " + str(epoch) + ", accuracy: " + str(accuracy.item()) + ", loss: " + str(loss))
        
        if (accuracy == 1.0):
            return
