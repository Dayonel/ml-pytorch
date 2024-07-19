import torchvision.transforms as transforms

# We create inputs for the neural network
# The expected of the target output will be the original color


def train(train_loader, device, model, criterion, optimizer):
    # We pass entire dataset to the model x10 times
    EPOCHS = 10  # usually we train we 30
    # Training loop
    for epoch in range(EPOCHS):
        for i, (image, _) in enumerate(train_loader):
            # Convert RGB image to grayscale
            transform = transforms.Grayscale()
            grayscale = transform(image).to(device)
            image = image.to(device)

            # Forward pass
            output = model(grayscale)
            # We compute the loss comparing the model's output to the original images
            loss = criterion(output, image)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print statistics
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i +
                  1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("Finished Training")
