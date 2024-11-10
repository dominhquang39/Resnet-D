transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Step 2: Load the label map
label_map_path = '/content/eurosat/EuroSAT/label_map.json'
with open(label_map_path) as f:
    label_map = json.load(f)

# Create a reverse mapping from index to class name
index_to_label = {v: k for k, v in label_map.items()}

# Step 3: Load the dataset (optional, for training and evaluation purposes)
data_dir = '/content/eurosat/EuroSAT'
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Step 4: Create DataLoader (optional, for training and evaluation purposes)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Step 5: Define the ResNet-D model
model = resnet50D(num_classes=len(label_map))  # Number of classes from the label map

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Step 6: Define loss function and optimizer (optional, for training)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

save_path = '/content/drive/MyDrive/DeepLearn/Weight'

# Step 7: (Optional) Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

         #Save model weight
        if (epoch + 1 == 10):
          checkpoint_path = f'{save_path}_epoch_{epoch+1}.pth'
          torch.save({
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': running_loss
          }, checkpoint_path)
          print(f'Model checkpoint saved at {checkpoint_path}')

# (Optional) Uncomment to train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Load the image and convert to RGB
    image = transform(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to predict the label of a single image
def predict_image(model, image_path):
    model.eval()  # Set model to evaluation mode
    image = load_and_preprocess_image(image_path)  # Load and preprocess the image
    image = image.to(device)  # Move image to the same device as the model

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class index

    return predicted.item()  # Return the predicted class index

# Input path to the single image you want to classify
image_path = '/content/eurosat/EuroSAT/Forest/Forest_4.jpg'  # Replace with your image path

# Predict the label
predicted_label_index = predict_image(model, image_path)

# Get the corresponding class label using the reverse mapping
predicted_label = index_to_label.get(predicted_label_index, "Unknown Label")
print(f'Predicted Label: {predicted_label}')

# If you have the actual label, compare it
actual_label = 'actual_label_here'  # Replace with the actual label if known
print('Actual Label: Forest')
