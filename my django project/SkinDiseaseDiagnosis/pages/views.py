from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.models import User  # Import User model for signup
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import IntegrityError
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
import os

def home(request):
    return render(request, 'pages/home.html')

def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = User.objects.create_user(username=username, password=password)
            messages.success(request, 'User created successfully.')
            return redirect('signup_success')  # Redirect to the success page after signup
        except IntegrityError:
            messages.error(request, 'Username already exists. Please choose a different username.')
            return render(request, 'pages/signup.html')  # Render the signup page again

    return render(request, 'pages/signup.html')  # Render signup page for GET requests

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Authenticate the user
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)  # Log the user in
            return redirect('login_success')  # Redirect to the success page
        else:
            messages.error(request, 'Invalid username or password.')  # Show error message
            return render(request, 'pages/login.html')  # Render login page again
    
    return render(request, 'pages/login.html')  # Render login page for GET requests

def signup_success(request):
    return render(request, 'pages/signup_success.html')  # Render the success page

def login_success(request):
    return render(request, 'pages/login_success.html')  # Render the login success page

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        input_filename = fs.save(image_file.name, image_file)
        input_image_path = fs.path(input_filename)

        # Load the image and process it with OpenCV
        image = cv2.imread(input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours to detect the affected area
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the image
        output_image = image.copy()
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

        # Save the processed image
        output_filename = 'processed_' + input_filename
        output_image_path = os.path.join(fs.location, output_filename)
        cv2.imwrite(output_image_path, output_image)

        # Calculate dimensions of the disease area
        dimensions = [cv2.boundingRect(contour) for contour in contours]
        dimensions_str = ", ".join([f"Width: {w}, Height: {h}" for (x, y, w, h) in dimensions])

        # Return the image URLs and dimensions to the template
        context = {
            'input_image': fs.url(input_filename),
            'output_image': fs.url(output_filename),
            'dimensions': dimensions_str
        }

        print(f"Input Image URL: {fs.url(input_filename)}")
        print(f"Output Image URL: {fs.url(output_filename)}")
        print(f"Dimensions: {dimensions_str}")

        return render(request, 'pages/upload.html', context)

    # Handle GET requests or invalid POSTs
    return render(request, 'pages/upload.html')
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']

        # Save image to disk
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_url = fs.url(filename)

        # Process image for model prediction
        img = Image.open(image)
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # Classify the image
        with torch.no_grad():
            outputs = model(img)

        # Get the predicted class
        _, predicted = torch.max(outputs, 1)

        # Here you can map predicted.item() to a skin disease label (this will depend on your model's training)
        disease_label = f"Predicted disease: {predicted.item()}"

        return render(request, 'upload_image.html', {
            'file_url': file_url,
            'disease_label': disease_label
        })

    return render(request, 'upload_image.html')