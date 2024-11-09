import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from PIL import Image
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
import io


# Define your ResNet model
def create_resnet_model(num_classes=1000):
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Identity()  # Change the final layer to Identity to get features
    return model


# Define image transformations
def transform_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Load your dataset
def load_dataset(data_dir, batch_size=32):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, len(dataset.classes)


# Extract features using the ResNet model
def extract_features(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set to evaluation mode
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(lbls.numpy())

    return np.concatenate(features), np.concatenate(labels)


# Train SVM classifier
def train_svm(features, labels):
    clf = svm.SVC(kernel="linear", probability=True)  # Enable probability estimates
    clf.fit(features, labels)
    return clf


# Evaluate SVM model
def evaluate_svm(clf, features, labels):
    predictions = clf.predict(features)
    accuracy = accuracy_score(labels, predictions)
    print(f"SVM Accuracy: {accuracy * 100:.2f}%")


# Predict the probability of melanoma
def predict_melanoma_probability(clf, model, image):
    single_image = transform_image(image)
    single_image = single_image.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    with torch.no_grad():
        feature = model(single_image).cpu().numpy()
        probabilities = clf.predict_proba(feature.reshape(1, -1))[
            0
        ]  # Get probabilities for each class

    return probabilities


# Plot Pie chart for the results
def plot_pie_chart(probabilities):
    labels = ["No Melanoma", "Melanoma"]
    colors = ["green", "red"]
    plt.figure(figsize=(5, 5))
    plt.pie(
        probabilities, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
    )
    plt.title("Prediction Probability Distribution")
    st.pyplot(plt)


# Function to generate PDF report
def generate_pdf_report(image, prediction_result, confidence, probabilities):
    # Create instance of FPDF class
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(255, 0, 0)  # Red color for the title
    pdf.cell(200, 10, txt="Melanoma Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Add Image with border
    image_path = "temp_image.jpg"
    image.save(image_path)
    pdf.set_line_width(1)
    pdf.image(image_path, x=50, y=45, w=100)
    pdf.ln(120)  # Adjust the vertical spacing after the image

    # Prediction result section
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)  # Black color for text
    pdf.cell(200, 10, txt="Prediction Result", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, txt=f"Prediction: {prediction_result}")
    pdf.multi_cell(0, 10, txt=f"Confidence: {confidence:.2f}%")
    pdf.multi_cell(0, 10, txt=f"Probability of No Melanoma: {probabilities[0]:.2f}")
    pdf.multi_cell(0, 10, txt=f"Probability of Melanoma: {probabilities[1]:.2f}")
    pdf.ln(10)

    # Add a colored border for the content section
    pdf.set_line_width(1)
    pdf.set_draw_color(255, 182, 193)  # Light pink border for content
    pdf.rect(10, 20, 190, 260)  # Draw border around the content area

    # Add some more detailed styling
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt="Detailed Explanation", ln=True)
    pdf.multi_cell(
        0,
        10,
        txt="The model uses ResNet-18 features, extracted from the image, to predict the presence of melanoma.",
    )
    pdf.multi_cell(
        0,
        10,
        txt="It analyzes the probability of melanoma presence, providing the confidence level.",
    )
    pdf.ln(10)

    # Add footer with page number
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, "C")

    # Save PDF
    pdf_output_path = "Melanoma_Prediction_Report.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path


# Streamlit UI for Image Upload and Prediction
def run_melanoma_prediction(image_file):
    # Paths to your dataset (update these with your local paths)
    train_data_dir = r"D:\main\Desktop\Ml\Maligant-ML\data\melanoma_cancer_dataset\train"  # Training dataset path
    test_data_dir = r"D:\main\Desktop\Ml\Maligant-ML\data\melanoma_cancer_dataset\test"  # Testing dataset path

    # Load datasets
    train_loader, num_classes = load_dataset(train_data_dir, batch_size=32)
    test_loader, _ = load_dataset(test_data_dir, batch_size=32)

    # Create the model
    model = create_resnet_model(num_classes=num_classes)

    # Extract features from the training dataset
    train_features, train_labels = extract_features(model, train_loader)

    # Train SVM classifier
    svm_classifier = train_svm(train_features, train_labels)

    # Extract features from the test dataset
    test_features, test_labels = extract_features(model, test_loader)

    # Evaluate the SVM classifier
    evaluate_svm(svm_classifier, test_features, test_labels)

    # Predicting the probability of melanoma for the given image
    probabilities = predict_melanoma_probability(svm_classifier, model, image_file)

    # Assuming class 0 is "No Melanoma" and class 1 is "Melanoma"
    result = "Melanoma" if probabilities[1] > probabilities[0] else "No Melanoma"
    result_color = "red" if result == "Melanoma" else "green"

    # Display results with color based on prediction
    st.markdown(f"## Prediction: {result}")
    st.markdown(
        f"<h3 style='color:{result_color};'>{result} Detected</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(f"### Confidence: {max(probabilities) * 100:.2f}%")

    # Plot pie chart showing the prediction confidence
    plot_pie_chart(probabilities)

    # Generate and offer PDF report for download
    pdf_path = generate_pdf_report(
        image_file, result, max(probabilities) * 100, probabilities
    )
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download Prediction Report",
            data=f,
            file_name="Melanoma_Prediction_Report.pdf",
            mime="application/pdf",
        )


# Streamlit Web Interface
def main():
    # Set background and text color to make UI more attractive
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #FAD6D6;
            color: black;
        }
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
        }
        .stButton > button {
            background-color: #3498db;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Melanoma Prediction using ResNet and SVM")
    st.markdown(
        """
    ### Upload an image of skin for melanoma prediction.
    The model predicts whether melanoma is present and provides the prediction confidence.
    """
    )

    # Allow the user to upload an image
    image_file = st.file_uploader(
        "Upload an image for melanoma prediction", type=["jpg", "jpeg", "png"]
    )

    if image_file is not None:
        # Open the image using PIL
        image = Image.open(image_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Show a processing loader
        with st.spinner("Processing image... Please wait."):
            run_melanoma_prediction(image)


if __name__ == "__main__":
    main()
