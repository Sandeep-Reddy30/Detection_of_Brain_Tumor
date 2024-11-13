function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select an image file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // Show the prediction result and uploaded image
        document.getElementById('result').style.display = 'block';
        document.getElementById('predictionText').textContent = `Predicted Class: ${data.prediction}`;
        document.getElementById('uploadedImage').src = data.image_path;  // Display the uploaded image
    })
    .catch(error => {
        console.error('Error:', error);
        alert("An error occurred while uploading the image.");
    });
}
