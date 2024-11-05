import requests

# Define the URL of the Flask API
url = "http://127.0.0.1:5000/predict"

# Path to the image file
image_path = "064.png"  # Replace with the path to your image

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Define the files dictionary for the POST request
    files = {"image": image_file}

    # Send the POST request with the image
    response = requests.post(url, files=files)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the JSON response with predictions
        print("Predictions:", response.json())
    else:
        print("Error:", response.status_code, response.text)
