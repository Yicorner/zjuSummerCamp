#!/bin/bash

# Create the data directory if it doesn't exist
mkdir -p ./data

# Go to the data directory
cd ./data

# Download the CIFAR-10 dataset in binary format
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Extract the tar.gz file
tar -xzf cifar-10-python.tar.gz

# Optionally, remove the tar.gz file to save space
rm cifar-10-python.tar.gz

# Print a message indicating the dataset has been downloaded
echo "CIFAR-10 dataset downloaded and extracted to ./data"
# Here's what to do:

# Create a new file called get_dataset.sh in your desired directory.
# Copy and paste the bash script above into the get_dataset.sh file.
# Save the file.
# Provide execution permissions to the script by running the command: chmod +x get_dataset.sh.
# Run the script using the command: bash get_dataset.sh.
# Once the script finishes running, the CIFAR-10 dataset will be downloaded and extracted to the ./data directory.





