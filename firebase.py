#------------------------------------------------------#
# Version Control:  v1.1                               #
# Date:             15-8-2018                          #
# Firebase Storage: sutd-rmp-2018-gotya.appspot.com    #
# Description:      Python 3.6 wrapper script.         #
#                   Allows access to Firebase Storage  #
#                     - Allows uploading of files      #
#                     - Allows downloading of files    #
#                     # Allows deletion of files       #
#------------------------------------------------------#

import pyrebase
import time

# Initialisation and Configuration
config = {
    "apiKey": "apiKey",
    "authDomain": "sutd-rmp-2018-gotya.firebaseapp.com",
    "databaseURL": "https://sutd-rmp-2018-dotya.firebaseio.com",
    "storageBucket": "sutd-rmp-2018-gotya.appspot.com"
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

# Function to post files onto Firebase Storage
# param 1: filepath - The filepath to the file saved on the computer
# param 2: dest - The destination filepath relative to the storage bucket
def upload_file(filepath, dest):
    storage.child(dest).put(filepath)
    # Example of function usage:
    # upload_file("C:/Users/.../test.txt", "test/test.txt")

# Function to download files from Firebase Storage
# param 1: src - The source filepath relative to the storage bucket
# param 2: filename - The preferred filename of the downloaded file
def download_file(src, filename):
    storage.child(src).download(filename)
    # Example of function usage:
    # download_file("test/test.txt", "C:/Users/.../test.txt")

# Function to delete files from Firebase Storage
# param 1: file - The filepath (to delete) relative to the storage bucket
# Remarks: Requires authenticated user to delete files
# def delete_file(file):
#     storage.delete(file)
#     # Example of function usage:
#     # delete_file("test/test.txt")
