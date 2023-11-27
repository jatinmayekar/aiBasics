import os
script_directory = os.path.dirname(os.path.realpath(__file__)) + r"\webcam_image.jpg"
script_directory = script_directory.replace("\\", "/")
print("Script Directory:", script_directory)