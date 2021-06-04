"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Used to download images from the imagenet dataset and to move concepts from the Broden dataset, rearranging them
in a format that is TCAV readable. Also enables creation of random folders from imagenet

Usage for Imagenet
  imagenet_dataframe = pandas.read_csv("imagenet_url_map.csv")
  fetch_imagenet_class(path="your_path", class_name="zebra", number_of_images=100,
                      imagenet_dataframe=imagenet_dataframe)
                    
Usage for broden:
First, make sure you downloaded and unzipped the broden_224 dataset to a location of your interest. Then, run:
  download_texture_to_working_folder(broden_path="path_were_you_saved_broden", 
                                      saving_path="your_path",
                                      texture_name="striped",
                                       number_of_images=100)
                                      
Usage for making random folders:
  imagenet_dataframe = pandas.read_csv("imagenet_url_map.csv")
  generate_random_folders(working_directory="your_path",
                            random_folder_prefix="random_500",
                            number_of_random_folders=11,
                            number_of_examples_per_folder=100,
                            imagenet_dataframe=imagenet_dataframe)

"""
import pandas as pd
import urllib.request
import os
import shutil
import PIL
from PIL import Image
import tensorflow as tf
import socket
import random

kImagenetBaseUrl = "http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid="
kBrodenTexturesPath = "broden1_224/images/dtd/"
kMinFileSize = 10000

####### Helper functions
""" Makes a dataframe matching imagenet labels with their respective url.

Reads a csv file containing matches between imagenet synids and the url in
which we can fetch them. Appending the synid to kImagenetBaseUrl will fetch all
the URLs of images for a given imagenet label

  Args: path_to_imagenet_classes: String. Points to a csv file matching
          imagenet labels with synids.

  Returns: a pandas dataframe with keys {url: _ , class_name: _ , synid: _}
"""
def make_imagenet_dataframe(path_to_imagenet_classes):
  urls_dataframe = pd.read_csv(path_to_imagenet_classes)
  urls_dataframe["url"] = kImagenetBaseUrl + urls_dataframe["synid"]
  return urls_dataframe


""" Downloads/migrates an image.

Downloads and image from a image path provided and saves it under path.
Filters away images that are corrupted or smaller than 10KB

  Args:
    path: Path to the folder where we're saving this image.
    orig_path: original full path to this image.

  Raises:
    Exception: Propagated from PIL.image.verify()
"""
def download_image(path, orig_path):
  image_name = orig_path.split("/")[-1]
  image_prefix = image_name.split(".")[0]
  saving_path = os.path.join(path, image_prefix + ".jpg")
  
  # copy and rename file
  shutil.copy(orig_path, path)
  old_name = os.path.join(path, image_name)
  os.rename(old_name, saving_path)


"""
  Fetches all nids within the tiny-imagenet-200 dataset
"""
def fetch_tiny_imagenet_nids():
    tiny_imagenet_wnids_path = "../../../../tiny-imagenet-200/wnids.txt"
    tiny_imagenet_class_index = open(tiny_imagenet_wnids_path, "r")
    tiny_imagenet_classes = tiny_imagenet_class_index.readlines()
    tiny_imagenet_nids = set()

    for line in tiny_imagenet_classes:
      tiny_imagenet_nids.add(line.strip())
    
    return tiny_imagenet_nids


"""
  Fetches all paths within the tiny-imagenet-200 dataset for images of a given concept.
"""
def fetch_all_paths_for_concept(imagenet_images_path, concept):
    tiny_imagenet_words_path = "../../../../tiny-imagenet-200/words.txt"
    tiny_imagenet_class_index = open(tiny_imagenet_words_path, "r")
    imagenet_classes = tiny_imagenet_class_index.readlines()
    tiny_imagenet_nids = fetch_tiny_imagenet_nids()
    concept_found = False
    concept_directory = None

    # get image folder for the concept passed in
    for line in imagenet_classes:
        split_line = line.split("\t", 1)
        if split_line[0] not in tiny_imagenet_nids:
          continue

        split_line[1] = split_line[1].strip()
        split_line[1] = split_line[1].split(", ")
        
        for word in split_line[1]:
          if concept.lower() in word.lower():
            # concept passed in exists, so our 'split_line' variable will contain the dir name
            concept_found = True
            concept_directory = split_line[0]
            break

    # concept does not exist in the tiny imagenet list of concepts/classes
    if not concept_found:
        raise tf.errors.NotFoundError(
            None, None, "Couldn't find any imagenet concept for '" + concept +
            "'. Make sure you're getting a valid concept")

    # create path to the imagenet class directory
    concept_dir_path = os.path.join(imagenet_images_path, concept_directory, "images")
    # print(concept_dir_path)
    if not os.path.exists(concept_dir_path):
        raise tf.errors.NotFoundError(
            None, None, "Path to images does not exist for imagenet concept " + concept +
            ". Make sure you're getting a valid concept")
    
    image_paths = os.listdir(concept_dir_path)
    all_full_image_paths = []
    for image_path in image_paths:
      all_full_image_paths.append(os.path.join(concept_dir_path, image_path))

    return all_full_image_paths


####### Main methods
""" For a given imagenet class, download images from the tiny-imagenet dataset.

  Args:
    path: String. Path where we're saving the data. Creates a new folder with
      path/class_name.
    class_name: String representing the name of the imagenet class.
    number_of_images: Integer representing number of images we're getting for
      this example.

"""
def fetch_imagenet_class(path, class_name, number_of_images, imagenet_dataframe):
  tf.compat.v1.logging.info("Fetching imagenet data for " + class_name)
  concept_path = os.path.join(path, class_name)
  tf.io.gfile.makedirs(concept_path)
  tf.compat.v1.logging.info("Saving images at " + concept_path)

  # Check to see if this class name exists. Fetch all image paths if so.
  tiny_imagenet_training_path = "../../../../tiny-imagenet-200/train"
  all_images = fetch_all_paths_for_concept(tiny_imagenet_training_path, class_name)
  
  # Fetch number_of_images images or as many as you can.
  num_downloaded = 0
  for image_path in all_images:
    try:
      download_image(concept_path, image_path)
      num_downloaded += 1

    except Exception as e:
      tf.compat.v1.logging.info("Problem downloading imagenet image. Exception was " +
                      str(e) + " for URL " + image_path)

    if num_downloaded >= number_of_images:
      break

  # If we reached the end, notify the user through the console.
  if num_downloaded < number_of_images:
    print("You requested " + str(number_of_images) +
          " but we were only able to find " +
          str(num_downloaded) +
          " good images from imageNet for concept " + class_name)
  else:
    print("Downloaded " + str(number_of_images) + " for " + class_name)


"""Moves all textures in a downloaded Broden to our working folder.

Assumes that you manually downloaded the broden dataset to broden_path.


  Args:
  broden_path: String.Path where you donwloaded broden.
  saving_path: String.Where we'll save the images. Saves under
    path/texture_name.
  texture_name: String representing DTD texture name i.e striped
  number_of_images: Integer.Number of images to move
"""
def download_texture_to_working_folder(broden_path, saving_path, texture_name,
                                       number_of_images):
  # Create new experiment folder where we're moving the data to
  texture_saving_path = os.path.join(saving_path, texture_name)
  tf.io.gfile.makedirs(texture_saving_path)

  # Get images from broden
  broden_textures_path = os.path.join(broden_path, kBrodenTexturesPath)
  tf.compat.v1.logging.info("Using path " + str(broden_textures_path) + " for texture: " +
                  str(texture_name))
  for root, dirs, files in os.walk(broden_textures_path):
    # Broden contains _color suffixed images. Those shouldn't be used by tcav.
    texture_files = [
        a for a in files if (a.startswith(texture_name) and "color" not in a)
    ]
    number_of_files_for_concept = len(texture_files)
    tf.compat.v1.logging.info("We have " + str(len(texture_files)) +
                    " images for the concept " + texture_name)

    # Make sure we can fetch as many as the user requested.
    if number_of_images > number_of_files_for_concept:
      raise Exception("Concept " + texture_name + " only contains " +
                      str(number_of_files_for_concept) +
                      " images. You requested " + str(number_of_images))

    # We are only moving data we are guaranteed to have, so no risk for infinite loop here.
    save_number = number_of_images
    while save_number > 0:
      for file in texture_files:
        path_file = os.path.join(root, file)
        texture_saving_path_file = os.path.join(texture_saving_path, file)
        tf.io.gfile.copy(
            path_file, texture_saving_path_file,
            overwrite=True)  # change you destination dir
        save_number -= 1
        # Break if we saved all images
        if save_number <= 0:
          break


def fetch_tiny_imagenet_classes():
    tiny_imagenet_nids = fetch_tiny_imagenet_nids()
    tiny_imagenet_words_path = "../../../../tiny-imagenet-200/words.txt"
    tiny_imagenet_class_index = open(tiny_imagenet_words_path, "r")
    tiny_imagenet_classes = tiny_imagenet_class_index.readlines()
    classes = []

    for line in tiny_imagenet_classes:
      split_line = line.split("\t", 1)
      if split_line[0] not in tiny_imagenet_nids:
        continue

      split_line[1] = split_line[1].strip()
      split_line[1] = split_line[1].split(", ")
      classes.append(split_line[1][0].lower())
    
    return classes


""" Creates folders with random examples under working directory.

They will be named with random_folder_prefix as a prefix followed by the number
of the folder. For example, if we have:

    random_folder_prefix = random500
    number_of_random_folders = 3

This function will create 3 folders, all with number_of_examples_per_folder
images on them, like this:
    random500_0
    random500_1
    random500_2


  Args:
    random_folder_prefix: String.The prefix for your folders. For example,
      random500_1, random500_2, ... , random500_n.
    number_of_random_folders: Integer. Number of random folders.
    number of examples_per_folder: Integer. Number of images that will be saved
      per folder.
"""
def generate_random_folders(working_directory, random_folder_prefix,
                            number_of_random_folders,
                            number_of_examples_per_folder, imagenet_dataframe):
  imagenet_concepts = fetch_tiny_imagenet_classes()
  for partition_number in range(number_of_random_folders):
    partition_name = random_folder_prefix + "_" + str(partition_number)
    partition_folder_path = os.path.join(working_directory, partition_name)
    tf.io.gfile.makedirs(partition_folder_path)

    # Select a random concept
    examples_selected = 0
    while examples_selected < number_of_examples_per_folder:
      random_concept = random.choice(imagenet_concepts)
      all_images = fetch_all_paths_for_concept("../../../../tiny-imagenet-200/train", random_concept)
      for image_path in all_images:
        try:
          download_image(partition_folder_path, image_path)
          examples_selected += 1
          if (examples_selected) % 10 == 0:
            tf.compat.v1.logging.info("Downloaded " + str(examples_selected) + "/" +
                            str(number_of_examples_per_folder) + " images for " +
                            partition_name)
          break  # Break if we successfully downloaded an image
        except:
            pass # try new url
