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
""" Downloads models and datasets for imagenet

    Content downloaded:
        - Imagenet images for the zebra class.
        - Full Broden dataset(http://netdissect.csail.mit.edu/)
        - Inception 5h model(https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception5h.py)
        - Mobilenet V2 model(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

    Functionality:
        - Downloads open source models(Inception and Mobilenet)
        - Downloads the zebra class from imagenet, to illustrate a target class
        - Extracts three concepts from the Broden dataset(striped, dotted, zigzagged)
        - Structures the data in a format that can be readily used by TCAV
        - Creates random folders with examples from Imagenet. Those are used by TCAV.

    Example usage:

    python download_and_make_datasets.py --source_dir=YOUR_FOLDER --number_of_images_per_folder=50 --number_of_random_folders=10
"""
import subprocess
import os
import argparse
from tensorflow.io import gfile
import imagenet_and_broden_fetcher as fetcher

def make_concepts_targets_and_randoms(source_dir, number_of_images_per_folder, number_of_random_folders, in_target):
    # Run script to download data to source_dir
    if not gfile.exists(source_dir):
        gfile.makedirs(source_dir)
    if not gfile.exists(os.path.join(source_dir,'broden1_224/')) or not gfile.exists(os.path.join(source_dir,'inception5h')):
        subprocess.call(['bash' , 'FetchDataAndModels.sh', source_dir])

    # Determine classes that we will fetch
    imagenet_classes = [in_target]
    broden_concepts = get_all_texture_names()

    # make targets from imagenet
    imagenet_dataframe = fetcher.make_imagenet_dataframe("./imagenet_url_map.csv")
    for image in imagenet_classes:
        fetcher.fetch_imagenet_class(source_dir, image, number_of_images_per_folder, imagenet_dataframe)

    # Make concepts from broden
    for concept in broden_concepts:
        fetcher.download_texture_to_working_folder(broden_path=os.path.join(source_dir, 'broden1_224'),
                                                   saving_path=source_dir,
                                                   texture_name=concept,
                                                   number_of_images=number_of_images_per_folder)

    # Make random folders. If we want to run N random experiments with tcav, we need N+1 folders.
    fetcher.generate_random_folders(
        working_directory=source_dir,
        random_folder_prefix="random500",
        number_of_random_folders=number_of_random_folders+1,
        number_of_examples_per_folder=number_of_images_per_folder,
        imagenet_dataframe=imagenet_dataframe
    )

"""
    Function to fetch all texture class names.
"""
def get_all_texture_names():
    textures = []

    # NOTE: Make relative path for texture CSV file
    texture_csv_path = "/Users/parkermitchell/Desktop/tcav3.0/tcav/tcav_examples/image_models/imagenet/IMGNET_DOWNLOAD/broden1_224/broden1_224/c_texture.csv"
    texture_file = open(texture_csv_path, "r")
    lines = texture_file.readlines()
    for line in lines[1:]:
        textures.append(line.split(',')[2])

    return textures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument('--source_dir', type=str,
                        help='Name for the directory where we will create the data.')
    parser.add_argument('--target', type=str,
                        help='Name of target object we wish to run TCAV for')
    parser.add_argument('--number_of_images_per_folder', type=int,
                        help='Number of images to be included in each folder')
    parser.add_argument('--number_of_random_folders', type=int,
                        help='Number of folders with random examples that we will generate for tcav')

    args = parser.parse_args()
    # create folder if it doesn't exist
    if not gfile.exists(args.source_dir):
        gfile.makedirs(os.path.join(args.source_dir))
        print("Created source directory at " + args.source_dir)
    # Make data
    make_concepts_targets_and_randoms(args.source_dir, args.number_of_images_per_folder, args.number_of_random_folders, args.target)
    print("Successfully created data at " + args.source_dir)
