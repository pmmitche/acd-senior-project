import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.utils_plot as utils_plot # utils_plot requires matplotlib
import os 
import tensorflow as tf
import absl
import subprocess
import traceback
import argparse


'''
This function performs automatic concept detection using TCAV.
'''
def run_tcav(in_target, n, show_plot):
    # This is the name of your model wrapper (InceptionV3 and GoogleNet are provided in model.py)
    model_to_run = 'GoogleNet'  
    user = 'username'
    # the name of the parent directory that results are stored (only if you want to cache)
    project_name = 'tcav_class_test'
    working_dir = "/tmp/" + user + '/' + project_name
    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir+ '/activations/'
    # where CAVs are stored. 
    # You can say None if you don't wish to store any.
    cav_dir = working_dir + '/cavs/'
    # where the images live.

    # TODO: replace 'YOUR_PATH' with path to downloaded models and images. 
    # NOTE: Make relative path
    source_dir = 'tcav/tcav_examples/image_models/imagenet/IMGNET_DOWNLOAD'
    # source_dir = '/Users/parkermitchell/Desktop/tcav3.0/tcav/tcav_examples/image_models/imagenet/IMGNET_DOWNLOAD'
    bottlenecks = [ 'mixed4c']  # @param 
        
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)

    # this is a regularizer penalty parameter for linear classifier to get CAVs. 
    alphas = [0.1]   

    # Enumerate target and texture classes
    target = in_target
    # concepts = ['blotchy', 'dotted', 'banded', 'striped']
    concepts = ['blotchy', 'dotted', 'banded', 'striped', 
                'bumpy', 'smeared', 'knitted', 'porous', 
                'pitted', 'fibrous', 'veined', 'perforated', 
                'woven', 'meshed', 'crosshatched', 'sprinkled', 
                'polka-dotted', 'marbled', 'stained', 'grid', 
                'gauzy', 'interlaced', 'frilly', 'zigzagged', 
                'spiralled', 'swirly', 'cracked', 'studded', 
                'matted', 'flecked', 'potholed', 'scaly', 
                'stratified', 'braided', 'lined', 'wrinkled', 
                'paisley', 'waffled', 'freckled', 'honeycombed', 
                'lacelike', 'chequered', 'crystalline', 
                'bubbly', 'grooved', 'pleated', 'cobwebbed']

    # Create TensorFlow session.
    sess = utils.create_session()

    # GRAPH_PATH is where the trained model is stored.
    GRAPH_PATH = source_dir + "/inception5h/tensorflow_inception_graph.pb"
    # LABEL_PATH is where the labels are stored. Each line contains one class, and they are ordered with respect to their index in 
    # the logit layer. (yes, id_to_label function in the model wrapper reads from this file.)
    # For example, imagenet_comp_graph_label_strings.txt looks like:
    # dummy                                                                                      
    # kit fox
    # English setter
    # Siberian husky ...

    LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"

    mymodel = model.GoogleNetWrapper_public(sess,
                                            GRAPH_PATH,
                                            LABEL_PATH)

    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)

    absl.logging.set_verbosity(0)
    num_random_exp=10
    mytcav = tcav.TCAV(sess,
                    target,
                    concepts,
                    bottlenecks,
                    act_generator,
                    alphas,
                    cav_dir=cav_dir,
                    num_random_exp=num_random_exp)#10)
    # NOTE: CHANGE NUM_RAND_EXP HIGHER
    print ('This may take a while... Go get coffee!')
    results = mytcav.run(run_parallel=False)
    print ('done!')

    tcav_scores = utils_plot.plot_results(results, num_random_exp=num_random_exp, show_figure=show_plot)
    tcav_scores.sort(key=lambda x: x[1], reverse=True)

    if n is None:
        n = len(tcav_scores)

    # once the matplotlib bar chart is closed, the n most important concepts will be written out to the command line
    print("\nMax TCAV score(s) for target '" + in_target + "':")
    for i in range(n):
        print("- " + tcav_scores[i][0] + ": " + str(round(tcav_scores[i][1], 4)))

    print()
    return (tcav_scores, tcav_scores[0:n])


'''
Function to do the following:
    - download imagenet images for target class
    - download broden dataset
    - download inception 5h model
    - download mobilenet v2 model
    - run automatic concept detection on said target class
'''
def execute_main(args):
    try:
        args.target = "\"" + args.target + "\""
        download_datasets_path = "tcav/tcav_examples/image_models/imagenet"
        return_path = "../../../../"
        os.chdir(download_datasets_path)
        print('Moved down to path:', os.getcwd())
        download_cmd = ("python3 download_and_make_datasets.py --source_dir=IMGNET_DOWNLOAD --target=" + args.target + " --number_of_images_per_folder=" +
                        str(args.number_of_images_per_folder) + " --number_of_random_folders=" + str(args.number_of_random_folders))
        download_err = subprocess.call(download_cmd, shell=True)
        if download_err != 0:
            raise Exception
        os.chdir(return_path)
        print('Moved back to path:', os.getcwd())

        if args.show_plot == None:
            show_plot = True
        elif ((args.show_plot).lower() == 'true'):
            show_plot = True
        elif ((args.show_plot).lower() == 'false'):
            show_plot = False
        else:
            show_plot = True

        args.target = args.target[1:-1]
        run_tcav(args.target, args.n, show_plot)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("\nTerminating.\n")


'''
Main driver for performing automatic concept detection. Takes four arguments as input.
    - target: string
        -- Name of target object we wish to run TCAV for
    - n: int
        -- Number of most important concepts to display. To display all, omit this flag
    - show_plot: string
        -- Argument to specify whether or not to show the matplotlib bar chart of concept importances. If omitted, the plot will be shown
    - number_of_images_per_folder: int
        -- Number of images to be included in each folder
    - number_of_random_folders: int
        -- Number of folders with random examples that we will generate for tcav
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument('--target', type=str,
                        help='Name of target object we wish to run ACD with TCAV for')
    parser.add_argument('--n', type=int,
                        help='Number of most important concepts to display. To display all, omit this flag')
    parser.add_argument('--show_plot', type=str,
                        help='Argument to specify whether or not to show the matplotlib bar chart of concept importances. If omitted, the plot will be shown')
    parser.add_argument('--number_of_images_per_folder', type=int,
                        help='Number of images to be included in each folder')
    parser.add_argument('--number_of_random_folders', type=int,
                        help='Number of folders with random examples that we will generate for tcav')

    args = parser.parse_args()
    execute_main(args)

# sample use of the command line interface
# python3 automatic_concept_detection_driver.py --target="volleyball" --number_of_images_per_folder=20 --number_of_random_folders=10
