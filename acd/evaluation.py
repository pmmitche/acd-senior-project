'''
File to evaluate sample Automatic Concept Detection runs.
'''

import os
import sys
import unittest
import traceback
import subprocess
import automatic_concept_detection_driver as acd


class TestACD(unittest.TestCase):
    
    # test to ensure that if a user passes the argument n, only the n most important concepts are displayed
    def test_displays_n_1(self):
        target = "Egyptian cat"
        n = 2
        show_plot = False

        # run tcav, which returns a tuple with the first element 
        # being the full list of tcav scores, and the second element
        # being a list of the n most important tcav scores
        tcav_scores, n_tcav_scores = acd.run_tcav(target, n, show_plot)

        # assert that the length of the n most important concepts we
        # compute is equal to the user input n, which is 2
        self.assertEqual(len(n_tcav_scores), n)


    # additional test to ensure that if a user passes the argument n, only the n most important concepts are displayed
    def test_displays_n_2(self):
        target = "Egyptian cat"
        n = 4
        show_plot = False

        # run tcav, which returns a tuple with the first element 
        # being the full list of tcav scores, and the second element
        # being a list of the n most important tcav scores
        tcav_scores, n_tcav_scores = acd.run_tcav(target, n, show_plot)

        # assert that the length of the n most important concepts we
        # compute is equal to the user input n, which is 4
        self.assertEqual(len(n_tcav_scores), n)


    # additional test to ensure that if a user passes the argument n, only the n most important concepts are displayed
    def test_displays_n_all(self):
        target = "Egyptian cat"
        n = None
        show_plot = False

        # run tcav, which returns a tuple with the first element 
        # being the full list of tcav scores, and the second element
        # being a list of the n most important tcav scores
        tcav_scores, n_tcav_scores = acd.run_tcav(target, n, show_plot)

        # assert that the length of all the most important concepts we
        # compute is equal to the number all texture concepts, which is 47
        self.assertEqual(len(n_tcav_scores), 47)


    # test to verify what concepts are deemed important when classifying an egyptian cat
    def test_egyptian_cat(self):
        target = "Egyptian cat"
        n = 5
        show_plot = False

        # Looking at the sample egyptian cat images in imagenet, it is clear to
        # see that most all cats are dotted or striped, which would be the best
        # texture concept indicators to see if our TCAV shows those are important as well
        expected_important_concepts = ['dotted', 'striped']

        # download required datasets
        download_datasets(target, 10, 5)

        # run tcav, which returns a tuple with the first element 
        # being the full list of tcav scores, and the second element
        # being a list of the n most important tcav scores
        tcav_scores, n_tcav_scores = acd.run_tcav(target, n, show_plot)

        # verify that at least one of the two expected important concepts
        # are present in the top five texture concepts
        is_present = False
        for c in n_tcav_scores:
            if c[0] in expected_important_concepts:
                is_present = True
        self.assertEqual(is_present, True)


    # test to verify what concepts are deemed important when classifying a trilobite
    def test_trilobite(self):
        target = "trilobite"
        n = 5
        show_plot = False

        # Looking at the sample trilobite images in imagenet, it is clear to
        # see that most all trilobites are grooved, wrinkled, or porous, which would be the best
        # texture concept indicators to see if our TCAV shows those are important as well
        expected_important_concepts = ['wrinkled', 'porous', 'grooved']

        # download required datasets
        download_datasets(target, 10, 5)

        # run tcav, which returns a tuple with the first element 
        # being the full list of tcav scores, and the second element
        # being a list of the n most important tcav scores
        tcav_scores, n_tcav_scores = acd.run_tcav(target, n, show_plot)

        # verify that at least one of the two expected important concepts
        # are present in the top five texture concepts
        is_present = False
        for c in n_tcav_scores:
            if c[0] in expected_important_concepts:
                is_present = True
        self.assertEqual(is_present, True)


    # test to verify what concepts are deemed important when classifying a monarch
    def test_monarch(self):
        target = "monarch"
        n = 5
        show_plot = False

        # Looking at the sample trilobite images in imagenet, it is clear to
        # see that most all trilobites are lined, or veined, which would be the best
        # texture concept indicators to see if our TCAV shows those are important as well
        expected_important_concepts = ['lined', 'veined']

        # download required datasets
        download_datasets(target, 10, 5)

        # run tcav, which returns a tuple with the first element 
        # being the full list of tcav scores, and the second element
        # being a list of the n most important tcav scores
        tcav_scores, n_tcav_scores = acd.run_tcav(target, n, show_plot)

        # verify that at least one of the two expected important concepts
        # are present in the top five texture concepts
        is_present = False
        for c in n_tcav_scores:
            if c[0] in expected_important_concepts:
                is_present = True
        self.assertEqual(is_present, True)


# helper function to download / store necessary images to run TCAV
def download_datasets(target, number_of_images_per_folder, number_of_random_folders):
    try:
        target = "\"" + target + "\""
        download_datasets_path = "tcav/tcav_examples/image_models/imagenet"
        return_path = "../../../../"
        os.chdir(download_datasets_path)
        print('Moved down to path:', os.getcwd())
        download_cmd = ("python3 download_and_make_datasets.py --source_dir=IMGNET_DOWNLOAD --target=" + target + " --number_of_images_per_folder=" +
                        str(number_of_images_per_folder) + " --number_of_random_folders=" + str(number_of_random_folders))
        download_err = subprocess.call(download_cmd, shell=True)
        if download_err != 0:
            raise Exception
        os.chdir(return_path)
        print('Moved back to path:', os.getcwd())

        target = target[1:-1]
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("\nTerminating.\n")


if __name__ == "__main__":
    unittest.main()
