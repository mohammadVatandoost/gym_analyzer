
import os
import string
import glob


def directory_list(directory: string):
    return [x[0] for x in os.walk(directory)]
    # return os.walk(directory)


def mp4_files_list(directory: string):
    return glob.glob(directory+"/*.mp4")
