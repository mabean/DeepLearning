import sys
import os
import zipfile
import shutil
import tfrecords_converter as converter

def extract_file(path, to_directory='.'):
    opener, mode = zipfile.ZipFile, 'r'
    cwd = os.getcwd()
    os.mkdir(os.path.join(cwd, to_directory))
    os.chdir(to_directory)
    print("Extract ", path)
    try:
        file = opener(os.path.join(cwd, path), mode)
        try: file.extractall()
        finally: file.close()
    finally:
        os.chdir(cwd)

def main(argv):
    dataset_file = argv[1]
    # os.chdir('data')
    # dataset_file = 'data.zip'
    unziped_dir = "_"
    extract_file(dataset_file, unziped_dir)
    converter.convert(unziped_dir)
    shutil.rmtree(unziped_dir)

if __name__ == '__main__':
    main(sys.argv)