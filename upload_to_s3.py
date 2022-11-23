#!/usr/bin/env python

"""
__author__ = ["Facundo Mercado, Brian Domecq"]
__contact__ = "facundomercado@deutschebahn.com"
__copyright__ = "Copyright 2022, DB Engineering & Consulting Gmbh"
__date__ = "2022/08/27"
__deprecated__ = False
__maintainer__ = "developer"
__status__ = "development"
__version__ = "0.0.1"
__Project__ = "Emova Censo"
__Python__Version__ = [3.10]
__venv__ = venv-mp
"""

if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description='Upload data to S3 bucket.')

    parser.add_argument('-i', '--input', type=str,
                        default='input file folder', help='Input folder.')

    parser.add_argument('-b', '--bucket', type=str,
                        default='S3 bucket uri', help='Bucket uri.')
    args = parser.parse_args()

    command = 'aws s3 cp {} s3://{} --recursive'.format(args.input, args.bucket)
    s3_folder_data = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
