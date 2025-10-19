#!/usr/bin/env python3

"""A simple python script template.
"""

import os
import sys
import argparse

def parseargs(parsearguments):
  parser = argparse.ArgumentParser(
           description=__doc__,
           formatter_class=argparse.RawDescriptionHelpFormatter)
  # Positional Arguments
  parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
  # Optional Arguments
  parser.add_argument('-o', '--outfile', help="Output file",
                      default=sys.stdout, type=argparse.FileType('w'))

  args = parser.parse_args(parsearguments)
  #print(args)
  return args


def main(mainarguments):
  """Main program"""
  # "#BEGINmain()#############"
  # If not using argparse, comment next line
  #args = parseargs(mainarguments)


  # "#ENDmain()###############"

if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
