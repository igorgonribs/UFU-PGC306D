#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for print decorations in python scripts

"""
# Imports

# For colors in console:
BLA='\033[0;30m'
RED='\033[0;31m'
RED1='\033[1;31m'
GRE='\033[0;32m'
GRE1='\033[1;32m'
YEL='\033[0;33m'
YEL1='\033[1;33m'
BLU='\033[0;34m'
BLU1='\033[1;34m'
MAG='\033[0;35m'
MAG1='\033[1;35m'
CYA='\033[0;36m'
CYA1='\033[1;36m'
WHI='\033[0;37m'
NC='\033[0m'

def banner(text, ch=' ', length=40)-> None:
  '''Print banner for program name'''
  print('+'+'-'*(length-2)+'+')
  text=("\033[1m"+text+"\033[0m")
  spaced_text = ' %s ' % text
  banner = "|"
  banner += spaced_text.center((length+6), ch)
  banner += "|"
  print(banner)
  print('+'+'-'*(length-2)+'+')

def line(color: str='\033[0;32m', level: int=1, text: str='exemple line' )-> None:
  '''Print colored line with level prefix char'''
  print(f"{color}", end='')
  for i in range(level):
    print("#", end='')
  print(f"> {text}\033[0m")

if __name__ == "__main__":
  import sys
  banner("Program -> Name of prog.")
  line(YEL, 1, "Exemple colored line with level")
  sys.exit(0)
