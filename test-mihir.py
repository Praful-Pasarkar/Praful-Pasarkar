#!/usr/bin/env python3
import os
FOLDER = 'C:/project_Downloads'

totalFiles = 0
for files in os.walk(FOLDER):
    for Files in files:
        totalFiles = totalFiles + 1

print('Number of files', totalFiles)