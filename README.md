# deep-learning-and-optics
Deep learning training code to optimize LED illumination of cells in a microscope.


1) MATLAB code for generating microspheres and saving them to disk. 

2) Tensorflow code for learning to recognise the generated microspheres


INSTRUCTIONS

 1) Put all the codes in same directory, and the data (already in a separate directory) into the same directory as the code.

 2) Change the location of the file's location to your current directory in the following lines:
     - line 234 of microspheres.py script. Change that to a directory where you want to save the training information to visualize in tensorboard.
     - line 298 of mnist_mine.py script. Change that to your current directory.
     
  3) Run the script microspheres.py in cmd or terminal.
  
  
  CREDIT:
  
  Codes written by Alex Muthumbi and Roarke Horstmeyer.
