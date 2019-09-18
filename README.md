# deep-learning-and-optics
Deep learning training code to optimize LED illumination of cells in a microscope.


1) MATLAB code for generating microspheres and saving them to disk. 

2) Tensorflow code for learning to recognise the generated microspheres

EXPLANATION.

The Matlab files are used to generate images of microspheres, and save them to disk. I am simulating here how these microspheres would appear if they are used as samples in a microscope where the illumination unit is replaced by an 5x5 LED array. Each of the LEDs illuminate the sample independently, and I save all the images from ALL the LEDs as 1 big array. Therefore, each sample "image" that I pass to the CNN has a dimension of 200x200x25. Where the 25 is for saving all the LED images.
After saving these images, I then preprocess them using the MNIST-mine.py file. In it, I have added a function (read_data_sets_microspheres) that reads in the images from disk, then re-arranges them into a specific format for the CNN file. After this pre-processing, I then pass these images to the microspheres.py file. 

In this file, i put the brightness of each of the 25 LEDs as a variable parameter, and it is learnt on top of classifying the nuumber of spheres in the files. The LED parameter is a physical characteristic of the microscope, and hence one can use these values to set the brightness of each individual LED for optimum classification of the microspheres. 

INSTRUCTIONS

 1) Put all the codes in same directory.
    
    Change lines 30 and 32 to increase (or decrease) the number of images to be created and saved to file in the script
    alex_microsphere_13.m

 2) For macOS... 
    Change the location of the file's location to your current directory in the following lines:
   - line 234 of microspheres.py script. Change that to a directory where you want to save the training information to visualize in tensorboard.
   - line 234 of mnist_mine.py script. Change that to your current directory.
     
  3) For Windows...
   Change the location of the file's location to your current directory in the following lines: 
     - line 231 of microspheres.py script. Change that to a directory where you want to save the training information to visualize in tensorboard.
     - line 235 of mnist_mine.py script. Change that to your current directory.
  
  
  Run the script microspheres.py in cmd or terminal.
  
  
  CREDIT:
  
  Codes written by Alex Muthumbi and Roarke Horstmeyer.
