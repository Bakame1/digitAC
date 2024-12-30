import sys
from Test.model2.model2 import model2_f
from Test.model3.model3 import model3_f
from Test.model4.model4 import model4_f

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call this function in a loop to create a progress bar.

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()





labels=[
    "25","25","25","25","25","24","24","24","24","24",
    "24","23","23","23","24","25","24","25","25","24",
    "26","26","26","26","26","26","25","25","26","26",
    "26","26","20","20","20","20","22","22","22","22",
    "25","25","25","25","23","23","23","23","26","26",
    "26","26","26","25","25","25","23","23","23","23",
    "21","21","21","21","21","26","26","26","26","26",
    "26","26","26","26","26","empty","empty","empty","empty","empty",
    "empty","empty","empty","26","26","26","26","26","26","26",
    "26","26","26","26","26","26","26","26","26","26",
    "26","26","26","25","25","25","26","26","26","26",
    "26","26","26","26","26","26","26","25","25","25",
    "25","25","25","24","24","24","26","26","26","26",
    "26","26","empty","empty","24","24","24","24","24","24",
    "26","26","26","26","26","26","26","26","26","21",
    "21","21","21","21","22","22","22","22","22","25",
    "25","25","25","25","27","27","27","27","27","26",
    "26","26","26","26","25","25","25","25","26","26",
    "26","26","26","25","25","25","25","25","20.5","20.5",
    "20","20","20","20","26","26","26","26","26","26",
    "26","26","26","26","26"
]

def displayPictureNotRead(tabPictureNotRead):
    """
    Display the pictures that the model failed to read.

    Parameters:
        tabPictureNotRead (list): A list of integers representing the indices of the pictures that the model failed to read.

    Returns:
        None
    """
    print("Here are the pictures we couldn't read: ")
    res=""
    for i in range (len(tabPictureNotRead)):
        if i%10 == 0:
            res=res+"\n"
        res=res+"n°"+str(tabPictureNotRead[i])+" |"
    print(res)


import time

def testModel(model):
    """
        Test the performance of a given model on a set of images.
        accuracy = number of pictures read correctly / number of pictures
        execution time = time it took to execute the model on all pictures
        pictures not read = number of pictures that the model failed to read

        Parameters:
            model (function): The model function to be tested.

        Returns:
            None
        """
    sumPictureRead=0
    PicturesNotRead=[]
    n=len(labels)
    start_time = time.time()
    for i in range(0,n):
        pathEachPicture='../../../Photos/Aircond/AC ('+str(i+1)+').jpg'
        #Testing the model on all pictures
        if model(pathEachPicture)==labels[i]:
            sumPictureRead=sumPictureRead+1
        else:
            PicturesNotRead.append(i+1)

        #Progression of the testing
        print_progress(i+1, n, prefix='Progress:')

    end_time = time.time()
    #Time elapsed
    execution_time = end_time - start_time
    print()
    print(f"Execution time: {execution_time} seconds")

    #Number of pictures read correctly
    print("This model read correctly: "+str(sumPictureRead)+"/"+str(n))
    print()
    #Display the pictures the model did not manage to read
    displayPictureNotRead(PicturesNotRead)

#TEST MODEL 2
#testModel(model2_f)

#TEST MODEL 3 WITH UPDGRADED CROP
#testModel(model3_f)

#TEST MODEL 4 WITH UPDGRADED image processing
testModel(model4_f)





