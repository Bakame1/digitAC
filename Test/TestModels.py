import sys
from Test.model2.model2 import model2_f

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
    print("Here are the pictures we couldn't read: ")
    res=""
    for i in range (len(tabPictureNotRead)):
        if i%10 == 0:
            res=res+"\n"
        res=res+"n°"+str(i+1)+" |"
    print(res)


import time



def testModel(model):
    purcentage="%"
    sumPictureRead=0
    PicturesNotRead=[]
    n=len(labels)
    start_time = time.time()
    for i in range(1,n):
        pathEachPicture='../../../Photos/Aircond/AC ('+str(i)+').jpg'
        #Testing the model on all pictures
        if model(pathEachPicture)==labels[i-1]:
            sumPictureRead=sumPictureRead+1
        else:
            PicturesNotRead.append(i)

        #Progression of the testing
        purcentage=str(int((i/n) *100))+"%"
        print_progress(i, n, prefix='Progress:', suffix=purcentage)

    end_time = time.time()
    #Time elapsed
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    #Number of pictures read correctly
    print("This model read correctly: "+str(sumPictureRead)+"/"+str(n))
    print()
    #Display the pictures the model did not manage to read
    displayPictureNotRead(PicturesNotRead)

testModel(model2_f)




