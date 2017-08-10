import pickle
import cPickle
import numpy as np
import cv2
import os
import scipy.io


def loadData( path ):
    fid = open( path, 'rb' )
    data = pickle.load( fid )
    fid.close()
    return data

def saveMat( dataList, path ):
    haha = {}
    count = 0 
    for data in dataList:
        haha['val' + str(count)] = data
        count += 1
    scipy.io.savemat( path, haha )    

def saveData( data, path ):
    fid = open( path, 'wb' )
    pickle.dump( data, fid, pickle.HIGHEST_PROTOCOL )
    fid.close()

def shuffleData( data, score ):
    assert( data.shape[0] > 0 )
    arr = np.arange( data.shape[0] )
    np.random.shuffle( arr )
    data = data[arr]
    score = score[arr]
    return [data, score]

def splitData( datascore, factor ):
    data = datascore[0]
    score = datascore[1]
    assert( data.shape[0] > 0 and factor > 0 and factor < 1 )

    splitLine = int( data.shape[0] * factor )
    trainData = data[:splitLine]
    trainScores = score[:splitLine]
    testData = data[splitLine:]
    testScores = score[splitLine:]

    data = [trainData, testData]
    score = [trainScores, testScores]
    return [data, score]
    
def normcol_equal( matin ):
    # matin./repmat(sqrt(sum(matin.*matin,1)+eps),size(matin,1),1);
    matout = matin / np.tile( np.sqrt( np.sum( matin * matin, axis = 0 ) ), ( matin.shape[0], 1 ) )
    return matout 

def saveParams( params, savedPath ):
    print 'Saving parameters...'

    fid = open( savedPath, 'wb' )

    for param_i in params:
        cPickle.dump( param_i.get_value( borrow = True ), fid, protocol = cPickle.HIGHEST_PROTOCOL )

    fid.close()
    print 'Saving finished!'

def loadParams( params, loadPath ):
    print 'Loading parameters...'

    fid = open( loadPath, 'rb' )

    for param_i in params:
        param_i.set_value( cPickle.load( fid ) )

    fid.close()
    print 'Loading finished!'
    
def saveImages( data, folderPath ):
#    print data.shape
    for i in xrange( data.shape[0] ):
        for c in xrange( data.shape[1] ):
            img = data[i, c]
            img = cv2.normalize( img, img, 0, 255, cv2.cv.CV_MINMAX )
            img = img.astype( 'uint8' )
#            print img
            cv2.imwrite( os.path.join( folderPath, str( i ) + '_' + str( c ) + '.png' ), img )
            
def saveData2File( data, path ):
    print data.shape
    assert( data.shape[0] != 0 and data.shape[1] != 0 )
    fid = open( path, 'w' )
    line = '%d %d\n' % ( data.shape[0], data.shape[1] )
    fid.writelines( line )
    for i in xrange( data.shape[0] ):
        line = ''
        for j in xrange( data.shape[1] ):
            line += '%e ' % data[i, j]
        line += '\n'
        fid.writelines( line )
    fid.close()    