import os.path
import pprint

class DataSetPathCall():
    def __init__(self, pathlist=['','','','']):
        #self.__homeDir = os.path.expanduser("~") + '/'
        self.__homeDir = '/db/s-sato/'
        self.__frameDir = pathlist[0]
        self.__trainDir = pathlist[0] + pathlist[1]
        self.__testDir = pathlist[0] + pathlist[2]
        
        if pathlist[3] is '' :
            self.__validationDir = ''
        else :
            self.__validationDir = pathlist[0] + pathlist[3]
        
        self.__pathlist = [self.__frameDir, self.__trainDir, self.__testDir, self.__validationDir]
    
    def setPath(self,pathlist):
        self.__frameDir = pathlist[0]
        self.__trainDir = pathlist[0] + pathlist[1]
        self.__testDir = pathlist[0] + pathlist[2]
        
        if pathlist[3] is '' :
            self.__validationDir = ''
        else :
            self.__validationDir = pathlist[0] + pathlist[3]
        
        self.__pathlist = [self.__frameDir, self.__trainDir, self.__testDir, self.__validationDir]
    
    def getHomeDir(self):
        return self.__homeDir
    
    def getFrameDir(self):
        return self.__frameDir
    
    def getTrainDir(self):
        return self.__trainDir
    
    def getTestDir(self):
        return self.__testDir
    
    def getValidationDir(self):
        return self.__validationDir
    
    def printPathList(self):
        pprint.pprint(self.__pathlist)
    
    def definePath(self, path):
        return self.getHomeDir() + path


class UCF101_PathCall(DataSetPathCall):
    def __init__(self):
        super().__init__()        
        pathlist = [
            self.getHomeDir() + 'dataset/UCF101/UCF-101_frame',
            '/train',
            '/test',
            ''
        ]
        self.setPath(pathlist)
        self.__dataListPath = self.getHomeDir() + 'dataset/UCF101/tvl1_flow/data_list.csv'
        self.__optFlowDir = self.getHomeDir() + 'dataset/UCF101/tvl1_flow'
        
    def getDataListPath(self):
        return self.__dataListPath
        
    def getOptFlowDir(self):
        return self.__optFlowDir
            
if __name__ == '__main__':
    a = UCF101_PathCall()
    a.printPathList()
    