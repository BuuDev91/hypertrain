import os
import codecs
import time

class Logger:
    """
    Class Logger is used to verbose logging

    Writes a File to /log/*.log for each execution
    """

    timeFormat = "%Y-%m-%d_%H-%M-%S"
    logLevels = ["error", "warn", "info", "debug"]
    class __impl:
        def __init__(self, directory='log', filename=None, level='info'):
            self.__logFile = self.createLogFile(directory, filename)
            self.setLogLevel(level)
          
        def getId(self):
            return id(self)
          
        def createLogFile(self, directory, filename):
            if not directory:
                directory = os.getcwd()
                
            if not filename:
                filename = "%s.log" %time.strftime(Logger.timeFormat)
            
            if not os.path.isdir(directory):
                os.mkdir(directory)

            fullpath = os.path.join(directory, filename)
            logFile = codecs.open(fullpath, "a+", "utf-8", buffering=0)
            return logFile 
        
        def setLogLevel(self, level="info"):
            self.__logLevel = Logger.logLevels.index(level)
        
        def log(self, msg, level=2):
            timeStr = time.strftime(Logger.timeFormat)
            levelStr = Logger.logLevels[level].upper()
            out = "%s [%s] %s" %(timeStr, levelStr, msg)
            if level <= self.__logLevel:
                print(out)
            self.__logFile.write(out)
            self.__logFile.write('\n')
        
        def debug(self, msg):
            self.log(msg, 3)
            
        def info(self, msg):
            self.log(msg, 2)
        
        def warn(self, msg):
            self.log(msg, 1)
        
        def error(self, msg):
            self.log(msg, 0)
        
        def logError(self, e):
            if type(e).__name__ == "str":
                self.error(e)
        
            if e.__class__:      
                msg = e.__class__.__name__
                if (e.args):
                    msg += " %s" %repr(e.args)        
                self.error(msg)

    __inst = None

    def __init__(self, directory='log', filename=None, level="info"):
        # Check whether we already have an instance
        if Logger.__inst is None:
            Logger.__inst = Logger.__impl(directory, filename, level)

        # Store instance reference in the handle
        self.__dict__["_Logger__inst"] = Logger.__inst

    # Delegate attribute getters/setters to instance
    def __getattr__(self, attr):
        return getattr(self.__inst, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__inst, attr, value)