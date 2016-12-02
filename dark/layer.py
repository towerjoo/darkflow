from utils import loader
import numpy as np

class Layer(object):

    def __init__(self, num, ltype, *args):
        self.signature = [ltype] + list(args)
        self.number = num
        self.type = ltype

        self.w = dict() # weights
        self.h = dict() # placeholders
        self.wshape = dict() # weight shape
        self.wsize = dict() # weight size
        self.setup(*args) # set attr up
        self.present()
        for var in self.wshape:
            shp = self.wshape[var]
            size = np.prod(shp)
            self.wsize[var] = size

    def load(self, src_loader):
        var_lay = src_loader.VAR_LAYER
        if self.type not in var_lay: return

        src_type = type(src_loader)
        if src_type is loader.weights_loader:
            wdict = self.load_weights(src_loader)
        else: 
            wdict = self.load_ckpt(src_loader)
        if wdict is not None: 
            self.recollect(wdict)

    def load_weights(self, src_loader):
        val = src_loader([self.presenter])
        if val is None: return None
        else: return val.w

    def load_ckpt(self, src_loader):
        result = dict()
        presenter = self.presenter
        for var in presenter.wshape:
            name = presenter.varsig(var)
            shape = presenter.wshape[var]
            key = [name, shape]
            val = src_loader(key)
            result[var] = val
        return result

    # For comparing two layers
    def __eq__(self, other):
        if type(other) is type(self):
            return self.signature == other.signature
        return False
    def __ne__(self, other):
        return not self.__eq__(other)

    def varsig(self, var):
        if var not in self.wshape:
            return None
        sig = str(self.number)
        sig += '-' + self.type
        sig += '/' + var
        return sig

    def recollect(self, w): self.w = w
    def present(self): 
        self.presenter = self
    def setup(self, *args): pass
    def finalize(self): pass 