# HACK: most modules in dir have underscores => PyCall.jl doesn't import these :(.
# This allows us to access them from julia.

import mt_model
import learn_mtmodel
import mtfixb_model
import learn_mtfixbmodel
import parseopts

class dictNamespace(object):
    """
    converts a dictionary into a namespace
    """
    def __init__(self, adict):
        self.__dict__.update(adict)


