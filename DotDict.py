class DotDict(dict):
    """
    DEPRECATED -- use dictlib instead!!

    dot.notation access to dictionary attributes
    src: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    usage:
        my_dict = DotDict({"val1": 1, "val2": 2})
        print(my_dict.val1)
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
