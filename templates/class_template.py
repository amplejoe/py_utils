#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
__description__ = """
[DESCRIPTION]
"""
# =============================================================================
# @author   : [AUTHOR]
# @date     : [DATE]
# @version  : 1.0
# =============================================================================

from py_utils import utils
import numpy as np
from tqdm import tqdm


class CLASS_NAME:
    """
    [DESCRIPTION]
    """

    def __init__(self, **kwargs):
        """
            kwargs:
                arg1 (str): purpose1
        """
        # if kwargs['arg1']:
        #     pass

    pass

    @classmethod
    def get_name(self):
        qualname = getattr(CLASS_NAME, '__qualname__', 'DefaultClassName')
        return qualname
