#!/usr/bin/env python3
#
# Python module of regression and support vector machine using random fourier features.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : October 10, 2020
##################################################### SOURCE START #####################################################

from .rfflearn_cpu_common     import seed
from .rfflearn_cpu_regression import RFFRegression, ORFRegression
from .rfflearn_cpu_svc        import RFFSVC, ORFSVC, RFFBatchSVC, ORFBatchSVC
from .rfflearn_cpu_gp         import RFFGPR, ORFGPR, RFFGPC, ORFGPC
from .rfflearn_cpu_pca        import RFFPCA, ORFPCA
from .rfflearn_cpu_cca        import RFFCCA, ORFCCA

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
