# -*- coding: utf-8 -*-
"""
Internal cvglmnet function. See also cvglmnet.

"""
import scipy
from glmnetPredict import glmnetPredict
from wtmean import wtmean
from cvcompute import cvcompute

def cvelnet(fit, \
            lambdau, \
            x, \
            y, \
            weights, \
            offset, \
            foldid, \
            ptype, \
            grouped, \
            keep = False):
    
    typenames = {'deviance':'Mean-Squared Error', 'mse':'Mean-Squared Error', 
                 'mae':'Mean Absolute Error'}
    if ptype == 'default':
        ptype = 'mse'

    ptypeList = ['mse', 'mae', 'deviance']    
    if not ptype in ptypeList:
        print('Warning: only ', ptypeList, 'available for Gaussian models; ''mse'' used')
        ptype = 'mse'
    if len(offset) > 0:
        y = y - offset

    predmat = scipy.ones([y.size, lambdau.size])*scipy.NAN               
    nfolds = scipy.amax(foldid) + 1
    nlams = [] 
    for i in range(nfolds):
        which = foldid == i
        fitobj = fit[i].copy()
        fitobj['offset'] = False
        preds = glmnetPredict(fitobj, x[which, ])
        nlami = scipy.size(fit[i]['lambdau'])
        predmat[which, 0:nlami] = preds
        nlams.append(nlami)
    # convert nlams to scipy array
    nlams = scipy.array(nlams, dtype = scipy.integer)

    N = y.shape[0] - scipy.sum(scipy.isnan(predmat), axis = 0)
    yy = scipy.tile(y, [1, lambdau.size])

    if ptype == 'mse':
        cvraw = (yy - predmat)**2
    elif ptype == 'deviance':
        cvraw = (yy - predmat)**2
    elif ptype == 'mae':
        cvraw = scipy.absolute(yy - predmat)
        
    if y.size/nfolds < 3 and grouped == True:
        print('Option grouped=false enforced in cv.glmnet, since < 3 observations per fold')
        grouped = False
        
    if grouped == True:
        cvob = cvcompute(cvraw, weights, foldid, nlams)
        cvraw = cvob['cvraw']
        weights = cvob['weights']
        N = cvob['N']
        
    cvm = wtmean(cvraw, weights)
    sqccv = (cvraw - cvm)**2
    cvsd = scipy.sqrt(wtmean(sqccv, weights)/(N-1))

    result = dict()
    result['cvm'] = cvm
    result['cvsd'] = cvsd
    result['name'] = typenames[ptype]

    if keep:
        result['fit_preval'] = predmat
        
    return(result)

# end of cvelnet
#=========================    
