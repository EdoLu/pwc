"""
LAWRENCE BERKELEY NATIONAL LABORATORY
RESEARCH & DEVELOPMENT, NON-COMMERCIAL USE ONLY, LICENSE

Copyright (c) 2015, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy).  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

(4) Use of the software, in source or binary form is FOR RESEARCH
& DEVELOPMENT, NON-COMMERCIAL USE, PURPOSES ONLY. All commercial use rights
for the software are hereby reserved. A separate commercial use license is
available from Lawrence Berkeley National Laboratory.

(5) In the event you create any bug fixes, patches, upgrades, updates,
modifications, derivative works or enhancements to the source code or
binary code of the software ("Enhancements") you hereby grant The Regents of
the University of California and the U.S. Government a paid-up,
non-exclusive, irrevocable, worldwide license in the Enhancements to
reproduce, prepare derivative works, distribute copies to the public,
perform publicly and display publicly, and to permit others to do so.  THIS
SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.  *** Copyright Notice *** FastKDE v1.0,
Copyright (c) 2015, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy).  All rights reserved.
If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Innovation & Partnerships Office at
IPO@lbl.gov.
NOTICE.  This software was developed under funding from the U.S. Department of Energy.  As such,
the U.S. Government has been granted for itself and others acting on its
behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, prepare derivative works, and perform publicly and
display publicly.  Beginning five (5) years after the date permission to
assert copyright is obtained from the U.S. Department of Energy, and
subject to any subsequent five (5) year renewals, the U.S. Government is
granted for itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable, worldwide license in the Software to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and
display publicly, and to permit others to do so.
****************************
"""

#!/usr/bin/env python
"""Code for sampling from a transition PDF"""
import scipy.stats as stat
from numpy import *

def underlyingFunction(x,x0=305,y0=200,yrange=4):
    """A sigmoid with a transition centered on x0"""
    return (yrange/2)*tanh(x-x0) + y0

def stochasticSample(x0=305,y0=200,yrange=4,numSamples=1e3, \
                     xmid=305,xp1=5,xp2=2, \
                     yp1 = 0, yp2 = None):
    if yp2 is None:
        #By default, have the range of the noise be larger than the range in y
        yp2 = 3*yrange
    #Generate random samples of X from the gamma distribution
    #Note: I flip the gamma distribution around here so that the upper range has a short tail
    x = -(random.gamma(xp1,xp2,int(numSamples))-xp1*xp2) + xmid
    #Generate random samples of y from x and add normally distributed noise
    y = underlyingFunction(x,x0,y0,yrange) + random.normal(loc=yp1,scale=yp2,size=numSamples)
    
    #Concatenate the paired samples together
    xy = concatenate((x[newaxis,:],y[newaxis,:]),axis=0)
    return xy

def conditionalPDF(y,x, \
                   x0=305,y0=200,yrange=4, \
                   xmid=305,xp1=5,xp2=2, \
                   yp1 = 0, yp2 = None):
    if yp2 is None:
        #By default, have the range of the noise be larger than the range in y
        yp2 = 3*yrange
        
    mu = underlyingFunction(x,x0,y0,yrange)
    
    return stat.norm.pdf(y,loc=mu,scale=yp2)

def marginalX(y,x, \
                   x0=305,y0=200,yrange=4, \
                   xmid=305,xp1=5,xp2=2, \
                   yp1 = 0, yp2 = None):
    xbar = xp1*xp2
    return stat.gamma.pdf(x0 - x + xbar,a=xp1,scale=xp2)

def jointXY(y,x, \
                   x0=305,y0=200,yrange=4, \
                   xmid=305,xp1=5,xp2=2, \
                   yp1 = 0, yp2 = None):
    #Return the product of the conditional and the marginal
    return      marginalX(y,x,x0,y0,yrange,xmid,xp1,xp2,yp1,yp2) * \
           conditionalPDF(y,x,x0,y0,yrange,xmid,xp1,xp2,yp1,yp2)

def getModeCurve(conditional,axes):
    """Extract the mode curve from a 2D conditional (assumes conditioning on axis 0)"""
    modeCurve = array([ axes[1][conditional[:,i].argmax()] for i in range(conditional.shape[1])])
    #Remove points whre all of the PDF is missing
    allMising = prod((conditional.mask),axis=0)
    modeCurve = ma.masked_where(allMising == 1,modeCurve)
    return modeCurve
