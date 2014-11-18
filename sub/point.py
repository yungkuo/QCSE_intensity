# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 20:47:36 2014

@author: Philip
"""
import numpy as np
import matplotlib.pyplot as plt

scan = 5  #scan 11x11 pixels to find localmax


def pIO(mov, ax, fig):
    print("Return to Figure 1, and click points of interests")
    print("If finished, press enter")
    nrow=len(mov[1,:,1])
    ncol=len(mov[1,1,:])
    pts = plt.ginput(0,0) 
    pts=np.array(pts)
    col_pts=np.around(pts[:,0])
    row_pts=np.around(pts[:,1])
    ax.plot(col_pts,row_pts, 'r+', markersize=20)
    ax.set_xlim([0,ncol])
    ax.set_ylim([nrow,0])
    fig.canvas.draw()    
    pts=pts.astype(int)
    pts_rc=zip(row_pts, col_pts)
    
    return pts_rc
    
def localmax(refimg, pts, ax, fig):
    nrow=len(refimg[:,0])
    ncol=len(refimg[0,:])
    drow_dcol= np.zeros((len(pts[:,0]),2))
    for i in range(len(pts[:,0])):
        local=mask(refimg, pts[i,:],scan)
        #mask = refimg[pts[i,0]-scan:pts[i,0]+scan+1,pts[i,1]-scan:pts[i,1]+scan+1]
        drow_dcol[i,:]=np.array(zip(*np.where(local==local.max())))   # * unpack the tuple, return its value as an input element of zip)
    drow_dcol=drow_dcol.astype(int)-scan
    pts_new=pts+drow_dcol
    
    ax.plot(pts_new[:,1], pts_new[:,0], 'y+', markersize=20)
    ax.plot((pts[:,1]+scan, pts[:,1]-scan, pts[:,1]-scan,pts[:,1]+scan,pts[:,1]+scan), (pts[:,0]-scan, pts[:,0]-scan, pts[:,0]+scan, pts[:,0]+scan,pts[:,0]-scan), '-+', color='w')
    for n in range(len(pts[:,0])):   
        ax.annotate(n+1,xy=(pts_new[n,1], pts_new[n,0]), xytext=(pts_new[n,1], pts_new[n,0]+20),color='w')   
    ax.set_xlim([0,ncol])
    ax.set_ylim([nrow,0])
    fig.canvas.draw()
    return pts_new
    
def mask(refimg, pts, scan):
    local=refimg[pts[0]-scan:pts[0]+scan+1,pts[1]-scan:pts[1]+scan+1]
    return local
    
if __name__ == "__main__": 
    print("Your scan pixel is +-%d" %scan)    
    