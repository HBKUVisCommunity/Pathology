import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import PIL.Image
from scipy import misc,ndimage
from skimage.transform import resize
from tqdm import tqdm

import multiprocessing as mp
import fastai
from fastai.vision import *
from fastai.torch_core import *
import pretrainedmodels
from fastai.callbacks import *
#import fastai.callbacks.all
from fastai.vision.models.efficientnet import *
#from fastai.vision.all import *
import os
from glob import glob
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm
from palettable.colorbrewer.qualitative import Set1_3
from palettable.colorbrewer.qualitative import Set1_5
from palettable.colorbrewer.qualitative import Set2_5
import pickle

from scipy import special
import json
import sys

os.environ['QT_PLUGIN_PATH']='C:\ProgramData\Miniconda3\Library\plugins'

# setup external GPU
torch.cuda.device(1)


def bbox(contour):            
    Xmin = np.min(contour[:,0]).astype(int)
    Xmax = np.max(contour[:,0]).astype(int)
    Ymin = np.min(contour[:,1]).astype(int)
    Ymax = np.max(contour[:,1]).astype(int)
    return Xmin,Xmax,Ymin,Ymax        
            
    
def contours_in_array(arr,size,thr=92.0):
    narr = (arr[0,:]+256.0*arr[1,:]+65536.0*arr[2,:])/(float(2**24)-1.0)
    r = np.ascontiguousarray(narr).reshape(size[0],size[1]).astype(np.float32)
    contours = measure.find_contours(r,thr/255.0)
    return contours

    
def is_equal(p1,p2, eps=1e-12):
  return np.sum((p1-p2)**2)<eps

def is_closed(contour, eps=1e-12):
  return is_equal(contour[0],contour[-1],eps)


def extract_nucleus(img,cont,size=(64,64,3),scale_factor=1.0):
    
    ret = np.zeros(shape=size,dtype=img.dtype)
    nucleus = np.zeros(shape=size, dtype=img.dtype)
    Xmin,Xmax,Ymin,Ymax = bbox(cont)
    if  (Xmax-Xmin) >= (size[0]/scale_factor):
        Xmax = Xmin + int(size[0]/scale_factor -1)
    if  (Ymax-Ymin) >= (size[1]/scale_factor):
        Ymax = Ymin + int(size[1]/scale_factor -1)

    # Create an empty image to store the masked array
    r_mask = np.zeros(shape=(img.shape[0],img.shape[1]), dtype='int')
    # Create a contour image by using the contour coordinates rounded to their nearest integer value
    r_mask[np.round(cont[:, 0]).astype('int'), np.round(cont[:, 1]).astype('int')] = 1
    # Fill in the hole created by the contour boundary
    r_mask = ndimage.binary_fill_holes(r_mask)
    mask = r_mask[Xmin:Xmax,Ymin:Ymax]
    img_crop = np.copy(img[Xmin:Xmax,Ymin:Ymax,:])
    idx=(mask==0)
    img_crop[idx]=0
    
    Xres = int(scale_factor*(Xmax-Xmin))
    Yres = int(scale_factor*(Ymax-Ymin))
    img_res = resize(img_crop,(Xres,Yres),preserve_range=True).astype(img.dtype)   
    org = (int((size[0]-Xres)/2),int((size[1]-Yres)/2))
    ret[org[0]:org[0]+Xres,org[1]:org[1]+Yres,:] = img_res
    return ret,int((Xmax+Xmin)/2),int((Ymax+Ymin)/2)


def extract_contours_in_wsi_tile(u,v,contdir,border,inner=1024,size=(64,64)):
    img_size = inner+2*border
    xoff,yoff= u*(img_size-border)-border,v*(img_size-border)-border
    if u == 0: xoff = 0
    if v == 0: yoff = 0
    name = str(u) + '_' + str(v) + '.png'
    cont_path = os.path.join(contdir, name)
    img_dict = {}
    if  os.path.exists(cont_path):
        img_seg = PIL.Image.open(cont_path)
        arr_seg = np.array(img_seg.getdata()).T.astype(np.float32)
        img_cont = contours_in_array(arr_seg,img_seg.size)
        for n,c in enumerate(img_cont):
            if c.shape[0] > 32 and is_closed(c):
                Xmin,Xmax,Ymin,Ymax = bbox(c)
                if  (Xmax-Xmin) >= size[0]:
                    Xmax = Xmin + int(size[0]-1)
                if  (Ymax-Ymin) >= size[1]:
                    Ymax = Ymin + int(size[1]-1)
                # axis are inverted by measure function for extracting contours
                yhat,xhat = int((Xmax+Xmin)/2),int((Ymax+Ymin)/2)
                img_dict[(xhat+xoff,yhat+yoff)] = c
    return img_dict
    


def extract_contours_in_roi(pool, roi, contdir, border, inner = 1024):
    nuclei_dict = {}
    for u in tqdm(range(roi[0],roi[1])):
        pairs = [ ((u,v),pool.apply(extract_contours_in_wsi_tile, args=(u,v,seg_dir,border))) for v in range(roi[2],roi[3])]
        nuclei_dict.update(pairs)
    return nuclei_dict


def extract_contours_in_tiles(pool, tiles, seg_dir, border, inner = 1024):
    nuclei_dict = { (u,v):pool.apply(extract_contours_in_wsi_tile, args=(u,v,seg_dir,border)) for (u,v) in tqdm(tiles) }
    return nuclei_dict


def extract_nuclei_in_wsi_tile(u,v,imgdir,contdir,outdir,border,inner=1024):
    img_size = inner+2*border
    xoff,yoff= u*(img_size-border)-border,v*(img_size-border)-border
    if u == 0: xoff = 0
    if v == 0: yoff = 0
    name = str(u) + '_' + str(v) + '.png'
    img_path = os.path.join(imgdir, name)
    cont_path = os.path.join(contdir, name)
    img_dict = {}
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if  os.path.exists(img_path) and os.path.exists(cont_path):
        img_seg = PIL.Image.open(cont_path)
        arr_seg = np.array(img_seg.getdata()).T.astype(np.float32)
        img_cont = contours_in_array(arr_seg,img_seg.size)
        img = PIL.Image.open(img_path)
        arr = np.array(img.getdata()).reshape(img.size[0],img.size[1],3).astype(int)
        for n,c in enumerate(img_cont):
            if c.shape[0] > 32 and is_closed(c):
                nucleus,x,y = extract_nucleus(arr,c)
                yhat=x
                xhat=y
                #nuclei_pixels.append( nucleus )
                #nuclei_centers.append( (x+xoff,y+yoff))
                outfile = outdir + '/' + str(xhat+xoff)+'_'+str(yhat+yoff)+ '.jpg'
                img_dict[(xhat+xoff,yhat+yoff)] = c
                plt.imsave(outfile,np.uint8(nucleus))
    return img_dict

                                      
def annotate_with_centers(u,v,dir_in,dir_out,label_dict,inner,border,scale=1,ps=1,use_label= True, cmap ='YlOrBr',channel = 0,
                          dpi=77, palette=Set1_5):
    name = str(u) + '_' + str(v) + '.png'
    if not os.path.exists(dir_in+'/'+name):
        return
    
    img = PIL.Image.open(dir_in+'/'+name)
    fig, ax = plt.subplots(figsize=((img.size[0]+0.5)/dpi,(img.size[1]+0.5)/dpi))
    ax.set_aspect(aspect='equal')
    plt.axis('off')
    ax.imshow(img)
    
    xstart,ystart= u*(inner+border)- border, v*(inner+border)-border
    if u==0 : xstart = 0
    if v==0 : ystart = 0
    xend,yend= xstart+img.size[0],ystart+img.size[1]

    x = []
    y = []
    c = []
    
    for k in label_dict:
        csi,theta = float(k[0]/scale),float(k[1]/scale)  
        # Hack for bug in offset computing
        if float(xstart+1) < csi < float(xend-1+border) and float(ystart+1)< theta< float(yend-1+border):
            x.append(  csi-xstart - border*(csi - xstart)/(inner+2*border))
            #x.append( (csi - xstart)/float(xend-xstart) * (img.size[0]-1) )
            y.append( theta-ystart - border*(theta - ystart)/(inner+2*border))
            #y.append( (theta - ystart)/float(yend-ystart) *(img.size[1]-1))
            if use_label:
                c.append( palette.mpl_colors[label_dict[k][0]] )
            else:
                c.append(label_dict[k][1][channel])
    plt.scatter(x,y,c=c,s=ps, cmap=cmap)
    plt.savefig(dir_out+'/'+name,bbox_inches='tight',pad_inches=0)
    plt.close()
    return v
    
            
def annotate_tile(u,v,dir_in,dir_out,cont_dict,label_dict,img_size,border, lwidth=1, dpi=77, palette=Set1_5):
    name = str(u) + '_' + str(v) + '.png'
    if not os.path.exists(dir_in+'/'+name):
        return
    
    img = PIL.Image.open(dir_in+'/'+name)
    fig, ax = plt.subplots(figsize=((img.size[0]+0.5)/dpi,(img.size[1]+0.5)/dpi))
    ax.set_aspect(aspect='equal')
    plt.axis('off')
    ax.imshow(img)
#    if draw_shape and scale==1:
#    cont_dict = nuclei_dict[(u,v)]
    for k in cont_dict:
        cont = cont_dict[k]
        label = 4
        if k in label_dict: label = label_dict[k][0]
        col = palette.mpl_colors[label]                
        ax.plot(cont[:, 1], cont[:, 0], linewidth=lwidth, color=col)

   # if draw_center:
   #      xstart,ystart= u*(img_size-border)-border,v*(img_size-border)-border
   #      if u==0: xstart = 0
   #      if v==0: ystart = 0
   #      xend,yend = xstart+img.size[0],ystart+img.size[1]
   #      x = []
   #      y = []
   #      c = []
   #      for k in label_dict:
   #          if xstart< k[0]< xend and ystart < k[1] < yend:
   #              x.append( k[0] - xstart  )
   #              y.append( k[1] - ystart  )
   #              c.append( label_dict[k][1][label_dict[k][0]])
   #      plt.scatter(x,y,c=c,s=ps,cmap='YlOrRd')
    plt.savefig(dir_out+'/'+name,bbox_inches='tight',pad_inches=0)
    plt.close()
    return v


def label_tiles(cont_dict,label_dict,zoom_levels):
    
    zoom_to_tile_labels = {}
    for l in zoom_levels:
        zoom_to_tile_labels[l] = {}
    for u,v in tqdm(cont_dict):
        tile_to_cont = cont_dict[u,v]
        tile_labels = { k:label_dict[k] for k in tile_to_cont if k in label_dict }
        for l in zoom_levels:
            for t in [ (u//l,v//l),(u//l-1,v//l),(u//l,v//l-1),(u//l-1,v//l-1)]:
                if t[0] >= 0 and t[1] >= 0:
                    if t not in zoom_to_tile_labels[l]:
                        zoom_to_tile_labels[l][t] = tile_labels
                    else:
                        zoom_to_tile_labels[l][t].update(tile_labels)
    return zoom_to_tile_labels


def annotate_wsi_hierarchy(pool,roi,tiles,nuclei_dict,label_dict,config):

    maxlevel=20
    distances  = config['resolutions']
    dpi = config['dpi']
    border = config['border']
    inner = config['inner']
    imgdir = config['img_dir']
    outdir = config['ann_dir']
    img_size = inner+2*border

    levels = [l for l in np.arange(0,maxlevel) if os.path.isdir(imgdir+'/'+str(l))]
    ann_levels = [ levels[-1-d] for d in distances]
    scales = [ 2**d for d in distances] 
    zoom_to_tile_labels = label_tiles(nuclei_dict,label_dict, scales)
    for n,level in enumerate(ann_levels):
        scale = scales[n]
        dir_in  = imgdir+'/'+str(level)
        dir_out = outdir+'/'+str(level)
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        print('Annotating  level ', level, '(', scale,'X)' )
        if scale == 1:
            macro = [ pool.apply(annotate_tile, args=(u,v,dir_in,dir_out,nuclei_dict[u,v],
                                                      zoom_to_tile_labels[scale][u,v],img_size,border,config['contour_width'],config['dpi']))
                      for u,v in tqdm(tiles) if (u,v) in nuclei_dict and (u,v) in zoom_to_tile_labels[scale] ]
        else:
            ustart_meso =  roi[0]//scale
            uend_meso =  (roi[1]-1)//scale+1
            vstart_meso =  roi[2]//scale
            vend_meso =  (roi[3]-1)//scale+1
            use_label = False
            if config['channels'][n] == -1:
                use_label = True
            meso = [ pool.apply(annotate_with_centers,
                                args=(u,v,dir_in,dir_out,zoom_to_tile_labels[scale][u,v],
                                      inner,border,scale,config['scatter_sizes'][n],
                                      use_label,config['channel_color_maps'][n],config['channels'][n],config['dpi']))
                     for u,v in tqdm(zoom_to_tile_labels[scale]) if ustart_meso <= u < uend_meso and vstart_meso <= v < vend_meso
                     and (u,v) in zoom_to_tile_labels[scale] ]

   

def extract_nuclei_in_tile(u,v,imgdir,img_dict,border,inner=1024):
    img_size = inner+2*border
    xoff,yoff= u*(img_size-border)-border,v*(img_size-border)-border
    name = str(u) + '_' + str(v) + '.png'
    img_path = os.path.join(imgdir, name)
    img = PIL.Image.open(img_path)
    arr = np.array(img.getdata()).reshape(img.size[0],img.size[1],3).astype(int)
    nuclei = {}
    for k in img_dict:
        c = img_dict[k]
        if c.shape[0] > 32 and is_closed(c):
            nucleus,x,y = extract_nucleus(arr,c)
            yhat=x
            xhat=y
            nuclei[(xhat+xoff,yhat+yoff)] = nucleus
    return nuclei

def export_nuclei( nuclei, outdir ):
    for (x,y) in nuclei: 
        plt.imsave(outdir+'/'+str(x)+'_'+str(y)+'.jpg',np.uint8(nuclei[x,y]))
    return 1
        
def extract_nuclei_in_tiles(pool,tiles,imgdir,cont_dict, outdir, borde, inner=1024):

    subdivisions  = 16
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
            
    for s in range(subdivisions):
        print('Extracting batch ',s)
        nuclei = { (u,v):pool.apply(extract_nuclei_in_tile, args=(u,v,imgdir,cont_dict[u,v],border))
                   for n,(u,v) in enumerate(tqdm(tiles)) if n%subdivisions == s}
        print('Saving nuclei images')
        dummy = [ export_nuclei( nuclei[(u,v)], outdir) for (u,v) in tqdm(nuclei) ]
    

        

def center(name):
    name = name.split('.')[-2]
    name = name.split('\\')[-1]
    xy   = name.split('_')
    return int(xy[0]),int(xy[1])


def squeezenet_inference(modeldir,imgdir):

    # create images panda frame
    files  = [ [ image.split('\\')[-1], center(image)[0], center(image)[1] ]\
          for image in glob(imgdir+'/*.jpg',recursive=True)]
    test = pd.DataFrame(files,columns=['name','x','y'])
    
    try:
        learner = learner.load(Path(modeldir+'/best_testing').absolute())
        learner.path = Path(modeldir).absolute()
        learner.export()
        del learner
        del databunch
        del data
    except:
        pass  

    tester = load_learner(Path(modeldir),
                      test=ImageList.from_df(path=imgdir,
                                             df=test
                                            )
                     ).to_fp16()
    p = tester.get_preds(ds_type=DatasetType.Test)[0]
    l = np.argmax(p,axis=1)
    labels = np.unique(l)
    prob = special.softmax(np.array(p).reshape(-1,labels.size),axis=1)
    x = [int(v) for v in test['x']]
    y = [int(v) for v in test['y']]
    return [x,y,l,prob]
    


def tile(f):
    s = f.split('\\')[-1].split('.')[0].split('_')
    return int(s[0]),int(s[1])


if __name__ ==  '__main__':

    config = {}
    config_json = 'config.json'
    if len(sys.argv) > 1:
        config_json = sys.argv[1]

    print('Using config ', config_json)
    with open(config_json) as json_file:
        config = json.load(json_file)

    img_dir = config['img_dir']
    seg_dir = config['seg_dir']
    out_dir = config['out_dir'] 
    ann_dir = config['ann_dir']
    model_dir = config['model_dir']
    labels_file =  out_dir + '/'+ config['labels_file']
    contours_file = out_dir + '/' + config['contours_file']
                                           
    levels = [l for l in np.arange(0,20) if os.path.isdir(img_dir+'/'+str(l))]
    macro_level = levels[-1]
    macro_dir_in  = img_dir+'/'+str(macro_level)

    tiles = [ tile(f) for f in glob(seg_dir+'/*.png')]
    tarray = np.array(tiles).reshape(-1,2)
    roi =(np.min(tarray[:,0]),np.max(tarray[:,0])+1, np.min(tarray[:,1]),np.max(tarray[:,1])+1)
    print('Computed roi = ', roi)
                                           
    config['use_custom_roi'] = False
    config['roi'] = (30,34,40,44)

    if  config['use_custom_roi']:
        roi = (30,34,40,44)
        tiles = [ (u,v) for (u,v) in tiles if roi[0] < u < roi[1] and roi[2] < v < roi[3] ]

    config['border']= 32
    config['inner'] = 1024 
    border = config['border']
    inner = config['inner']
                                           
    pool = mp.Pool(mp.cpu_count())
    nuclei_dir = out_dir + '/' + config['nuclei_dir'] 
    if not os.path.isdir(nuclei_dir):
        os.mkdir(nuclei_dir)

    nuclei_dict = {}
    if config['extract_contours']:
       print('Contours extraction')
       nuclei_dict = extract_contours_in_tiles(pool,tiles,seg_dir, border)
       print('Saving contours  in ',contours_file)
       f = open(contours_file,'wb')
       pickle.dump(nuclei_dict,f)
       f.close()
    else:
       print('Open contours in ', contours_file)
       f = open(contours_file,'rb')
       nuclei_dict = pickle.load(f)
       f.close()

    if config['extract_nuclei']:
        print('Nuclei extraction')
        extract_nuclei_in_tiles(pool,tiles,macro_dir_in,nuclei_dict,nuclei_dir,border)
        print('Saving contours  in ',contours_file)
        f = open(contours_file,'wb')
        pickle.dump(nuclei_dict,f)
        f.close()

    if config['inference']:
        print('Squeezenet inference')
        x,y,l,p = squeezenet_inference(model_dir,nuclei_dir)
        label_dict = dict(zip(zip(x,y), zip(l,p)))
        print('Saving labels  in ',labels_file)
        f = open(labels_file,'wb')
        pickle.dump(x,f)
        pickle.dump(y,f)
        pickle.dump(l,f)
        pickle.dump(p,f)
        f.close()
    else:
        print('Open labels in ', labels_file) 
        f = open(labels_file,'rb')
        x = pickle.load(f)
        y = pickle.load(f)
        l = pickle.load(f)
        p = pickle.load(f)
        f.close()
        label_dict = dict(zip(zip(x,y), zip(l,p)))

    if config['annotations']:
        print('Annotating ',ann_dir)
        annotate_wsi_hierarchy(pool,roi,tiles,nuclei_dict,label_dict,config)

#    with open('config.json', 'w') as outfile:
#        json.dump(config, outfile)

